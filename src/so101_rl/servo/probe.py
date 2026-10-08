"""Read-only joint probe (camera parity unit SUC, task C.2.1).

Reads the present position of the six STS3215s without the servo process, for camera calibration
(``perception.camera_calibrate``) and the real-vs-sim overlay (``perception.camera_overlay``). The arm is never
moved, and nothing is ever written to a servo:

* **No write path, structurally.** The probe talks to the bus through :class:`ReadOnlyLink`, whose whole surface
  is ``open`` / ``close`` / ``ping_all`` / ``read_block`` / ``read_u16`` / ``read_u8``. The real link
  (:class:`ScsReadOnlyLink`) is built directly on ``scservo_sdk``'s ``PortHandler`` + PING (0x01) + READ (0x02) /
  ``GroupSyncRead`` (SYNC_READ, 0x82): those three instructions and no others; it never imports lerobot (whose ``disconnect()`` disables torque by default, a register
  write) and never calls a write instruction. ``tests/test_probe.py`` checks the source for write calls and runs
  the probe on a ``FakeBus`` that records every goal and torque write: zero, and spies on
  ``PortHandler.writePort`` to assert every transmitted instruction byte is in {1, 2, 0x82}.
* **Never next to the servo process.** The probe takes the same ``SerialPortLock`` the servo process holds for its
  bus port (``servo.shm.serial_lock_path``, keyed on the resolved device path). If the servo holds it, the probe
  refuses (:class:`ProbeRefused`); while the probe holds it, a servo start fails with exit 4.
* **Torque is reported, never changed.** ``Torque_Enable`` (address 40) is read back and returned.

Usage: ``python -m so101_rl.servo.probe [--port /dev/ttyACM0] [--calibration PATH] [--reads 5]`` (pixi task
``joint-probe``) prints the ticks, the joint angles and the homing-offset check as JSON. ``--reads`` must be >= 1;
``--timeout`` bounds the whole probe in wall-clock seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, load_calibration, ticks_to_rad
from so101_rl.servo.bus import BLOCK_ADDR, BLOCK_LEN, BlockRead
from so101_rl.servo.shm import SerialPortLock, ServoAlreadyRunning

BAUD = 1_000_000
ADDR_HOMING_OFFSET = 31  # 2 bytes, sign bit 11 (STS series)
ADDR_TORQUE_ENABLE = 40  # 1 byte
_SIGN_BIT_POS = 15
_SIGN_BIT_HOMING = 11
_OFF = {"pos": 0, "vel": 2, "load": 4, "volt": 6, "temp": 7, "status": 9, "moving": 10}
DEFAULT_TIMEOUT_S = 10.0  # wall-clock bound on the whole probe (ping + reads); a wedged bus must not hang the operator
MAX_SPREAD_TICKS = 3  # present-position spread over the reads above which the arm is "moving"


class ProbeRefused(RuntimeError):
    """The probe will not run (the servo process owns the port, a motor is silent, reads fail)."""


def _decode_sm(value: int, sign_bit: int) -> int:
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if (value >> sign_bit) & 1 else magnitude


@runtime_checkable
class ReadOnlyLink(Protocol):
    """Everything the probe may do on the bus. There is deliberately no write method."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def ping_all(self) -> dict[int, bool]: ...

    def read_block(self, out: BlockRead) -> bool: ...

    def read_u16(self, address: int) -> dict[int, int] | None: ...

    def read_u8(self, address: int) -> dict[int, int] | None: ...


class ScsReadOnlyLink:
    """``scservo_sdk`` PING + SYNC_READ on one port. Holds the packet handler privately; exposes reads only."""

    def __init__(self, port: str, ids: tuple[int, ...], *, baud: int = BAUD) -> None:
        self._port_name = port
        self._ids = tuple(int(i) for i in ids)
        self._baud = int(baud)
        self._port = None
        self._ph = None
        self._scs = None

    def open(self) -> None:
        import scservo_sdk as scs
        import serial

        port = scs.PortHandler(self._port_name)
        try:
            opened = port.openPort()
        except (serial.SerialException, OSError) as e:  # no such device, permission denied, port busy
            raise ProbeRefused(f"cannot open {self._port_name}: {e}") from e
        if not opened:
            raise ProbeRefused(f"cannot open {self._port_name}")
        try:
            baud_ok = port.setBaudRate(self._baud)
        except (serial.SerialException, OSError) as e:
            port.closePort()
            raise ProbeRefused(f"cannot set {self._baud} baud on {self._port_name}: {e}") from e
        if not baud_ok:
            port.closePort()
            raise ProbeRefused(f"cannot set {self._baud} baud on {self._port_name}")
        self._scs, self._port, self._ph = scs, port, scs.PacketHandler(0)

    def close(self) -> None:
        if self._port is not None:
            port, self._port = self._port, None
            port.closePort()  # closing the port sends nothing to the servos

    def ping_all(self) -> dict[int, bool]:
        assert self._port is not None and self._ph is not None
        out = {}
        for i in self._ids:
            _model, comm, _err = self._ph.ping(self._port, i)
            out[i] = comm == self._scs.COMM_SUCCESS
        return out

    def _sync_read(self, address: int, length: int):
        assert self._port is not None and self._ph is not None
        reader = self._scs.GroupSyncRead(self._port, self._ph, address, length)
        for i in self._ids:
            reader.addParam(i)
        try:
            comm = reader.txRxPacket()
        except (OSError, ConnectionError):
            return None
        if comm != self._scs.COMM_SUCCESS:
            return None
        if not all(reader.isAvailable(i, address, length) for i in self._ids):
            return None
        return reader

    def read_block(self, out: BlockRead) -> bool:
        reader = self._sync_read(BLOCK_ADDR, BLOCK_LEN)
        if reader is None:
            out.ok = False
            return False
        for j, i in enumerate(self._ids):
            g = reader.getData
            out.position[j] = _decode_sm(g(i, BLOCK_ADDR + _OFF["pos"], 2), _SIGN_BIT_POS)
            out.velocity[j] = _decode_sm(g(i, BLOCK_ADDR + _OFF["vel"], 2), _SIGN_BIT_POS)
            out.load[j] = _decode_sm(g(i, BLOCK_ADDR + _OFF["load"], 2), 10)
            out.voltage_dv[j] = g(i, BLOCK_ADDR + _OFF["volt"], 1)
            out.temperature_c[j] = g(i, BLOCK_ADDR + _OFF["temp"], 1)
            out.status[j] = g(i, BLOCK_ADDR + _OFF["status"], 1)
            out.moving[j] = g(i, BLOCK_ADDR + _OFF["moving"], 1)
        out.ok = True
        return True

    def read_u16(self, address: int) -> dict[int, int] | None:
        reader = self._sync_read(address, 2)
        return None if reader is None else {i: int(reader.getData(i, address, 2)) for i in self._ids}

    def read_u8(self, address: int) -> dict[int, int] | None:
        reader = self._sync_read(address, 1)
        return None if reader is None else {i: int(reader.getData(i, address, 1)) for i in self._ids}


class BusReadAdapter:
    """A ``servo.bus.Bus`` (e.g. ``FakeBus``) seen through :class:`ReadOnlyLink`: only ``connect`` / ``disconnect``
    / ``ping_all`` / ``read_block`` / ``read_reg`` of the bus are ever called. Used by the tests."""

    def __init__(self, bus, ids: tuple[int, ...]) -> None:
        self._bus = bus
        self._ids = tuple(ids)

    def open(self) -> None:
        self._bus.connect()

    def close(self) -> None:
        self._bus.disconnect()

    def ping_all(self) -> dict[int, bool]:
        return self._bus.ping_all()

    def read_block(self, out: BlockRead) -> bool:
        return self._bus.read_block(out)

    def read_u16(self, address: int) -> dict[int, int] | None:
        if address != ADDR_HOMING_OFFSET:
            return None
        out = {}
        for i in self._ids:  # the decoded register value, re-encoded as the servo sends it (sign-magnitude, bit 11)
            v = int(self._bus.read_reg("Homing_Offset", i))
            out[i] = (1 << _SIGN_BIT_HOMING) | -v if v < 0 else v
        return out

    def read_u8(self, address: int) -> dict[int, int] | None:
        if address != ADDR_TORQUE_ENABLE:
            return None
        return {i: int(self._bus.read_reg("Torque_Enable", i)) for i in self._ids}


@dataclass(frozen=True, eq=False)
class ProbeResult:
    ticks: np.ndarray  # int32[6], median over the reads, JOINT_NAMES order
    spread_ticks: np.ndarray  # int32[6], max - min over the reads
    q_rad: np.ndarray  # float64[6]: arm joints by ticks_to_rad; the gripper entry is NaN (needs sim.gripper_map)
    torque_enabled: dict[str, int] | None
    homing_offset: dict[str, int] | None  # read back from the servos (decoded)
    homing_matches_calibration: bool | None
    temperature_c: np.ndarray
    voltage_dv: np.ndarray
    status: np.ndarray
    n_reads: int

    def to_json(self) -> dict:
        return {
            "joints": list(JOINT_NAMES),
            "ticks": self.ticks.tolist(),
            "spread_ticks": self.spread_ticks.tolist(),
            "q_rad_arm": [None if not np.isfinite(x) else float(x) for x in self.q_rad],
            "torque_enabled": self.torque_enabled,
            "homing_offset": self.homing_offset,
            "homing_matches_calibration": self.homing_matches_calibration,
            "temperature_c": self.temperature_c.tolist(),
            "voltage_dv": self.voltage_dv.tolist(),
            "status": self.status.tolist(),
            "n_reads": self.n_reads,
        }


def probe_joints(
    calibration: dict[str, MotorCalibration],
    *,
    port: str,
    n_reads: int = 5,
    link_factory: Callable[[str, tuple[int, ...]], ReadOnlyLink] | None = None,
    require_still: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ProbeResult:
    """Ping and block-read the six motors (see the module docstring). Raises :class:`ProbeRefused` when the servo
    process owns ``port``, a motor does not answer, a read fails, or (``require_still``) the arm moves."""
    if n_reads < 1:
        raise ValueError("n_reads must be >= 1")
    if not timeout_s > 0:
        raise ValueError("timeout_s must be > 0")
    deadline = time.monotonic() + float(timeout_s)

    def _check_deadline(what: str) -> None:
        if time.monotonic() > deadline:
            raise ProbeRefused(f"probe timed out after {timeout_s:g} s ({what})")

    ids = tuple(calibration[n].id for n in JOINT_NAMES)
    try:
        lock = SerialPortLock(port)
    except ServoAlreadyRunning as e:
        raise ProbeRefused(f"the servo process owns {port}; the probe never shares the bus ({e})") from e
    try:
        link = (link_factory or (lambda p, i: ScsReadOnlyLink(p, i)))(port, ids)
        link.open()
        try:
            pings = link.ping_all()
            _check_deadline("ping")
            silent = [i for i in ids if not pings.get(i, False)]
            if silent:
                raise ProbeRefused(f"motors {silent} do not answer a ping (servo power off?)")
            blocks = []
            for _ in range(n_reads):
                b = BlockRead()
                if not link.read_block(b):
                    raise ProbeRefused("block read failed")
                blocks.append(b)
                _check_deadline(f"block read {len(blocks)}/{n_reads}")
            raw_h = link.read_u16(ADDR_HOMING_OFFSET)
            raw_t = link.read_u8(ADDR_TORQUE_ENABLE)
            _check_deadline("register reads")
        finally:
            link.close()
    finally:
        lock.release()
    pos = np.stack([b.position for b in blocks]).astype(np.int64)
    ticks = np.rint(np.median(pos, axis=0)).astype(np.int32)
    spread = (pos.max(axis=0) - pos.min(axis=0)).astype(np.int32)
    if require_still and int(spread.max()) > MAX_SPREAD_TICKS:
        raise ProbeRefused(f"the arm moved during the probe (spread {spread.tolist()} ticks)")
    q = ticks_to_rad(ticks, calibration)
    q[N_JOINTS - 1] = np.nan  # gripper: not mid-range centred (sim.gripper_map converts it)
    by_id = {calibration[n].id: n for n in JOINT_NAMES}
    homing = None if raw_h is None else {by_id[i]: _decode_sm(v, _SIGN_BIT_HOMING) for i, v in raw_h.items()}
    torque = None if raw_t is None else {by_id[i]: int(v) for i, v in raw_t.items()}
    match = None if homing is None else all(homing[n] == calibration[n].homing_offset for n in JOINT_NAMES)
    last = blocks[-1]
    return ProbeResult(
        ticks=ticks,
        spread_ticks=spread,
        q_rad=q,
        torque_enabled=torque,
        homing_offset=homing,
        homing_matches_calibration=match,
        temperature_c=last.temperature_c.copy(),
        voltage_dv=last.voltage_dv.copy(),
        status=last.status.copy(),
        n_reads=n_reads,
    )


def checked_ticks(res: ProbeResult, *, require_torque_off: bool = False) -> np.ndarray:
    """The probed ticks, refused (:class:`ProbeRefused`) when the servos' homing offsets differ from the
    calibration file (the joint angles would be wrong) or, with ``require_torque_off``, when any motor has torque on
    (the operator is about to move the arm by hand)."""
    if res.homing_matches_calibration is False:
        raise ProbeRefused(f"the servos' homing offsets {res.homing_offset} differ from the calibration file")
    if require_torque_off and (res.torque_enabled is None or any(res.torque_enabled.values())):
        raise ProbeRefused(f"torque must be off to move the arm by hand (Torque_Enable {res.torque_enabled})")
    return res.ticks


def main(argv: list[str] | None = None) -> int:
    from so101_rl.config import load_config

    ap = argparse.ArgumentParser(prog="python -m so101_rl.servo.probe", description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=None, help="default: so101.yaml port")
    ap.add_argument("--calibration", default=None, help="default: so101.yaml calibration_path")
    ap.add_argument("--reads", type=int, default=5, help="block reads to take (>= 1)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="wall-clock bound for the probe, s")
    args = ap.parse_args(argv)
    if args.reads < 1:
        ap.error("--reads must be >= 1")
    cfg = load_config(calibration=load_calibration(Path(args.calibration)) if args.calibration else None)
    cal = load_calibration(Path(args.calibration or cfg.calibration_path).expanduser())
    try:
        res = probe_joints(cal, port=args.port or cfg.port, n_reads=args.reads, timeout_s=args.timeout)
    except ProbeRefused as e:
        print(f"probe refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps(res.to_json(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

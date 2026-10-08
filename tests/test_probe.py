"""Read-only joint probe (camera parity C.2.1): no write path, zero writes on a FakeBus, refuses next to the servo."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES, load_calibration, ticks_to_rad
from so101_rl.servo import probe as probe_mod
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.probe import BusReadAdapter, ProbeRefused, ReadOnlyLink, ScsReadOnlyLink, probe_joints
from so101_rl.servo.shm import SerialPortLock
from so101_rl.servo.timing import FakeClock

CAL = load_calibration(Path(__file__).parent / "data" / "so101_follower.calib.json")
TICKS = [2050, 2200, 2301, 2745, 2040, 2065]
WRITE_NAMES = {"write", "set_torque", "enable_torque", "disable_torque", "action", "reboot", "factory", "regwrite", "syncwrite"}


class SpyBus:
    """Records every attribute of the bus that is touched (a write would show up here)."""

    def __init__(self, bus):
        self._bus = bus
        self.touched: list[str] = []

    def __getattr__(self, name):
        self.touched.append(name)
        return getattr(self._bus, name)


def _port(tmp_path):
    p = tmp_path / "ttyFAKE"
    p.write_text("")
    return str(p)


def _fake(**kw):
    return FakeBus(FakeClock(), calibration=CAL, initial_ticks=TICKS, **kw)


def test_probe_on_fakebus_reads_ticks_and_writes_nothing(tmp_path):
    bus = _fake()
    spy = SpyBus(bus)
    ids = tuple(CAL[n].id for n in JOINT_NAMES)
    res = probe_joints(CAL, port=_port(tmp_path), link_factory=lambda p, i: BusReadAdapter(spy, ids))
    assert res.ticks.tolist() == TICKS
    assert res.spread_ticks.max() == 0
    np.testing.assert_allclose(res.q_rad[:5], ticks_to_rad(np.array(TICKS), CAL)[:5])
    assert np.isnan(res.q_rad[5])
    assert res.homing_matches_calibration is True
    assert res.homing_offset == {n: CAL[n].homing_offset for n in JOINT_NAMES}
    assert res.torque_enabled == dict.fromkeys(JOINT_NAMES, 0)
    # zero writes: no goal, no torque event, and no write-like method of the bus was even looked up
    assert bus.written_goals == [] and bus.torque_events == []
    assert not any(any(w in t.lower() for w in WRITE_NAMES) for t in spy.touched), spy.touched
    assert set(spy.touched) <= {"connect", "disconnect", "ping_all", "read_block", "read_reg"}


def test_probe_refuses_when_the_servo_process_holds_the_port(tmp_path):
    port = _port(tmp_path)
    lock = SerialPortLock(port)  # what servo.process.run holds for its bus port
    try:
        bus = _fake()
        with pytest.raises(ProbeRefused, match="servo process owns"):
            probe_joints(CAL, port=port, link_factory=lambda p, i: BusReadAdapter(bus, i))
        assert bus.pings == 0  # refused before any bus traffic
    finally:
        lock.release()


def test_probe_holds_the_port_lock_while_running(tmp_path):
    port = _port(tmp_path)
    seen = []

    class Link(BusReadAdapter):
        def ping_all(self):
            from so101_rl.servo.shm import ServoAlreadyRunning

            with pytest.raises(ServoAlreadyRunning):
                SerialPortLock(port)  # a servo started now would exit 4
            seen.append(True)
            return super().ping_all()

    bus = _fake()
    probe_joints(CAL, port=port, link_factory=lambda p, i: Link(bus, i))
    assert seen == [True]
    SerialPortLock(port).release()  # released afterwards


def test_silent_motor_and_failed_read_are_refused(tmp_path):
    bus = _fake()
    bus.faults.drop_motor(3)
    with pytest.raises(ProbeRefused, match=r"\[3\]"):
        probe_joints(CAL, port=_port(tmp_path), link_factory=lambda p, i: BusReadAdapter(bus, i))
    bus = _fake()
    bus.faults.fail_next_read(1)
    with pytest.raises(ProbeRefused, match="block read failed"):
        probe_joints(CAL, port=_port(tmp_path), link_factory=lambda p, i: BusReadAdapter(bus, i))


def test_moving_arm_is_refused(tmp_path):
    class Moving(BusReadAdapter):
        k = 0

        def read_block(self, out):
            ok = super().read_block(out)
            Moving.k += 1
            out.position[1] += 10 * Moving.k
            return ok

    bus = _fake()
    with pytest.raises(ProbeRefused, match="moved"):
        probe_joints(CAL, port=_port(tmp_path), link_factory=lambda p, i: Moving(bus, i))


def test_link_surface_has_no_write_method():
    for cls in (ReadOnlyLink, ScsReadOnlyLink, BusReadAdapter):
        public = [n for n, _ in inspect.getmembers(cls) if not n.startswith("_")]
        assert not [n for n in public if any(w in n.lower() for w in WRITE_NAMES)], (cls, public)


def test_probe_source_never_calls_a_write_and_never_imports_lerobot():
    tree = ast.parse(Path(probe_mod.__file__).read_text())
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            calls.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            assert not any("lerobot" in m or "feetech_bus" in m for m in mods), mods
    bad = {c for c in calls if any(w in c.lower() for w in WRITE_NAMES)}
    assert not bad, bad
    # the scservo instructions used: PING and SYNC_READ only
    assert {"ping", "GroupSyncRead", "txRxPacket"} <= calls


def test_checked_ticks_refuses_wrong_homing_and_torque_on_for_hand_moves(tmp_path):
    from dataclasses import replace

    from so101_rl.servo.probe import checked_ticks

    bus = _fake()
    res = probe_joints(CAL, port=_port(tmp_path), link_factory=lambda p, i: BusReadAdapter(bus, i))
    assert checked_ticks(res, require_torque_off=True).tolist() == TICKS
    with pytest.raises(ProbeRefused, match="homing"):
        checked_ticks(replace(res, homing_matches_calibration=False))
    on = replace(res, torque_enabled={**res.torque_enabled, "elbow_flex": 1})
    assert checked_ticks(on).tolist() == TICKS  # one pose: the arm is not moved
    with pytest.raises(ProbeRefused, match="torque must be off"):
        checked_ticks(on, require_torque_off=True)


# ---- byte-level spy, serial errors, --reads 0, wall-clock timeout (verification-review NITs) ---------------------
def _spy_on_port_handler(monkeypatch):
    """Patch ``scservo_sdk.PortHandler`` so nothing touches a device: open/close succeed, every transmitted packet is
    recorded, nothing ever answers (every read times out)."""
    import scservo_sdk as scs

    sent: list[bytes] = []
    monkeypatch.setattr(scs.PortHandler, "openPort", lambda self: True)
    monkeypatch.setattr(scs.PortHandler, "closePort", lambda self: None)
    monkeypatch.setattr(scs.PortHandler, "setBaudRate", lambda self, b: True)
    monkeypatch.setattr(scs.PortHandler, "clearPort", lambda self: None)
    monkeypatch.setattr(scs.PortHandler, "readPort", lambda self, n: [])
    monkeypatch.setattr(scs.PortHandler, "writePort", lambda self, packet: (sent.append(bytes(packet)), len(packet))[1])
    return sent


def test_every_transmitted_instruction_byte_is_ping_read_or_sync_read(monkeypatch):
    sent = _spy_on_port_handler(monkeypatch)
    ids = tuple(CAL[n].id for n in JOINT_NAMES)
    link = ScsReadOnlyLink("/dev/null-not-opened", ids)
    link.open()
    link.ping_all()
    link.read_block(__import__("so101_rl.servo.bus", fromlist=["BlockRead"]).BlockRead())
    link.read_u16(31)
    link.read_u8(40)
    link.close()
    assert sent, "the spy saw no packet: it is not wired to the transmit path"
    for pkt in sent:
        assert pkt[:2] == b"\xff\xff"
        assert pkt[4] in {0x01, 0x02, 0x82}, f"instruction byte {pkt[4]:#x} in {pkt.hex()}"
    assert {p[4] for p in sent} >= {0x01, 0x82}  # PING and SYNC_READ both exercised


def test_the_byte_spy_catches_a_write_instruction(monkeypatch):
    """Shown-to-fail twin: a WRITE (0x03) / SYNC_WRITE (0x83) packet through the same port handler is caught."""
    import scservo_sdk as scs

    sent = _spy_on_port_handler(monkeypatch)
    ph, port = scs.PacketHandler(0), scs.PortHandler("/dev/null-not-opened")
    ph.write1ByteTxOnly(port, 1, 40, 0)  # torque off: a register write
    assert sent and sent[0][4] not in {0x01, 0x02, 0x82}


def test_serial_exception_on_open_is_a_probe_refusal(monkeypatch):
    import scservo_sdk as scs
    import serial

    def boom(self):
        raise serial.SerialException("could not open port /dev/ttyACM0: [Errno 13] Permission denied")

    monkeypatch.setattr(scs.PortHandler, "openPort", boom)
    with pytest.raises(ProbeRefused, match="Permission denied"):
        ScsReadOnlyLink("/dev/ttyACM0", (1, 2, 3, 4, 5, 6)).open()


def test_reads_zero_is_rejected_cleanly(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        probe_mod.main(["--reads", "0", "--port", _port(tmp_path), "--calibration", str(Path(__file__).parent / "data" / "so101_follower.calib.json")])
    assert e.value.code == 2
    assert "--reads must be >= 1" in capsys.readouterr().err
    with pytest.raises(ValueError, match="n_reads"):
        probe_joints(CAL, port=_port(tmp_path), n_reads=0, link_factory=lambda p, i: BusReadAdapter(_fake(), i))


def test_probe_has_a_wall_clock_timeout(tmp_path, monkeypatch):
    clock = iter(range(0, 1000, 5))  # every monotonic() call advances 5 s
    monkeypatch.setattr(probe_mod.time, "monotonic", lambda: float(next(clock)))
    bus = _fake()
    with pytest.raises(ProbeRefused, match="timed out"):
        probe_joints(CAL, port=_port(tmp_path), timeout_s=7.0, link_factory=lambda p, i: BusReadAdapter(bus, i))
    with pytest.raises(ValueError, match="timeout_s"):
        probe_joints(CAL, port=_port(tmp_path), timeout_s=0, link_factory=lambda p, i: BusReadAdapter(bus, i))

"""Real STS3215 bus over lerobot 0.6.1 (plan F2, Task 1.2).

Only the servo process may construct this. The 30 Hz loop drives the SDK directly: ONE ``GroupSyncRead`` of the
11 bytes at 56..66 per tick (position, velocity, load, voltage, temperature, Status, Moving), decoded here
because the raw ``GroupSyncRead`` path bypasses lerobot's sign handling. ``disconnect()`` never changes torque;
the safe-state sequence owns that.

Register layout of the block (offset from 56): 0 Present_Position(2), 2 Present_Velocity(2), 4 Present_Load(2),
6 Present_Voltage(1), 7 Present_Temperature(1), 9 Status(1), 10 Moving(1). Byte 8 (address 64) is unused.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, So101Config
from so101_rl.servo.bus import BLOCK_ADDR, BLOCK_LEN, BlockRead
from so101_rl.servo.timing import Clock, RealClock

_SIGN_BIT_16 = 15  # Present_Position, Present_Velocity, Goal_Position
_SIGN_BIT_LOAD = 10
_TICK_MAX = 4095
_OFF = {"pos": 0, "vel": 2, "load": 4, "volt": 6, "temp": 7, "status": 9, "moving": 10}
_ROWS = ("position", "velocity", "load", "voltage_dv", "temperature_c", "status", "moving")


def _decode_sm(value: int, sign_bit: int) -> int:
    """Sign-magnitude decode (same definition as ``lerobot.motors.encoding_utils.decode_sign_magnitude``)."""
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if (value >> sign_bit) & 1 else magnitude


class FeetechBus:
    def __init__(
        self,
        cfg: So101Config,
        calibration: dict[str, MotorCalibration],
        *,
        clock: Clock | None = None,
    ) -> None:
        # Imported lazily: lerobot is heavy and absent from envs that only run the fake backends.
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import MotorCalibration as LeMotorCalibration

        self._cfg = cfg
        self._calibration = calibration
        self._clock: Clock = clock if clock is not None else RealClock()
        self._names = JOINT_NAMES
        self._ids = [calibration[n].id for n in JOINT_NAMES]
        self._name_of_id = {calibration[n].id: n for n in JOINT_NAMES}
        motors = {n: Motor(calibration[n].id, "sts3215", MotorNormMode.RANGE_M100_100) for n in JOINT_NAMES}
        le_calib = {
            n: LeMotorCalibration(
                id=c.id,
                drive_mode=c.drive_mode,
                homing_offset=c.homing_offset,
                range_min=c.range_min,
                range_max=c.range_max,
            )
            for n, c in calibration.items()
        }
        self._bus = FeetechMotorsBus(port=cfg.port, motors=motors, calibration=le_calib)
        self._reader = None
        self._scratch = np.zeros((len(_ROWS), N_JOINTS), dtype=np.int32)

    # -- lifecycle -----------------------------------------------------------------------------
    def connect(self) -> None:
        self._bus.connect(handshake=True)
        self._build_reader()

    def _build_reader(self) -> None:
        import scservo_sdk as scs

        self._scs = scs
        reader = scs.GroupSyncRead(self._bus.port_handler, self._bus.packet_handler, BLOCK_ADDR, BLOCK_LEN)
        for motor_id in self._ids:
            reader.addParam(motor_id)
        self._reader = reader

    def disconnect(self) -> None:
        """Closes the port. Deliberately leaves torque alone (lerobot's default would disable it)."""
        if self._bus.is_connected:
            self._bus.disconnect(disable_torque=False)
        self._reader = None

    # -- Bus protocol --------------------------------------------------------------------------
    def ping_all(self) -> dict[int, bool]:
        return {motor_id: self._bus.ping(motor_id) is not None for motor_id in self._ids}

    def read_block(self, out: BlockRead) -> bool:
        if self._reader is None:
            raise RuntimeError("FeetechBus.read_block before connect()")
        reader = self._reader
        try:
            comm = reader.txRxPacket()
        except (OSError, ConnectionError):
            comm = -1
        t_ns = self._clock.monotonic_ns()
        if comm != self._scs.COMM_SUCCESS:
            out.ok = False
            return False
        s = self._scratch
        for j, motor_id in enumerate(self._ids):
            if not reader.isAvailable(motor_id, BLOCK_ADDR, BLOCK_LEN):
                out.ok = False  # missing id (no reply): keep the PREVIOUS values
                return False
            s[0, j] = _decode_sm(reader.getData(motor_id, BLOCK_ADDR + _OFF["pos"], 2), _SIGN_BIT_16)
            s[1, j] = _decode_sm(reader.getData(motor_id, BLOCK_ADDR + _OFF["vel"], 2), _SIGN_BIT_16)
            s[2, j] = _decode_sm(reader.getData(motor_id, BLOCK_ADDR + _OFF["load"], 2), _SIGN_BIT_LOAD)
            s[3, j] = reader.getData(motor_id, BLOCK_ADDR + _OFF["volt"], 1)
            s[4, j] = reader.getData(motor_id, BLOCK_ADDR + _OFF["temp"], 1)
            s[5, j] = reader.getData(motor_id, BLOCK_ADDR + _OFF["status"], 1)
            s[6, j] = reader.getData(motor_id, BLOCK_ADDR + _OFF["moving"], 1)
        for row, name in enumerate(_ROWS):
            getattr(out, name)[:] = s[row]
        out.t_mono_ns = t_ns
        out.ok = True
        return True

    def write_goal(self, goal_ticks: np.ndarray) -> bool:
        goal = np.asarray(goal_ticks)
        if goal.shape != (N_JOINTS,):
            raise ValueError(f"goal_ticks must have shape ({N_JOINTS},), got {goal.shape}")
        if goal.min() < 0 or goal.max() > _TICK_MAX:
            raise ValueError(f"goal_ticks outside 0..{_TICK_MAX}: {goal}")
        values = {n: int(goal[j]) for j, n in enumerate(self._names)}
        try:
            self._bus.sync_write("Goal_Position", values, normalize=False)
        except (ConnectionError, OSError):
            return False
        return True

    def set_torque(self, enabled: bool, ids: Sequence[int] | None = None) -> bool:
        names = None if ids is None else [self._name_of_id[i] for i in ids]
        try:
            if enabled:
                self._bus.enable_torque(names)
            else:
                self._bus.disable_torque(names)
        except (ConnectionError, OSError, RuntimeError):
            return False
        return True

    def read_reg(self, name: str, motor_id: int) -> int:
        return int(self._bus.read(name, self._name_of_id[motor_id], normalize=False))

    def write_reg(self, name: str, motor_id: int, value: int) -> None:
        self._bus.write(name, self._name_of_id[motor_id], int(value), normalize=False)

    def read_calibration_regs(self) -> dict[str, MotorCalibration]:
        offsets = self._bus.sync_read("Homing_Offset", normalize=False)
        mins = self._bus.sync_read("Min_Position_Limit", normalize=False)
        maxs = self._bus.sync_read("Max_Position_Limit", normalize=False)
        return {
            n: MotorCalibration(
                id=self._calibration[n].id,
                drive_mode=self._calibration[n].drive_mode,  # not stored on the servo; from config
                homing_offset=int(offsets[n]),
                range_min=int(mins[n]),
                range_max=int(maxs[n]),
            )
            for n in self._names
        }

    # -- configuration (not part of the Bus protocol) ------------------------------------------
    def write_registers(self, registers=None) -> None:
        """Write EVERY F1 register to every motor, including the ones LeRobot never writes
        (``Protection_Current``, ``Overload_Torque``, ``Over_Current_Protection_Time``).

        Refuses while any motor has torque on: these are EEPROM registers and a failed/ignored write on a live
        arm must not look like success. Leaves ``Lock`` at 0; ``set_torque(True)`` sets it back to 1.
        """
        reg = registers if registers is not None else self._cfg.registers
        for n in self._names:
            if self._bus.read("Torque_Enable", n, normalize=False):
                raise RuntimeError(f"write_registers requires torque off; {n} has torque enabled")
        for n in self._names:
            self._bus.write("Lock", n, 0, normalize=False)
        self._bus.configure_motors(
            return_delay_time=reg.return_delay_time,
            maximum_acceleration=reg.maximum_acceleration,
            acceleration=reg.acceleration,
        )
        done = {"return_delay_time", "maximum_acceleration", "acceleration"}
        for f in dataclasses.fields(reg):
            if f.name in done:
                continue
            table_name = "_".join(w.capitalize() for w in f.name.split("_"))
            for n in self._names:
                self._bus.write(table_name, n, int(getattr(reg, f.name)), normalize=False)

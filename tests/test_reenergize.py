"""Re-energise order (plan task 2.4). Uses a minimal local Bus double with a call log (not U1's FakeBus)."""

from dataclasses import asdict, replace
from itertools import pairwise

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration
from so101_rl.safety.calib_check import CalibrationMismatch
from so101_rl.safety.reenergize import reenergize, register_name
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.timing import FakeClock

S = 1_000_000_000


class LoggingBus:
    """Ideal plant: position jumps to the written goal; every Bus call is appended to ``log``."""

    def __init__(self, calibration: dict[str, MotorCalibration], clock: FakeClock, present=None):
        self.clock = clock
        self.cal = calibration
        self.ids = [calibration[n].id for n in JOINT_NAMES]
        self.position = np.array(present if present is not None else [1800, 2200, 2000, 2100, 2047, 2300], dtype=np.int32)
        self.goal_reg = {i: 0 for i in self.ids}
        self.torque = False
        self.log: list[str] = []
        self.regs: dict[tuple[str, int], int] = {}
        self.missing_until_ns = 0  # ping_all reports motor 4 missing until this time
        self.corrupt_offset: str | None = None
        self.block_fail_until_ns = 0
        self.goal_readback_error = False

    def ping_all(self):
        self.log.append("ping_all")
        gone = self.clock.monotonic_ns() < self.missing_until_ns
        return {i: not (gone and i == 4) for i in self.ids}

    def read_calibration_regs(self):
        self.log.append("read_calibration_regs")
        out = dict(self.cal)
        if self.corrupt_offset:
            c = out[self.corrupt_offset]
            out[self.corrupt_offset] = replace(c, homing_offset=c.homing_offset + 1)
        return out

    def write_reg(self, name, motor_id, value):
        self.log.append(f"write_reg:{name}")
        self.regs[(name, motor_id)] = value

    def read_block(self, out: BlockRead):
        self.log.append("read_block")
        if self.clock.monotonic_ns() < self.block_fail_until_ns:
            out.ok = False
            return False
        out.position[...] = self.position
        out.ok = True
        return True

    def write_goal(self, goal):
        self.log.append("write_goal")
        assert goal.dtype == np.int32 and goal.min() >= 0 and goal.max() <= 4095
        for i, g in zip(self.ids, goal, strict=True):
            self.goal_reg[i] = int(g)
        self.position[...] = goal
        return True

    def read_reg(self, name, motor_id):
        self.log.append(f"read_reg:{name}")
        v = self.goal_reg[motor_id]
        return v + 1 if self.goal_readback_error else v

    def set_torque(self, enabled, ids=None):
        self.log.append(f"set_torque:{enabled}")
        self.torque = enabled
        return True


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def bus(calibration, clock):
    return LoggingBus(calibration, clock)


def phases(log):
    """Collapse the call log to its ordered phases (consecutive duplicates merged)."""
    out = []
    for e in log:
        key = "write_reg" if e.startswith("write_reg:") else e
        if not out or out[-1] != key:
            out.append(key)
    return out


def test_frozen_order(cfg, calibration, bus, clock):
    reenergize(bus, cfg, calibration, clock, BlockRead())
    head = phases(bus.log)[:6]
    assert head == [
        "ping_all",
        "read_calibration_regs",
        "write_reg",
        "read_block",
        "write_goal",
        "read_reg:Goal_Position",
    ]
    i = bus.log.index("set_torque:True")
    before = bus.log[:i]
    # exactly: ping -> calib -> 60 register writes -> block -> goal=present -> 6 read-backs -> torque
    n_regs = len(asdict(cfg.registers)) * N_JOINTS
    assert before[0] == "ping_all" and before[1] == "read_calibration_regs"
    assert all(e.startswith("write_reg:") for e in before[2 : 2 + n_regs])
    assert before[2 + n_regs :] == ["read_block", "write_goal"] + ["read_reg:Goal_Position"] * N_JOINTS
    assert bus.log.count("set_torque:True") == 1 and bus.torque


def test_every_register_written_to_every_motor_with_feetech_names(cfg, calibration, bus, clock):
    reenergize(bus, cfg, calibration, clock, BlockRead())
    for field, value in asdict(cfg.registers).items():
        for motor_id in bus.ids:
            assert bus.regs[(register_name(field), motor_id)] == value
    assert register_name("p_coefficient") == "P_Coefficient"
    assert register_name("over_current_protection_time") == "Over_Current_Protection_Time"


def test_goal_is_present_before_torque_so_nothing_jumps(cfg, calibration, bus, clock):
    reenergize(bus, cfg, calibration, clock, BlockRead())
    # the first goal ever written equals the pre-energise present position
    first_goal_idx = bus.log.index("write_goal")
    assert first_goal_idx < bus.log.index("set_torque:True")


def test_ends_at_home(cfg, calibration, bus, clock):
    reenergize(bus, cfg, calibration, clock, BlockRead())
    home = np.asarray(cfg.safe_state.home_ticks)
    assert np.abs(bus.position - home).max() <= 25


def test_ramp_home_respects_per_tick_step(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, present=[1700, 2400, 1800, 2100, 2047, 2300])
    goals = []
    orig = bus.write_goal
    bus.write_goal = lambda g: (goals.append(g.copy()), orig(g))[1]
    reenergize(bus, cfg, calibration, clock, BlockRead())
    step = cfg.safe_state.ramp_ticks_per_tick
    for a, b in pairwise(goals[1:]):
        assert np.abs(b.astype(np.int64) - a).max() <= step


def test_missing_motor_beyond_timeout_raises_timeout_and_never_energises(cfg, calibration, bus, clock):
    bus.missing_until_ns = 100 * S
    with pytest.raises(TimeoutError):
        reenergize(bus, cfg, calibration, clock, BlockRead(), timeout_s=1.0)
    assert "set_torque:True" not in bus.log and "read_calibration_regs" not in bus.log
    assert clock.monotonic_ns() <= 1 * S + 1


def test_motor_that_appears_within_timeout_is_accepted(cfg, calibration, bus, clock):
    bus.missing_until_ns = S // 2
    reenergize(bus, cfg, calibration, clock, BlockRead(), timeout_s=2.0)
    assert bus.torque
    assert bus.log.count("ping_all") > 1


def test_corrupted_calibration_raises_and_never_enables_torque(cfg, calibration, bus, clock):
    bus.corrupt_offset = "shoulder_lift"  # homing_offset + 1
    with pytest.raises(CalibrationMismatch) as ei:
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert "shoulder_lift" in ei.value.report.mismatches
    assert "set_torque:True" not in bus.log
    assert not any(e.startswith("write_reg") or e == "write_goal" for e in bus.log)  # nothing written at all
    assert not bus.torque


def test_goal_readback_mismatch_blocks_torque(cfg, calibration, bus, clock):
    bus.goal_readback_error = True
    with pytest.raises(RuntimeError, match="Goal_Position"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert "set_torque:True" not in bus.log


def test_failed_block_read_is_never_treated_as_present_zero(cfg, calibration, bus, clock):
    bus.block_fail_until_ns = S // 3
    reenergize(bus, cfg, calibration, clock, BlockRead(), timeout_s=2.0)
    first_goal_calls = [e for e in bus.log if e == "write_goal"]
    assert first_goal_calls  # wrote only after a successful read
    assert bus.log.index("write_goal") > bus.log.index("read_block")
    assert bus.log.count("read_block") > 1


def test_block_read_that_never_recovers_times_out_without_torque(cfg, calibration, bus, clock):
    bus.block_fail_until_ns = 100 * S
    with pytest.raises(TimeoutError):
        reenergize(bus, cfg, calibration, clock, BlockRead(), timeout_s=1.0)
    assert "write_goal" not in bus.log and "set_torque:True" not in bus.log


def test_present_outside_range_is_refused(cfg, calibration, bus, clock):
    bus.position[2] = 5000
    with pytest.raises(RuntimeError, match="outside 0..4095"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert "set_torque:True" not in bus.log


def test_incomplete_calibration_is_rejected(cfg, calibration, bus, clock):
    partial = {k: v for k, v in calibration.items() if k != "gripper"}
    with pytest.raises(ValueError):
        reenergize(bus, cfg, partial, clock, BlockRead())
    assert bus.log == []

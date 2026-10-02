"""Re-energise order (plan task 2.4, review B1 d / B2 / B3 / S5 / S12). Uses a minimal local Bus double with a call log
(not U1's FakeBus; the backends are exercised in tests/test_reenergize_backends.py and tests/test_safety_e2e_*.py)."""

from dataclasses import asdict, replace
from itertools import pairwise

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration
from so101_rl.safety.calib_check import CalibrationMismatch
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.reenergize import ReenergizeAborted, UnsafeStartPose, reenergize, register_name
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.timing import FakeClock

S = 1_000_000_000
NEAR_REST = [2060, 2190, 2310, 2740, 2047, 2300]  # within rest_tolerance_ticks of the shipped rest pose
IN_BOX = [1900, 1300, 2600, 2700, 2047, 2300]  # tip inside the box, far from rest
OUTSIDE = [2047, 2047, 2047, 2047, 2047, 2047]  # tip at x = 0.39 m: outside the box and far from rest


class LoggingBus:
    """Bus double with a call log. Plant: position moves toward the written goal by ``follow`` of the error each
    write (1.0 = instant). Joints in ``blocked`` never move. Register writes are stored and read back."""

    def __init__(self, calibration: dict[str, MotorCalibration], clock: FakeClock, present=None, *, follow=1.0):
        self.clock = clock
        self.cal = calibration
        self.ids = [calibration[n].id for n in JOINT_NAMES]
        self.position = np.array(present if present is not None else NEAR_REST, dtype=np.int32)
        self.goal_reg = {i: 0 for i in self.ids}
        self.torque = False
        self.torque_reg_reads_on = False  # Torque_Enable reads back 1 (torque left on by someone else)
        self.log: list[str] = []
        self.regs: dict[tuple[str, int], int] = {}
        self.missing_until_ns = 0  # ping_all reports motor 4 missing until this time
        self.corrupt_offset: str | None = None
        self.corrupt_reg: str | None = None  # this register reads back +1
        self.block_fail_until_ns = 0
        self.goal_readback_error = False
        self.follow = follow
        self.blocked: set[int] = set()
        self.temperature = np.full(N_JOINTS, 30, dtype=np.int32)
        self.status = np.zeros(N_JOINTS, dtype=np.int32)
        self.write_goal_results: list[bool] = []  # popped per write_goal; empty -> True
        self.set_torque_true_result = True
        self.write_goal_calls = 0
        self.max_write_goal_calls = 20_000  # a ramp that never times out must fail the test, not hang it

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
        out.temperature_c[...] = self.temperature
        out.status[...] = self.status
        out.ok = True
        return True

    def write_goal(self, goal):
        self.log.append("write_goal")
        self.write_goal_calls += 1
        assert self.write_goal_calls < self.max_write_goal_calls, "ramp never ended (deadline check missing?)"
        assert goal.dtype == np.int32 and goal.min() >= 0 and goal.max() <= 4095
        if self.write_goal_results and not self.write_goal_results.pop(0):
            return False
        for i, g in zip(self.ids, goal, strict=True):
            self.goal_reg[i] = int(g)
        if self.torque:
            step = np.rint((goal.astype(np.float64) - self.position) * self.follow).astype(np.int32)
            for j in self.blocked:
                step[j] = 0
            self.position += step
        return True

    def read_reg(self, name, motor_id):
        self.log.append(f"read_reg:{name}")
        if name == "Goal_Position":
            v = self.goal_reg[motor_id]
            return v + 1 if self.goal_readback_error else v
        if name == "Torque_Enable":
            return int(self.torque or self.torque_reg_reads_on)
        v = self.regs[(name, motor_id)]
        return v + 1 if name == self.corrupt_reg else v

    def set_torque(self, enabled, ids=None):
        self.log.append(f"set_torque:{enabled}")
        if enabled and not self.set_torque_true_result:
            return False
        self.torque = enabled
        return True


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def bus(calibration, clock):
    return LoggingBus(calibration, clock)


@pytest.fixture
def fk(cfg, calibration):
    return So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)


def phases(log):
    """Collapse the call log to its ordered phases (consecutive duplicates merged)."""
    out = []
    for e in log:
        if not out or out[-1] != e:
            out.append(e)
    return out


def test_frozen_order(cfg, calibration, bus, clock):
    reenergize(bus, cfg, calibration, clock, BlockRead())
    i = bus.log.index("set_torque:True")
    before = bus.log[:i]
    fields = [register_name(f) for f in asdict(cfg.registers)]
    expected = (
        ["ping_all", "read_calibration_regs"]
        + ["write_reg:Lock"] * N_JOINTS
        + ["read_reg:Torque_Enable"] * N_JOINTS
        + [f"write_reg:{f}" for f in fields for _ in range(N_JOINTS)]
        + [f"read_reg:{f}" for f in fields for _ in range(N_JOINTS)]
        + ["read_block", "write_goal"]
        + ["read_reg:Goal_Position"] * N_JOINTS
    )
    assert before == expected
    assert bus.log.count("set_torque:True") == 1 and bus.torque
    assert "set_torque:False" not in bus.log  # success leaves torque on


def test_every_register_written_to_every_motor_with_feetech_names(cfg, calibration, bus, clock):
    reenergize(bus, cfg, calibration, clock, BlockRead())
    for field, value in asdict(cfg.registers).items():
        for motor_id in bus.ids:
            assert bus.regs[(register_name(field), motor_id)] == value
    assert all(bus.regs[("Lock", i)] == 0 for i in bus.ids)
    assert register_name("p_coefficient") == "P_Coefficient"
    assert register_name("over_current_protection_time") == "Over_Current_Protection_Time"


def test_torque_already_on_refuses_register_writes(cfg, calibration, bus, clock):
    bus.torque_reg_reads_on = True
    with pytest.raises(RuntimeError, match="torque on before the register writes"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    written = [e for e in bus.log if e.startswith("write_reg:")]
    assert set(written) == {"write_reg:Lock"}  # only the unlock; no protection register was touched
    assert "set_torque:True" not in bus.log


@pytest.mark.parametrize("reg", ["Protection_Current", "Over_Current_Protection_Time", "P_Coefficient"])
def test_register_readback_mismatch_raises_before_torque(cfg, calibration, bus, clock, reg):
    bus.corrupt_reg = reg
    with pytest.raises(RuntimeError, match=reg):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert "set_torque:True" not in bus.log and "write_goal" not in bus.log


def test_goal_is_present_before_torque_so_nothing_jumps(cfg, calibration, bus, clock):
    goals = []
    orig = bus.write_goal
    bus.write_goal = lambda g: (goals.append(g.copy()), orig(g))[1]
    reenergize(bus, cfg, calibration, clock, BlockRead())
    assert bus.log.index("write_goal") < bus.log.index("set_torque:True")
    np.testing.assert_array_equal(goals[0], NEAR_REST)


@pytest.mark.parametrize("follow", [1.0, 0.3])
def test_ends_at_home(cfg, calibration, clock, follow):
    bus = LoggingBus(calibration, clock, follow=follow)  # 0.3: a lagging plant, not an instant follower
    reenergize(bus, cfg, calibration, clock, BlockRead())
    home = np.asarray(cfg.safe_state.home_ticks)
    assert np.abs(bus.position - home).max() <= 25


def test_ramp_home_respects_per_tick_step(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, present=IN_BOX, follow=0.5)
    goals = []
    orig = bus.write_goal
    bus.write_goal = lambda g: (goals.append(g.copy()), orig(g))[1]
    reenergize(bus, cfg, calibration, clock, BlockRead())
    step = cfg.safe_state.ramp_ticks_per_tick
    assert len(goals) > 10
    for a, b in pairwise(goals):
        assert np.abs(b.astype(np.int64) - a).max() <= step


def test_ramp_from_rest_never_takes_the_target_below_the_table_floor(cfg, calibration, clock, fk):
    bus = LoggingBus(calibration, clock, present=cfg.safe_state.rest_ticks, follow=0.5)
    goals = []
    orig = bus.write_goal
    bus.write_goal = lambda g: (goals.append(g.copy()), orig(g))[1]
    reenergize(bus, cfg, calibration, clock, BlockRead())
    floor = cfg.safe_state.table_z_m + cfg.safe_state.table_clearance_m
    xyz = np.zeros(3)
    assert min(fk.gripper_xyz_from_ticks(g, xyz)[2] for g in goals) >= floor


# ---- B1 (d): start pose -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("present", [NEAR_REST, IN_BOX])
def test_start_near_rest_or_inside_the_box_is_accepted(cfg, calibration, clock, present):
    bus = LoggingBus(calibration, clock, present=present)
    reenergize(bus, cfg, calibration, clock, BlockRead())
    assert bus.torque


def test_start_pose_neither_near_rest_nor_in_box_is_refused_before_any_goal_or_torque(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, present=OUTSIDE)
    with pytest.raises(UnsafeStartPose, match="neither"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert "write_goal" not in bus.log and "set_torque:True" not in bus.log


def test_rest_tolerance_is_per_arm_joint(cfg, calibration, clock):
    rest = np.asarray(cfg.safe_state.rest_ticks)
    tol = cfg.safe_state.rest_tolerance_ticks
    at_edge = rest.copy()
    at_edge[2] += tol  # elbow exactly at the tolerance: still "near rest"
    reenergize(LoggingBus(calibration, clock, present=at_edge), cfg, calibration, clock, BlockRead())
    beyond = rest.copy()
    beyond[2] += tol + 1
    beyond[1] += tol + 1  # shoulder too, so the tip is not in the box either
    with pytest.raises(UnsafeStartPose):
        reenergize(LoggingBus(calibration, clock, present=beyond), cfg, calibration, clock, BlockRead())


# ---- S5: temperatures and status before energising; torque off after any failure once energised ---------------
@pytest.mark.parametrize("temp", [60, 75, 0, 255])
def test_hot_or_dead_temperature_refuses_before_torque(cfg, calibration, bus, clock, temp):
    bus.temperature[3] = temp
    with pytest.raises(RuntimeError, match="temperatures"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert "set_torque:True" not in bus.log and "write_goal" not in bus.log


def test_protection_status_bit_refuses_before_torque(cfg, calibration, bus, clock):
    bus.status[1] = 32  # overload
    with pytest.raises(RuntimeError, match="protection status"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert "set_torque:True" not in bus.log


def test_ramp_that_cannot_finish_times_out_and_turns_torque_off(cfg, calibration, clock):
    """A blocked joint stalls the ramp (tracking clamp). The ramp deadline must end it (S12): with the deadline
    check removed this test fails on the bus double's call guard instead of hanging."""
    bus = LoggingBus(calibration, clock, present=IN_BOX)
    bus.blocked = {1}  # shoulder_lift cannot move
    t0 = clock.monotonic_ns()
    with pytest.raises(TimeoutError, match="ramp home"):
        reenergize(bus, cfg, calibration, clock, BlockRead(), timeout_s=1.0)
    assert bus.log[-1] == "set_torque:False" and not bus.torque
    # bounded: timeout + slack + the nominal ramp time of the longest joint
    worst = np.abs(np.asarray(cfg.safe_state.home_ticks) - IN_BOX).max()
    budget = 1.0 + 1.0 + worst / cfg.safe_state.ramp_ticks_per_tick / cfg.rates.servo_hz
    assert (clock.monotonic_ns() - t0) / S <= budget + 0.1


def test_write_goal_false_during_ramp_raises_and_turns_torque_off(cfg, calibration, bus, clock):
    bus.write_goal_results = [True, True, True, False]  # present, 2 ramp ticks, then the bus refuses
    with pytest.raises(RuntimeError, match="write_goal failed during ramp"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert bus.log[-1] == "set_torque:False" and not bus.torque


def test_write_goal_present_false_raises_and_never_energises(cfg, calibration, bus, clock):
    bus.write_goal_results = [False]
    with pytest.raises(RuntimeError, match=r"write_goal\(present\) failed"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert "set_torque:True" not in bus.log


def test_set_torque_false_raises_and_turns_torque_off(cfg, calibration, bus, clock):
    bus.set_torque_true_result = False
    with pytest.raises(RuntimeError, match="set_torque"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert bus.log[-1] == "set_torque:False"


def test_exception_inside_ramp_turns_torque_off(cfg, calibration, bus, clock):
    calls = {"n": 0}
    orig = bus.read_block

    def flaky(out):
        calls["n"] += 1
        if calls["n"] == 5:
            raise ConnectionError("usb unplugged")
        return orig(out)

    bus.read_block = flaky
    with pytest.raises(ConnectionError):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert bus.log[-1] == "set_torque:False" and not bus.torque


# ---- B2: per-tick hook ----------------------------------------------------------------------------------------
def test_tick_hook_once_per_tick_true_only_after_good_read_and_write(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, follow=0.5)
    bus.block_fail_until_ns = S // 10  # the first reads fail
    events = []
    orig_write = bus.write_goal
    bus.write_goal = lambda g: (events.append("write"), orig_write(g))[1]
    orig_read = bus.read_block

    def read(out):
        ok = orig_read(out)
        events.append("read_ok" if ok else "read_fail")
        return ok

    bus.read_block = read
    hook = []
    reenergize(bus, cfg, calibration, clock, BlockRead(), tick_hook=lambda ok: (hook.append(ok), events.append(ok)))
    assert hook[0] is False  # a failed read before the first good tick
    assert hook.count(True) > 10
    # every True closes a tick that contained a good read and an accepted write
    tick: list[str] = []
    for e in events:
        if isinstance(e, bool):
            if e:
                assert "read_ok" in tick and "write" in tick, tick
            tick = []
        else:
            tick.append(e)
    # the failed reads all come before the first good tick (the write_goal(present) tick)
    assert hook.index(True) == hook.count(False)


def test_tick_hook_reports_false_when_a_ramp_read_fails(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, follow=0.5)
    n = {"reads": 0}
    orig = bus.read_block

    def read(out):
        n["reads"] += 1
        if n["reads"] == 4:  # one failed read in the middle of the ramp
            out.ok = False
            return False
        return orig(out)

    bus.read_block = read
    hook = []
    reenergize(bus, cfg, calibration, clock, BlockRead(), tick_hook=hook.append)
    assert hook[:3] == [True, True, True] and hook[3] is False and all(hook[4:])


def test_with_hook_the_pre_torque_phase_must_reach_a_good_tick_within_the_heartbeat_gap(cfg, calibration, bus, clock):
    bus.missing_until_ns = int(0.9 * S)  # motors answer only after 0.9 s: the MCU would time out
    with pytest.raises(TimeoutError, match="ping"):
        reenergize(bus, cfg, calibration, clock, BlockRead(), tick_hook=lambda ok: None)
    assert "set_torque:True" not in bus.log
    assert clock.monotonic_ns() <= int(0.8 * S) + 1


def test_without_hook_the_same_slow_start_is_accepted(cfg, calibration, bus, clock):
    bus.missing_until_ns = int(0.9 * S)
    reenergize(bus, cfg, calibration, clock, BlockRead())
    assert bus.torque


def test_with_hook_a_ramp_without_good_ticks_for_the_heartbeat_gap_aborts(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, follow=0.5)
    n = {"reads": 0}
    orig = bus.read_block

    def read(out):
        n["reads"] += 1
        if n["reads"] >= 4:
            out.ok = False
            return False
        return orig(out)

    bus.read_block = read
    with pytest.raises(TimeoutError, match="heartbeat gap"):
        reenergize(bus, cfg, calibration, clock, BlockRead(), tick_hook=lambda ok: None)
    assert bus.log[-1] == "set_torque:False"


# ---- unchanged guarantees ---------------------------------------------------------------------------------------
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


def test_explicit_fk_argument_is_used(cfg, calibration, bus, clock, fk):
    reenergize(bus, cfg, calibration, clock, BlockRead(), fk=fk)
    assert bus.torque


def test_a_dead_bus_during_the_torque_off_does_not_mask_the_original_error(cfg, calibration, bus, clock):
    bus.write_goal_results = [True, False]  # the first ramp write fails

    def set_torque(enabled, ids=None):
        bus.log.append(f"set_torque:{enabled}")
        if not enabled:
            raise ConnectionError("bus gone")
        bus.torque = True
        return True

    bus.set_torque = set_torque
    with pytest.raises(RuntimeError, match="write_goal failed during ramp"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert bus.log[-1] == "set_torque:False"


def test_a_refused_torque_off_after_a_failure_is_retried_and_noted(cfg, calibration, bus, clock):
    bus.write_goal_results = [True, False]
    orig = bus.set_torque

    def set_torque(enabled, ids=None):
        if enabled:
            return orig(enabled, ids)
        bus.log.append("set_torque:False")
        return False

    bus.set_torque = set_torque
    with pytest.raises(RuntimeError, match="write_goal failed during ramp") as ei:
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert bus.log.count("set_torque:False") == 3
    assert any("torque-off after the failure ALSO failed" in n for n in ei.value.__notes__)


def test_a_joint_heating_during_the_ramp_ends_it_with_torque_off(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, follow=0.3)
    n = {"writes": 0}
    orig = bus.write_goal

    def write(goal):
        n["writes"] += 1
        if n["writes"] == 10:
            bus.temperature[2] = cfg.thermal.trip_c  # elbow reaches the trip temperature mid-ramp
        return orig(goal)

    bus.write_goal = write
    with pytest.raises(RuntimeError, match="temperatures"):
        reenergize(bus, cfg, calibration, clock, BlockRead())
    assert bus.log[-1] == "set_torque:False" and not bus.torque


def test_with_hook_a_slow_register_phase_is_caught_before_torque(cfg, calibration, bus, clock):
    """The ping answered at once but the register phase took ~1 s: the first good tick is already too late."""
    orig = bus.write_reg

    def slow_write(name, motor_id, value):
        clock.advance(15_000_000)  # 15 ms per register write: 66 writes ~ 1 s
        orig(name, motor_id, value)

    bus.write_reg = slow_write
    with pytest.raises(TimeoutError, match="heartbeat gap"):
        reenergize(bus, cfg, calibration, clock, BlockRead(), tick_hook=lambda ok: None)
    assert "write_goal" not in bus.log and "set_torque:True" not in bus.log


# ---- should_abort (U3 review item 7) ----------------------------------------------------------------------------
def test_abort_before_torque_never_energises(cfg, calibration, bus, clock):
    with pytest.raises(ReenergizeAborted):
        reenergize(bus, cfg, calibration, clock, BlockRead(), should_abort=lambda: True)
    assert "set_torque:True" not in bus.log and not bus.torque


def test_abort_during_the_ramp_turns_torque_off_within_one_tick(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, present=list(cfg.safe_state.rest_ticks), follow=0.3)
    state = {"t_abort": None}

    def abort():
        if bus.torque and state["t_abort"] is None:
            state["t_abort"] = clock.monotonic_ns() + 200_000_000  # 200 ms into the ramp
        return state["t_abort"] is not None and clock.monotonic_ns() >= state["t_abort"]

    with pytest.raises(ReenergizeAborted):
        reenergize(bus, cfg, calibration, clock, BlockRead(), should_abort=abort)
    assert not bus.torque and bus.log[-1] == "set_torque:False"
    assert clock.monotonic_ns() - state["t_abort"] < int(1e9 / cfg.rates.servo_hz) + 1  # honoured within one tick


# ---- hand_over_on_abort (plan section 10, 2026-10-02 "Stop during re-energise") ------------------------------
def test_abort_during_the_ramp_with_hand_over_leaves_torque_on_for_the_caller(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, present=list(cfg.safe_state.rest_ticks), follow=0.3)
    state = {"t_abort": None}

    def abort():
        if bus.torque and state["t_abort"] is None:
            state["t_abort"] = clock.monotonic_ns() + 200_000_000
        return state["t_abort"] is not None and clock.monotonic_ns() >= state["t_abort"]

    with pytest.raises(ReenergizeAborted) as ei:
        reenergize(bus, cfg, calibration, clock, BlockRead(), should_abort=abort, hand_over_on_abort=True)
    assert ei.value.torque_on
    assert bus.torque and "set_torque:False" not in bus.log  # the caller ramps to rest; no torque off in place
    assert clock.monotonic_ns() - state["t_abort"] < int(1e9 / cfg.rates.servo_hz) + 1  # within one tick


def test_abort_before_torque_with_hand_over_still_never_energises(cfg, calibration, bus, clock):
    with pytest.raises(ReenergizeAborted) as ei:
        reenergize(bus, cfg, calibration, clock, BlockRead(), should_abort=lambda: True, hand_over_on_abort=True)
    assert not ei.value.torque_on
    assert "set_torque:True" not in bus.log and not bus.torque


def test_other_failures_during_the_ramp_still_turn_torque_off_with_hand_over(cfg, calibration, clock):
    bus = LoggingBus(calibration, clock, present=list(cfg.safe_state.rest_ticks), follow=0.3)
    calls = {"n": 0}

    def hook(ok):
        calls["n"] += 1
        if bus.torque and calls["n"] > 5:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        reenergize(bus, cfg, calibration, clock, BlockRead(), tick_hook=hook, should_abort=lambda: False, hand_over_on_abort=True)
    assert not bus.torque and bus.log[-1] == "set_torque:False"

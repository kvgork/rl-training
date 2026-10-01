"""SafeStateSequencer (plan task 2.4, review B1 c / B2 / B5 / S2): bounded RAMP_HOME -> REST -> TORQUE_OFF."""

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_rl.config import JOINT_NAMES, N_JOINTS
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.safe_state import Phase, SafeStateSequencer, ramp_toward
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.timing import FakeClock

S = 1_000_000_000
TICK_NS = S // 30


def i32(*v):
    return np.array(v, dtype=np.int32)


IN_BOX = i32(1900, 1300, 2600, 2700, 2047, 2300)
FAR = i32(2500, 1000, 3000, 1000, 3000, 2500)


@pytest.fixture
def fk(cfg, calibration):
    return So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)


@pytest.fixture
def ids(calibration):
    return {n: c.id for n, c in calibration.items()}


@pytest.fixture
def make(cfg, fk, ids):
    def _make(clock, scfg=None, *, E=None, motor_ids=None):
        return SafeStateSequencer(
            scfg if scfg is not None else cfg.safe_state,
            clock,
            tracking_error_ticks=cfg.action.tracking_error_ticks if E is None else E,
            motor_ids=ids if motor_ids is None else motor_ids,
            fk=fk,
        )

    return _make


def run(seq, clock, present, *, follow=1.0, dt_ns=TICK_NS, max_steps=5000):
    """Drive the sequencer; the plant moves ``follow`` of the way to the written target each tick."""
    target = np.zeros(N_JOINTS, dtype=np.int32)
    steps = []
    t0 = clock.monotonic_ns()
    for _ in range(max_steps):
        clock.advance(dt_ns)
        step = seq.next(present, target)
        steps.append((clock.monotonic_ns() - t0, step, target.copy()))
        if step.write_target and follow:
            present[...] = np.rint(present + (target - present) * follow).astype(np.int32)
        if step.done:
            break
    return steps


@pytest.mark.parametrize("follow", [1.0, 0.5])
def test_completes_within_the_bound_and_ends_at_rest(cfg, make, follow):
    clock = FakeClock()
    seq = make(clock)
    present = IN_BOX.copy()
    seq.start(present)
    steps = run(seq, clock, present, follow=follow)
    elapsed_ns, last, _ = steps[-1]
    assert last.done and not last.timed_out
    assert elapsed_ns < cfg.safe_state.bound_s * S
    assert np.abs(present - cfg.safe_state.rest_ticks).max() <= 25  # home, then rest


def test_torque_off_uses_the_given_motor_ids_in_order(cfg, make):
    """S2: the ids come from the caller (calibration), never an implicit 1..6."""
    clock = FakeClock()
    odd_ids = {n: 10 + i for i, n in enumerate(JOINT_NAMES)}
    seq = make(clock, motor_ids=odd_ids)
    present = np.asarray(cfg.safe_state.rest_ticks).copy()
    seq.start(present)
    steps = run(seq, clock, present)
    last = steps[-1][1]
    assert last.torque_off_ids == tuple(odd_ids[n] for n in cfg.safe_state.torque_off_order) == (15, 14, 13, 12, 11, 10)
    assert all(s.torque_off_ids == () for _, s, _ in steps[:-1])  # never before the arm reached rest


@pytest.mark.parametrize("missing", ["tracking_error_ticks", "motor_ids", "fk"])
def test_tracking_clamp_motor_ids_and_fk_are_required(cfg, fk, ids, missing):
    kw = {"tracking_error_ticks": cfg.action.tracking_error_ticks, "motor_ids": ids, "fk": fk}
    del kw[missing]
    with pytest.raises(TypeError):
        SafeStateSequencer(cfg.safe_state, FakeClock(), **kw)
    kw[missing] = None
    with pytest.raises(TypeError):
        SafeStateSequencer(cfg.safe_state, FakeClock(), **kw)


def test_motor_ids_must_cover_every_joint(cfg, fk, ids):
    partial = {k: v for k, v in ids.items() if k != "gripper"}
    with pytest.raises(ValueError, match="gripper"):
        SafeStateSequencer(
            cfg.safe_state, FakeClock(), tracking_error_ticks=cfg.action.tracking_error_ticks, motor_ids=partial, fk=fk
        )


def test_ramp_step_never_exceeds_limit_and_passes_through_home_before_rest(cfg, make):
    clock = FakeClock()
    seq = make(clock)
    present = FAR.copy()
    seq.start(present)
    prev = present.copy()
    steps = run(seq, clock, present)
    reached_home = False
    for _, s, tgt in steps:
        if s.write_target:
            assert np.abs(tgt - prev).max() <= cfg.safe_state.ramp_ticks_per_tick
            prev = tgt
            reached_home = reached_home or bool((tgt == cfg.safe_state.home_ticks).all())
    assert reached_home
    assert steps[-1][1].done and not steps[-1][1].timed_out


def test_targets_never_go_below_the_table_floor(cfg, make, fk):
    clock = FakeClock()
    seq = make(clock)
    present = IN_BOX.copy()
    seq.start(present)
    steps = run(seq, clock, present, follow=0.4)
    floor = cfg.safe_state.table_z_m + cfg.safe_state.table_clearance_m
    xyz = np.zeros(3)
    zs = [fk.gripper_xyz_from_ticks(t, xyz)[2] for _, s, t in steps if s.write_target]
    assert min(zs) >= floor
    assert steps[-1][1].done and not steps[-1][1].timed_out


def test_a_rest_pose_the_transit_cannot_reach_times_out_instead_of_crossing_the_table(cfg, make, fk):
    """The planner holds every joint rather than lower the tip through the floor; the bound then ends it."""
    clock = FakeClock()
    sunk = np.asarray(cfg.safe_state.rest_ticks).copy()
    sunk[1] += 300  # shoulder further down: this "rest" puts the tip 4 cm into the table
    xyz = np.zeros(3)
    floor = cfg.safe_state.table_z_m + cfg.safe_state.table_clearance_m
    assert fk.gripper_xyz_from_ticks(sunk, xyz)[2] < floor - 0.01
    seq = make(clock, replace(cfg.safe_state, rest_ticks=sunk))
    present = IN_BOX.copy()
    seq.start(present)
    steps = run(seq, clock, present)
    assert steps[-1][1].timed_out
    zs = [fk.gripper_xyz_from_ticks(t, xyz)[2] for _, s, t in steps if s.write_target]
    assert min(zs) >= floor


def test_starts_from_present_not_from_a_stale_target(cfg, make):
    clock = FakeClock()
    seq = make(clock)
    home = np.asarray(cfg.safe_state.home_ticks)
    seq.start(i32(100, 100, 100, 100, 100, 100))
    seq.start(home.copy())  # restart: must forget the first call
    out = np.zeros(N_JOINTS, dtype=np.int32)
    clock.advance(TICK_NS)
    seq.next(home.copy(), out)
    np.testing.assert_array_equal(out, home)


def test_bus_failing_mid_ramp_falls_through_to_torque_off_at_exactly_bound(cfg, make, ids):
    clock = FakeClock()
    seq = make(clock)
    present = FAR.copy()
    seq.start(present)
    bound_ns = int(cfg.safe_state.bound_s * S)
    steps = run(seq, clock, present, follow=0, dt_ns=S // 10)  # present never follows
    elapsed_ns, last, _ = steps[-1]
    assert last.done and last.timed_out
    assert elapsed_ns == bound_ns
    assert last.torque_off_ids == tuple(ids[n] for n in cfg.safe_state.torque_off_order)
    assert not last.write_target
    assert steps[-2][0] == bound_ns - S // 10 and not steps[-2][1].timed_out and steps[-2][1].write_target


def test_bound_is_exact_to_the_nanosecond(cfg, make):
    clock = FakeClock()
    seq = make(clock)
    present = FAR.copy()
    seq.start(present)
    out = np.zeros(N_JOINTS, dtype=np.int32)
    clock.advance(int(cfg.safe_state.bound_s * S) - 1)
    assert not seq.next(present, out).timed_out
    clock.advance(1)
    step = seq.next(present, out)
    assert step.timed_out and step.done


def test_blocked_joint_stalls_ramp_instead_of_winding_target_up(cfg, make):
    clock = FakeClock()
    E = cfg.action.tracking_error_ticks
    seq = make(clock)
    present = i32(2700, 1000, 3000, 1000, 3000, 2500)  # far from home
    seq.start(present)
    target = np.zeros(N_JOINTS, dtype=np.int32)
    for _ in range(200):  # the arm does NOT move (blocked)
        clock.advance(TICK_NS)
        step = seq.next(present, target)
        if step.done:
            break
        assert np.all(np.abs(target.astype(np.int64) - present) <= E)


def test_failed_block_read_holds_the_target_and_only_the_bound_advances(cfg, make):
    clock = FakeClock()
    seq = make(clock)
    present = IN_BOX.copy()
    seq.start(present)
    out = np.zeros(N_JOINTS, dtype=np.int32)
    clock.advance(TICK_NS)
    seq.next(present, out)
    held = out.copy()
    for _ in range(20):
        clock.advance(TICK_NS)
        step = seq.next(i32(0, 0, 0, 0, 0, 0), out, block_ok=False)  # stale garbage must not be used
        assert step.write_target and not step.done
        np.testing.assert_array_equal(out, held)
    clock.advance(int(cfg.safe_state.bound_s * S))
    assert seq.next(present, out, block_ok=False).timed_out


def test_next_after_done_is_inert(cfg, make):
    clock = FakeClock()
    seq = make(clock)
    present = np.asarray(cfg.safe_state.rest_ticks).copy()
    seq.start(present)
    run(seq, clock, present)
    assert seq.phase is Phase.DONE
    out = np.full(N_JOINTS, 7, dtype=np.int32)
    step = seq.next(present, out)
    assert step.done and not step.write_target and step.torque_off_ids == ()
    np.testing.assert_array_equal(out, 7)


def test_next_before_start_is_inert(cfg, make):
    seq = make(FakeClock())
    step = seq.next(np.asarray(cfg.safe_state.home_ticks).copy(), np.zeros(N_JOINTS, dtype=np.int32))
    assert step.done and not step.write_target


def test_requires_home_and_rest(cfg, make):
    with pytest.raises(ValueError):
        make(FakeClock(), replace(cfg.safe_state, home_ticks=None))


def test_timed_out_flag_persists_on_later_calls(cfg, make):
    clock = FakeClock()
    seq = make(clock)
    present = FAR.copy()
    seq.start(present)
    clock.advance(int(cfg.safe_state.bound_s * S) + S)
    out = np.zeros(N_JOINTS, dtype=np.int32)
    assert seq.next(present, out).timed_out
    assert seq.next(present, out).timed_out and seq.timed_out


# ---- run(): the blocking driver with the per-tick hook (B2) -----------------------------------------------------
class PlantBus:
    """Minimal bus for run(): the arm follows the goal by ``follow``; reads can be made to fail."""

    def __init__(self, present, clock, *, follow=0.5):
        self.clock = clock
        self.position = np.asarray(present, dtype=np.int32).copy()
        self.follow = follow
        self.fail_reads = False
        self.torque_off: list[tuple[int, ...]] = []
        self.writes = 0

    def read_block(self, out):
        if self.fail_reads:
            out.ok = False
            return False
        out.position[...] = self.position
        out.ok = True
        return True

    def write_goal(self, goal):
        self.writes += 1
        self.position = np.rint(self.position + (goal - self.position) * self.follow).astype(np.int32)
        return True

    def set_torque(self, enabled, ids=None):
        assert not enabled
        self.torque_off.append(tuple(ids))
        return True


def test_run_drives_to_rest_and_calls_the_hook_once_per_tick(cfg, make, ids):
    clock = FakeClock()
    seq = make(clock)
    bus = PlantBus(IN_BOX, clock)
    hook = []
    t0 = clock.monotonic_ns()
    last = seq.run(bus, BlockRead(), servo_hz=cfg.rates.servo_hz, tick_hook=hook.append)
    assert last.done and not last.timed_out
    assert bus.torque_off == [tuple(ids[n] for n in cfg.safe_state.torque_off_order)]
    assert len(hook) == bus.writes + 1 and all(hook)  # one per write tick + the torque-off tick
    assert np.abs(bus.position - cfg.safe_state.rest_ticks).max() <= 25
    assert (clock.monotonic_ns() - t0) < cfg.safe_state.bound_s * S


def test_run_with_a_dead_bus_reports_false_ticks_and_torques_off_at_the_bound(cfg, make, ids):
    clock = FakeClock()
    seq = make(clock)
    bus = PlantBus(IN_BOX, clock)
    bus.fail_reads = True
    hook = []
    t0 = clock.monotonic_ns()
    last = seq.run(bus, BlockRead(), servo_hz=cfg.rates.servo_hz, tick_hook=hook.append)
    assert last.timed_out and not any(hook)
    assert bus.torque_off == [tuple(ids[n] for n in cfg.safe_state.torque_off_order)]
    assert clock.monotonic_ns() - t0 == pytest.approx(cfg.safe_state.bound_s * S, abs=TICK_NS)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.integers(0, 4095), min_size=N_JOINTS, max_size=N_JOINTS),
    st.lists(st.integers(0, 4095), min_size=N_JOINTS, max_size=N_JOINTS),
    st.integers(1, 50),
)
def test_ramp_toward_properties(cur, goal, step):
    cur, goal = np.array(cur, dtype=np.int32), np.array(goal, dtype=np.int32)
    out = np.empty(N_JOINTS, dtype=np.int32)
    ramp_toward(cur, goal, step, out)
    assert np.all(np.abs(out - cur) <= step)
    assert np.all(np.abs(out - goal) <= np.abs(cur - goal))  # never moves away
    near = np.abs(goal - cur) <= step
    np.testing.assert_array_equal(out[near], goal[near])


def test_run_raises_when_the_final_torque_off_is_refused(cfg, make):
    clock = FakeClock()
    seq = make(clock)
    bus = PlantBus(cfg.safe_state.rest_ticks, clock)
    calls = []

    def refuse(enabled, ids=None):
        calls.append(ids)
        return False

    bus.set_torque = refuse
    with pytest.raises(RuntimeError, match="torque off of motors"):
        seq.run(bus, BlockRead(), servo_hz=cfg.rates.servo_hz)
    assert len(calls) == 3

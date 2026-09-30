"""SafeStateSequencer (plan task 2.4): bounded RAMP_HOME -> REST -> TORQUE_OFF."""

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_rl.config import JOINT_NAMES, N_JOINTS
from so101_rl.safety.safe_state import Phase, SafeStateSequencer, ramp_toward
from so101_rl.servo.timing import FakeClock

S = 1_000_000_000
TICK_NS = S // 30


def i32(*v):
    return np.array(v, dtype=np.int32)


HOME = i32(2047, 2000, 2100, 2047, 2047, 1800)
REST = i32(2047, 1500, 2600, 2300, 2047, 1800)


@pytest.fixture
def scfg(cfg):
    return replace(cfg.safe_state, home_ticks=HOME, rest_ticks=REST)


def run(seq, clock, present, *, follow=True, dt_ns=TICK_NS, max_steps=1000):
    """Drive the sequencer; the plant follows the written target instantly when ``follow``."""
    target = np.zeros(N_JOINTS, dtype=np.int32)
    steps = []
    t0 = clock.monotonic_ns()
    for _ in range(max_steps):
        clock.advance(dt_ns)
        step = seq.next(present, target)
        steps.append((clock.monotonic_ns() - t0, step, target.copy()))
        if step.write_target and follow:
            present[...] = target
        if step.done:
            break
    return steps


def test_completes_well_under_five_seconds_on_a_healthy_plant(scfg, cfg):
    clock = FakeClock()
    seq = SafeStateSequencer(scfg, clock)
    present = HOME + i32(300, -200, 150, 0, -100, 100)
    seq.start(present)
    steps = run(seq, clock, present)
    elapsed_ns, last, _ = steps[-1]
    assert last.done and not last.timed_out
    assert elapsed_ns < 5 * S
    np.testing.assert_array_equal(present, REST)  # home, then rest


def test_torque_off_order_and_ids(scfg):
    clock = FakeClock()
    seq = SafeStateSequencer(scfg, clock)
    present = REST.copy()
    seq.start(present)
    steps = run(seq, clock, present)
    last = steps[-1][1]
    expected = tuple(JOINT_NAMES.index(n) + 1 for n in scfg.torque_off_order)
    assert last.torque_off_ids == expected == (6, 5, 4, 3, 2, 1)
    # torque is never turned off before the arm reached rest
    assert all(s.torque_off_ids == () for _, s, _ in steps[:-1])


def test_ramp_step_never_exceeds_limit_and_passes_through_home_before_rest(scfg):
    clock = FakeClock()
    seq = SafeStateSequencer(scfg, clock)
    present = i32(2500, 1000, 3000, 1000, 3000, 2500)
    seq.start(present)
    prev = present.copy()
    steps = run(seq, clock, present, max_steps=2000)
    reached_home = False
    for _, s, tgt in steps:
        if s.write_target:
            assert np.abs(tgt - prev).max() <= scfg.ramp_ticks_per_tick
            prev = tgt
            reached_home = reached_home or bool((tgt == HOME).all())
    assert reached_home


def test_starts_from_present_not_from_a_stale_target(scfg):
    clock = FakeClock()
    seq = SafeStateSequencer(scfg, clock)
    seq.start(i32(100, 100, 100, 100, 100, 100))
    seq.start(HOME.copy())  # restart: must forget the first call
    out = np.zeros(N_JOINTS, dtype=np.int32)
    clock.advance(TICK_NS)
    seq.next(HOME.copy(), out)
    np.testing.assert_array_equal(out, HOME)


def test_bus_failing_mid_ramp_falls_through_to_torque_off_at_exactly_bound(scfg):
    clock = FakeClock()
    seq = SafeStateSequencer(scfg, clock)
    present = i32(2500, 1000, 3000, 1000, 3000, 2500)
    seq.start(present)
    # the bus dies after the first few ticks: present stops following the target
    steps = run(seq, clock, present, follow=False, dt_ns=S // 10, max_steps=200)
    assert steps[-1][1].done and steps[-1][1].timed_out
    elapsed_ns, last, _ = steps[-1]
    assert elapsed_ns == 5 * S  # exactly bound_s
    assert last.torque_off_ids == (6, 5, 4, 3, 2, 1)
    assert not last.write_target
    # the tick before the bound was still ramping
    assert steps[-2][0] == 5 * S - S // 10 and not steps[-2][1].timed_out and steps[-2][1].write_target


def test_bound_is_exact_to_the_nanosecond(scfg):
    clock = FakeClock()
    seq = SafeStateSequencer(scfg, clock)
    present = i32(2500, 1000, 3000, 1000, 3000, 2500)
    seq.start(present)
    out = np.zeros(N_JOINTS, dtype=np.int32)
    clock.advance(5 * S - 1)
    assert not seq.next(present, out).timed_out
    clock.advance(1)
    step = seq.next(present, out)
    assert step.timed_out and step.done


def test_blocked_joint_stalls_ramp_instead_of_winding_target_up(scfg, cfg):
    clock = FakeClock()
    E = cfg.action.tracking_error_ticks
    seq = SafeStateSequencer(scfg, clock, tracking_error_ticks=E)
    present = i32(3000, 1000, 3000, 1000, 3000, 2500)  # far from home
    seq.start(present)
    target = np.zeros(N_JOINTS, dtype=np.int32)
    for _ in range(120):  # the arm does NOT move (blocked)
        clock.advance(TICK_NS)
        step = seq.next(present, target)
        if step.done:
            break
        assert np.all(np.abs(target.astype(np.int64) - present) <= E)


def test_without_tracking_clamp_target_runs_away_when_blocked(scfg):
    """Contrast to the test above: documents why the clamp matters."""
    clock = FakeClock()
    seq = SafeStateSequencer(scfg, clock)
    present = i32(3000, 1000, 3000, 1000, 3000, 2500)
    seq.start(present)
    target = np.zeros(N_JOINTS, dtype=np.int32)
    for _ in range(60):
        clock.advance(TICK_NS)
        seq.next(present, target)
    assert np.abs(target - present).max() > 200


def test_next_after_done_is_inert(scfg):
    clock = FakeClock()
    seq = SafeStateSequencer(scfg, clock)
    present = REST.copy()
    seq.start(present)
    run(seq, clock, present)
    assert seq.phase is Phase.DONE
    out = np.full(N_JOINTS, 7, dtype=np.int32)
    step = seq.next(present, out)
    assert step.done and not step.write_target and step.torque_off_ids == ()
    np.testing.assert_array_equal(out, 7)


def test_next_before_start_is_inert(scfg):
    seq = SafeStateSequencer(scfg, FakeClock())
    step = seq.next(HOME.copy(), np.zeros(N_JOINTS, dtype=np.int32))
    assert step.done and not step.write_target


def test_requires_home_and_rest(cfg):
    with pytest.raises(ValueError):
        SafeStateSequencer(replace(cfg.safe_state, home_ticks=None), FakeClock())


def test_timed_out_flag_persists_on_later_calls(scfg):
    clock = FakeClock()
    seq = SafeStateSequencer(scfg, clock)
    present = i32(2500, 1000, 3000, 1000, 3000, 2500)
    seq.start(present)
    clock.advance(6 * S)
    out = np.zeros(N_JOINTS, dtype=np.int32)
    assert seq.next(present, out).timed_out
    assert seq.next(present, out).timed_out and seq.timed_out


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

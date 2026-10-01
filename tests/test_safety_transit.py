"""TransitPlanner (review B1 c): one ramp tick with lift-first ordering and the table-floor check."""

from dataclasses import replace

import numpy as np
import pytest

from so101_rl.config import N_JOINTS
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.safe_state import SafeStateSequencer
from so101_rl.safety.transit import TransitPlanner, ramp_toward, simulate_transit
from so101_rl.servo.timing import FakeClock

OLD_HOME = np.full(N_JOINTS, 2047, dtype=np.int32)  # the pre-fix placeholder home


@pytest.fixture
def fk(cfg, calibration):
    return So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)


@pytest.fixture
def planner(cfg, fk):
    return TransitPlanner(fk, cfg.safe_state)


def z(fk, ticks):
    return float(fk.gripper_xyz_from_ticks(np.asarray(ticks), np.zeros(3))[2])


def naive_path(start, goal, step):
    cur = np.asarray(start, dtype=np.int32).copy()
    out = np.empty_like(cur)
    path = []
    while not (cur == goal).all():
        ramp_toward(cur, goal, step, out)
        cur[...] = out
        path.append(cur.copy())
    return path


def planner_path(planner, start, goal):
    cur = np.asarray(start, dtype=np.int32).copy()
    out = np.empty_like(cur)
    path = []
    for _ in range(2000):
        if (cur == goal).all():
            break
        planner.next_target(cur, goal, out)
        cur[...] = out
        path.append(cur.copy())
    return path


def test_floor_values_come_from_the_config(cfg, planner):
    ss = cfg.safe_state
    assert planner.floor_z == pytest.approx(ss.table_z_m + ss.table_clearance_m)
    assert planner.zone_top_z == pytest.approx(ss.table_z_m + ss.table_clearance_m + ss.transit_zone_m)
    assert planner.step_ticks == ss.ramp_ticks_per_tick


def test_a_goal_through_the_table_is_held_at_the_floor(cfg, fk, planner):
    """Goal 4 cm inside the table: the planner never emits a target below the floor and finally holds."""
    rest = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32)
    sunk = rest.copy()
    sunk[1] += 300
    assert z(fk, sunk) < planner.floor_z - 0.03
    path = planner_path(planner, cfg.safe_state.home_ticks, sunk)
    assert min(z(fk, p) for p in path) >= planner.floor_z
    _n, _, reached = simulate_transit(planner, cfg.safe_state.home_ticks, sunk, 2000)
    assert not reached  # stalled: the caller's time bound ends it with torque off
    # and the naive ramp WOULD have gone through the table
    assert min(z(fk, p) for p in naive_path(cfg.safe_state.home_ticks, sunk, planner.step_ticks)) < planner.floor_z


def test_below_the_floor_only_moves_that_do_not_lower_the_tip_are_taken(cfg, fk, planner):
    rest = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32)
    low = rest.copy()
    low[1] += 120  # start a little below the floor
    assert z(fk, low) < planner.floor_z
    out = np.empty_like(low)
    planner.next_target(low, cfg.safe_state.home_ticks, out)  # home is up: lifting moves are allowed
    assert z(fk, out) >= z(fk, low)
    assert not (out == low).all()
    deeper = low.copy()
    deeper[1] += 200
    planner.next_target(low, deeper, out)  # further down is refused outright
    np.testing.assert_array_equal(out, low)


def test_acceptable(cfg, fk, planner):
    home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int32)
    rest = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32)
    sunk = rest.copy()
    sunk[1] += 300
    assert planner.acceptable(home, rest)  # above the floor
    assert not planner.acceptable(rest, sunk)  # below the floor and lower
    sunk2 = sunk.copy()
    sunk2[1] -= 8
    assert planner.acceptable(sunk, sunk2)  # below the floor but going up


def test_at_goal_is_a_no_op(cfg, planner):
    home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int32)
    out = np.zeros_like(home)
    planner.next_target(home, home, out)
    np.testing.assert_array_equal(out, home)


def test_sequencer_ramps_exactly_along_the_planner_path(cfg, fk, calibration, planner):
    """The sequencer's RAMP_HOME targets are the planner's (lift first), not the naive per-joint ramp."""
    rest = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32)
    expected = planner_path(planner, rest, OLD_HOME)
    naive = naive_path(rest, OLD_HOME, planner.step_ticks)
    assert len(expected) != len(naive) or any((a != b).any() for a, b in zip(expected, naive, strict=False))
    clock = FakeClock()
    seq = SafeStateSequencer(
        replace(cfg.safe_state, home_ticks=OLD_HOME),
        clock,
        tracking_error_ticks=cfg.action.tracking_error_ticks,
        motor_ids={n: c.id for n, c in calibration.items()},
        fk=fk,
    )
    present = rest.copy()
    seq.start(present)
    out = np.zeros(N_JOINTS, dtype=np.int32)
    got = []
    for _ in range(len(expected)):
        clock.advance(1_000_000_000 // 30)
        seq.next(present, out)
        present[...] = out
        got.append(out.copy())
    for a, b in zip(got, expected, strict=True):
        np.testing.assert_array_equal(a, b)

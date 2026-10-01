"""Runtime wiring of the transit planner (review B1 c, B2): reenergize's ramp home and the safe-state sequencer must
use the planner, keep its post-clamp ``acceptable()`` guard, fall back to hold-all, and report a failed read to the
heartbeat hook. Each test fails if the corresponding line is reverted (mutation-checked, see the commit message).

The shipped rest -> home path never dips under the plain per-joint ramp, so these tests use synthetic geometry."""

from dataclasses import replace

import numpy as np
import pytest
from test_reenergize import LoggingBus

from so101_rl.config import N_JOINTS
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.reenergize import reenergize
from so101_rl.safety.safe_state import SafeStateSequencer
from so101_rl.safety.transit import TransitPlanner, ramp_toward
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.timing import FakeClock

# Synthetic rest/home pair: the plain per-joint ramp from this rest bottoms out 2 mm below the floor
# (table_z + 5 mm clearance), the rest pose itself being above it. Same pair as tests/test_safety_config_geometry.py.
DIP_REST = np.array([2050, 1951, 2017, 2964, 2040, 2065], dtype=np.int32)
DIP_TABLE_Z = 0.08446094929348767


@pytest.fixture
def fk(cfg, calibration):
    return So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)


def z_of(fk, ticks):
    return float(fk.gripper_xyz_from_ticks(np.asarray(ticks, dtype=np.int32), np.zeros(3))[2])


def record_goals(bus):
    goals = []
    orig = bus.write_goal
    bus.write_goal = lambda g: (goals.append(g.copy()), orig(g))[1]
    return goals


# ---- reenergize: the ramp home goes through the planner ----------------------------------------------------------
def test_reenergize_ramp_never_dips_where_the_plain_ramp_would(cfg, calibration, fk):
    ss = replace(cfg.safe_state, rest_ticks=DIP_REST, table_z_m=DIP_TABLE_Z)
    c = replace(cfg, safe_state=ss)
    floor = ss.table_z_m + ss.table_clearance_m
    home = np.asarray(ss.home_ticks, dtype=np.int32)
    # precondition (non-vacuous): the plain ramp from this rest DOES go below the floor
    cur, out, plain_min = DIP_REST.copy(), np.empty(N_JOINTS, dtype=np.int32), 1.0
    while not (cur == home).all():
        ramp_toward(cur, home, ss.ramp_ticks_per_tick, out)
        cur[...] = out
        plain_min = min(plain_min, z_of(fk, cur))
    assert z_of(fk, DIP_REST) > floor > plain_min

    clock = FakeClock()
    bus = LoggingBus(calibration, clock, present=DIP_REST, follow=1.0)
    goals = record_goals(bus)
    reenergize(bus, c, calibration, clock, BlockRead(), fk=fk)
    assert np.abs(bus.position - home).max() <= 25  # it still gets home
    assert min(z_of(fk, g) for g in goals) >= floor


def test_reenergize_post_clamp_guard_keeps_the_tracking_clamp_from_lowering_the_tip(cfg, calibration, fk):
    """Shoulder blocked at rest: the tracking clamp pins its target while the planner's other joints move, which
    would lower the tip below the floor. ``planner.acceptable`` refuses that target; the ramp then times out."""
    rest = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32)
    floor_z = z_of(fk, rest)  # floor exactly at the rest tip: any lowering is below the floor
    ss = replace(cfg.safe_state, table_z_m=floor_z - cfg.safe_state.table_clearance_m, transit_zone_m=0.0)
    clock = FakeClock()
    bus = LoggingBus(calibration, clock, present=rest, follow=0.5)
    bus.blocked = {1}
    goals = record_goals(bus)
    with pytest.raises(TimeoutError, match="ramp home"):
        reenergize(bus, replace(cfg, safe_state=ss), calibration, clock, BlockRead(), fk=fk)
    assert len(goals) > 20
    assert min(z_of(fk, g) for g in goals) >= floor_z - 1e-9


# ---- safe-state sequencer: same guard ----------------------------------------------------------------------------
def test_sequencer_post_clamp_guard_keeps_the_tracking_clamp_from_lowering_the_tip(cfg, calibration, fk):
    rest = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32)
    floor_z = z_of(fk, rest)
    ss = replace(cfg.safe_state, table_z_m=floor_z - cfg.safe_state.table_clearance_m, transit_zone_m=0.0)
    clock = FakeClock()
    seq = SafeStateSequencer(
        ss,
        clock,
        tracking_error_ticks=cfg.action.tracking_error_ticks,
        motor_ids={n: c.id for n, c in calibration.items()},
        fk=fk,
    )
    present = rest.copy()
    seq.start(present)
    out = np.zeros(N_JOINTS, dtype=np.int32)
    zs = []
    for _ in range(120):
        clock.advance(1_000_000_000 // 30)
        seq.next(present, out)
        zs.append(z_of(fk, out))
        pinned = present[1]
        present[...] = out
        present[1] = pinned  # the shoulder is blocked: it never follows
    assert min(zs) >= floor_z - 1e-9
    assert not (out == np.asarray(ss.home_ticks)).all()  # and the arm did not get home: the guard held targets back


# ---- planner: hold-all fall-back ----------------------------------------------------------------------------------
class CouplingFK:
    """z = Z0 - K * (t0 - s0) * (t1 - s1): each of joints 0 and 1 alone leaves the tip where it is, both together
    lower it. No per-joint lowering is detectable, so only the hold-all fall-back can stop the move."""

    Z0, K = 0.10, 1e-4

    def __init__(self, start):
        self.start = np.asarray(start, dtype=np.int64)

    def gripper_xyz_from_ticks(self, ticks, out):
        d = np.asarray(ticks, dtype=np.int64) - self.start
        out[:] = (0.0, 0.0, self.Z0 - self.K * float(d[0] * d[1]))
        return out


def test_hold_all_when_the_joints_together_lower_the_tip_below_the_floor(cfg):
    start = np.array([2000, 2000, 2000, 2000, 2000, 2000], dtype=np.int32)
    goal = start + 100
    fk = CouplingFK(start)
    ss = replace(cfg.safe_state, ramp_ticks_per_tick=8, table_z_m=CouplingFK.Z0 - 0.005 - 1e-6, transit_zone_m=0.0)
    planner = TransitPlanner(fk, ss)  # type: ignore[arg-type]
    out = np.empty(N_JOINTS, dtype=np.int32)
    planner.next_target(start, goal, out)
    cand = np.empty(N_JOINTS, dtype=np.int32)
    ramp_toward(start, goal, 8, cand)
    assert planner.tip_z(cand) < planner.floor_z and planner.tip_z(cand) < planner.tip_z(start)  # the plain step dips
    np.testing.assert_array_equal(out, start)  # held entirely


# ---- B2: the sequencer's heartbeat hook after start ----------------------------------------------------------------
class ScriptedBus:
    def __init__(self, present, fail_reads: set[int]):
        self.position = np.asarray(present, dtype=np.int32).copy()
        self.fail_reads = fail_reads
        self.reads = 0
        self.torque_off = []

    def read_block(self, out):
        self.reads += 1
        if self.reads in self.fail_reads:
            out.ok = False
            return False
        out.position[...] = self.position
        out.ok = True
        return True

    def write_goal(self, goal):
        self.position = np.rint(self.position + (goal - self.position) * 0.5).astype(np.int32)
        return True

    def set_torque(self, enabled, ids=None):
        self.torque_off.append(tuple(ids))
        return True


def test_run_reports_false_to_the_hook_for_a_failed_read_after_start(cfg, calibration, fk):
    clock = FakeClock()
    seq = SafeStateSequencer(
        cfg.safe_state,
        clock,
        tracking_error_ticks=cfg.action.tracking_error_ticks,
        motor_ids={n: c.id for n, c in calibration.items()},
        fk=fk,
    )
    bus = ScriptedBus(cfg.safe_state.home_ticks, fail_reads={5, 6})  # mid-ramp, after the first good read started it
    hook: list[bool] = []
    last = seq.run(bus, BlockRead(), servo_hz=cfg.rates.servo_hz, tick_hook=hook.append)
    assert last.done and not last.timed_out
    assert hook[:4] == [True] * 4
    assert hook[4:6] == [False, False]  # ticks 5 and 6: read failed (the write of the held target still succeeded)
    assert all(hook[6:])

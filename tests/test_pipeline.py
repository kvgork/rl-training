"""SafetyPipeline (plan task 2.5): order, interpolation, anti-windup, BOX vs BOX_PRESENT, import fence."""

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from so101_rl.config import N_JOINTS
from so101_rl.safety import pipeline as pipeline_mod
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.pipeline import SafetyPipeline
from so101_rl.safety.stuck import StuckActionMonitor
from so101_rl.safety.thermal import ThermalMonitor
from so101_rl.safety.types import TripCause, Verdict
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.timing import FakeClock

T = TripCause
# tip at (0.284, 0, 0.198): comfortably inside the default box
POSE = np.array([2047, 1395, 2699, 2047, 2047, 2047], dtype=np.int32)
# shoulder_lift 1930 puts the tip at z ~ 0.015, 3 cm above the box floor (-0.015 = table + 5 mm since plan §10
# 2026-10-06; it was 0.010): repeated +45-tick lifts reach the floor within 2 commands
LOW_POSE = np.array([2047, 1930, 2699, 2047, 2047, 2047], dtype=np.int32)


class Rig:
    """Pipeline + an ideal plant that follows the written target exactly."""

    def __init__(self, cfg, calibration, pose=POSE):
        self.clock = FakeClock(start_ns=100 * 10**9)
        self.cfg = cfg
        fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)
        self.fk = fk
        self.thermal = ThermalMonitor(cfg.thermal, self.clock)
        self.stuck = StuckActionMonitor()
        self.pipe = SafetyPipeline(cfg, fk, self.thermal, self.stuck)
        self.block = BlockRead()
        self.block.position[...] = pose
        self.block.temperature_c[...] = 35
        self.block.ok = True
        self.v = Verdict()
        self.pipe.resync(self.block.position)

    def command(self, delta):
        return self.pipe.new_command(np.asarray(delta, dtype=np.float64), self.v)

    def tick(self, remaining, follow=True):
        self.clock.advance(33_333_333)
        self.pipe.apply(self.block, remaining, self.v)
        if follow:
            self.block.position[...] = self.v.target_ticks
        return self.v

    def cycle(self, delta, follow=True):
        """One 10 Hz command = 3 ticks; returns the list of (target, trips, mask) per tick."""
        self.command(delta)
        rows = []
        for rem in (3, 2, 1):
            v = self.tick(rem, follow)
            rows.append((v.target_ticks.copy(), v.trips, v.clamped_mask.copy()))
        return rows

    def xyz(self, ticks):
        return self.fk.gripper_xyz_from_ticks(ticks, np.zeros(3))

    def floor_z(self, ticks):
        """Lowest wrist/gripper point (plan §10 2026-10-06 / 2026-10-07 gripper floor)."""
        return self.fk.tip_and_floor_z_from_ticks(ticks, np.zeros(3))

    def inside(self, ticks):
        """The pipeline's exact box test (tip box + gripper floor)."""
        xyz = np.zeros(3)
        zf = self.fk.tip_and_floor_z_from_ticks(ticks, xyz)
        return self.fk.in_box_gripper(xyz, zf, self.cfg.box)


def tip_only(cfg):
    """The shipped config with the gripper floor out of the way, to test the TIP box rules on their own."""
    return dataclasses.replace(cfg, box=dataclasses.replace(cfg.box, gripper_floor_z_m=-1.0))


@pytest.fixture
def rig(cfg, calibration):
    return Rig(cfg, calibration)


def d(*vals):
    out = np.zeros(N_JOINTS)
    out[: len(vals)] = vals
    return out


# ---- interpolation --------------------------------------------------------------------------------------------------
def test_interpolation_reaches_goal_in_exactly_three_ticks(rig):
    start = rig.block.position.copy()
    rows = rig.cycle(d(30, -21, 9, 0, 0, 15))
    goal = start + np.array([30, -21, 9, 0, 0, 15])
    t1, t2, t3 = (r[0] for r in rows)
    assert not np.array_equal(t1, goal) and not np.array_equal(t2, goal)  # no step input
    np.testing.assert_array_equal(t3, goal)
    # monotone, no overshoot, roughly equal thirds
    assert np.all(np.abs(t1 - start) <= np.abs(goal - start)) and np.all(np.abs(t2 - start) >= np.abs(t1 - start))
    np.testing.assert_array_equal(rig.v.goal_ticks, goal)


def test_applied_delta_is_the_post_clamp_goal_change(rig):
    rig.cycle(d(30, -21, 9, 0, 0, 15))
    np.testing.assert_array_equal(rig.v.applied_delta_ticks, [30, -21, 9, 0, 0, 15])
    assert rig.v.applied_delta_ticks.dtype == np.float32
    rig.cycle(d(-7, 0, 0, 0, 0, 0))
    np.testing.assert_array_equal(rig.v.applied_delta_ticks, [-7, 0, 0, 0, 0, 0])  # per command, not cumulative


def test_delta_above_max_is_clipped_and_reported_in_log_value(rig, cfg):
    rig.cycle(d(500, -500, 1e30, -1e30, 0, 500))
    np.testing.assert_array_equal(rig.v.applied_delta_ticks, [45, -45, 45, -45, 0, 60])


def test_ticks_remaining_below_one_lands_on_goal(rig):
    rig.command(d(10))
    rig.pipe.apply(rig.block, 0, rig.v)
    assert rig.v.target_ticks[0] == POSE[0] + 10


def test_apply_before_any_resync_syncs_to_present(cfg, calibration):
    r = Rig(cfg, calibration)
    fresh = SafetyPipeline(cfg, r.fk, r.thermal, r.stuck)
    v = Verdict()
    fresh.apply(r.block, 1, v)
    np.testing.assert_array_equal(v.target_ticks, POSE)


# ---- NaN ------------------------------------------------------------------------------------------------------------------
def test_nan_delta_trips_and_holds(rig):
    before = rig.pipe.goal.copy()
    assert rig.command(d(1, float("nan"))) == T.NAN
    assert rig.pipe.goal.tolist() == before.tolist()
    v = rig.tick(3)
    assert v.trips & T.NAN
    np.testing.assert_array_equal(v.target_ticks, POSE)
    assert not v.applied_delta_ticks.any()
    v = rig.tick(2)
    assert not v.trips & T.NAN  # reported once; the state machine owns the reaction


def test_infinite_delta_is_nan_class(rig):
    assert rig.command(d(float("inf"))) == T.NAN


# ---- clamps ------------------------------------------------------------------------------------------------------------------
def test_joint_limit_clamps_goal_and_marks_mask(cfg, calibration):
    pose = POSE.copy()
    pose[0] = cfg.joints.max_ticks[0] - 10  # 10 ticks below the pan limit
    r = Rig(cfg, calibration, pose)
    rows = r.cycle(d(45))
    assert r.v.goal_ticks[0] == cfg.joints.max_ticks[0]
    assert r.v.applied_delta_ticks[0] == 10  # what actually happened, not what was asked
    assert rows[0][2][0] and rows[0][1] & T.JOINT_LIMIT
    assert all(rows[i][0][0] <= cfg.joints.max_ticks[0] for i in range(3))


def test_velocity_clamp_limits_the_per_tick_step(rig, cfg):
    rig.pipe.goal[...] = POSE + 100  # injected: within E of present but far beyond one tick of velocity
    v = rig.tick(1)
    vmax = cfg.action.max_ticks_per_tick
    assert np.all(np.abs(v.target_ticks - POSE) <= vmax)
    assert v.clamped_mask.all()


def test_tracking_clamp_and_anti_windup_keep_goal_within_E_when_blocked(rig, cfg):
    E = cfg.action.tracking_error_ticks
    saw_tracking = False
    for _ in range(20):  # the arm is blocked: present never follows; 20 cmds x 3 ticks < the stuck window
        rows = rig.cycle(d(45, 0, 0, 0, 0, 0), follow=False)
        for tgt, trips, _mask in rows:
            assert np.all(np.abs(tgt.astype(np.int64) - POSE) <= E)
            saw_tracking |= bool(trips & T.TRACKING)
        assert np.all(np.abs(rig.pipe.goal.astype(np.int64) - POSE) <= E)
    assert saw_tracking
    assert rig.pipe.goal[0] - POSE[0] == E[0]  # saturated, not wound up to +900
    # the policy reverses: the goal responds immediately instead of unwinding an integrator
    rig.cycle(d(-45, 0, 0, 0, 0, 0), follow=False)
    assert rig.pipe.goal[0] - POSE[0] == E[0] - 45


def test_gripper_uses_its_own_larger_E(rig, cfg):
    assert cfg.action.tracking_error_ticks[5] > cfg.action.tracking_error_ticks[0]
    for _ in range(20):
        rig.cycle(d(0, 0, 0, 0, 0, 60), follow=False)
    assert rig.pipe.goal[5] - POSE[5] == cfg.action.tracking_error_ticks[5]


# ---- box -------------------------------------------------------------------------------------------------------------------------
def test_box_on_commanded_target_reverts_to_previous_target(cfg, calibration):
    r = Rig(cfg, calibration, LOW_POSE)
    assert r.fk.in_box(r.xyz(LOW_POSE), cfg.box)
    box_seen = False
    for _ in range(4):
        for tgt, trips, mask in r.cycle(d(0, 45)):
            assert r.inside(tgt), "a target outside the box (or below the gripper floor) was emitted"
            if trips & T.BOX:
                box_seen = True
                assert mask[1]
        assert not r.v.trips & T.BOX_PRESENT
    assert box_seen
    # anti-windup: the stored goal is the reverted one, so it cannot push against the wall
    assert r.inside(r.pipe.goal)
    np.testing.assert_array_equal(r.v.goal_ticks, r.pipe.goal)


def test_box_revert_is_a_clamp_not_a_trip(rig, cfg, calibration):
    r = Rig(cfg, calibration, LOW_POSE)
    rows = r.cycle(d(0, 45))
    for _t, trips, _m in rows:
        assert not trips & (T.BOX_PRESENT | T.RANGE | T.NAN)


def test_box_present_is_a_trip(cfg, calibration):
    q0 = np.full(N_JOINTS, 2047, dtype=np.int32)  # tip at x = 0.39 m, outside the default box
    r = Rig(cfg, calibration, q0)
    assert not r.fk.in_box(r.xyz(q0), cfg.box)
    r.command(d(1))
    v = r.tick(3)
    assert v.trips & T.BOX_PRESENT
    np.testing.assert_array_equal(v.target_ticks, q0)  # holds; does not push further


def _pose_below_floor(r, cfg, depth_m, *, points=False):
    """LOW_POSE with shoulder_lift lowered until the tip (or, ``points``, the lowest gripper point) is ``depth_m``
    below its floor (box.z[0] / box.gripper_floor_z_m), within 1.5 mm."""
    q = LOW_POSE.copy()
    floor = cfg.box.gripper_floor_z_m if points else cfg.box.z[0]
    for lift in range(1930, 2300):
        q[1] = lift
        z = r.floor_z(q) if points else r.xyz(q)[2]
        if z <= floor - depth_m:
            assert z >= floor - depth_m - 0.0015
            return q
    raise AssertionError("no such pose")


@pytest.mark.parametrize("depth_mm", [1.0, 3.0, 4.0])
def test_present_sag_within_the_tolerance_is_not_a_trip(cfg, calibration, depth_mm):
    """S3: a few mm of TIP sag below the box floor must not end the run; commands are still clamped to the box.
    (Gripper floor moved out of the way: it has its own tolerance, tested below.)"""
    cfg = tip_only(cfg)
    r = Rig(cfg, calibration, LOW_POSE)
    q = _pose_below_floor(r, cfg, depth_mm / 1000)
    assert depth_mm / 1000 < cfg.box.present_tolerance_m
    r = Rig(cfg, calibration, q)
    for tgt, trips, _m in r.cycle(d(0, 45)):  # pushing further down
        assert not trips & T.BOX_PRESENT
        assert r.xyz(tgt)[2] <= r.xyz(q)[2] + 1e-9  # the command did not get the target any lower than present...
    assert not r.v.trips & T.BOX_PRESENT


def test_present_beyond_the_tolerance_is_a_trip(cfg, calibration):
    cfg = tip_only(cfg)
    r = Rig(cfg, calibration, LOW_POSE)
    q = _pose_below_floor(r, cfg, cfg.box.present_tolerance_m + 0.002)
    r = Rig(cfg, calibration, q)
    r.command(d(0, -10))
    assert r.tick(3).trips & T.BOX_PRESENT


@pytest.mark.parametrize("depth_mm", [0.5, 1.5])
def test_gripper_point_sag_within_its_tolerance_is_not_a_trip(cfg, calibration, depth_mm):
    """Plan §10 2026-10-07: the lowest gripper point may sag up to gripper_present_tolerance_m (2 mm) below its floor
    (table + 1 mm) without a trip; commands never take it lower."""
    r = Rig(cfg, calibration, LOW_POSE)
    q = _pose_below_floor(r, cfg, depth_mm / 1000, points=True)
    assert r.fk.in_box(r.xyz(q), cfg.box, margin=cfg.box.present_tolerance_m)  # the tip alone would not trip
    r = Rig(cfg, calibration, q)
    for tgt, trips, _m in r.cycle(d(0, 45)):
        assert not trips & T.BOX_PRESENT
        assert r.floor_z(tgt) <= r.floor_z(q) + 1e-9 or r.inside(tgt)
    assert not r.v.trips & T.BOX_PRESENT


def test_gripper_point_beyond_its_tolerance_is_a_trip_although_the_tip_is_fine(cfg, calibration):
    r = Rig(cfg, calibration, LOW_POSE)
    q = _pose_below_floor(r, cfg, cfg.box.gripper_present_tolerance_m + 0.0005, points=True)
    assert r.fk.in_box(r.xyz(q), cfg.box, margin=cfg.box.present_tolerance_m)  # tip-only would NOT trip
    r = Rig(cfg, calibration, q)
    r.command(d(0, -10))
    assert r.tick(3).trips & T.BOX_PRESENT


def test_commanded_target_never_takes_the_lowest_gripper_point_below_its_floor(cfg, calibration):
    r = Rig(cfg, calibration, LOW_POSE)
    for _ in range(8):
        for tgt, _trips, _m in r.cycle(d(0, 45)):
            assert r.floor_z(tgt) >= cfg.box.gripper_floor_z_m


def test_commanded_target_is_clamped_to_the_exact_box_not_the_tolerance_band(cfg, calibration):
    r = Rig(cfg, calibration, LOW_POSE)
    for _ in range(6):
        for tgt, _trips, _m in r.cycle(d(0, 45)):
            assert r.xyz(tgt)[2] >= cfg.box.z[0]


# ---- thermal / protection / stuck -----------------------------------------------------------------------------------------
def test_thermal_and_protection_bits_reported(rig):
    rig.block.temperature_c[3] = 61
    v = rig.tick(3)
    assert v.trips & T.THERMAL
    rig.block.temperature_c[3] = 35
    rig.block.status[2] = 4
    # latched thermal keeps reporting until cooled; protection bit appears alongside
    v = rig.tick(2)
    assert v.trips & T.PROTECTION_BIT and v.trips & T.THERMAL


def test_stuck_when_same_command_repeats_against_a_blocked_arm(rig):
    tripped_at = None
    for i in range(40):
        rows = rig.cycle(d(20), follow=False)  # same delta, arm not moving (block.moving == 0)
        if any(r[1] & T.STUCK for r in rows):
            tripped_at = i
            break
    assert tripped_at is not None
    assert 28 <= tripped_at <= 30  # 90 ticks = 30 commands


def test_moving_arm_never_trips_stuck(rig):
    for _ in range(60):
        rig.block.moving[...] = 1
        rows = rig.cycle(d(20, 0, 0, 0, 0, 0), follow=False)
        assert not any(r[1] & T.STUCK for r in rows)


# ---- range check is last ---------------------------------------------------------------------------------------------------------
def test_range_failure_is_a_trip_and_the_bad_target_is_not_hidden(cfg, calibration, monkeypatch):
    pose = POSE.copy()
    pose[4] = 4090
    r = Rig(cfg, calibration, pose)
    # defeat the joint-limit clamp (simulating a future bug) and inject a goal within E of present but > 4095
    monkeypatch.setattr(
        pipeline_mod.clamp, "clamp_joint_limits", lambda t, lim, out, mask: (out.__setitem__(slice(None), t), mask.fill(False))[0]
    )
    r.pipe.goal[4] = 4090 + 20
    v = r.tick(1)
    assert v.trips & T.RANGE
    assert v.target_ticks[4] == 4110  # left as-is so the core's own check_range rejects it as well


def test_order_of_checks_matches_architecture_6_3(rig, monkeypatch):
    calls = []

    def spy(mod, name):
        orig = getattr(mod, name)

        def wrapper(*a, **k):
            calls.append(name)
            return orig(*a, **k)

        monkeypatch.setattr(mod, name, wrapper)

    for n in ("clamp_joint_limits", "clamp_velocity", "clamp_tracking", "check_range"):
        spy(pipeline_mod.clamp, n)
    spy(rig.fk, "tip_and_floor_z_from_ticks")
    spy(rig.thermal, "update")
    spy(rig.stuck, "update")
    rig.tick(1)
    first = {}
    for i, n in enumerate(calls):
        first.setdefault(n, i)
    # thermal.update and stuck.update share the name "update": both run after the box FK and before check_range
    assert first["clamp_joint_limits"] < first["clamp_velocity"] < first["clamp_tracking"]
    assert first["clamp_tracking"] < first["tip_and_floor_z_from_ticks"] < first["update"]
    assert calls.index("check_range") == len(calls) - 1  # the very last call
    assert calls.count("update") == 2 and calls[-3:-1] == ["update", "update"]


def test_earlier_check_wins_over_a_later_one(rig, cfg):
    """A value that would PASS a later check (range 0..4095, within E of present) but fails an earlier one (joint
    limits) must be stopped by the earlier one, not just by check_range."""
    pose = POSE.copy()
    pose[0] = cfg.joints.max_ticks[0]
    r = rig
    r.block.position[...] = pose
    r.pipe.resync(r.block.position)
    r.pipe.goal[0] = cfg.joints.max_ticks[0] + 60  # valid tick, within E, but beyond the calibrated range
    v = r.tick(1)
    assert v.target_ticks[0] == cfg.joints.max_ticks[0]
    assert v.clamped_mask[0] and not v.trips & T.RANGE


def test_monitor_only_feeds_thermal_without_touching_targets(rig):
    goal, prev = rig.pipe.goal.copy(), rig.pipe.prev_target.copy()
    rig.block.temperature_c[1] = 62
    v = rig.pipe.monitor_only(rig.block, rig.v)
    assert v.trips == T.THERMAL
    np.testing.assert_array_equal(rig.pipe.goal, goal)
    np.testing.assert_array_equal(rig.pipe.prev_target, prev)
    rig.block.temperature_c[1] = 45
    for _ in range(31):  # cooldown dwell advances although apply() is never called
        rig.clock.advance(10**9)
        rig.pipe.monitor_only(rig.block, rig.v)
    assert rig.thermal.may_resume()
    assert rig.pipe.monitor_only(rig.block, rig.v).trips == T.NONE


# ---- state ------------------------------------------------------------------------------------------------------------------------
def test_resync_sets_goal_and_previous_target_to_present(rig):
    rig.cycle(d(45), follow=False)
    new = POSE + 7
    rig.block.position[...] = new
    rig.pipe.resync(new)
    np.testing.assert_array_equal(rig.pipe.goal, new)
    np.testing.assert_array_equal(rig.pipe.prev_target, new)
    v = rig.tick(1)
    np.testing.assert_array_equal(v.target_ticks, new)
    assert not v.applied_delta_ticks.any()


def test_pipeline_buffers_are_reused(rig):
    a, b = rig.v.target_ticks, rig.v.goal_ticks
    rig.cycle(d(5))
    assert rig.v.target_ticks is a and rig.v.goal_ticks is b


# ---- property: the two safety invariants --------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_random_commands_never_violate_invariants(cfg, calibration, seed):
    rng = np.random.default_rng(seed)
    r = Rig(cfg, calibration)
    E = cfg.action.tracking_error_ticks
    vmax = cfg.action.max_ticks_per_tick
    prev = r.block.position.copy()
    for _ in range(150):
        delta = rng.uniform(-100, 100, N_JOINTS)
        if rng.random() < 0.05:
            delta[rng.integers(N_JOINTS)] = rng.choice([np.nan, np.inf, -np.inf, 1e30])
        r.command(delta)
        for rem in (3, 2, 1):
            present_before = r.block.position.copy()
            v = r.tick(rem, follow=bool(rng.random() < 0.8))
            tgt = v.target_ticks.astype(np.int64)
            assert np.all((tgt >= cfg.joints.min_ticks) & (tgt <= cfg.joints.max_ticks))
            assert np.all(np.abs(tgt - present_before) <= E)
            assert np.all(np.abs(tgt - prev) <= vmax)
            assert np.isfinite(v.applied_delta_ticks).all()
            assert not v.trips & T.RANGE
            prev = v.target_ticks.copy()


# ---- layering ---------------------------------------------------------------------------------------------------------------------
def test_safety_never_imports_rl_wm_or_env():
    root = Path(pipeline_mod.__file__).parent
    forbidden = {"rl", "wm", "env"}
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = ("." * node.level) + (node.module or "")
                names = [base] + [f"{base}.{a.name}" for a in node.names]
            for n in names:
                parts = n.lstrip(".").split(".")
                if (parts[0] == "so101_rl" and len(parts) > 1 and parts[1] in forbidden) or (
                    n.startswith("..") and parts[0] in forbidden
                ):
                    offenders.append(f"{path.name}: {n}")
    assert offenders == []


def test_apply_before_any_sync_with_a_failed_read_never_emits_a_writable_target(cfg, calibration):
    from so101_rl.safety import clamp

    clock = FakeClock()
    fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)
    pipe = SafetyPipeline(cfg, fk, ThermalMonitor(cfg.thermal, clock), StuckActionMonitor())
    block = BlockRead()  # ok=False, position zeros (stale)
    v = pipe.apply(block, 3, Verdict())
    assert v.trips & T.RANGE
    assert not clamp.check_range(v.target_ticks)  # the core's pre-write check refuses it
    block.position[...] = POSE
    block.temperature_c[...] = 35
    block.ok = True
    v = pipe.apply(block, 3, Verdict())  # the first good read syncs
    np.testing.assert_array_equal(v.target_ticks, POSE)

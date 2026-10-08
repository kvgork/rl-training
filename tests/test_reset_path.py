"""The deploy reset path (``env/reset_path.py``; plan §10 2026-10-08 "Deploy reset: lift then return")."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so101_rl.config import load_calibration, load_config
from so101_rl.env.reset_path import (
    IK_JOINTS,
    WALL_MARGIN_M,
    densify,
    lift_point,
    path_in_box,
    plan_lift,
    plan_reset_path,
)
from so101_rl.safety.fk import KinematicChain, So101FK

CALIB = Path(__file__).parent / "data" / "so101_follower.calib.json"
# where the M6 policy left the arm on the seed-7 wall goal [0.254, -0.154, 0.032]: tip ON the y = -0.22 wall
# (servo goal ticks read from the shm state of the M7 run whose reset ramp tripped STUCK)
WALL_START = np.array([2675, 2262, 2764, 1288, 1894, 2310])


@pytest.fixture(scope="module")
def rig():
    cal = load_calibration(CALIB)
    cfg = load_config(calibration=cal)
    return cfg, So101FK(KinematicChain.from_yaml(cfg.kinematics_path), cal)


def _tip(fk, q):
    return fk.gripper_xyz_from_ticks(np.asarray(q, np.float64), np.empty(3)).copy()


def _wall_dists(p, box):
    return np.array([p[0] - box.x[0], box.x[1] - p[0], p[1] - box.y[0], box.y[1] - p[1], p[2] - box.z[0],
                     box.z[1] - p[2]])  # fmt: skip


def _assert_lift_never_approaches_a_wall(fk, cfg, start, lift):
    """Along the lift, the tip's distance to each face never drops (beyond the 1 mm IK tolerance) unless it stays at
    least the wall margin."""
    prev = _wall_dists(_tip(fk, start), cfg.box)
    for q in lift:
        d = _wall_dists(_tip(fk, q), cfg.box)
        assert np.all(d >= np.minimum(prev, WALL_MARGIN_M) - 1.5e-3), (q, prev, d)
        prev = d


def _old_ramp(start, q0, step):
    """The pre-v0.27 reset: every joint at its max step toward q0 independently."""
    g, out = np.asarray(start, np.int64), []
    while np.any(g != q0):
        g = g + np.clip(q0 - g, -step, step)
        out.append(g.copy())
    return np.array(out)


def test_wall_start_lifts_away_from_the_wall_then_returns(rig):
    cfg, fk = rig
    home = np.asarray(cfg.safe_state.home_ticks, np.int64)
    step = cfg.action.max_ticks_per_tick
    # the old ramp pushes the tip out of the box (the servo's clamp holds it -> STUCK); the new path does not
    assert not path_in_box(fk, cfg, WALL_START, _old_ramp(WALL_START, home, step))
    path = plan_reset_path(fk, cfg, WALL_START, home)
    targets = densify(WALL_START, path, step)
    assert path_in_box(fk, cfg, WALL_START, targets)
    assert np.array_equal(path[-1], home) and np.array_equal(targets[-1], home)
    lift = plan_lift(fk, cfg, WALL_START, home)
    assert lift and all(np.array_equal(a, b) for a, b in zip(lift, path, strict=False))
    _assert_lift_never_approaches_a_wall(fk, cfg, WALL_START, lift)
    end = _tip(fk, lift[-1])
    want = lift_point(_tip(fk, WALL_START), cfg, _tip(fk, home)[2])
    assert np.allclose(end, want, atol=1.5e-3)  # up to the start pose's height, >= 3 cm off the y wall
    assert end[1] >= cfg.box.y[0] + WALL_MARGIN_M - 1.5e-3
    for q in lift:  # only the arm joints move during the lift; wrist and gripper hold
        assert np.array_equal(np.delete(q, IK_JOINTS), np.delete(WALL_START, IK_JOINTS))


def test_densify_steps_are_bounded_and_hit_every_waypoint(rig):
    cfg, _fk = rig
    step = cfg.action.max_ticks_per_tick
    wps = np.array([[2100, 1500, 2600, 2000, 2047, 2047], [2047, 960, 2700, 2748, 2047, 2047]])
    t = densify(WALL_START, wps, step)
    d = np.diff(np.vstack([WALL_START, t]), axis=0)
    assert np.all(np.abs(d) <= step)
    rows = {tuple(r) for r in t}
    assert all(tuple(w) in rows for w in wps) and np.array_equal(t[-1], wps[-1])


def test_near_wall_starts_stay_in_the_box(rig):
    """Seeded poses around home whose tip is within 2 cm of a box face (where an episode that pushed into a wall
    ends): every densified reset target (with the servo's 3-tick interpolation) is inside the box, the lift never
    approaches a wall, and the path ends on q0."""
    cfg, fk = rig
    home = np.asarray(cfg.safe_state.home_ticks, np.int64)
    lo, hi = cfg.joints.min_ticks, cfg.joints.max_ticks
    step = cfg.action.max_ticks_per_tick
    rng = np.random.default_rng(3)
    xyz = np.empty(3)
    n = 0
    while n < 60:
        q = np.clip(home + rng.integers(-900, 901, 6), lo, hi)
        z_low = fk.tip_and_floor_z_from_ticks(q.astype(np.float64), xyz)
        if not fk.in_box_gripper(xyz, z_low, cfg.box) or _wall_dists(xyz, cfg.box).min() > 0.02:
            continue
        n += 1
        path = plan_reset_path(fk, cfg, q, home)
        assert np.array_equal(path[-1], home)
        assert path_in_box(fk, cfg, q, densify(q, path, step)), q
        lift = plan_lift(fk, cfg, q, home)
        _assert_lift_never_approaches_a_wall(fk, cfg, q, lift)
        for w in lift:
            assert np.all(w >= lo) and np.all(w <= hi)

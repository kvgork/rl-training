"""L1 feasibility of the die task under the shipped workspace box (plan §10 2026-10-06, "Workspace box floor").

With the box floor at table + 5 mm (z = -0.015) the jaws must be able to reach a die resting on the table at the
start well and at the zone through the clamp chain, i.e. with the commanded tip inside the exact box. With the old
floor (z = 0.010, 3 cm above the table) no such pose exists: that twin shows the check can fail.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pytest

from so101_rl.env.goals import fk_tip_batch
from so101_rl.env.tasks import DIE_HALF_EDGE_M, DIE_TO_ZONE
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.sim.model import TABLE_Z_M, load_model

_RAD = 2 * np.pi / 4096
DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def setup():
    from so101_rl.config import load_calibration, load_config

    cal = load_calibration(DATA / "so101_follower.calib.json")
    cfg = load_config(calibration=cal)
    fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), cal)
    return cfg, cal, fk


def _arm_grid(cfg) -> np.ndarray:
    """Pan centred; shoulder / elbow / wrist_flex on a grid around the gripper-down poses; roll centred."""
    sl, el, wf = np.meshgrid(np.arange(1900, 2301, 10), np.arange(2100, 2801, 10), np.arange(2300, 2901, 20), indexing="ij")
    t = np.zeros((sl.size, 6), np.int64)
    t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5] = 2047, sl.ravel(), el.ravel(), wf.ravel(), 2047, 2350
    lo, hi = cfg.joints.min_ticks.astype(np.int64), cfg.joints.max_ticks.astype(np.int64)
    return t[((t >= lo) & (t <= hi)).all(axis=1)]


def _jaw_tip_z(model, data, ticks) -> np.ndarray:
    data.qpos[:] = 0
    data.qpos[:5] = (np.asarray(ticks[:5], np.float64) - 2047) * _RAD  # drive_mode 0 on every arm joint
    mujoco.mj_kinematics(model, data)
    return np.array([data.geom_xpos[model.geom(g).id][2] for g in ("jaw_fixed_tip", "jaw_moving_tip")])


def _reach_resting_die(cfg, fk, box, target_xy) -> tuple[np.ndarray, float] | None:
    """A pose whose FK tip is inside ``box`` (exact, as the BOX clamp tests the target), within 5 mm of
    ``target_xy``, with both jaw tips between the table and a resting die's top face, or None."""
    model, data = load_model()
    die_top = TABLE_Z_M + 2 * DIE_HALF_EDGE_M
    grid = _arm_grid(cfg)
    xyz = fk_tip_batch(fk, grid)
    r_t = float(np.hypot(*target_xy))
    lo = np.array([box.x[0], box.y[0], box.z[0]])
    hi = np.array([box.x[1], box.y[1], box.z[1]])
    z_ok = (xyz[:, 2] >= box.z[0]) & (xyz[:, 2] <= die_top)
    cand = grid[z_ok & (np.abs(np.hypot(xyz[:, 0], xyz[:, 1]) - r_t) < 0.004)]
    for t in cand:
        jz = _jaw_tip_z(model, data, t)  # pan does not change heights
        if not ((jz > TABLE_Z_M).all() and (jz < die_top).all()):
            continue
        pans = np.tile(t, (801, 1))
        pans[:, 0] = np.arange(1647, 2448)
        p = fk_tip_batch(fk, pans)
        inside = ((p >= lo) & (p <= hi)).all(axis=1)
        err = np.where(inside, np.hypot(p[:, 0] - target_xy[0], p[:, 1] - target_xy[1]), np.inf)
        k = int(np.argmin(err))
        if err[k] < 0.005:
            return pans[k], float(p[k, 2])
    return None


@pytest.mark.parametrize("where", ["well", "zone"])
def test_jaws_reach_a_resting_die_inside_the_shipped_box(setup, where):
    cfg, _, fk = setup
    assert cfg.box.z[0] == pytest.approx(TABLE_Z_M + 0.005)
    xy = DIE_TO_ZONE.die_start.well_center_xy_m if where == "well" else DIE_TO_ZONE.zone.center_xy_m
    found = _reach_resting_die(cfg, fk, cfg.box, xy)
    assert found is not None, f"no in-box pose reaches a die resting at the {where} {xy}"
    ticks, tip_z = found
    model, data = load_model()
    jz = _jaw_tip_z(model, data, ticks)
    die_top = TABLE_Z_M + 2 * DIE_HALF_EDGE_M
    # both jaw tips are between the table and the die's top face: they can close on a resting die
    assert (jz > TABLE_Z_M).all() and (jz < die_top).all(), (where, ticks.tolist(), tip_z, jz)


def test_with_the_old_floor_no_pose_reaches_a_resting_die(setup):
    """Shown to fail: the pre-2026-10-06 floor (z = 0.010) keeps the tip 3 cm above the table."""
    cfg, _, fk = setup
    old = dataclasses.replace(cfg.box, z=(0.010, cfg.box.z[1]))
    assert _reach_resting_die(cfg, fk, old, DIE_TO_ZONE.die_start.well_center_xy_m) is None

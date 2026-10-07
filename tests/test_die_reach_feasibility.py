"""L1 feasibility of the die task under the shipped safety box (plan §10 2026-10-06 "Workspace box floor" and
"Gripper floor check", 2026-10-07 "Gripper floor clearance").

The jaws must be able to reach a die resting on the table at the start well and at the zone THROUGH THE CLAMP
CHAIN: the commanded pose passes ``So101FK.in_box_gripper`` (tip box with the floor at z = -0.015, AND the lowest
wrist/gripper floor point at or above ``box.gripper_floor_z_m`` = table + 1 mm). On top of the safety points the pose
is checked against the EXACT MuJoCo collision meshes (wrist, gripper, moving jaw at the commanded gripper angle):
their lowest vertex must also be at or above the gripper floor, and both jaw-tip collision boxes must have their
centres between the table and the resting die's top face. The test reports the margin of the best pose.

Shown to fail: the old box floor (z = 0.010) and the superseded 5 mm gripper clearance (best margin -1.1 mm on the
exact mesh, 2026-10-07) both leave no such pose.
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
from so101_rl.sim.gripper_map import gripper_ticks_to_rad
from so101_rl.sim.model import TABLE_Z_M, load_model

_RAD = 2 * np.pi / 4096
DATA = Path(__file__).parent / "data"
DIE_TOP = TABLE_Z_M + 2 * DIE_HALF_EDGE_M
MESH_BODIES = ("wrist", "gripper", "moving_jaw_so101_v1")


@pytest.fixture(scope="module")
def setup():
    from so101_rl.config import load_calibration, load_config

    cal = load_calibration(DATA / "so101_follower.calib.json")
    cfg = load_config(calibration=cal)
    fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), cal)
    return cfg, cal, fk


def _arm_grid(cfg) -> np.ndarray:
    """Pan centred; shoulder / elbow / wrist_flex on a grid around the gripper-down poses; roll centred, gripper
    open (2350)."""
    sl, el, wf = np.meshgrid(np.arange(1900, 2301, 5), np.arange(2100, 2801, 5), np.arange(2300, 2901, 10), indexing="ij")
    t = np.zeros((sl.size, 6), np.int64)
    t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5] = 2047, sl.ravel(), el.ravel(), wf.ravel(), 2047, 2350
    lo, hi = cfg.joints.min_ticks.astype(np.int64), cfg.joints.max_ticks.astype(np.int64)
    return t[((t >= lo) & (t <= hi)).all(axis=1)]


class ExactMesh:
    """MuJoCo collision meshes of the wrist / gripper / moving jaw (group 3) and the jaw-tip collision boxes."""

    def __init__(self, cal) -> None:
        self.cal = cal
        self.model, self.data = load_model()
        m = self.model
        self.geoms = []
        for b in MESH_BODIES:
            bid = m.body(b).id
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 3:
                    mid = m.geom_dataid[g]
                    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                    self.geoms.append((g, v.astype(np.float64)))
        self.jaws = [m.geom(n).id for n in ("jaw_fixed_tip", "jaw_moving_tip")]

    def heights(self, ticks) -> tuple[float, np.ndarray]:
        """(lowest mesh vertex z, jaw-tip box centre z [2]) with the gripper at its commanded angle. Pan does not
        change heights."""
        d = self.data
        d.qpos[:] = 0
        d.qpos[:5] = (np.asarray(ticks[:5], np.float64) - 2047) * _RAD  # drive_mode 0 on every arm joint
        d.qpos[5] = gripper_ticks_to_rad(int(ticks[5]), self.cal)
        mujoco.mj_kinematics(self.model, d)
        low = min(
            float((v @ d.geom_xmat[g].reshape(3, 3)[2] + d.geom_xpos[g][2]).min()) for g, v in self.geoms
        )
        return low, np.array([d.geom_xpos[j][2] for j in self.jaws])


def _reach_resting_die(cfg, cal, fk, box, target_xy) -> tuple[np.ndarray, float, dict] | None:
    """The pose with the largest margin that passes the safety box test (exact, as the BOX clamp tests the target),
    has its exact-mesh lowest vertex at or above ``box.gripper_floor_z_m``, the tip within 5 mm of ``target_xy`` and
    both jaw-tip centres between the table and the die's top face; or None."""
    mesh = ExactMesh(cal)
    grid = _arm_grid(cfg)
    xyz = fk_tip_batch(fk, grid)
    r_t = float(np.hypot(*target_xy))
    near = (np.abs(np.hypot(xyz[:, 0], xyz[:, 1]) - r_t) < 0.004) & (xyz[:, 2] >= box.z[0]) & (xyz[:, 2] <= DIE_TOP)
    best = None
    tmp = np.zeros(3)
    for t in grid[near]:
        floor_z = fk.tip_and_floor_z_from_ticks(t, tmp)  # pan-invariant
        if floor_z < box.gripper_floor_z_m:
            continue
        low, jz = mesh.heights(t)
        margin = min(low - box.gripper_floor_z_m, DIE_TOP - jz.max(), jz.min() - TABLE_Z_M)
        if margin <= 0 or (best is not None and margin <= best[1]):
            continue
        pans = np.tile(t, (801, 1))
        pans[:, 0] = np.arange(1647, 2448)
        p = fk_tip_batch(fk, pans)
        inside = np.array([fk.in_box_gripper(pp, floor_z, box) for pp in p])
        err = np.where(inside, np.hypot(p[:, 0] - target_xy[0], p[:, 1] - target_xy[1]), np.inf)
        k = int(np.argmin(err))
        if err[k] < 0.005:
            info = {"tip_z": float(p[k, 2]), "points_floor_z": floor_z, "mesh_low_z": low, "jaw_z": jz.round(4).tolist()}
            best = (pans[k], margin, info)
    return best


@pytest.mark.parametrize("where", ["well", "zone"])
def test_jaws_reach_a_resting_die_inside_the_shipped_box(setup, where):
    cfg, cal, fk = setup
    assert cfg.box.z[0] == pytest.approx(TABLE_Z_M + 0.005)
    assert cfg.box.gripper_floor_z_m == pytest.approx(TABLE_Z_M + 0.001)
    xy = DIE_TO_ZONE.die_start.well_center_xy_m if where == "well" else DIE_TO_ZONE.zone.center_xy_m
    found = _reach_resting_die(cfg, cal, fk, cfg.box, xy)
    assert found is not None, f"no in-box pose reaches a die resting at the {where} {xy}"
    ticks, margin, info = found
    print(f"{where}: ticks {ticks.tolist()} margin {margin * 1000:.2f} mm {info}")
    xyz = np.zeros(3)
    zf = fk.tip_and_floor_z_from_ticks(ticks, xyz)
    assert fk.in_box_gripper(xyz, zf, cfg.box)  # passes the BOX clamp
    assert margin >= 0.001, f"{where}: margin {margin * 1000:.2f} mm"  # reported; >= 1 mm to the nearest limit


@pytest.mark.parametrize(
    "variant",
    ["old_box_floor", "gripper_clearance_5mm"],
)
def test_superseded_floors_leave_no_pose(setup, variant):
    """Shown to fail: the pre-2026-10-06 tip floor (z = 0.010) and the superseded 5 mm gripper clearance."""
    cfg, cal, fk = setup
    if variant == "old_box_floor":
        box = dataclasses.replace(cfg.box, z=(0.010, cfg.box.z[1]))
    else:
        box = dataclasses.replace(cfg.box, gripper_floor_z_m=TABLE_Z_M + 0.005)
    assert _reach_resting_die(cfg, cal, fk, box, DIE_TO_ZONE.die_start.well_center_xy_m) is None

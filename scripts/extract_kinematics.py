"""Extract the SO-101 kinematic chain from the upstream MJCF into config/so101_kinematics.yaml.

The yaml is the only input of the pure-numpy FK (safety/fk.py, no placo/mujoco at runtime, L7).
Per joint: body pose relative to its parent body, hinge axis (in the body frame), range.
`gripper_tip` is the `gripperframe` site of the MJCF (attached to the `gripper` body, i.e. after wrist_roll).

`floor_points` (plan §10 2026-10-06 "Gripper floor check", 2026-10-07 "Gripper floor clearance"): fixed points in the
`wrist` and `gripper` body frames; their lowest z must stay >= box.gripper_floor_z_m (table + 1 mm), so no part of
the gripper is driven into the table.
Derived from the MJCF collision meshes: the vertices of the `wrist` body, of the `gripper` body (fixed jaw + gripper
servo) and of the moving jaw swept over the WHOLE gripper joint range (so the check does not depend on the gripper
reading, conservatively), reduced to their extreme points (support points over 20 000 directions), then a greedy
selection until, for every one of 8 000 directions on the sphere, the selected points' support is within
`FLOOR_TOLERANCE_M` of the full cloud's: in ANY orientation the lowest selected point is at most that far above the
lowest mesh vertex (tests/test_gripper_floor.py re-checks it against MuJoCo on random poses).

    python scripts/extract_kinematics.py [--mjcf PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

from so101_rl.config import JOINT_NAMES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MJCF = ROOT / "assets" / "so101" / "upstream" / "so101_new_calib.xml"
DEFAULT_OUT = ROOT / "src" / "so101_rl" / "config" / "so101_kinematics.yaml"
TIP_SITE = "gripperframe"
FLOOR_BODIES = ("wrist", "gripper")  # point sets, in these body frames (after wrist_flex / wrist_roll)
MOVING_JAW_BODY = "moving_jaw_so101_v1"
FLOOR_TOLERANCE_M = 0.0005
_JAW_SWEEP_STEPS = 25
_HULL_DIRS = 20_000
_SELECT_DIRS = 8_000


def _floats(a: np.ndarray) -> list[float]:
    return [float(f"{x:.9g}") for x in a]


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _quat_mat(q: np.ndarray) -> np.ndarray:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, np.float64))
    return r.reshape(3, 3)


def _body_mesh_vertices(model: mujoco.MjModel, body: str) -> np.ndarray:
    """Vertices of the body's collision meshes (geom group 3), in the body frame."""
    bid = model.body(body).id
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 3:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        out.append(v.astype(np.float64) @ _quat_mat(model.geom_quat[g]).T + model.geom_pos[g])
    if not out:
        raise SystemExit(f"body {body!r}: no collision mesh")
    return np.concatenate(out)


def _support_points(cloud: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    idx: set[int] = set()
    for k in range(0, len(dirs), 500):
        idx.update(np.argmax(cloud @ dirs[k : k + 500].T, axis=0).tolist())
    return cloud[sorted(idx)]


def _greedy_cover(cloud: np.ndarray, dirs: np.ndarray, tol: float) -> np.ndarray:
    """Fewest points (greedy) whose support is within ``tol`` of the cloud's in every direction of ``dirs``."""
    h = (cloud @ dirs.T).max(axis=0)
    cur = np.full(len(dirs), -np.inf)
    chosen: list[int] = []
    while True:
        gap = h - cur
        j = int(np.argmax(gap))
        if gap[j] <= tol:
            return cloud[chosen]
        v = int(np.argmax(cloud @ dirs[j]))
        chosen.append(v)
        cur = np.maximum(cur, cloud[v] @ dirs.T)


def floor_points(model: mujoco.MjModel) -> dict:
    """``{"wrist": [[x, y, z], ...], "gripper": [...]}`` (body frames) per the module docstring."""
    jid = model.joint("gripper").id
    if [float(a) for a in model.jnt_axis[jid]] != [0.0, 0.0, 1.0] or np.any(model.jnt_pos[jid] != 0):
        raise SystemExit("gripper joint: expected a +z hinge at the moving-jaw body origin")
    jaw = _body_mesh_vertices(model, MOVING_JAW_BODY)
    jb = model.body(MOVING_JAW_BODY).id
    if model.body(int(model.body_parentid[jb])).name != "gripper":
        raise SystemExit("the moving jaw must be a child of the gripper body")
    p, r = model.body_pos[jb], _quat_mat(model.body_quat[jb])
    sweep = []
    for a in np.linspace(*model.jnt_range[jid], _JAW_SWEEP_STEPS):
        c, s = np.cos(a), np.sin(a)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        sweep.append(jaw @ (r @ rz).T + p)
    clouds = {
        "wrist": _body_mesh_vertices(model, "wrist"),
        "gripper": np.concatenate([_body_mesh_vertices(model, "gripper"), *sweep]),
    }
    hull_dirs, sel_dirs = _fibonacci_sphere(_HULL_DIRS), _fibonacci_sphere(_SELECT_DIRS)
    out = {}
    for body in FLOOR_BODIES:
        pts = _greedy_cover(_support_points(clouds[body], hull_dirs), sel_dirs, FLOOR_TOLERANCE_M)
        out[body] = [_floats(np.round(q, 6)) for q in pts]
    return out


def extract(mjcf: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    joints = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"joint {name!r} not found in {mjcf}")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise SystemExit(f"joint {name!r} is not a hinge")
        bid = int(model.jnt_bodyid[jid])
        joints.append(
            {
                "name": name,
                "mjcf_joint": name,
                "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid),
                "parent_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[bid])),
                "parent_body_pos": _floats(model.body_pos[bid]),
                "parent_body_quat": _floats(model.body_quat[bid]),  # wxyz
                "axis": _floats(model.jnt_axis[jid]),
                "range_rad": _floats(model.jnt_range[jid]),
            }
        )
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
    if sid < 0:
        raise SystemExit(f"site {TIP_SITE!r} not found; add a 0.0 m offset site on the fixed jaw")
    tip_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[sid]))
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    return {
        "source": {"mjcf": mjcf.name, "note": "generated by scripts/extract_kinematics.py; do not edit"},
        "convention": {"quat": "wxyz", "angle_unit": "rad", "zero_pose": "LeRobot homing pose (qpos=0)"},
        "base": {"body": "base", "pos": _floats(model.body_pos[base]), "quat": _floats(model.body_quat[base])},
        "joints": joints,
        "gripper_tip": {
            "site": TIP_SITE,
            "body": tip_body,
            "pos": _floats(model.site_pos[sid]),
            "quat": _floats(model.site_quat[sid]),
            "note": "MJCF `gripperframe` site on the fixed jaw body (child of the wrist_roll joint)",
        },
        "floor_points": {
            "note": (
                "box FLOOR points (plan sec. 10, 2026-10-06 'Gripper floor check'): extreme points of the wrist / gripper "
                "collision meshes, moving jaw swept over its whole range; lowest point at most tolerance_m above the "
                "lowest mesh vertex in any orientation (scripts/extract_kinematics.py)"
            ),
            "tolerance_m": FLOOR_TOLERANCE_M,
            **floor_points(model),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    data = extract(args.mjcf)
    args.out.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None))
    fp = data["floor_points"]
    print(f"wrote {args.out} ({len(data['joints'])} joints, floor points: {len(fp['wrist'])} wrist, {len(fp['gripper'])} gripper)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

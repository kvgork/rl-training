"""Extract the arm's visual surface into src/so101_rl/perception/arm_points.npz (the arm mask, plan section 10
2026-10-17 "Perception: arm mask").

The perception process masks the arm's own pixels (``perception/arm_mask.py``) without MuJoCo at runtime: it places
these points with the numpy FK of ``config/so101_kinematics.yaml``. The points are :class:`sim.arm_cloud.ArmSurface`
samples (the visual meshes, group 2, area-weighted, 3 mm spacing, seed 0: the cloud the arm-shape camera
calibration fits) expressed in their bodies' frames, plus the 8 corners of each of the four ``jaw_col`` boxes
(``sim/model.py``: the fingers, whose hull is the grasp region the mask leaves open so a die held between the jaws
stays visible). ``tests/test_arm_mask.py`` regenerates them and compares.

    python scripts/extract_arm_points.py [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "src" / "so101_rl" / "perception" / "arm_points.npz"
SPACING_M = 0.003


def arm_points(model: mujoco.MjModel | None = None) -> dict[str, np.ndarray]:
    """``{"bodies": [B] str, "body": [P] uint8 index into bodies, "points": [P, 3] float32, "grasp_body": [K] uint8,
    "grasp_corners": [K, 3] float32}``, points and corners in their bodies' frames (metres)."""
    from so101_rl.perception.arm_mask import ARM_BODIES
    from so101_rl.sim.arm_cloud import ArmSurface
    from so101_rl.sim.model import _JAW_BOXES, load_model

    if model is None:
        model, _ = load_model(camera=False)
    surf = ArmSurface(model, spacing_m=SPACING_M, seed=0)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    pw, _ = surf.points(model.qpos0)
    gbody = model.geom_bodyid[surf.geom]
    names = [model.body(int(b)).name for b in gbody]
    if set(names) - set(ARM_BODIES):
        raise ValueError(f"arm meshes on bodies outside ARM_BODIES: {sorted(set(names) - set(ARM_BODIES))}")
    body = np.array([ARM_BODIES.index(n) for n in names], dtype=np.uint8)
    R = data.xmat[gbody].reshape(-1, 3, 3)
    pb = np.einsum("pji,pj->pi", R, pw - data.xpos[gbody])  # R^T (p - x)

    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    gb, gc = [], []
    for name, *_ in _JAW_BOXES:
        g = model.geom(name)
        rg = np.zeros(9)
        mujoco.mju_quat2Mat(rg, g.quat)
        gc.append((signs * g.size) @ rg.reshape(3, 3).T + g.pos)  # geom frame -> body frame
        gb += [ARM_BODIES.index(model.body(int(g.bodyid[0])).name)] * len(signs)
    return {
        "bodies": np.array(ARM_BODIES),
        "body": body,
        "points": pb.astype(np.float32),
        "grasp_body": np.array(gb, dtype=np.uint8),
        "grasp_corners": np.concatenate(gc).astype(np.float32),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    d = arm_points()
    np.savez_compressed(a.out, **d)
    print(f"{a.out}: {len(d['points'])} points on {len(d['bodies'])} bodies, {len(d['grasp_corners'])} grasp corners")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

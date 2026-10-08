"""``camera-overlay``: the real D435 frame next to the sim render at the probed joint angles (camera parity C.4).

``pixi run camera-overlay [--frames 5]``

Captures aligned colour + depth (streaming only), probes the joint angles read-only (``servo/probe.py``), renders the
scene from ``d435_color`` (``calib/camera.yaml``) at those angles, and writes ``runs/camera-overlay/<ts>.png``:
real | sim | 50 % blend, with both arm silhouettes outlined (real green, sim magenta). The arm silhouettes should
coincide. ``<ts>.json`` holds the numbers.

**Arm-mask IoU (gate >= 0.8).** Sim mask: the pixels whose segmentation id is an arm geom (``base`` and below).
Real mask, from depth only (independent of the sim's silhouette shape): pixels whose point (through
``T_base_color``) lies more than 8 mm above the table, inside the sim mask dilated by ``--search-px`` (a generous
box around the arm's expected projection; clutter elsewhere is ignored), and within ``--depth-tol-m`` of the sim's
arm depth nearby (so a cup next to the arm is not counted as arm). IoU = |real AND sim| / |real OR sim|.

Exit codes: 0 IoU >= 0.8, 1 below, 3 hardware refused (probe or camera).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_rl.perception.camera_calib import CAMERA_YAML, CameraCalib, load_camera_calib

RUNS = Path("runs/camera-overlay")
MIN_IOU = 0.8
ABOVE_TABLE_M = 0.008


@dataclass
class OverlayResult:
    iou: float
    real_px: int
    sim_px: int
    sim_rgb: np.ndarray
    sim_mask: np.ndarray
    real_mask: np.ndarray
    image: np.ndarray  # BGR side-by-side


def render_sim(calib: CameraCalib, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rgb, depth, arm mask) of the scene from ``d435_color`` at ``qpos``."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from so101_rl.sim.d435 import COLOR_CAM
    from so101_rl.sim.model import build_spec

    spec = build_spec(camera=calib)
    spec.visual.global_.offwidth = calib.color.width
    spec.visual.global_.offheight = calib.color.height
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    base = model.body("base").id
    arm_geoms = np.zeros(model.ngeom + 1, bool)
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        while b > 0 and b != base:
            b = int(model.body_parentid[b])
        arm_geoms[g] = b == base
    r = mujoco.Renderer(model, calib.color.height, calib.color.width)
    try:
        r.update_scene(data, camera=COLOR_CAM)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=COLOR_CAM)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=COLOR_CAM)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
    finally:
        r.close()
    gid = seg[..., 0]
    is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    mask = is_geom & arm_geoms[np.where(gid >= 0, gid, model.ngeom)]
    return rgb, depth, mask


def real_arm_mask(depth_m: np.ndarray, calib: CameraCalib, table_z_m: float, sim_mask: np.ndarray, sim_depth: np.ndarray, *, search_px: int = 25, depth_tol_m: float = 0.06) -> np.ndarray:
    import cv2

    T = calib.T_base_color
    H, W = depth_m.shape
    v, u = np.mgrid[0:H, 0:W]
    p = calib.color.deproject(u, v, depth_m)
    z = p @ T[2, :3] + T[2, 3]
    valid = depth_m > 0
    k = np.ones((2 * search_px + 1, 2 * search_px + 1), np.uint8)
    region = cv2.dilate(sim_mask.astype(np.uint8), k).astype(bool)
    sd = np.where(sim_mask, sim_depth, np.float32(1e3)).astype(np.float32)
    near_sim_depth = cv2.erode(sd, k)  # nearest-ish sim arm depth around each pixel (min filter)
    return valid & (z > table_z_m + ABOVE_TABLE_M) & region & (np.abs(depth_m - near_sim_depth) < depth_tol_m)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


def compose(real_rgb: np.ndarray, sim_rgb: np.ndarray, real_mask: np.ndarray | None, sim_mask: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    real = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2BGR)
    sim = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
    blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0.0)
    for mask, col in ((sim_mask, (255, 0, 255)), (real_mask, (0, 255, 0))):
        if mask is None:
            continue
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(blend, cs, -1, col, 1)
    img = np.concatenate([real, sim, blend], axis=1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def overlay(calib: CameraCalib, real_rgb: np.ndarray, real_depth: np.ndarray | None, qpos: np.ndarray, table_z_m: float, **kw) -> OverlayResult:
    sim_rgb, sim_depth, sim_mask = render_sim(calib, qpos)
    rm = None if real_depth is None else real_arm_mask(real_depth, calib, table_z_m, sim_mask, sim_depth, **kw)
    score = float("nan") if rm is None else iou(rm, sim_mask)
    pose = calib.require_pose()
    lines = [
        f"real | sim d435_color ({pose.method} pose, {pose.date}) | 50% blend: real arm green, sim arm magenta",
        f"arm-mask IoU {score:.3f} (gate >= {MIN_IOU})" if rm is not None else "arm-mask IoU: n/a (no depth)",
    ]
    img = compose(real_rgb, sim_rgb, rm, sim_mask, lines)
    return OverlayResult(score, 0 if rm is None else int(rm.sum()), int(sim_mask.sum()), sim_rgb, sim_mask, rm if rm is not None else np.zeros_like(sim_mask), img)


def main(
    argv: list[str] | None = None,
    *,
    probe: Callable[[], np.ndarray] | None = None,
    camera: Callable[[CameraCalib, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> int:
    import cv2

    from so101_rl.config import load_calibration, load_config
    from so101_rl.perception.camera_calibrate import capture_d435
    from so101_rl.perception.table_plane import median_depth
    from so101_rl.servo.probe import ProbeRefused, checked_ticks, probe_joints
    from so101_rl.sim.arm_cloud import qpos_from_ticks
    from so101_rl.sim.model import TABLE_Z_M, load_model

    ap = argparse.ArgumentParser(prog="python -m so101_rl.perception.camera_overlay", description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--camera-yaml", type=Path, default=CAMERA_YAML)
    ap.add_argument("--out-dir", type=Path, default=RUNS)
    ap.add_argument("--port", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--search-px", type=int, default=25)
    ap.add_argument("--depth-tol-m", type=float, default=0.06)
    a = ap.parse_args(argv)
    calib = load_camera_calib(a.camera_yaml)
    cfg = load_config()
    cal = load_calibration(Path(a.calibration or cfg.calibration_path).expanduser())
    try:
        ticks = np.asarray(probe() if probe else checked_ticks(probe_joints(cal, port=a.port or cfg.port)), dtype=np.int32)
    except ProbeRefused as e:
        print(f"camera-overlay: probe refused: {e}", file=sys.stderr)
        return 3
    try:
        rgb, frames = (camera or capture_d435)(calib, a.frames)
    except RuntimeError as e:
        print(f"camera-overlay: camera failed: {e}", file=sys.stderr)
        return 3
    model = load_model(camera=False)[0]
    q = qpos_from_ticks(model, ticks, cal)
    res = overlay(calib, rgb, median_depth(frames), q, TABLE_Z_M, search_px=a.search_px, depth_tol_m=a.depth_tol_m)
    ts = _dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    png = a.out_dir / f"{ts}.png"
    cv2.imwrite(str(png), res.image)
    rep = {"iou": res.iou, "gate": MIN_IOU, "passed": res.iou >= MIN_IOU, "real_arm_px": res.real_px, "sim_arm_px": res.sim_px,
           "ticks": ticks.tolist(), "pose_method": calib.require_pose().method, "png": str(png)}  # fmt: skip
    (a.out_dir / f"{ts}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if res.iou >= MIN_IOU else 1


if __name__ == "__main__":
    sys.exit(main())

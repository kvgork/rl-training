"""``camera-calibrate --method arm-shape``: camera-only extrinsics of the D435 (camera parity C.2).

``pixi run camera-calibrate --method arm-shape [--poses K] [--frames 10]``

Per pose: the read-only joint probe (``servo/probe.py``; refuses if the servo process owns the port, never writes a
register) and ``--frames`` aligned colour + depth frames (``D435Camera``, streaming only). With ``--poses K > 1``
the operator moves the arm BY HAND with torque off between poses (attended; optional). Then:

1. the table plane (``perception/table_plane.py``) on the per-pixel median depth of the first pose;
2. the arm-shape registration (``perception/arm_registration.py``) of the FK surface at the probed angles against
   the depth cloud, constrained by the plane, from several seeds;
3. the gate (RMS <= 5 mm, model coverage >= 0.8, plane-vs-6-DOF <= 5 mm / 2 deg, FK tip <= 8 px).

Artifacts go to ``runs/camera-calib/<ts>/``: ``report.json``, ``scene_<k>.ply`` / ``model_<k>.ply`` (base frame,
registered), ``overlay_<k>.png`` (the colour frame with the registered model points and the FK tip), and the raw
capture ``capture_<k>.npz``. ``calib/camera.yaml`` is rewritten (pose method ``arm-shape``, residual, bootstrap
uncertainty, table plane; intrinsics untouched) ONLY when the gate passes; otherwise the previous file stays.

Exit codes: 0 PASS (file written), 1 gate FAIL (nothing written), 3 hardware refused (probe or camera).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_rl.perception.arm_registration import ArmShapeRegistration, ArmShapeResult, Observation
from so101_rl.perception.camera_calib import (
    CAMERA_YAML,
    CameraCalib,
    CameraPose,
    TablePlane,
    load_camera_calib,
    save_camera_calib,
    today,
)
from so101_rl.perception.table_plane import PlaneFit, depth_to_points, fit_table_plane, median_depth

RUNS = Path("runs/camera-calib")


@dataclass
class Capture:
    rgb: np.ndarray  # [H, W, 3] uint8 (last frame)
    depth_frames: np.ndarray  # [K, H, W] float32 metres, aligned to colour
    ticks: np.ndarray  # int32[6] probed


def capture_d435(calib: CameraCalib, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """``n_frames`` aligned frames from the D435 of ``calib`` (streaming only)."""
    from so101_rl.perception.camera import D435Camera

    c = calib.color
    cam = D435Camera(serial=calib.serial, width=c.width, height=c.height, fps=c.fps, timeout_ms=5000)
    cam.start()
    try:
        rgb = np.zeros((c.height, c.width, 3), np.uint8)
        d = np.zeros((c.height, c.width), np.float32)
        frames = []
        for _ in range(n_frames):
            if not cam.read(rgb, d).ok:
                raise RuntimeError("D435 frame timeout")
            frames.append(d.copy())
        return rgb.copy(), np.stack(frames)
    finally:
        cam.stop()


def observations(model, calibration, calib: CameraCalib, captures: list[Capture]) -> tuple[list[Observation], PlaneFit]:
    from so101_rl.sim.arm_cloud import ArmSurface, qpos_from_ticks

    surf = ArmSurface(model)
    obs = []
    plane = None
    for cap in captures:
        dm = median_depth(cap.depth_frames)
        pts, pix = depth_to_points(dm, calib.color)
        if plane is None:
            plane = fit_table_plane(pts)
        q = qpos_from_ticks(model, cap.ticks, calibration)
        mp, mn = surf.points(q)
        obs.append(Observation(pts, pix, mp, mn, surf.site(q, "gripper_tip"), dm))
    assert plane is not None
    return obs, plane


def calibrate(model, calibration, calib: CameraCalib, captures: list[Capture], table_z_m: float, *, seed: int = 0, bootstrap: int = 20) -> tuple[ArmShapeResult, PlaneFit, list[Observation]]:
    obs, plane = observations(model, calibration, calib, captures)
    reg = ArmShapeRegistration(obs, plane, calib.color, table_z_m, seed=seed)
    return reg.run(bootstrap=bootstrap), plane, obs


def write_ply(path: Path, pts: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, pts, fmt="%.5f")


def overlay_png(path: Path, rgb: np.ndarray, T: np.ndarray, calib: CameraCalib, model_pts: np.ndarray, tip: np.ndarray, text: list[str]) -> None:
    import cv2

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    Ti = np.linalg.inv(T)
    pc = model_pts @ Ti[:3, :3].T + Ti[:3, 3]
    uv = calib.color.project(pc[pc[:, 2] > 0.05])
    for u, v in uv[:: max(1, len(uv) // 4000)].astype(int):
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            img[v, u] = (0, 255, 0)
    tc = Ti[:3, :3] @ tip + Ti[:3, 3]
    tu = calib.color.project(tc[None])[0].astype(int)
    cv2.drawMarker(img, (int(tu[0]), int(tu[1])), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    for i, s in enumerate(text):
        cv2.putText(img, s, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def save_artifacts(out: Path, res: ArmShapeResult, plane: PlaneFit, obs: list[Observation], captures: list[Capture], calib: CameraCalib, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    T = res.T_base_color
    for k, (o, cap) in enumerate(zip(obs, captures, strict=True)):
        write_ply(out / f"scene_{k}.ply", o.pts_cam[:: max(1, len(o.pts_cam) // 50_000)] @ T[:3, :3].T + T[:3, 3])
        write_ply(out / f"model_{k}.ply", o.model_pts)
        np.savez_compressed(out / f"capture_{k}.npz", rgb=cap.rgb, depth=cap.depth_frames, ticks=cap.ticks)
        verdict = "PASS" if res.passed else "FAIL"
        overlay_png(out / f"overlay_{k}.png", cap.rgb, T, calib, o.model_pts, o.tip_base, [
            f"arm-shape {verdict}: rms {res.rms_mm:.2f} mm  coverage {res.coverage:.2f}  tip {res.tip_px_err:.1f} px",
            f"plane dh {res.plane_height_err_mm:.1f} mm  dtilt {res.plane_tilt_err_deg:.2f} deg",
        ])  # fmt: skip
    rep = {
        **res.to_json(),
        "table_plane": {"normal_up_color": plane.normal_up.tolist(), "height_m": plane.height_m, "inlier_fraction": plane.inlier_fraction},
        **extra,
    }
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))


def updated_calib(calib: CameraCalib, res: ArmShapeResult, plane: PlaneFit, n_frames: int, n_poses: int) -> CameraCalib:
    pose = CameraPose(
        matrix=res.T_base_color,
        method="arm-shape",
        date=today(),
        residual_mm=res.rms_mm,
        trans_mm=res.uncertainty_trans_mm,
        rot_deg=res.uncertainty_rot_deg,
        notes=f"camera-calibrate --method arm-shape, {n_poses} pose(s); coverage {res.coverage:.2f}, tip {res.tip_px_err:.1f} px, "
        f"plane dh {res.plane_height_err_mm:.1f} mm / {res.plane_tilt_err_deg:.2f} deg; uncertainty = 2 sigma bootstrap",
    )
    tp = TablePlane(plane.normal_up, plane.height_m, plane.inlier_fraction, n_frames, today())
    return calib.with_pose(pose, tp)


def main(
    argv: list[str] | None = None,
    *,
    probe: Callable[[], np.ndarray] | None = None,
    camera: Callable[[CameraCalib, int], tuple[np.ndarray, np.ndarray]] | None = None,
    wait: Callable[[str], None] | None = None,
) -> int:
    from so101_rl.config import load_calibration, load_config
    from so101_rl.servo.probe import ProbeRefused, checked_ticks, probe_joints
    from so101_rl.sim.model import TABLE_Z_M, load_model

    ap = argparse.ArgumentParser(prog="python -m so101_rl.perception.camera_calibrate", description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=("arm-shape",), required=True)
    ap.add_argument("--poses", type=int, default=1, help="K >= 1; K > 1 = the operator moves the arm by hand, torque off")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--camera-yaml", type=Path, default=CAMERA_YAML)
    ap.add_argument("--out-dir", type=Path, default=RUNS)
    ap.add_argument("--port", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.poses < 1 or a.frames < 1:
        ap.error("--poses and --frames must be >= 1")
    calib = load_camera_calib(a.camera_yaml)
    cfg = load_config()
    cal = load_calibration(Path(a.calibration or cfg.calibration_path).expanduser())
    port = a.port or cfg.port
    do_probe = probe or (lambda: checked_ticks(probe_joints(cal, port=port), require_torque_off=a.poses > 1))
    do_cam = camera or capture_d435
    do_wait = wait or (lambda msg: input(msg))
    captures = []
    for k in range(a.poses):
        if k > 0:
            do_wait(f"pose {k + 1}/{a.poses}: move the arm BY HAND (torque stays off), let go, then press Enter ")
        try:
            ticks = np.asarray(do_probe(), dtype=np.int32)
        except ProbeRefused as e:
            print(f"camera-calibrate: probe refused: {e}", file=sys.stderr)
            return 3
        try:
            rgb, depth = do_cam(calib, a.frames)
        except RuntimeError as e:
            print(f"camera-calibrate: camera failed: {e}", file=sys.stderr)
            return 3
        captures.append(Capture(rgb, depth, ticks))
    model = load_model(camera=False)[0]
    res, plane, obs = calibrate(model, cal, calib, captures, TABLE_Z_M, seed=a.seed)
    ts = _dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out = a.out_dir / ts
    save_artifacts(out, res, plane, obs, captures, calib, {"poses": a.poses, "frames": a.frames, "ticks": [c.ticks.tolist() for c in captures]})
    print(json.dumps({k: v for k, v in res.to_json().items() if k != "seeds"}, indent=1, default=float))
    if not res.passed:
        print(f"camera-calibrate: GATE FAIL ({'; '.join(res.failures)}); {a.camera_yaml} NOT changed; artifacts {out}", file=sys.stderr)
        return 1
    save_camera_calib(updated_calib(calib, res, plane, a.frames, a.poses), a.camera_yaml)
    print(f"camera-calibrate: PASS; wrote {a.camera_yaml}; artifacts {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

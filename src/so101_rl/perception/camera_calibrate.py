"""``camera-calibrate --method arm-shape``: camera-only extrinsics of the D435 (camera parity C.2).

``pixi run camera-calibrate --method arm-shape [--poses K] [--frames 10]``
``pixi run camera-calibrate --method arm-shape --from-capture runs/camera-calib/<ts>`` (offline replay)

Per pose: the read-only joint probe (``servo/probe.py``; refuses if the servo process owns the port, never writes a
register) and ``--frames`` aligned colour + depth frames (``D435Camera``, streaming only). With ``--poses K > 1``
the operator moves the arm BY HAND with torque off between poses (attended; optional): the prompt
(:func:`hand_pose_prompt`) asks to turn ONLY the base ~60 deg (>= 45), the second pose that best separates a tilted
base from joint-zero errors (``diagnostics.tilt_vs_offsets``). The arm needs no particular posture: any posture that
stays put by itself with torque off and the gripper resting on the table (e.g. upper arm leaning forward, wrist bent
down), the whole arm incl. the gripper in the image and the operator out of view. Then:

1. the table plane (``perception/table_plane.py``) on the per-pixel median depth of the first pose;
2. the arm-shape registration (``perception/arm_registration.py``) of the FK surface at the probed angles against
   the depth cloud, constrained by the plane, from several seeds;
3. the table plane again on the workspace only (``refit_in_workspace``: 6-35 cm from the base axis under the first
   registration, so people / the bed / far clutter cannot tilt it), and the registration again on that plane;
4. the gate (RMS <= 5 mm, model coverage >= 0.8, plane-vs-6-DOF <= 5 mm / 2 deg, FK tip <= 8 px), plus report-only
   diagnostics (``report.json`` ``diagnostics``: the plane in the base frame, the tilt axis, the ICP information,
   the level-base / joint-offset alternative, the tilt + joint-offset joint fit with its verdict
   ``tilt_vs_offsets`` (printed as one line), the global-vs-workspace plane).

``--from-capture DIR`` replays the ``capture_<k>.npz`` files (colour, depth frames, probed ticks) of an earlier run:
no probe, no camera, no bus. It is REPORT-ONLY: it never writes ``calib/camera.yaml`` (the camera may have moved
since the capture), whatever the gate says; artifacts go to ``runs/camera-calib/<ts>-replay/``.

Artifacts go to ``runs/camera-calib/<ts>/``: ``report.json``, ``scene_<k>.ply`` / ``model_<k>.ply`` (base frame,
registered), ``overlay_<k>.png`` (the colour frame with the registered model points and the FK tip), and the raw
capture ``capture_<k>.npz``. ``calib/camera.yaml`` is rewritten (pose method ``arm-shape``, residual, bootstrap
uncertainty, table plane; intrinsics untouched) ONLY when the gate passes; otherwise the previous file stays.

Exit codes: 0 PASS (file written; replay: nothing written), 1 gate FAIL (nothing written), 3 hardware refused (probe
or camera), 2 usage (also: no capture in the ``--from-capture`` directory).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_rl.perception.arm_registration import (
    OFFSET_JOINTS,
    ArmShapeRegistration,
    ArmShapeResult,
    Observation,
)
from so101_rl.perception.camera_calib import (
    CAMERA_YAML,
    CameraCalib,
    CameraPose,
    TablePlane,
    load_camera_calib,
    save_camera_calib,
    today,
)
from so101_rl.perception.table_plane import (
    PlaneFit,
    depth_to_points,
    fit_table_plane,
    median_depth,
    refit_in_workspace,
)

RUNS = Path("runs/camera-calib")
# The hand-moved pose that best separates a tilted base from joint-zero errors (``tilt_vs_offsets``; 2026-10-10
# synthetic study, ARCHITECTURE §5): what matters is the BASE TURN (the shoulder_lift axis then lies at an angle to the
# tilt axis); raising the upper arm adds little. Any posture that stays put by itself with torque off will do (gripper
# resting on the table); only the base turn changes between poses.
SECOND_POSE_TURN_DEG = 60
SECOND_POSE_MIN_TURN_DEG = 45


def hand_pose_prompt(k: int, n: int) -> str:
    """The operator prompt before hand-moved pose ``k`` (0-based, ``k >= 1``) of ``n``. Either turn direction works
    equally well (synthetic study); a third pose turns the other way."""
    where = "either way" if k == 1 else "the other way, past the first pose"
    return (
        f"pose {k + 1}/{n}: BY HAND, torque stays off: turn ONLY the base about {SECOND_POSE_TURN_DEG} deg {where} "
        f"from the first pose (at least {SECOND_POSE_MIN_TURN_DEG} deg: this turn is what separates a tilted base from "
        "joint-zero errors); leave the other joints as they are. The arm needs no particular posture, only one that "
        "stays put by itself with the gripper resting on the table (e.g. upper arm leaning forward, wrist bent down). "
        "The whole arm including the gripper must be in the camera's view. Let go, step out of view, then press Enter "
    )


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
    adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in OFFSET_JOINTS])
    obs = []
    plane = None
    for cap in captures:
        dm = median_depth(cap.depth_frames)
        pts, pix = depth_to_points(dm, calib.color)
        if plane is None:
            plane = fit_table_plane(pts)
        q = qpos_from_ticks(model, cap.ticks, calibration)
        mp, mn = surf.points(q)

        def model_fn(delta: np.ndarray, q: np.ndarray = q) -> tuple[np.ndarray, np.ndarray]:
            qq = q.copy()
            qq[adr] += delta
            return surf.points(qq)

        obs.append(Observation(pts, pix, mp, mn, surf.site(q, "gripper_tip"), dm, model_fn))
    assert plane is not None
    return obs, plane


def _plane_json(p: PlaneFit) -> dict:
    return {"normal_up_color": p.normal_up.tolist(), "height_m": p.height_m, "inlier_fraction": p.inlier_fraction}


def calibrate(
    model, calibration, calib: CameraCalib, captures: list[Capture], table_z_m: float, *, seed: int = 0,
    bootstrap: int = 20, workspace_plane: bool = True, diagnostics: bool = True,
) -> tuple[ArmShapeResult, PlaneFit, list[Observation]]:  # fmt: skip
    """Register (module docstring, steps 1-4). With ``workspace_plane`` a first registration on the global plane
    places the base, the plane is refitted on the workspace (``refit_in_workspace``), and the registration runs again
    on it; the plane returned is the one the gate used."""
    obs, plane = observations(model, calibration, calib, captures)
    plane_global = plane
    seeds = None
    if workspace_plane:
        T1, refined = ArmShapeRegistration(obs, plane, calib.color, table_z_m, seed=seed).register()
        ws = refit_in_workspace(obs[0].pts_cam, plane, T1, seed=seed)
        if ws is not None:
            plane = ws
            # the plane moved little: the first pass's refined (x, y, yaw) seed the second, without a new coarse search
            seeds = [(r["coarse_score"], r["x"], r["y"], math.radians(r["yaw_deg"])) for r in refined]
    reg = ArmShapeRegistration(obs, plane, calib.color, table_z_m, seed=seed)
    res = reg.run(bootstrap=bootstrap, seeds=seeds, diagnostics=diagnostics)
    if diagnostics:
        res.diagnostics["table_plane_global"] = _plane_json(plane_global)
        res.diagnostics["table_plane_used"] = "workspace" if plane is not plane_global else "global"
        ang = float(np.degrees(np.arccos(np.clip(plane.normal_up @ plane_global.normal_up, -1.0, 1.0))))
        res.diagnostics["workspace_vs_global_plane"] = {"angle_deg": ang, "height_mm": 1000.0 * (plane.height_m - plane_global.height_m)}
    return res, plane, obs


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
    rep = {**res.to_json(), "table_plane": _plane_json(plane), **extra}
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


def tilt_vs_offsets_line(res: ArmShapeResult) -> str:
    """One line for the operator: the tilt-vs-joint-offset verdict of the diagnostics (report only)."""
    t = res.diagnostics.get("tilt_vs_offsets")
    if not t:
        return "tilt vs joint offsets: not computed"
    o = t["offsets_deg"]
    line = (
        f"tilt vs joint offsets: {t['verdict']} (separation VIF {t['separation_vif']:.2f}, needs <= {t['max_separation_vif']}); "
        f"base tilt {t['tilt_deg']:.2f} deg, table z {t['table_z_mm']:.1f} mm; offsets shoulder_lift {o['shoulder_lift']:+.2f}, "
        f"elbow_flex {o['elbow_flex']:+.2f}, wrist_flex {o['wrist_flex']:+.2f} deg"
    )
    if t["verdict"] == "ambiguous":
        line += f"; to separate them rerun with --poses 2 (second pose: base turned ~{SECOND_POSE_TURN_DEG} deg)"
    return line


def load_captures(run_dir: Path) -> list[Capture]:
    """The ``capture_<k>.npz`` files of an earlier ``camera-calibrate`` run, in pose order."""
    files = sorted(Path(run_dir).glob("capture_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    out = []
    for f in files:
        with np.load(f) as z:
            out.append(Capture(z["rgb"], z["depth"].astype(np.float32), np.asarray(z["ticks"], dtype=np.int32)))
    return out


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
    ap.add_argument("--from-capture", type=Path, default=None, metavar="DIR",
                    help="replay the capture_<k>.npz of an earlier run offline (no probe, no camera); report-only")  # fmt: skip
    a = ap.parse_args(argv)
    if a.poses < 1 or a.frames < 1:
        ap.error("--poses and --frames must be >= 1")
    calib = load_camera_calib(a.camera_yaml)
    if a.from_capture is not None:
        return _replay(a, calib)
    cfg = load_config()
    cal = load_calibration(Path(a.calibration or cfg.calibration_path).expanduser())
    port = a.port or cfg.port
    do_probe = probe or (lambda: checked_ticks(probe_joints(cal, port=port), require_torque_off=a.poses > 1))
    do_cam = camera or capture_d435
    do_wait = wait or (lambda msg: input(msg))
    captures = []
    for k in range(a.poses):
        if k > 0:
            do_wait(hand_pose_prompt(k, a.poses))
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
    print(json.dumps({k: v for k, v in res.to_json().items() if k not in ("seeds", "diagnostics")}, indent=1, default=float))
    print(tilt_vs_offsets_line(res))
    if not res.passed:
        print(f"camera-calibrate: GATE FAIL ({'; '.join(res.failures)}); {a.camera_yaml} NOT changed; artifacts {out}", file=sys.stderr)
        return 1
    save_camera_calib(updated_calib(calib, res, plane, a.frames, a.poses), a.camera_yaml)
    print(f"camera-calibrate: PASS; wrote {a.camera_yaml}; artifacts {out}")
    return 0


def _replay(a: argparse.Namespace, calib: CameraCalib) -> int:
    from so101_rl.config import load_calibration, load_config
    from so101_rl.sim.model import TABLE_Z_M, load_model

    captures = load_captures(a.from_capture)
    if not captures:
        print(f"camera-calibrate: no capture_<k>.npz in {a.from_capture}", file=sys.stderr)
        return 2
    cal = load_calibration(Path(a.calibration or load_config().calibration_path).expanduser())
    model = load_model(camera=False)[0]
    res, plane, obs = calibrate(model, cal, calib, captures, TABLE_Z_M, seed=a.seed)
    ts = _dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out = a.out_dir / f"{ts}-replay"
    extra = {"poses": len(captures), "frames": int(captures[0].depth_frames.shape[0]), "ticks": [c.ticks.tolist() for c in captures],
             "replay_of": str(a.from_capture)}  # fmt: skip
    save_artifacts(out, res, plane, obs, captures, calib, extra)
    print(json.dumps({k: v for k, v in res.to_json().items() if k not in ("seeds", "diagnostics")}, indent=1, default=float))
    print(tilt_vs_offsets_line(res))
    verdict = "PASS" if res.passed else f"GATE FAIL ({'; '.join(res.failures)})"
    print(f"camera-calibrate (replay of {a.from_capture}): {verdict}; report-only, {a.camera_yaml} NOT changed; artifacts {out}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())

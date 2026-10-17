"""Hand-eye calibration driver (port of LIT/scripts/calibrate_hand_eye.py to the F9/F7 interfaces; H8).

Procedure: visit >= 10 poses spread over the workspace (greedy farthest-point selection), at each one record
the tip-marker position from forward kinematics (base frame) and from the die tracker looking at a red marker on
the tip (camera frame), then fit ``T_base_cam`` with :func:`fit_hand_eye` and write ``hand_eye.yaml``.

* ``--dry-run`` uses a synthetic rig (ground truth: the camera of ``calib/camera.yaml``, rendered marker, real
  tracker) and never touches the arm or a camera. It proves the pipeline and the residual gate end to end.
* ``--execute`` (H8, attended; plan §10 2026-10-17 "H8 software") moves the arm. Order: the reach policy is loaded
  (default the 3 mm export, as ``scripts/point_tip.py``), the camera is opened on its own thread
  (:class:`FrameGrabber`; BEFORE the arm can move: plan §10 2026-10-17 "Perception order"), then the servo is found
  and classified like ``deploy-policy`` (the real arm needs ``--attended``, a ``--no-mcu`` real servo
  ``--accept-no-watchdog``), attached and waited for in RUN (``rl/tip_control.py``). Targets: a 2 cm grid over the
  reach task's goal bounds (3 cm inside) AND the workspace box (2 cm margin) AND z 0.04..0.15 m AND >= 0.17 m from
  the base axis (the policy's weak region in front of the base: STUCK trips in sim), kept when camera.yaml's pose
  projects them 40 px inside the image; visited in farthest-point order, ``--n-poses`` (default 12, min 10) wanted
  out of up to twice as many tried. Per target (:class:`ArmCameraRig`): one reach episode, 0.5 s settle and a 1 s
  window held with zero-delta RUN commands (never HOLD: STUCK); the FK marker (``So101FK`` tip + ``--marker-offset-m``
  in the gripper frame) is averaged over the window's present ticks, the tracker's camera-frame detection (its
  surface point, pushed on along the ray by ``--marker-radius-m`` for a round marker; blobs within 25 mm of the FK
  marker mapped by camera.yaml's pose only) over the window's frames. A pose is skipped
  when the marker was seen in < 70 % of the frames, the detections spread > 2 mm rms or the arm moved > 1 mm. Then
  commanding stops (the servo's stale-command stop takes the arm to rest), ``fit_hand_eye``, and the outputs:
  ``runs/hw/hand-eye-<ts>/`` {``poses.json``, ``hand_eye.yaml``, ``report.json``} always; ``calib/hand_eye.yaml``
  only when residual <= 3 mm and n >= 10 (exit 1 otherwise). ``report.json`` compares the fit with camera.yaml's
  pose (rotation / translation difference against the root-sum-square of the two 2-sigma bootstrap uncertainties)
  and prints the camera-parity plan C.6 recommendation: agree = keep the lower residual (fold a better hand-eye into
  camera.yaml BY HAND), disagree = investigate before training any die task. camera.yaml is never written here.
  ``--camera-backend sim`` (``SimRenderCamera`` following the servo's ticks with a red sphere on the tip) runs the
  whole path against ``servo --backend mujoco`` (``tests/test_hand_eye_execute.py``). Exit 2 refused, 3 camera
  refused, 130 interrupted (nothing written).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from so101_rl.config.loader import load_geometry
from so101_rl.config.schema import WorkspaceBox
from so101_rl.perception.camera import DEFAULT_SERIAL, FakeCamera, Intrinsics
from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.perception.die_tracker import HsvDepthDieTracker
from so101_rl.perception.frames import (
    MAX_RESIDUAL_MM,
    MIN_POSES,
    HandEye,
    fit_hand_eye,
    load_hand_eye,
    save_hand_eye,
)

DEFAULT_DRY_OUT = Path("runs/hand-eye-dry/hand_eye.yaml")
DEFAULT_REAL_OUT = Path("calib/hand_eye.yaml")


class CalibrationRig(Protocol):
    def measure(self, target_xyz_base: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Move the marker to ``target_xyz_base`` and return ``(fk_xyz_base, cam_xyz)``; None = not detected."""
        ...


def farthest_point_subset(points: np.ndarray, k: int, start: int = 0) -> np.ndarray:
    """Indices of ``k`` points chosen greedily to maximise the minimum pairwise distance."""
    n = len(points)
    if not 1 <= k <= n:
        raise ValueError(f"cannot pick {k} of {n} points")
    chosen = [start]
    d = np.linalg.norm(points - points[start], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen)


def sample_box_points(box: WorkspaceBox, n: int, rng: np.random.Generator, margin_m: float = 0.02) -> np.ndarray:
    lo = np.array([box.x[0], box.y[0], box.z[0]]) + margin_m
    hi = np.array([box.x[1], box.y[1], box.z[1]]) - margin_m
    return rng.uniform(lo, hi, (n, 3))


class SyntheticRig:
    """Ground-truth camera model: renders a red square marker and runs the real tracker on it."""

    def __init__(self, T_true_base_cam: np.ndarray, intr: Intrinsics, tracker: HsvDepthDieTracker, marker_px: int = 15):
        self.T = T_true_base_cam
        self.intr = intr
        self.tracker = tracker
        self.half = marker_px // 2
        self.size = 2 * self.half + 1  # odd: the centroid is an exact pixel
        self._rgb = np.full((intr.height, intr.width, 3), 120, np.uint8)
        self._depth = np.full((intr.height, intr.width), 0.75, np.float32)
        self.cam = FakeCamera(self._rgb, self._depth, intr)
        self.cam.start()

    def measure(self, target_xyz_base: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        R, t = self.T[:3, :3], self.T[:3, 3]
        p_cam = R.T @ (np.asarray(target_xyz_base, dtype=np.float64) - t)
        if p_cam[2] <= 0.05:
            return None
        u = round(self.intr.fx * p_cam[0] / p_cam[2] + self.intr.cx)
        v = round(self.intr.fy * p_cam[1] / p_cam[2] + self.intr.cy)
        h = self.half
        if not (h + 3 <= u < self.intr.width - h - 3 and h + 3 <= v < self.intr.height - h - 3):
            return None
        rgb = self._rgb.copy()
        depth = self._depth.copy()
        rgb[v - h : v + h + 1, u - h : u + h + 1] = (200, 20, 20)
        depth[v - h : v + h + 1, u - h : u + h + 1] = p_cam[2]
        self.cam.set_frames(rgb, depth)
        rgb_buf = np.empty_like(rgb)
        depth_buf = np.empty_like(depth)
        if not self.cam.read(rgb_buf, depth_buf).ok:
            return None
        det = self.tracker.detect(rgb_buf, depth_buf)
        if det is None:
            return None
        return np.asarray(target_xyz_base, dtype=np.float64), det.xyz_cam


def collect_correspondences(
    rig: CalibrationRig, candidates_base: np.ndarray, *, n_poses: int, extra: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Visit up to ``n_poses + extra`` farthest-point candidates until ``n_poses`` are usable: ``(p_base [N,3],
    p_cam [N,3])``. Raises if fewer than MIN_POSES are usable."""
    if n_poses < MIN_POSES:
        raise ValueError(f"n_poses must be >= {MIN_POSES}")
    order = farthest_point_subset(candidates_base, min(len(candidates_base), n_poses + extra))
    p_base: list[np.ndarray] = []
    p_cam: list[np.ndarray] = []
    for idx in order:
        m = rig.measure(candidates_base[idx])
        if m is None:
            continue
        p_base.append(m[0])
        p_cam.append(m[1])
        if len(p_base) == n_poses:
            break
    if len(p_base) < MIN_POSES:
        raise RuntimeError(f"only {len(p_base)} usable poses, need >= {MIN_POSES}")
    return np.array(p_base), np.array(p_cam)


def run_calibration(
    rig: CalibrationRig,
    candidates_base: np.ndarray,
    *,
    n_poses: int,
    camera_serial: str,
    date: str | None = None,
    extra: int = 4,
) -> HandEye:
    """Visit farthest-point poses, collect correspondences, fit. Raises if fewer than MIN_POSES are usable."""
    p_base, p_cam = collect_correspondences(rig, candidates_base, n_poses=n_poses, extra=extra)
    T, residual = fit_hand_eye(p_base, p_cam)
    return HandEye(T, residual, len(p_base), date or _dt.datetime.now(_dt.UTC).date().isoformat(), camera_serial)


def _dry_run(out: Path, n_poses: int, seed: int) -> int:
    # the synthetic ground truth is the camera of calib/camera.yaml (intrinsics and pose; plan C.1 single source)
    calib = load_camera_calib()
    intr = calib.color_intrinsics()
    tracker = HsvDepthDieTracker.from_yaml(intr)
    T_true = calib.T_base_color
    rig = SyntheticRig(T_true, intr, tracker)
    rng = np.random.default_rng(seed)
    cands = sample_box_points(load_geometry()[1], 200, rng)
    he = run_calibration(rig, cands, n_poses=n_poses, camera_serial="dry-run")
    save_hand_eye(he, out)
    load_hand_eye(out)  # the artifact must pass its own gate
    err_mm = float(np.max(np.abs(he.T_base_cam - T_true)) * 1000)
    print(f"dry-run: {he.n_poses} poses, residual {he.residual_mm:.3f} mm, max |T - T_true| {err_mm:.3f} mm -> {out}")
    return 0


# ------------------------------------------------------------------------------------------------ --execute (H8)
EXEC_HOLD_S = 1.0  # detection window per pose (zero-delta RUN hold)
EXEC_SETTLE_S = 0.5  # hold before the window (the arm settles after the reach episode)
MIN_DETECT_FRACTION = 0.7  # of the window's frames
MAX_SPREAD_MM = 2.0  # rms distance of the window's detections from their mean
MAX_FK_SPREAD_MM = 1.0  # the arm must stay still during the window
PRIOR_GATE_MM = 25.0  # a blob farther than this from the FK marker mapped by camera.yaml's pose is not the marker
Z_RANGE_M = (0.04, 0.15)
GRID_STEP_M = 0.02
VIEW_MARGIN_PX = 40
BOX_MARGIN_M = 0.02
GOAL_MARGIN_M = 0.03  # inside the reach goal bounds: the policy is weakest at their faces / corners
# the 3 mm policy on the sim servo (SimBackend, 80 candidates): every STUCK trip / miss was in front of the base,
# horizontal radius 0.139..0.153 m (e.g. (0.13, -0.05, 0.06)); with >= 0.17 m: 0 trips, 3 misses (5-17 mm) in 150
MIN_RADIUS_XY_M = 0.17
BOOTSTRAP_N = 200
RUNS_HW = Path("runs/hw")
EXIT_FIT_REFUSED, EXIT_REFUSED, EXIT_CAMERA = 1, 2, 3


def candidate_targets(bounds: Any, box: WorkspaceBox, T_base_cam: np.ndarray, intr: Any, *,
                      z_range: tuple[float, float] = Z_RANGE_M, step_m: float = GRID_STEP_M,
                      margin_px: float = VIEW_MARGIN_PX, box_margin_m: float = BOX_MARGIN_M,
                      goal_margin_m: float = GOAL_MARGIN_M, min_radius_xy_m: float = MIN_RADIUS_XY_M) -> np.ndarray:  # fmt: skip
    """[N, 3] grid points (``step_m``) in the reach task's goal bounds shrunk by ``goal_margin_m`` (x, y and the
    radius limit) AND the workspace box shrunk by ``box_margin_m`` AND ``z_range`` (heights where the tip marker faces
    the oblique camera) AND at least ``min_radius_xy_m`` from the base axis, whose projection with ``T_base_cam``
    (camera.yaml's prior pose) lies ``margin_px`` inside the colour image."""
    m, g = box_margin_m, goal_margin_m
    lo = np.maximum(np.asarray(bounds.lo) + [g, g, 0.0], [box.x[0] + m, box.y[0] + m, max(box.z[0] + m, z_range[0])])
    hi = np.minimum(np.asarray(bounds.hi) - [g, g, 0.0], [box.x[1] - m, box.y[1] - m, min(box.z[1] - m, z_range[1])])
    if np.any(hi < lo):
        return np.zeros((0, 3))
    axes = [np.arange(a, b + 1e-9, step_m) for a, b in zip(lo, hi, strict=True)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    r = np.linalg.norm(pts, axis=1)
    r_xy = np.linalg.norm(pts[:, :2], axis=1)
    pts = pts[bounds.contains_np(pts) & (r <= bounds.radius_max_m - g) & (r_xy >= min_radius_xy_m)]
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    pc = (pts - t) @ R
    z = pc[:, 2]
    ok = z > 0.1
    zs = np.where(ok, z, 1.0)
    u = intr.fx * pc[:, 0] / zs + intr.cx
    v = intr.fy * pc[:, 1] / zs + intr.cy
    ok &= (u >= margin_px) & (u <= intr.width - margin_px) & (v >= margin_px) & (v <= intr.height - margin_px)
    return pts[ok]


class MarkerFK:
    """Base-frame position of the tip marker from present ticks: the FK tip of ``safety/fk.py`` (``So101FK``, the
    ``gripperframe`` point) plus ``offset_m`` in the gripper body frame (``arm_mask.ArmGeometry``, same kinematics)."""

    def __init__(self, fk: Any, calibration: Any, offset_m: Any = (0.0, 0.0, 0.0)) -> None:
        self.fk = fk
        self.calibration = calibration
        self.offset_m = np.asarray(offset_m, dtype=np.float64).reshape(3)
        self._geo = None
        if np.any(self.offset_m != 0):
            from so101_rl.perception.arm_mask import ArmGeometry

            self._geo = ArmGeometry()

    def __call__(self, ticks: np.ndarray) -> np.ndarray:
        t = np.asarray(ticks, dtype=np.float64)
        p = np.array(self.fk.gripper_xyz_from_ticks(t, np.empty(3)), dtype=np.float64)
        if self._geo is not None:
            from so101_rl.perception.arm_mask import ARM_BODIES, joint_angles

            T = self._geo.body_frames(joint_angles(t, self.calibration))[ARM_BODIES.index("gripper")]
            p = p + T[:3, :3] @ self.offset_m
        return p


class _Window:
    def __init__(self, accept: Any) -> None:
        self.accept = accept
        self.frames = 0
        self.xyz: list[np.ndarray] = []


class FrameGrabber:
    """The camera on its own thread: opened there (``process.open_checked``: serial and intrinsics of camera.yaml;
    a GL-backed fake must render on the thread that created it), read continuously (fresh frames, a drained queue),
    and the tracker run only inside a :meth:`begin` / :meth:`end` window. :meth:`start` returns once the camera is
    open, BEFORE anything can move the arm (plan §10 2026-10-17: heavy start-up during arm motion starves the hold
    loop)."""

    def __init__(self, camera: Any, calib: Any, tracker: HsvDepthDieTracker) -> None:
        import threading

        self.camera, self.calib, self.tracker = camera, calib, tracker
        self.frames = 0
        self.failed = 0
        self.error: BaseException | None = None
        self._win: _Window | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hand-eye-camera", daemon=True)

    def start(self, timeout_s: float = 60.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout_s):
            self._stop.set()
            raise RuntimeError(f"camera did not open within {timeout_s:.0f} s")
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        from so101_rl.perception.process import open_checked

        try:
            open_checked(self.camera, self.calib)
        except BaseException as e:  # noqa: BLE001 - handed to start()
            self.error = e
            self._ready.set()
            return
        c = self.calib.color
        rgb = np.zeros((c.height, c.width, 3), np.uint8)
        depth = np.zeros((c.height, c.width), np.float32)
        self._ready.set()
        try:
            while not self._stop.is_set():
                if not self.camera.read(rgb, depth).ok:
                    self.failed += 1
                    continue
                self.frames += 1
                win = self._win
                if win is None:
                    continue
                det = self.tracker.detect(rgb, depth, accept=win.accept)
                with self._lock:
                    if win is self._win:
                        win.frames += 1
                        if det is not None:
                            win.xyz.append(det.xyz_cam.copy())
        except BaseException as e:  # noqa: BLE001 - reported by the rig (a dead camera skips every pose)
            self.error = e
        finally:
            self.camera.stop()

    def begin(self, accept: Any) -> None:
        with self._lock:
            self._win = _Window(accept)

    def end(self) -> _Window:
        with self._lock:
            win, self._win = self._win, None
        if win is None:
            raise RuntimeError("FrameGrabber.end without begin")
        return win

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)


class ArmCameraRig:
    """The H8 rig (:class:`CalibrationRig`): the reach policy drives the tip marker to a target
    (``rl/tip_control.TipController``), the arm is held still with zero-delta RUN commands, the FK marker position is
    averaged over the hold's present ticks and the tracker's camera-frame detection over the hold's frames. A pose is
    skipped (None) when the marker was detected in fewer than ``min_detect_fraction`` of the frames, the detections
    spread more than ``max_spread_mm`` rms, or the arm moved more than ``max_fk_spread_mm``. Only blobs within
    ``prior_gate_mm`` of the FK marker mapped by camera.yaml's pose are taken (not the die, not the arm's orange parts)."""

    def __init__(self, ctl: Any, grabber: FrameGrabber, marker_fk: MarkerFK, T_prior: np.ndarray, *,
                 hold_s: float = EXEC_HOLD_S, settle_s: float = EXEC_SETTLE_S,
                 min_detect_fraction: float = MIN_DETECT_FRACTION, max_spread_mm: float = MAX_SPREAD_MM,
                 max_fk_spread_mm: float = MAX_FK_SPREAD_MM, prior_gate_mm: float = PRIOR_GATE_MM,
                 marker_radius_m: float = 0.0, log: Any = print) -> None:  # fmt: skip
        self.ctl, self.grabber, self.marker_fk = ctl, grabber, marker_fk
        self.T_prior = np.asarray(T_prior, dtype=np.float64)
        self.hold_s, self.settle_s = float(hold_s), float(settle_s)
        self.min_detect_fraction, self.max_spread_mm = float(min_detect_fraction), float(max_spread_mm)
        self.max_fk_spread_mm, self.prior_gate_m = float(max_fk_spread_mm), float(prior_gate_mm) / 1000.0
        self.marker_radius_m = float(marker_radius_m)
        self.log = log
        self.records: list[dict[str, Any]] = []

    def _centre(self, xyz_cam: np.ndarray) -> np.ndarray:
        """The marker centre from the tracker's surface point: on along the viewing ray by ``marker_radius_m`` (0 =
        a flat sticker, the surface point IS the marker; a sphere / bead shows its near surface)."""
        return xyz_cam * (1.0 + self.marker_radius_m / np.linalg.norm(xyz_cam, axis=-1, keepdims=True))

    def measure(self, target_xyz_base: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        from so101_rl.perception.frames import cam_to_base

        target = np.asarray(target_xyz_base, dtype=np.float64)
        rec: dict[str, Any] = {"index": len(self.records), "target": [round(float(v), 4) for v in target]}
        self.records.append(rec)
        ep = self.ctl.reach(target)  # ServoLeftRun propagates: stop everything
        rec.update(reach_success=bool(ep["success"]), reach_error_mm=round(ep["final_error_m"] * 1000, 2),
                   reach_steps=int(ep["steps"]))  # fmt: skip
        info = self.ctl.hold(self.settle_s)
        fk0 = self.marker_fk(info["present_ticks"])
        T, gate = self.T_prior, self.prior_gate_m

        def accept(det: Any) -> bool:
            return bool(np.linalg.norm(cam_to_base(self._centre(det.xyz_cam), T) - fk0) <= gate)

        ticks: list[np.ndarray] = []
        self.grabber.begin(accept)
        try:
            self.ctl.hold(self.hold_s, on_step=lambda i: ticks.append(np.array(i["present_ticks"], np.float64)))
        finally:
            win = self.grabber.end()
        fk = np.array([self.marker_fk(t) for t in ticks])
        fk_mean = fk.mean(axis=0)
        fk_spread = float(np.linalg.norm(fk - fk_mean, axis=1).max() * 1000)
        rec.update(fk_marker=[round(float(v), 5) for v in fk_mean], fk_spread_mm=round(fk_spread, 3),
                   frames=win.frames, detected=len(win.xyz))  # fmt: skip
        why = None
        if self.grabber.error is not None:
            why = f"camera error: {self.grabber.error}"
        elif win.frames == 0:
            why = "no camera frames during the hold"
        elif len(win.xyz) / win.frames < self.min_detect_fraction:
            why = f"marker detected in {len(win.xyz)}/{win.frames} frames (< {self.min_detect_fraction:.0%})"
        elif fk_spread > self.max_fk_spread_mm:
            why = f"the arm moved {fk_spread:.2f} mm during the hold (> {self.max_fk_spread_mm} mm)"
        cam = None
        if why is None:
            xyz = self._centre(np.array(win.xyz))
            cam = xyz.mean(axis=0)
            spread = float(np.sqrt(np.mean(np.sum((xyz - cam) ** 2, axis=1))) * 1000)
            rec.update(cam_marker=[round(float(v), 5) for v in cam], cam_spread_mm=round(spread, 3),
                       prior_err_mm=round(float(np.linalg.norm(cam_to_base(cam, T) - fk_mean)) * 1000, 2))  # fmt: skip
            if spread > self.max_spread_mm:
                why = f"detection spread {spread:.2f} mm rms (> {self.max_spread_mm} mm)"
        rec["used"] = why is None
        rec["skip_reason"] = why
        n_used = sum(r["used"] for r in self.records)
        self.log(f"pose {rec['index']:2d} target {rec['target']}: reach {rec['reach_error_mm']:.1f} mm, "
                 + (f"USED ({n_used}), detected {len(win.xyz)}/{win.frames}, prior err {rec['prior_err_mm']:.1f} mm"
                    if why is None else f"skipped: {why}"))  # fmt: skip
        return None if why is not None or cam is None else (fk_mean, cam)


def bootstrap_uncertainty(p_base: np.ndarray, p_cam: np.ndarray, *, n: int = BOOTSTRAP_N, seed: int = 0) -> tuple[float, float]:
    """(trans_mm, rot_deg): 2 sigma of ``fit_hand_eye`` over correspondence resamples, the definition of
    camera.yaml's arm-shape ``uncertainty`` (``arm_registration.ArmShapeRegistration.bootstrap``)."""
    from so101_rl.perception.arm_registration import rotation_angle_deg

    T0, _ = fit_hand_eye(p_base, p_cam)
    rng = np.random.default_rng(seed)
    ts, rs = [], []
    k = len(p_base)
    for _ in range(n):
        idx = rng.integers(0, k, k)
        if len(np.unique(idx)) < 3:
            continue
        Tb, _ = fit_hand_eye(p_base[idx], p_cam[idx])
        ts.append(Tb[:3, 3])
        rs.append(rotation_angle_deg(Tb[:3, :3] @ T0[:3, :3].T))
    return float(2 * np.linalg.norm(np.array(ts).std(0)) * 1000), float(2 * np.sqrt(np.mean(np.square(rs))))


def compare_with_camera_yaml(he: HandEye, unc: tuple[float, float], pose: Any, p_cam: np.ndarray) -> dict[str, Any]:
    """The camera-parity plan C.6 rule: hand-eye vs camera.yaml's pose (rotation / translation difference against
    their combined uncertainty, root-sum-square of the two 2-sigma values; also how far the measured marker points
    move between the two poses). Agree: keep the lower residual. Disagree: investigate before any die task."""
    from so101_rl.perception.arm_registration import rotation_angle_deg

    A, B = he.T_base_cam, pose.matrix
    rot = rotation_angle_deg(A[:3, :3] @ B[:3, :3].T)
    trans = float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000)
    tol_t, tol_r = float(np.hypot(unc[0], pose.trans_mm)), float(np.hypot(unc[1], pose.rot_deg))
    disp = np.linalg.norm((p_cam @ A[:3, :3].T + A[:3, 3]) - (p_cam @ B[:3, :3].T + B[:3, 3]), axis=1) * 1000
    agree = rot <= tol_r and trans <= tol_t
    other_res = pose.residual_mm
    if agree:
        keep = "hand-eye" if other_res is None or he.residual_mm < other_res else pose.method
        rec = f"AGREE within the stated uncertainty: keep the lower residual = {keep}" + (
            " (fold it into calib/camera.yaml as method hand-eye, by hand, and shrink the camera DR range)"
            if keep == "hand-eye"
            else " (calib/camera.yaml stays as it is)"
        )
    else:
        keep = None
        rec = ("DISAGREE beyond the stated uncertainty: investigate before training any die task (camera-parity plan "
               "C.6); calib/camera.yaml is NOT changed")  # fmt: skip
    return {
        "camera_yaml_method": pose.method,
        "camera_yaml_date": pose.date,
        "camera_yaml_residual_mm": other_res,
        "camera_yaml_uncertainty": {"trans_mm": pose.trans_mm, "rot_deg": pose.rot_deg},
        "hand_eye_residual_mm": he.residual_mm,
        "hand_eye_uncertainty": {"trans_mm": unc[0], "rot_deg": unc[1]},
        "rotation_diff_deg": rot,
        "translation_diff_mm": trans,
        "combined_uncertainty": {"trans_mm": tol_t, "rot_deg": tol_r},
        "marker_point_displacement_mm": {"mean": float(disp.mean()), "max": float(disp.max())},
        "agree": bool(agree),
        "keep": keep,
        "recommendation": rec,
    }


def _execute(args: argparse.Namespace, camera: Any | None = None) -> int:
    """``--execute`` (H8, attended): the real-rig adapter. Order: load the policy, open the camera (its own thread),
    THEN find the servo (refusals: the real arm without ``--attended``, a ``--no-mcu`` real servo without
    ``--accept-no-watchdog``), attach, wait for RUN, visit the poses, stop commanding (the servo's stale-command stop
    takes the arm to rest), fit and write."""
    import json

    def log(msg: str) -> None:
        print(f"hand-eye: {msg}", file=sys.stderr, flush=True)

    if args.n_poses < MIN_POSES:
        log(f"refused: --n-poses must be >= {MIN_POSES}")
        return EXIT_REFUSED
    from so101_rl.env.tasks import REACH
    from so101_rl.rl import deploy_policy as dp
    from so101_rl.rl import tip_control as tc

    calib = load_camera_calib(args.camera_yaml)
    pose = calib.require_pose(calibrated=True)
    intr = calib.color_intrinsics()
    T_prior = pose.matrix
    cands = candidate_targets(REACH.goal_bounds, load_geometry()[1], T_prior, intr,
                              z_range=tuple(args.z_range), step_m=args.grid_step_m)  # fmt: skip
    if len(cands) < args.n_poses:
        log(f"refused: only {len(cands)} candidate targets in view (need {args.n_poses})")
        return EXIT_REFUSED
    try:
        import torch

        torch.set_num_threads(1)
        policy, task = tc.load_reach_policy(args.policy)
    except (dp.Refused, OSError, ValueError, KeyError) as e:
        log(f"refused: reach policy {args.policy}: {e}")
        return EXIT_REFUSED
    tracker = HsvDepthDieTracker.from_camera_calib(calib)
    if camera is None:
        camera = _make_camera(args, calib)
    grabber = FrameGrabber(camera, calib, tracker)
    try:
        grabber.start()
    except Exception as e:  # noqa: BLE001 - camera refused / not connected
        log(f"camera refused: {e}")
        grabber.stop()
        return EXIT_CAMERA
    log(f"camera {calib.serial} open; {len(cands)} candidate targets, {args.n_poses} poses wanted")
    ts = _dt.datetime.now().astimezone()
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_HW / f"hand-eye-{ts:%Y%m%d-%H%M%S}"
    rig: ArmCameraRig | None = None
    p_base = p_cam = None
    failure = ""
    try:
        try:
            rs = tc.resolve(attended=args.attended, accept_no_watchdog=args.accept_no_watchdog,
                            wait_servo_s=args.wait_servo_s, shm_name=args.shm_name, log=log)  # fmt: skip
        except dp.Refused as e:
            log(f"refused: {e}")
            return EXIT_REFUSED
        if args.camera_backend == "sim" and getattr(camera, "follow", "n/a") is None:
            from so101_rl.perception.arm_mask import ServoStateReader

            camera.follow = (ServoStateReader(rs.cfg.shm_name), rs.calibration)  # the render follows the servo
        from so101_rl.env.backend import ServoNotRunning, ServoStateStale

        try:
            ctl = tc.attach(rs, policy, task, wait_servo_s=args.wait_servo_s, log=log)
        except (ServoNotRunning, ServoStateStale) as e:  # nothing moved yet
            log(f"refused: {e}")
            return EXIT_REFUSED
        marker_fk = MarkerFK(ctl.backend.fk(), rs.calibration, args.marker_offset_m)
        rig = ArmCameraRig(ctl, grabber, marker_fk, T_prior, hold_s=args.hold_s, settle_s=args.settle_s,
                           min_detect_fraction=args.min_detect_fraction, max_spread_mm=args.max_spread_mm,
                           prior_gate_mm=args.prior_gate_mm, marker_radius_m=args.marker_radius_m, log=log)  # fmt: skip
        try:
            p_base, p_cam = collect_correspondences(rig, cands, n_poses=args.n_poses, extra=args.extra_targets)
        except tc.ServoLeftRun as e:
            failure = f"stopped: {e}"
        except RuntimeError as e:  # too few usable poses, or a reset that failed (env.backend.ResetFailed)
            failure = f"{type(e).__name__}: {e}"
    except KeyboardInterrupt:
        log("interrupted: commanding stopped (the servo's stale-command stop takes the arm to rest); nothing written")
        return 130
    finally:
        grabber.stop()
        follow = getattr(camera, "follow", None)
        if follow is not None:
            follow[0].close()
        if rig is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "poses.json").write_text(json.dumps(rig.records, indent=2) + "\n")
    log("commanding stopped: the servo's stale-command stop takes the arm back to rest")
    report: dict[str, Any] = {
        "date": ts.isoformat(timespec="seconds"),
        "camera_serial": calib.serial,
        "camera_backend": args.camera_backend,
        "policy": str(args.policy),
        "marker_offset_m": [float(v) for v in args.marker_offset_m],
        "marker_radius_m": float(args.marker_radius_m),
        "servo": {"backend": rs.servo.backend, "pid": rs.servo.pid, "real": rs.servo.is_real},
        "camera_yaml": calib.pose_meta(),
        "poses_tried": len(rig.records),
        "poses_used": 0 if p_base is None else len(p_base),
    }
    if p_base is None or p_cam is None:
        report["result"] = failure
        (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        log(f"FAILED: {failure}; nothing written to calib/; poses in {run_dir}/poses.json")
        return EXIT_FIT_REFUSED
    T, residual = fit_hand_eye(p_base, p_cam)
    he = HandEye(T, residual, len(p_base), ts.date().isoformat(), calib.serial)
    save_hand_eye(he, run_dir / "hand_eye.yaml")
    unc = bootstrap_uncertainty(p_base, p_cam)
    err = np.linalg.norm(p_base - (p_cam @ T[:3, :3].T + T[:3, 3]), axis=1) * 1000
    cmp_ = compare_with_camera_yaml(he, unc, pose, p_cam)
    report.update(T_base_cam=T.tolist(), residual_mm=residual, n_poses=len(p_base),
                  per_pose_residual_mm=[round(float(e), 3) for e in err], comparison=cmp_)  # fmt: skip
    gate_ok = residual <= MAX_RESIDUAL_MM and len(p_base) >= MIN_POSES
    out = Path(args.out) if args.out else DEFAULT_REAL_OUT
    if gate_ok:
        save_hand_eye(he, out)
        load_hand_eye(out)  # the artifact must pass its own gate
        report["written"] = str(out)
    else:
        report["written"] = None
        report["refused"] = f"residual {residual:.2f} mm > {MAX_RESIDUAL_MM} mm or n {len(p_base)} < {MIN_POSES}"
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    cu = cmp_["combined_uncertainty"]
    log(f"fit: {len(p_base)} poses, residual {residual:.2f} mm (gate {MAX_RESIDUAL_MM} mm), bootstrap 2 sigma "
        f"{unc[0]:.2f} mm / {unc[1]:.3f} deg")  # fmt: skip
    log(f"vs calib/camera.yaml ({pose.method}, residual {pose.residual_mm} mm): rotation {cmp_['rotation_diff_deg']:.3f} "
        f"deg, translation {cmp_['translation_diff_mm']:.2f} mm (combined uncertainty {cu['rot_deg']:.3f} deg / "
        f"{cu['trans_mm']:.2f} mm); the marker points move {cmp_['marker_point_displacement_mm']['mean']:.1f} mm "
        "mean between the two poses")  # fmt: skip
    log(f"RECOMMENDATION: {cmp_['recommendation']}")
    if not gate_ok:
        log(f"REFUSED to write {out}: {report['refused']}; the fit is in {run_dir}")
        return EXIT_FIT_REFUSED
    log(f"wrote {out} and {run_dir}/ (hand_eye.yaml, poses.json, report.json)")
    return 0


def _make_camera(args: argparse.Namespace, calib: Any) -> Any:
    c = calib.color
    if args.camera_backend == "sim":
        from so101_rl.perception.sim_camera import SimRenderCamera

        return SimRenderCamera(calib, realtime=True, tip_marker_radius_m=args.sim_marker_radius_m,
                               tip_marker_offset_m=tuple(args.marker_offset_m))  # fmt: skip
    from so101_rl.perception.camera import D435Camera

    return D435Camera(serial=calib.serial, width=c.width, height=c.height, fps=c.fps)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hand-eye", description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="synthetic rig; no arm, no camera")
    mode.add_argument("--execute", action="store_true", help="move the arm (H8; the real arm needs --attended)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-poses", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serial", default=DEFAULT_SERIAL)
    ex = ap.add_argument_group("--execute")
    ex.add_argument("--attended", action="store_true", help="a human at the e-stop (required on the real arm)")
    ex.add_argument("--accept-no-watchdog", action="store_true", help="a real servo with --no-mcu (hand on power)")
    ex.add_argument("--policy", type=Path, default=None, help="reach export (default: the 3 mm reach policy)")
    ex.add_argument("--shm-name", default=None, help="servo shm (default so101.yaml shm_name)")
    ex.add_argument("--wait-servo-s", type=float, default=120.0)
    ex.add_argument("--camera-backend", choices=("d435", "sim"), default="d435",
                    help="sim: MuJoCo renders following the servo's ticks, a red sphere on the tip (servo --backend mujoco)")  # fmt: skip
    ex.add_argument("--camera-yaml", type=Path, default=None)
    ex.add_argument("--marker-offset-m", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                    help="tip marker centre relative to the FK tip (gripperframe), gripper body frame, metres")  # fmt: skip
    ex.add_argument("--marker-radius-m", type=float, default=None,
                    help="a round marker's radius: its centre lies this far behind the seen surface (default 0 = a "
                    "flat sticker; --camera-backend sim: the sim sphere's radius)")  # fmt: skip
    ex.add_argument("--sim-marker-radius-m", type=float, default=0.006)
    ex.add_argument("--hold-s", type=float, default=EXEC_HOLD_S)
    ex.add_argument("--settle-s", type=float, default=EXEC_SETTLE_S)
    ex.add_argument("--min-detect-fraction", type=float, default=MIN_DETECT_FRACTION)
    ex.add_argument("--max-spread-mm", type=float, default=MAX_SPREAD_MM)
    ex.add_argument("--prior-gate-mm", type=float, default=PRIOR_GATE_MM)
    ex.add_argument("--z-range", type=float, nargs=2, default=Z_RANGE_M, metavar=("ZMIN", "ZMAX"))
    ex.add_argument("--grid-step-m", type=float, default=GRID_STEP_M)
    ex.add_argument("--extra-targets", type=int, default=None, help="targets tried beyond --n-poses (default n)")
    ex.add_argument("--run-dir", type=Path, default=None, help="default runs/hw/hand-eye-<ts>")
    return ap


def main(argv: list[str] | None = None, *, camera: Any | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return _dry_run(args.out or DEFAULT_DRY_OUT, args.n_poses, args.seed)
    from so101_rl.rl.tip_control import DEFAULT_REACH_POLICY

    args.policy = args.policy or DEFAULT_REACH_POLICY
    args.extra_targets = args.n_poses if args.extra_targets is None else args.extra_targets
    if args.marker_radius_m is None:
        args.marker_radius_m = args.sim_marker_radius_m if args.camera_backend == "sim" else 0.0
    return _execute(args, camera)

if __name__ == "__main__":
    raise SystemExit(main())

"""ArUco fiducial "camera moved" gate (plan F9, Phase 0 camera-moved gate) and its capture CLI (H8).

A fixed marker is glued to the table: ArUco ``DICT_4X4_50`` id 0, **50 mm**, centred at base (0.09, 0.05), sides
parallel to the base x / y axes, printed upright with its top edge towards +x (away from the base; plan §10
2026-10-17 "H8 software"). The MJCF ``fiducial`` site is the same square (``tests/test_fiducial.py``).
``calib/fiducial.yaml`` records where its four corners appear in the image; if the detected corners drift by more than
``tol_px`` (mean corner distance) the camera or the marker moved and training must halt. A marker that is not found
(occluded by the arm or a hand) is no evidence either way (the perception process counts it, ``process.py``).

``pixi run fiducial-capture`` (``python -m so101_rl.perception.fiducial``) writes ``calib/fiducial.yaml``: with the arm
at rest or out of view (no servo needed) it opens the calibrated D435 (``process.open_checked``: serial and intrinsics
of ``calib/camera.yaml``), detects the marker in ``--frames`` frames (default 30), averages the corners and writes
``dictionary``, ``marker_id``, ``expected_corners_px``, ``tol_px`` 2, ``size_mm`` 50, ``xy_base``, ``date``,
``camera_serial``, ``camera_yaml_sha256`` and ``prior_err_px`` (mean distance to the corners predicted from
``camera.yaml``'s pose; informational). Refused (exit 1, nothing written) when the marker is found in fewer than 90 %
of the frames or a corner jitters by more than 0.5 px rms. Exit 2 calibration refused, 3 camera refused.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

FIDUCIAL_YAML = Path("calib/fiducial.yaml")
DICTIONARY = "DICT_4X4_50"
MARKER_ID = 0
SIZE_MM = 50.0
XY_BASE = (0.09, 0.05)  # = the MJCF fiducial site (tests/test_fiducial.py)
TOL_PX = 2.0
CAPTURE_FRAMES = 30
MIN_FOUND_FRACTION = 0.9
MAX_JITTER_PX = 0.5
PRIOR_WARN_PX = 10.0


@dataclass
class FiducialReport:
    ok: bool
    reproj_err_px: float
    found: bool


class FiducialCheck:
    def __init__(
        self,
        marker_id: int,
        expected_corners_px: np.ndarray,
        *,
        dictionary: int = cv2.aruco.DICT_4X4_50,
        tol_px: float = TOL_PX,
        camera_serial: str | None = None,
    ) -> None:
        self.marker_id = int(marker_id)
        self.expected = np.asarray(expected_corners_px, dtype=np.float64).reshape(4, 2)
        self.tol_px = float(tol_px)
        self.camera_serial = camera_serial
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), params)

    @classmethod
    def from_yaml(cls, path: Path = FIDUCIAL_YAML) -> FiducialCheck:
        d = yaml.safe_load(Path(path).read_text())
        dictionary = getattr(cv2.aruco, d["dictionary"])
        serial = d.get("camera_serial")
        return cls(
            int(d["marker_id"]),
            np.array(d["expected_corners_px"], dtype=np.float64),
            dictionary=dictionary,
            tol_px=float(d.get("tol_px", TOL_PX)),
            camera_serial=None if serial is None else str(serial),
        )

    def detect_corners(self, rgb: np.ndarray) -> np.ndarray | None:
        """Corners ``[4,2]`` (TL, TR, BR, BL) of the configured marker id, or None."""
        gray = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return None
        for c, i in zip(corners, ids.ravel(), strict=True):
            if int(i) == self.marker_id:
                return c.reshape(4, 2).astype(np.float64)
        return None

    def check(self, rgb: np.ndarray) -> FiducialReport:
        found = self.detect_corners(rgb)
        if found is None:
            return FiducialReport(ok=False, reproj_err_px=float("inf"), found=False)
        err = float(np.mean(np.linalg.norm(found - self.expected, axis=1)))
        return FiducialReport(ok=err <= self.tol_px, reproj_err_px=err, found=True)


# ------------------------------------------------------------------------------------------------ geometry
def marker_corners_base(xy: tuple[float, float] = XY_BASE, size_mm: float = SIZE_MM, z: float | None = None) -> np.ndarray:
    """[4, 3] base-frame corners in ArUco order (TL, TR, BR, BL) of the marker printed upright with its top edge
    towards +x: TL = (+x, +y), TR = (+x, -y), BR = (-x, -y), BL = (-x, +y) from its centre, on the table top
    (``z`` default: config ``safe_state.table_z_m``)."""
    if z is None:
        from so101_rl.config.loader import load_geometry

        z = load_geometry()[0]
    h = size_mm / 2000.0
    x, y = xy
    return np.array([[x + h, y + h, z], [x + h, y - h, z], [x - h, y - h, z], [x - h, y + h, z]], dtype=np.float64)


def project_base(points_base: np.ndarray, T_base_cam: np.ndarray, intr: Any) -> np.ndarray:
    """[N, 2] pixel coordinates of base-frame points (pinhole, the colour stream)."""
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    p = (np.asarray(points_base, np.float64) - t) @ R  # = R.T (p - t), row-wise
    return np.stack([intr.fx * p[:, 0] / p[:, 2] + intr.cx, intr.fy * p[:, 1] / p[:, 2] + intr.cy], axis=1)


def prior_error_px(corners_px: np.ndarray, predicted_px: np.ndarray) -> tuple[float, int]:
    """(mean corner distance, quarter turns) for the best cyclic order: a marker glued rotated by k x 90 deg shows
    its corners shifted by k (still a valid fiducial; only the documented orientation predicts k = 0)."""
    errs = [float(np.mean(np.linalg.norm(np.roll(corners_px, k, axis=0) - predicted_px, axis=1))) for k in range(4)]
    k = int(np.argmin(errs))
    return errs[k], k


# ------------------------------------------------------------------------------------------------ capture
@dataclass
class FiducialCapture:
    frames: int
    found: int
    corners_px: np.ndarray | None  # [4, 2] mean over the frames where it was found
    jitter_px: float  # worst corner's rms distance from its mean

    @property
    def found_fraction(self) -> float:
        return self.found / self.frames if self.frames else 0.0


def capture_corners(camera: Any, check: FiducialCheck, n_frames: int, shape: tuple[int, int]) -> FiducialCapture:
    """Read ``n_frames`` good frames from the started ``camera`` and average the detected corners."""
    h, w = shape
    rgb = np.zeros((h, w, 3), np.uint8)
    depth = np.zeros((h, w), np.float32)
    got: list[np.ndarray] = []
    frames = failed = 0
    while frames < n_frames:
        if not camera.read(rgb, depth).ok:
            failed += 1
            if failed > 3 * n_frames + 30:
                raise RuntimeError(f"camera: {failed} failed reads")
            continue
        frames += 1
        c = check.detect_corners(rgb)
        if c is not None:
            got.append(c)
    if not got:
        return FiducialCapture(frames, 0, None, float("inf"))
    a = np.stack(got)
    mean = a.mean(axis=0)
    jitter = float(np.sqrt(np.mean(np.sum((a - mean) ** 2, axis=2), axis=0)).max())
    return FiducialCapture(frames, len(got), mean, jitter)


def save_fiducial_yaml(path: Path, doc: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "# ArUco camera-moved fiducial (H8; schema calib/README.md). Written by `pixi run fiducial-capture`.\n"
    path.write_text(head + yaml.safe_dump(doc, sort_keys=False))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fiducial-capture", description="Record calib/fiducial.yaml (H8)")
    ap.add_argument("--out", type=Path, default=FIDUCIAL_YAML)
    ap.add_argument("--camera-yaml", type=Path, default=None, help="default calib/camera.yaml")
    ap.add_argument("--frames", type=int, default=CAPTURE_FRAMES)
    ap.add_argument("--marker-id", type=int, default=MARKER_ID)
    ap.add_argument("--size-mm", type=float, default=SIZE_MM)
    ap.add_argument("--xy", type=float, nargs=2, default=XY_BASE, metavar=("X", "Y"), help="marker centre, base frame")
    ap.add_argument("--tol-px", type=float, default=TOL_PX)
    ap.add_argument("--min-found", type=float, default=MIN_FOUND_FRACTION)
    ap.add_argument("--max-jitter-px", type=float, default=MAX_JITTER_PX)
    return ap


def main(argv: list[str] | None = None, *, camera: Any | None = None) -> int:
    from so101_rl.perception.camera_calib import CameraNotCalibrated, load_camera_calib
    from so101_rl.perception.process import EXIT_CALIB, EXIT_CAMERA, CameraRefused, open_checked

    a = build_parser().parse_args(argv)
    try:
        calib = load_camera_calib(a.camera_yaml)
        pose = calib.require_pose()
    except (CameraNotCalibrated, FileNotFoundError, ValueError, TypeError) as e:
        print(f"fiducial-capture: calibration refused: {e}", file=sys.stderr)
        return EXIT_CALIB
    c = calib.color
    if camera is None:
        from so101_rl.perception.camera import D435Camera

        camera = D435Camera(serial=calib.serial, width=c.width, height=c.height, fps=c.fps)
    try:
        open_checked(camera, calib)
    except (CameraRefused, RuntimeError, ImportError) as e:
        print(f"fiducial-capture: camera refused: {e}", file=sys.stderr)
        return EXIT_CAMERA
    check = FiducialCheck(a.marker_id, np.zeros((4, 2)), dictionary=getattr(cv2.aruco, DICTIONARY), tol_px=a.tol_px)
    try:
        cap = capture_corners(camera, check, a.frames, (c.height, c.width))
    finally:
        camera.stop()
    print(f"fiducial-capture: {DICTIONARY} id {a.marker_id} found in {cap.found}/{cap.frames} frames, corner jitter "
          f"{cap.jitter_px:.3f} px rms", file=sys.stderr)  # fmt: skip
    if cap.corners_px is None or cap.found_fraction < a.min_found:
        print(f"fiducial-capture: REFUSED: the marker must be found in >= {a.min_found:.0%} of the frames (arm at rest "
              "or out of view, nothing over the marker, light on it); nothing written", file=sys.stderr)  # fmt: skip
        return 1
    if cap.jitter_px > a.max_jitter_px:
        print(f"fiducial-capture: REFUSED: corner jitter {cap.jitter_px:.3f} px > {a.max_jitter_px} px (camera or "
              "marker moving, or poor light); nothing written", file=sys.stderr)  # fmt: skip
        return 1
    predicted = project_base(marker_corners_base(tuple(a.xy), a.size_mm), pose.matrix, calib.color_intrinsics())
    prior_err, turns = prior_error_px(cap.corners_px, predicted)
    if turns:
        print(f"fiducial-capture: note: the marker is glued rotated by {turns * 90} deg from the documented orientation "
              "(top edge towards +x); the gate does not care", file=sys.stderr)  # fmt: skip
    if prior_err > PRIOR_WARN_PX:
        print(f"fiducial-capture: WARNING the corners are {prior_err:.1f} px from where calib/camera.yaml predicts a "
              f"{a.size_mm:g} mm marker at {list(a.xy)}: check its position (the zone / tray clearances assume it)",
              file=sys.stderr)  # fmt: skip
    doc = {
        "dictionary": DICTIONARY,
        "marker_id": int(a.marker_id),
        "tol_px": float(a.tol_px),
        "size_mm": float(a.size_mm),
        "xy_base": [float(v) for v in a.xy],
        "expected_corners_px": [[round(float(v), 3) for v in row] for row in cap.corners_px],
        "date": _dt.datetime.now().astimezone().date().isoformat(),
        "camera_serial": str(calib.serial),
        "camera_yaml_sha256": calib.source_sha256 or None,
        "frames": int(cap.frames),
        "found_frames": int(cap.found),
        "jitter_px": round(cap.jitter_px, 4),
        "prior_err_px": round(prior_err, 2),
    }
    save_fiducial_yaml(a.out, doc)
    print(f"fiducial-capture: wrote {a.out} (prior error {prior_err:.1f} px)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

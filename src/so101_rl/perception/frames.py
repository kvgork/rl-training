"""Camera geometry and the hand-eye artifact (plan F9).

``T_base_cam`` is the 4x4 transform taking a point in the camera frame to the robot base frame. The camera pose
every consumer uses is ``calib/camera.yaml``'s ``T_base_color`` (camera parity C.1; :func:`default_T_base_cam`).
``calib/hand_eye.yaml`` is the H8 hand-eye artifact (schema in ``calib/README.md``): its result is folded into
``camera.yaml`` (method ``hand-eye``) once H8 has run (plan C.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from so101_rl.perception.camera import Intrinsics

MAX_RESIDUAL_MM = 3.0
MIN_POSES = 10


@dataclass(frozen=True, eq=False)
class HandEye:
    T_base_cam: np.ndarray
    residual_mm: float
    n_poses: int
    date: str
    camera_serial: str


def default_T_base_cam(path: Path | None = None, *, require_calibrated: bool = False) -> np.ndarray:
    """``T_base_color`` of ``calib/camera.yaml`` (or ``path``): the colour optical frame, which the aligned depth
    shares. ``require_calibrated`` refuses a provisional pose."""
    from so101_rl.perception.camera_calib import load_camera_calib

    return load_camera_calib(path, require_calibrated=require_calibrated).T_base_color


def pixel_to_cam(u: float, v: float, z_m: float, intr: Intrinsics) -> np.ndarray:
    """Pinhole deprojection (no distortion; the D435 colour stream is effectively undistorted)."""
    return np.array([(u - intr.cx) * z_m / intr.fx, (v - intr.cy) * z_m / intr.fy, z_m], dtype=np.float64)


def cam_to_base(xyz_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    p = np.asarray(xyz_cam, dtype=np.float64)
    return T_base_cam[:3, :3] @ p + T_base_cam[:3, 3]


def fit_hand_eye(p_base: np.ndarray, p_cam: np.ndarray) -> tuple[np.ndarray, float]:
    """Umeyama rigid fit (rotation + translation, no scale): ``p_base ~ R p_cam + t``.

    Returns ``(T_base_cam 4x4, rms residual in mm)``.
    """
    a = np.asarray(p_cam, dtype=np.float64)
    b = np.asarray(p_base, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"need matching [N,3] arrays, got {a.shape} and {b.shape}")
    if a.shape[0] < 3:
        raise ValueError("need at least 3 correspondences")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite correspondences")
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    cov = (b - mb).T @ (a - ma) / a.shape[0]
    u, _, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1.0  # no reflection
    r = u @ s @ vt
    t = mb - r @ ma
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    err = b - (a @ r.T + t)
    return T, float(np.sqrt(np.mean(np.sum(err**2, axis=1))) * 1000.0)


def save_hand_eye(he: HandEye, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "T_base_cam": [[float(x) for x in row] for row in he.T_base_cam],
        "residual_mm": float(he.residual_mm),
        "n_poses": int(he.n_poses),
        "date": str(he.date),
        "camera_serial": str(he.camera_serial),
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def load_hand_eye(path: Path = Path("calib/hand_eye.yaml")) -> HandEye:
    """Load and validate. Refuses ``residual_mm > 3.0``, ``n_poses < 10`` or a non-rigid matrix."""
    path = Path(path)
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict):
        raise TypeError(f"{path}: expected a mapping")
    missing = {"T_base_cam", "residual_mm", "n_poses", "date", "camera_serial"} - set(doc)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    T = np.array(doc["T_base_cam"], dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{path}: T_base_cam must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError(f"{path}: T_base_cam last row must be [0,0,0,1]")
    r = T[:3, :3]
    if not (np.allclose(r @ r.T, np.eye(3), atol=1e-6) and math.isclose(np.linalg.det(r), 1.0, abs_tol=1e-6)):
        raise ValueError(f"{path}: T_base_cam rotation is not a proper rotation")
    residual = float(doc["residual_mm"])
    n = doc["n_poses"]
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"{path}: n_poses must be an integer")
    if not math.isfinite(residual) or residual > MAX_RESIDUAL_MM:
        raise ValueError(f"{path}: hand-eye residual {residual} mm exceeds {MAX_RESIDUAL_MM} mm; recalibrate")
    if n < MIN_POSES:
        raise ValueError(f"{path}: only {n} poses, need >= {MIN_POSES}; recalibrate")
    return HandEye(T, residual, n, str(doc["date"]), str(doc["camera_serial"]))

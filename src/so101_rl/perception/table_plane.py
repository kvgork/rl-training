"""Table-plane fit on D435 depth (camera parity C.2.2).

:func:`fit_table_plane` is ported from ``fit_table_plane`` in LIT ``scripts/_object_pose_depth.py``
(lerobot-isaac-training, commit 1f3451a): RANSAC over three-point planes on the pixels inside a depth band, 300
iterations, 5 mm inlier tolerance. Changes in the port: it takes a point cloud (the caller deprojects, usually the
per-pixel median of several aligned frames), it keeps only planes whose up direction points towards the camera and
up in the image (the table, not a wall or the bed), it refines the winner by a least-squares fit on its inliers, and
it orients the normal UP (towards the camera) explicitly instead of by the median-side heuristic.

The result fixes three of the camera's six DOF relative to the table: its height above the table and its roll and
pitch. The table is the base frame's plane ``z = table_z_m``, so these become ``plane_constrained_pose`` in
``perception/camera_calib.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_rl.perception.camera_calib import StreamIntrinsics

DEPTH_BAND_M = (0.15, 1.5)
PLANE_TOL_M = 0.005
ITERATIONS = 300
MIN_UP_IMAGE = 0.3  # the up normal's component along image-up (-y): rejects walls and steep surfaces


@dataclass(frozen=True, eq=False)
class PlaneFit:
    normal_up: np.ndarray  # unit [3] in the camera frame, pointing towards the camera
    height_m: float  # camera centre above the plane
    inlier_fraction: float
    inliers: np.ndarray  # bool [P]

    def signed_height(self, pts: np.ndarray) -> np.ndarray:
        """Height of camera-frame points above the plane (positive towards the camera)."""
        return np.asarray(pts) @ self.normal_up + self.height_m


def depth_to_points(depth_m: np.ndarray, intr: StreamIntrinsics, *, band: tuple[float, float] = DEPTH_BAND_M) -> tuple[np.ndarray, np.ndarray]:
    """(points [P, 3] camera frame, pixel indices [P, 2] as (v, u)) of the valid depth pixels inside ``band``."""
    d = np.asarray(depth_m, dtype=np.float64)
    ok = np.isfinite(d) & (d > band[0]) & (d < band[1])
    v, u = np.nonzero(ok)
    return intr.deproject(u, v, d[ok]), np.stack([v, u], axis=1)


def median_depth(frames: np.ndarray) -> np.ndarray:
    """Per-pixel median over ``frames [K, H, W]`` ignoring zeros (holes); 0 where every frame is a hole."""
    f = np.where(np.asarray(frames) > 0, frames, np.nan).astype(np.float64)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(f, axis=0)
    return np.nan_to_num(m, nan=0.0)


def fit_table_plane(pts: np.ndarray, *, seed: int = 0, iterations: int = ITERATIONS, tol_m: float = PLANE_TOL_M) -> PlaneFit:
    """RANSAC the table in camera-frame points (module docstring). Raises ``ValueError`` without a candidate."""
    P = np.asarray(pts, dtype=np.float64)
    if len(P) < 5000:
        raise ValueError(f"only {len(P)} points in the depth band")
    rng = np.random.default_rng(seed)
    sub = P[rng.choice(len(P), size=min(len(P), 60_000), replace=False)]
    best = None
    for _ in range(iterations):
        p = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ p[0]
        if d < 0:  # orient towards the camera: the camera (origin) is on the positive side
            n, d = -n, -d
        if -n[1] < MIN_UP_IMAGE:
            continue
        inl = int((np.abs(sub @ n + d) < tol_m).sum())
        if best is None or inl > best[0]:
            best = (inl, n, d)
    if best is None:
        raise ValueError("no table-like plane found")
    _, n, d = best
    inliers = np.abs(P @ n + d) < tol_m
    Q = P[inliers]
    c = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n2 = vt[2] if vt[2] @ n > 0 else -vt[2]
    d2 = -n2 @ c
    inliers = np.abs(P @ n2 + d2) < tol_m
    return PlaneFit(n2, float(d2), float(inliers.mean()), inliers)

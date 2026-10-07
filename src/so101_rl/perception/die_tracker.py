"""HSV + depth red-die tracker (plan F9).

``detect`` returns ``None`` whenever the die is not reliably visible (no blob, blob outside the area band, a blob
cut by the image border, inconsistent or missing depth, or a nearer object touching the blob = occlusion). The verifier maps ``None``
to UNKNOWN, never SUCCESS (L11).

**Position convention (integration 2026-10-06).** ``Detection.xyz_cam`` is the deprojected median depth at the
blob centre, i.e. the point where the camera ray through the blob centre ENTERS the die (a point on its visible
surface; it is also the right point for the hand-eye marker, which is a flat surface at the FK tip). Everything
published as a die position -- the object table's ``xyz_base``, FS3's ``die_xyz``, the reward and the success
predicate -- is the die CENTROID, the convention of the sim tracker (``sim/sim_tracker.py``) and of
``env/tasks.py``. Convert with :func:`die_centroid_base` / :meth:`HsvDepthDieTracker.centroid_base`: base frame,
then push the surface point ON ALONG THE VIEWING RAY by the cube's entry depth, half an edge / max_i |d_i| for the
unit ray d in the die frame (camera parity C.4: the real D435 is oblique, 64 deg from vertical, so the blob centre
lies on a side face, not on the top face). The die is assumed face-up with an unknown yaw: max_i |d_i| =
max(|d_z|, s (1 + 1/sqrt 2) / 2) with s the horizontal part of d (the yaw average of the two extremes). Overhead
(d = -z) this is exactly "half an edge along the table normal", the previous rule. Error budget: the yaw
assumption <= ~1.5 mm, a die tilted in the gripper a few mm, inside the 16 mm height band.
``tests/test_die_convention.py`` and ``tests/test_camera_parity.py`` check the real and sim trackers against each
other on scenes rendered from the calibrated camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from so101_rl.perception.camera import Intrinsics
from so101_rl.perception.camera_calib import CameraCalib, load_camera_calib
from so101_rl.perception.frames import cam_to_base, pixel_to_cam

DEFAULT_HSV_YAML = Path(__file__).with_name("hsv_red_die.yaml")
_OPEN_KERNEL = np.ones((3, 3), np.uint8)
_RING_KERNEL = np.ones((7, 7), np.uint8)
_OCCLUDED_RING_FRACTION = 0.15  # share of the blob's surrounding ring that is nearer than the blob
TABLE_NORMAL_BASE = np.array([0.0, 0.0, 1.0])  # the table is the base frame's horizontal plane (z = table_z_m)


def die_entry_depth(ray_base: np.ndarray, die_edge_m: float) -> float:
    """Distance from where the unit viewing ray ``ray_base`` enters a face-up die of unknown yaw to its centroid
    (module docstring)."""
    d = np.asarray(ray_base, dtype=np.float64)
    d = d / np.linalg.norm(d)
    s = float(np.hypot(d[0], d[1]))
    m = max(abs(float(d @ TABLE_NORMAL_BASE)), s * (1.0 + 1.0 / np.sqrt(2.0)) / 2.0)
    return 0.5 * float(die_edge_m) / m


def die_centroid_base(xyz_cam: np.ndarray, T_base_cam: np.ndarray, die_edge_m: float) -> np.ndarray:
    """Die centroid in the base frame from the deprojected surface point ``xyz_cam`` (module docstring): the point
    in the base frame pushed along the viewing ray by :func:`die_entry_depth`."""
    if not die_edge_m > 0:
        raise ValueError(f"die_edge_m must be > 0, got {die_edge_m}")
    p = np.asarray(xyz_cam, dtype=np.float64)
    ray = T_base_cam[:3, :3] @ (p / np.linalg.norm(p))
    return cam_to_base(p, T_base_cam) + die_entry_depth(ray, die_edge_m) * ray


@dataclass
class Detection:
    u: float
    v: float
    area_px: int
    xyz_cam: np.ndarray  # visible TOP-face point, camera frame (NOT the centroid: use centroid_base)
    confidence: float


class HsvDepthDieTracker:
    def __init__(
        self,
        hsv_lo: np.ndarray,
        hsv_hi: np.ndarray,
        intr: Intrinsics,
        *,
        min_area_px: int = 40,
        max_area_px: int = 4000,
        depth_window_m: float = 0.02,
        die_edge_m: float = 0.016,
    ) -> None:
        self.hsv_lo = np.asarray(hsv_lo, dtype=np.uint8).reshape(3)
        self.hsv_hi = np.asarray(hsv_hi, dtype=np.uint8).reshape(3)
        self.intr = intr
        self.min_area_px = int(min_area_px)
        self.max_area_px = int(max_area_px)
        self.depth_window_m = float(depth_window_m)
        self.die_edge_m = float(die_edge_m)
        if not 0 < self.min_area_px < self.max_area_px:
            raise ValueError("need 0 < min_area_px < max_area_px")
        if self.depth_window_m <= 0:
            raise ValueError("depth_window_m must be > 0")
        if not self.die_edge_m > 0:
            raise ValueError("die_edge_m must be > 0")

    @classmethod
    def from_camera_calib(cls, calib: CameraCalib | None = None, path: Path = DEFAULT_HSV_YAML) -> HsvDepthDieTracker:
        """The tracker on the colour stream of ``calib/camera.yaml`` (or ``calib``): depth is aligned to colour."""
        c = calib if calib is not None else load_camera_calib()
        return cls.from_yaml(c.color_intrinsics(), path)

    @classmethod
    def from_yaml(cls, intr: Intrinsics, path: Path = DEFAULT_HSV_YAML) -> HsvDepthDieTracker:
        d = yaml.safe_load(Path(path).read_text())
        return cls(
            np.array(d["hsv_lo"]),
            np.array(d["hsv_hi"]),
            intr,
            min_area_px=int(d["min_area_px"]),
            max_area_px=int(d["max_area_px"]),
            depth_window_m=float(d["depth_window_m"]),
            die_edge_m=float(d["die_edge_m"]),
        )

    def centroid_base(self, det: Detection, T_base_cam: np.ndarray) -> np.ndarray:
        """The die CENTROID (base frame) of a detection: what the object table publishes (module docstring)."""
        return die_centroid_base(det.xyz_cam, T_base_cam, self.die_edge_m)

    def _mask(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
        lo, hi = self.hsv_lo, self.hsv_hi
        if lo[0] <= hi[0]:
            mask = cv2.inRange(hsv, lo, hi)
        else:  # hue wrap-around: two bands
            mask = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], np.uint8))
            mask |= cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)

    def detect(self, rgb: np.ndarray, depth_m: np.ndarray) -> Detection | None:
        mask = self._mask(rgb)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = -1
        best_area = 0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if self.min_area_px <= area <= self.max_area_px and area > best_area:
                best, best_area = i, area
        if best < 0:
            return None
        x0, y0 = int(stats[best, cv2.CC_STAT_LEFT]), int(stats[best, cv2.CC_STAT_TOP])
        x1, y1 = x0 + int(stats[best, cv2.CC_STAT_WIDTH]), y0 + int(stats[best, cv2.CC_STAT_HEIGHT])
        if x0 == 0 or y0 == 0 or x1 >= mask.shape[1] or y1 >= mask.shape[0]:
            return None  # cut by the image border: its centre is not the die's (camera parity C.4; the sim
            # tracker calls a die whose centre leaves the image invisible)
        u, v = float(cents[best][0]), float(cents[best][1])

        h, w = depth_m.shape
        ui, vi = round(u), round(v)
        win = depth_m[max(vi - 2, 0) : min(vi + 3, h), max(ui - 2, 0) : min(ui + 3, w)]
        valid = win[np.isfinite(win) & (win > 0)]
        if valid.size < max(win.size // 2, 1):
            return None
        spread = float(valid.max() - valid.min())
        if spread > self.depth_window_m:
            return None
        z = float(np.median(valid))

        blob = labels == best
        if self._occluded(blob, depth_m, z):
            return None

        area_fit = min(1.0, best_area / (3.0 * self.min_area_px))
        consistency = 1.0 - spread / self.depth_window_m
        conf = float(np.clip(area_fit * consistency, 0.0, 1.0))
        return Detection(u, v, best_area, pixel_to_cam(u, v, z, self.intr), conf)

    def _occluded(self, blob: np.ndarray, depth_m: np.ndarray, z: float) -> bool:
        """True when a large part of the blob's surrounding ring is clearly nearer than the blob."""
        ring = cv2.dilate(blob.astype(np.uint8), _RING_KERNEL).astype(bool) & ~blob
        d = depth_m[ring]
        d = d[np.isfinite(d) & (d > 0)]
        if d.size == 0:
            return False
        return float(np.mean(d < z - self.depth_window_m)) > _OCCLUDED_RING_FRACTION

"""HSV + depth red-die tracker (plan F9).

``detect`` returns ``None`` whenever the die is not reliably visible (no blob, blob outside the area band,
inconsistent or missing depth, or a nearer object touching the blob = occlusion). The verifier maps ``None``
to UNKNOWN, never SUCCESS (L11).

**Position convention (integration 2026-10-06).** ``Detection.xyz_cam`` is the deprojected median depth at the
blob centre, i.e. a point on the die's visible TOP face, about half an edge (8 mm) above its centre (it is also
the right point for the hand-eye marker, which is a flat surface at the FK tip). Everything published as a die
position -- the object table's ``xyz_base``, FS3's ``die_xyz``, the reward and the success predicate -- is the
die CENTROID, the convention of the sim tracker (``sim/sim_tracker.py``) and of ``env/tasks.py``. Convert with
:func:`die_centroid_base` / :meth:`HsvDepthDieTracker.centroid_base`: base frame, then half an edge along the
TABLE NORMAL (base +z: the table is the base frame's horizontal plane) subtracted. This assumes the die rests
face-up (on the table or flat in the gripper); a die tilted by theta reads up to half an edge x (sqrt(2)-1)
(about 3 mm at 45 deg about one axis) off, inside the 16 mm height band of the success predicate.
``tests/test_die_convention.py`` checks the real and sim trackers against each other on rendered scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from so101_rl.perception.camera import Intrinsics
from so101_rl.perception.frames import cam_to_base, pixel_to_cam

DEFAULT_HSV_YAML = Path(__file__).with_name("hsv_red_die.yaml")
_OPEN_KERNEL = np.ones((3, 3), np.uint8)
_RING_KERNEL = np.ones((7, 7), np.uint8)
_OCCLUDED_RING_FRACTION = 0.15  # share of the blob's surrounding ring that is nearer than the blob
TABLE_NORMAL_BASE = np.array([0.0, 0.0, 1.0])  # the table is the base frame's horizontal plane (z = table_z_m)


def die_centroid_base(xyz_cam: np.ndarray, T_base_cam: np.ndarray, die_edge_m: float) -> np.ndarray:
    """Die centroid in the base frame from the deprojected TOP-face point ``xyz_cam`` (see the module docstring):
    ``cam_to_base(xyz_cam) - die_edge_m / 2 * TABLE_NORMAL_BASE``."""
    if not die_edge_m > 0:
        raise ValueError(f"die_edge_m must be > 0, got {die_edge_m}")
    return cam_to_base(xyz_cam, T_base_cam) - 0.5 * float(die_edge_m) * TABLE_NORMAL_BASE


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

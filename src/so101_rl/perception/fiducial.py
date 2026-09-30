"""ArUco fiducial "camera moved" gate (plan F9, Phase 0 camera-moved gate).

A fixed marker is glued to the table. ``calib/fiducial.yaml`` records where its four corners appear in the
image when the hand-eye calibration was made. If the detected corners drift by more than ``tol_px`` (or the
marker is not found), the camera or the marker moved and training must halt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


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
        tol_px: float = 2.0,
    ) -> None:
        self.marker_id = int(marker_id)
        self.expected = np.asarray(expected_corners_px, dtype=np.float64).reshape(4, 2)
        self.tol_px = float(tol_px)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), params)

    @classmethod
    def from_yaml(cls, path: Path = Path("calib/fiducial.yaml")) -> FiducialCheck:
        d = yaml.safe_load(Path(path).read_text())
        dictionary = getattr(cv2.aruco, d["dictionary"])
        return cls(
            int(d["marker_id"]),
            np.array(d["expected_corners_px"], dtype=np.float64),
            dictionary=dictionary,
            tol_px=float(d.get("tol_px", 2.0)),
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

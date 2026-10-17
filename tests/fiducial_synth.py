"""Synthetic D435 frames of the H8 fiducial: the printed ArUco marker (on its white paper) warped by the homography of
``calib/camera.yaml``'s pose onto a grey table, and the table plane's aligned depth (no rendering, no hardware)."""

from __future__ import annotations

import time

import cv2
import numpy as np

from so101_rl.perception.fiducial import (
    DICTIONARY,
    MARKER_ID,
    SIZE_MM,
    XY_BASE,
    marker_corners_base,
    project_base,
)

PAPER_SCALE = 1.6  # the white paper around the marker (quiet zone), relative to the marker size


def marker_frame(calib, *, xy=XY_BASE, size_mm=SIZE_MM, shift_px=(0.0, 0.0), marker_id=MARKER_ID, present=True,
                 bg=90) -> np.ndarray:  # fmt: skip
    """RGB uint8 [H, W, 3]: the marker at ``xy`` seen by the calibrated camera, every pixel shifted by ``shift_px``
    (a camera that moved); ``present=False`` = an empty grey table."""
    c = calib.color
    img = np.full((c.height, c.width, 3), bg, np.uint8)
    if not present:
        return img
    src_px = 600
    m_px = round(src_px / PAPER_SCALE / 6) * 6  # 6 cells (4x4 + border), whole pixels per cell
    paper = np.full((src_px, src_px), 255, np.uint8)
    o = (src_px - m_px) // 2
    paper[o : o + m_px, o : o + m_px] = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY)), marker_id, m_px
    )
    scale = src_px / m_px  # paper size / marker size, exactly
    dst = project_base(marker_corners_base(xy, size_mm * scale), calib.T_base_color, calib.color_intrinsics())
    dst = dst + np.asarray(shift_px, np.float64)
    # pixel-centre convention: source pixel i covers [i, i+1); the paper corners are the outer pixel edges
    src = np.array([[0, 0], [src_px, 0], [src_px, src_px], [0, src_px]], np.float64) - 0.5
    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    warped = cv2.warpPerspective(paper, H, (c.width, c.height), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpPerspective(np.full_like(paper, 255), H, (c.width, c.height), flags=cv2.INTER_LINEAR, borderValue=0)
    a = mask.astype(np.float32)[..., None] / 255.0
    out = img.astype(np.float32) * (1 - a) + warped.astype(np.float32)[..., None] * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def predicted_corners(calib, *, xy=XY_BASE, size_mm=SIZE_MM, shift_px=(0.0, 0.0)) -> np.ndarray:
    return project_base(marker_corners_base(xy, size_mm), calib.T_base_color, calib.color_intrinsics()) + np.asarray(
        shift_px, np.float64
    )


def plane_depth(calib) -> np.ndarray:
    """float32 [H, W] depth (m) of camera.yaml's table plane along every colour pixel's ray."""
    c = calib.color
    v, u = np.mgrid[0 : c.height, 0 : c.width]
    rays = np.stack([(u - c.cx) / c.fx, (v - c.cy) / c.fy, np.ones(u.shape)], axis=-1)
    n, h = np.asarray(calib.table_plane.normal_color, np.float64), float(calib.table_plane.height_m)
    z = -h / (rays @ n)
    return np.where(z > 0, z, 0.0).astype(np.float32)


class SequenceCamera:
    """A started-on-demand fake camera serving ``frames()`` (a callable returning (rgb, depth)) on every read, with
    the calibration's serial (passes ``process.open_checked``)."""

    def __init__(self, calib, frames) -> None:
        self.calib = calib
        self.frames = frames
        self.device_serial = calib.serial
        self.reads = 0
        self.started = False

    def start(self):
        self.started = True
        return self.calib.color_intrinsics()

    def read(self, rgb_out, depth_out):
        from so101_rl.perception.camera import Frame

        rgb, depth = self.frames()
        rgb_out[:] = rgb
        depth_out[:] = depth
        self.reads += 1
        return Frame(time.monotonic_ns(), True)

    def stop(self) -> None:
        self.started = False

"""The arm's own pixels in the colour frame, so orange / red parts of the SO-101 are never taken for the die (plan
section 10 2026-10-17 "Perception: arm mask").

The table-only rule (2026-10-17) rejects red blobs off the table, but the arm stands ON the table workspace: its
servo horns, clips and cables are orange / red and lie inside the workspace whenever the arm is in view.
:class:`ArmMasker` renders where the arm is, from the joint ticks and the calibrated camera, and the die tracker skips
every candidate blob of which more than :data:`ARM_MAJORITY` (half) lies on the arm (``HsvDepthDieTracker.detect(...,
exclude=)``).

**Geometry, no MuJoCo at runtime.** ``arm_points.npz`` (``scripts/extract_arm_points.py``) holds the visual-mesh
surface samples of :class:`sim.arm_cloud.ArmSurface` (3 mm, 28 k points, the cloud the arm-shape calibration fits) in
their bodies' frames, and the corners of the four ``jaw_col`` boxes. :class:`ArmGeometry` places them with a numpy FK
of ``config/so101_kinematics.yaml`` (all six joints, the moving jaw included; ``tests/test_arm_mask.py`` pins it to
MuJoCo). The base never moves, so its 8.5 k points are z-buffered once.

**Per frame** (:meth:`ArmMasker.render`): FK -> 6 body transforms -> project the 19.6 k moving points (pinhole, the
D435 colour stream is undistorted) -> a z-buffer of the nearest arm point per pixel (with the static base layer) ->
a min filter of ``2 DILATE_PX + 1`` square (separable) inside the arm's bounding box. The min filter both fills the
gaps between samples (3 px already covers every arm pixel MuJoCo draws) and grows the silhouette for calibration error.

**Gates** (:meth:`ArmMasker.mask`): a covered pixel is "arm" unless

* its measured depth is clearly NEARER than the arm (``depth < arm_z - DEPTH_TOL_M``): something in front of the arm
  is not the arm; no depth = arm;
* it shows the **grasp region**: the convex hull of the facing faces of the two fingertip boxes (fixed tip, moving
  tip; each grown by ``GRASP_MARGIN_M``), with a measured depth inside that region's depth range +- ``DEPTH_TOL_M``.
  A die held between the jaws touches them, so the silhouette and the depth gate alone call it arm (sim: every die
  between closed jaws was lost); the region gives it back, while arm parts seen THROUGH the open jaws (farther) stay
  masked.

A red part ON the arm reads the arm's own depth (calibration 4.5 mm + D435 noise, inside 10 mm). The table around the
arm reads farther than the arm, so the dilation band counts as arm: a die on the table next to the arm loses at most
that band (sim: never more than half).

**Thresholds.** ``DILATE_PX = 6``: the arm-shape calibration residual is 4.5 mm (``calib/camera.yaml``), which is
4.6 px at the far end of the arm (0.58 m) and 7.9 px at the near end (0.38 m) with fx 617; 6 px is 4.5 mm at the
middle of that range (0.46 m). ``DEPTH_TOL_M = 0.010``, ``GRASP_MARGIN_M = 0.004``, ``ARM_MAJORITY = 0.5``. Cost
1.6 ms p50 / 1.9 ms p95 per 640 x 480 frame (CPU). Not masked: a red part of the fingertips themselves (inside the
grasp region) and the real arm's cables where they hang off the links by more than the dilation.

:class:`ServoStateReader` is the joint source of the perception process: it ATTACHES to the servo shm and only ever
calls ``read_state`` (never writes, never creates or unlinks, never touches the serial bus; CLAUDE.md). A record older
than :data:`MAX_STATE_AGE_NS` (100 ms), a failed bus read, or no servo at all = no ticks: the process then runs that
frame without an arm mask, as before, and counts it (:class:`ArmMaskSource`).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from so101_rl.config import DEFAULT_CONFIG_PATH, JOINT_NAMES, MotorCalibration, ticks_to_rad
from so101_rl.perception.camera_calib import CameraCalib
from so101_rl.perception.die_tracker import EXCLUDE_MAJORITY

__all__ = [
    "ARM_BODIES",
    "ARM_MAJORITY",
    "DEPTH_TOL_M",
    "DILATE_PX",
    "MAX_STATE_AGE_NS",
    "ArmGeometry",
    "ArmMaskSource",
    "ArmMasker",
    "JointSource",
    "ServoStateReader",
]

ARM_POINTS_NPZ = Path(__file__).with_name("arm_points.npz")
KINEMATICS_YAML = DEFAULT_CONFIG_PATH.with_name("so101_kinematics.yaml")
ARM_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")

DILATE_PX = 6
DEPTH_TOL_M = 0.010
GRASP_MARGIN_M = 0.004  # the jaw boxes grown by this for the grasp region
# arm_points.npz holds the corners of sim/model.py's _JAW_BOXES in order: fixed tip, fixed base, moving tip, moving
# base; corner index 4 ix + 2 iy + iz (i = 0 for the minus side)
GRASP_BOX_BODIES = ("gripper", "gripper", "moving_jaw_so101_v1", "moving_jaw_so101_v1")
GRASP_PAIRS = ((0, 2),)  # fingertips only: adding the proximal pair (1, 3) unmasked the gripper servo horn (sim)
GRASP_AXIS_CORNERS = [4, 2, 1]
ARM_MAJORITY = EXCLUDE_MAJORITY  # a blob is the arm's when MORE than this share of its pixels is arm (detect)
MAX_STATE_AGE_NS = 100_000_000
REATTACH_EVERY_NS = 1_000_000_000
_FAR = np.float32(1e3)  # "no arm" in the z-buffer (finite: cv2's min filter)


def _quat_wxyz(q: list[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose(pos: list[float], quat: list[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _quat_wxyz(quat)
    t[:3, 3] = np.asarray(pos, dtype=np.float64)
    return t


class ArmGeometry:
    """The arm's surface points per body and the FK that places them (module docstring)."""

    def __init__(self, kinematics: Path = KINEMATICS_YAML, points: Path = ARM_POINTS_NPZ) -> None:
        kin = yaml.safe_load(Path(kinematics).read_text())
        joints = kin["joints"]
        if tuple(j["name"] for j in joints) != JOINT_NAMES:
            raise ValueError(f"{kinematics}: joints != {JOINT_NAMES}")
        if (kin["base"]["body"], *(j["body"] for j in joints)) != ARM_BODIES:
            raise ValueError(f"{kinematics}: bodies != {ARM_BODIES}")
        for j in joints:
            if [float(a) for a in j["axis"]] != [0.0, 0.0, 1.0]:
                raise ValueError(f"joint {j['name']}: only +z axes are supported")
        self.base_T = _pose(kin["base"]["pos"], kin["base"]["quat"])
        self.parent = [ARM_BODIES.index(j["parent_body"]) for j in joints]
        if any(p > i for i, p in enumerate(self.parent)):
            raise ValueError("kinematics yaml: a joint's parent body must come before it")
        self.parent_T = np.stack([_pose(j["parent_body_pos"], j["parent_body_quat"]) for j in joints])
        d = np.load(points)
        if tuple(str(b) for b in d["bodies"]) != ARM_BODIES:
            raise ValueError(f"{points}: bodies != {ARM_BODIES} (rerun scripts/extract_arm_points.py)")
        body, pts = d["body"].astype(np.int64), d["points"].astype(np.float64)
        self.points = [np.ascontiguousarray(pts[body == b]) for b in range(len(ARM_BODIES))]
        self.grasp_body = d["grasp_body"].astype(np.int64)  # [K] body of each jaw-box corner
        self.grasp_corners = d["grasp_corners"].astype(np.float64)  # [K, 3] in that body's frame
        want = np.repeat([ARM_BODIES.index(b) for b in GRASP_BOX_BODIES], 8)
        if self.grasp_corners.shape != (32, 3) or not np.array_equal(self.grasp_body, want):
            raise ValueError(f"{points}: grasp corners are not the 4 jaw boxes (rerun scripts/extract_arm_points.py)")

    def grasp_corners_base(self, T: np.ndarray, margin_m: float = 0.0) -> np.ndarray:
        """Base-frame corners of the space BETWEEN the jaws for the body poses ``T`` of :meth:`body_frames`: for each
        box pair of ``GRASP_PAIRS`` (the fingertips), per box the 4 corners of the face that faces the other box, each
        box grown by ``margin_m``. Their convex hull is the grasp region."""
        R, t = T[self.grasp_body, :3, :3], T[self.grasp_body, :3, 3]
        boxes = (np.einsum("kij,kj->ki", R, self.grasp_corners) + t).reshape(-1, 8, 3)  # [4, 8, 3] base frame
        centre = boxes.mean(axis=1)
        out = []
        for a, b in GRASP_PAIRS:
            for me, other in ((a, b), (b, a)):
                c = boxes[me]
                ax = c[GRASP_AXIS_CORNERS] - c[0]  # [3, 3] the box edges from corner 0 (signs -,-,-)
                half = 0.5 * np.linalg.norm(ax, axis=1)
                ax = ax / (2 * half[:, None])
                d = centre[other] - centre[me]
                i = int(np.argmax(np.abs(ax @ d) / half))  # the face the other box lies beyond
                s = np.sign(ax[i] @ d)
                grow = c + margin_m * np.sign((c - centre[me]) @ ax.T) @ ax
                face = np.sign((c - centre[me]) @ ax[i]) == s
                out.append(grow[face])
        return np.concatenate(out)

    def body_frames(self, q: np.ndarray) -> np.ndarray:
        """[7, 4, 4] base-frame poses of :data:`ARM_BODIES` at joint angles ``q`` [6] (rad, JOINT_NAMES order)."""
        T = np.empty((len(ARM_BODIES), 4, 4))
        T[0] = self.base_T
        rz = np.eye(4)
        for i in range(len(JOINT_NAMES)):
            c, s = math.cos(float(q[i])), math.sin(float(q[i]))
            rz[0, 0], rz[0, 1], rz[1, 0], rz[1, 1] = c, -s, s, c
            T[i + 1] = T[self.parent[i]] @ self.parent_T[i] @ rz
        return T


def joint_angles(ticks: np.ndarray, calibration: dict[str, MotorCalibration]) -> np.ndarray:
    """Present_Position ticks [6] -> joint angles [6] rad: the arm through ``config.ticks_to_rad``, the gripper
    through ``sim.gripper_map`` (pure; the same conversion as ``MujocoBus`` and ``sim.arm_cloud.qpos_from_ticks``)."""
    from so101_rl.sim.gripper_map import gripper_ticks_to_rad

    q = ticks_to_rad(np.asarray(ticks, dtype=np.float64), calibration)
    q[-1] = gripper_ticks_to_rad(float(ticks[-1]), calibration)
    return q


class ArmMasker:
    """Arm z-buffer and arm mask in the colour frame of ``calib`` (module docstring)."""

    def __init__(
        self,
        calib: CameraCalib,
        *,
        geometry: ArmGeometry | None = None,
        dilate_px: int = DILATE_PX,
        depth_tol_m: float = DEPTH_TOL_M,
        grasp_margin_m: float = GRASP_MARGIN_M,
    ) -> None:
        self.geometry = geometry if geometry is not None else ArmGeometry()
        c = calib.color
        self.w, self.h = int(c.width), int(c.height)
        self.fx, self.fy, self.cx, self.cy = float(c.fx), float(c.fy), float(c.cx), float(c.cy)
        self.T_color_base = np.linalg.inv(np.asarray(calib.T_base_color, dtype=np.float64))
        self.dilate_px = int(dilate_px)
        self.depth_tol_m = float(depth_tol_m)
        self.grasp_margin_m = float(grasp_margin_m)
        if self.dilate_px < 0 or not self.depth_tol_m > 0:
            raise ValueError("need dilate_px >= 0 and depth_tol_m > 0")
        self._kernel = np.ones((2 * self.dilate_px + 1, 2 * self.dilate_px + 1), np.uint8)
        self._static = np.full(self.h * self.w, _FAR, np.float32)
        self._static_box = self._splat(self._static, self.geometry.points[0], self.geometry.base_T)
        self.zbuf = np.full((self.h, self.w), _FAR, np.float32)  # the last render, dilated (FAR = no arm)
        self.box: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1) of the last render, exclusive
        self._T = self.geometry.body_frames(np.zeros(len(JOINT_NAMES)))

    def _splat(self, flat: np.ndarray, pts: np.ndarray, T_base_body: np.ndarray) -> tuple[int, int, int, int] | None:
        """z-buffer ``pts`` (body frame) into ``flat`` [h*w] (min); the pixel box they cover, or None."""
        M = self.T_color_base @ T_base_body
        p = pts @ M[:3, :3].T + M[:3, 3]
        z = p[:, 2]
        ok = z > 0.05
        u = np.rint(self.fx * p[ok, 0] / z[ok] + self.cx).astype(np.int64)
        v = np.rint(self.fy * p[ok, 1] / z[ok] + self.cy).astype(np.int64)
        z = z[ok]
        inside = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        if not inside.any():
            return None
        u, v, z = u[inside], v[inside], z[inside].astype(np.float32)
        np.minimum.at(flat, v * self.w + u, z)
        return int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1

    def render(self, q: np.ndarray) -> np.ndarray:
        """The dilated arm z-buffer [h, w] float32 at joint angles ``q`` (FAR where there is no arm)."""
        g = self.geometry
        T = self._T = g.body_frames(q)
        flat = self._static.copy()
        boxes = [self._static_box]
        for b in range(1, len(g.points)):
            boxes.append(self._splat(flat, g.points[b], T[b]))
        boxes = [bx for bx in boxes if bx is not None]
        self.zbuf.fill(_FAR)
        if not boxes:
            self.box = None
            return self.zbuf
        r = self.dilate_px
        x0 = max(min(bx[0] for bx in boxes) - r, 0)
        y0 = max(min(bx[1] for bx in boxes) - r, 0)
        x1 = min(max(bx[2] for bx in boxes) + r, self.w)
        y1 = min(max(bx[3] for bx in boxes) + r, self.h)
        roi = flat.reshape(self.h, self.w)[y0:y1, x0:x1]
        self.zbuf[y0:y1, x0:x1] = cv2.erode(roi, self._kernel, borderType=cv2.BORDER_REPLICATE) if r else roi
        self.box = (x0, y0, x1, y1)
        return self.zbuf

    def mask(self, q: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """bool [h, w]: the arm covers the pixel, the measured depth is not clearly nearer than the arm, and the pixel
        does not show the grasp region (module docstring)."""
        zb = self.render(q)
        out = np.zeros((self.h, self.w), bool)
        if self.box is None:
            return out
        x0, y0, x1, y1 = self.box
        z = zb[y0:y1, x0:x1]
        d = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float32)
        valid = np.isfinite(d) & (d > 0)
        arm = (z < _FAR) & ~(valid & (d < z - self.depth_tol_m))
        grasp = self._grasp_region(x0, y0, x1, y1)
        if grasp is not None:
            poly, zlo, zhi = grasp
            arm &= ~(poly & valid & (d >= zlo - self.depth_tol_m) & (d <= zhi + self.depth_tol_m))
        out[y0:y1, x0:x1] = arm
        return out

    def _grasp_region(self, x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, float, float] | None:
        """(bool [y1-y0, x1-x0] the projected grasp hull, its nearest and farthest corner depth), or None when a
        corner is behind the camera."""
        P = self.geometry.grasp_corners_base(self._T, self.grasp_margin_m)
        M = self.T_color_base
        p = P @ M[:3, :3].T + M[:3, 3]
        z = p[:, 2]
        if not np.all(z > 0.05):
            return None
        uv = np.stack([self.fx * p[:, 0] / z + self.cx - x0, self.fy * p[:, 1] / z + self.cy - y0], axis=1)
        hull = cv2.convexHull(np.rint(uv).astype(np.int32))
        poly = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(poly, hull, 1)
        return poly.astype(bool), float(z.min()), float(z.max())


class JointSource(Protocol):
    """Fresh Present_Position ticks [6] (JOINT_NAMES order), or None when there is no fresh joint state."""

    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


class ServoStateReader:
    """Read-only joint source on the servo shm segment ``name`` (module docstring). ``status`` says why the last
    :meth:`read` gave None: ``no_servo`` (no segment), ``stale`` (older than ``max_age_ns``, or the servo stopped
    publishing), ``read_failed`` (the servo's bus read failed this tick)."""

    def __init__(self, name: str, *, max_age_ns: int = MAX_STATE_AGE_NS, reattach_every_ns: int = REATTACH_EVERY_NS) -> None:
        from so101_rl.servo.shm import new_state

        self.name = str(name)
        self.max_age_ns = int(max_age_ns)
        self.reattach_every_ns = int(reattach_every_ns)
        self._st = new_state()
        self._shm = None
        self._next_attach = 0
        self._last_good = 0
        self.status = "no_servo"

    def _attach(self, now: int) -> None:
        from so101_rl.servo.shm import ServoShm

        if now < self._next_attach:
            return
        self._next_attach = now + self.reattach_every_ns
        if self._shm is not None:  # a stale segment: the servo may have restarted on a new one
            self._shm.close()
            self._shm = None
        try:
            self._shm = ServoShm.attach(self.name)  # policy-side attach: never creates, never unlinks
        except (OSError, RuntimeError, ValueError):
            self._shm = None

    def read(self) -> np.ndarray | None:
        now = time.monotonic_ns()
        if self._shm is None or now - self._last_good > self.reattach_every_ns:
            self._attach(now)
        if self._shm is None:
            self.status = "no_servo"
            return None
        st = self._st
        if not self._shm.read_state(st, retries=8):
            self.status = "stale"
            return None
        if now - int(st["t_mono_ns"][0]) > self.max_age_ns:
            self.status = "stale"
            return None
        if not int(st["read_ok"][0]):
            self.status = "read_failed"
            return None
        self._last_good = now
        self.status = "ok"
        return st["present_ticks"][0].astype(np.float64)

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()  # attach side: unmaps only
            self._shm = None


class ArmMaskSource:
    """What the perception process calls once per frame: fresh joint ticks from ``joints`` -> :class:`ArmMasker`
    mask, or None (no fresh state: the frame runs without an arm mask, as before the mask existed). Counts both."""

    def __init__(self, masker: ArmMasker, joints: JointSource, calibration: dict[str, MotorCalibration]) -> None:
        self.masker, self.joints, self.calibration = masker, joints, calibration
        self.active = 0
        self.inactive: dict[str, int] = {}
        self.ms: list[float] = []

    def mask(self, depth_m: np.ndarray) -> np.ndarray | None:
        ticks = self.joints.read()
        if ticks is None:
            why = str(getattr(self.joints, "status", "no_state"))
            self.inactive[why] = self.inactive.get(why, 0) + 1
            return None
        t0 = time.perf_counter_ns()
        m = self.masker.mask(joint_angles(ticks, self.calibration), depth_m)
        self.ms.append((time.perf_counter_ns() - t0) / 1e6)
        self.active += 1
        return m

    def close(self) -> None:
        self.joints.close()

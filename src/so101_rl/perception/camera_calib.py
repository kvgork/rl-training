"""``calib/camera.yaml``: the ONE source of the D435's intrinsics and pose (camera parity unit SUC, task C.1).

Plan §10, 2026-10-06 "Camera parity": the intrinsics come from the device (``camera-intrinsics``, pyrealsense2,
streaming only), the pose from a camera-only calibration (``camera-calibrate --method arm-shape``; later H8
hand-eye refines it). Every consumer reads this file: the sim cameras ``d435_color`` / ``d435_depth``
(``sim/d435.py``, ``sim/model.py``), ``SimDieTracker`` (``CameraModel.default``), the rollout videos
(``viz/video.py``), the real ``HsvDepthDieTracker`` (``from_camera_calib``) and ``perception/frames``
(``default_T_base_cam``). ``tests/test_camera_single_source.py`` greps the sources for hard-coded intrinsics or
camera poses.

Frames. ``color`` / ``depth`` are the RealSense optical frames (x right, y down, z forward, metres).
``T_base_color`` maps colour-frame points to the robot base frame. ``depth_to_color`` maps depth-frame points to the
colour frame (``p_color = T @ p_depth``; pyrealsense2 ``get_extrinsics_to``, whose rotation is column-major).

Schema (strict: unknown or missing keys are errors; ``version`` must be :data:`CAMERA_YAML_VERSION`)::

    version: 1
    serial: "819312071240"
    firmware: "5.11.1.100"
    color: {width, height, fps, fx, fy, cx, cy, distortion: {model, coeffs: [5]}}
    depth: {width, height, fps, fx, fy, cx, cy, distortion: {model, coeffs: [5]}, depth_scale}
    depth_to_color: [[4x4]]
    T_base_color:            # null = never calibrated (consumers that need a pose refuse)
      matrix: [[4x4]]
      method: arm-shape | hand-eye | provisional
      date: "YYYY-MM-DD"
      residual_mm: float | null      # null only for provisional
      uncertainty: {trans_mm, rot_deg}
      notes: str
    table_plane:             # null = not measured
      normal_color: [3]      # unit, pointing UP (towards the camera), colour frame
      height_m: float        # camera centre above the table plane
      inlier_fraction: float
      n_frames: int
      date: "YYYY-MM-DD"

Distortion: every consumer uses the pinhole model, so the loader refuses non-zero coefficients (the D435 reports
0 for both streams; a firmware that reports otherwise must be handled explicitly, not silently ignored).
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from so101_rl.perception.camera import Intrinsics

CAMERA_YAML_VERSION: Final[int] = 1
CAMERA_YAML: Final[Path] = Path(__file__).resolve().parents[3] / "calib" / "camera.yaml"
METHODS: Final[frozenset[str]] = frozenset({"arm-shape", "hand-eye", "provisional"})
CALIBRATED_METHODS: Final[frozenset[str]] = frozenset({"arm-shape", "hand-eye"})
DISTORTION_MODELS: Final[frozenset[str]] = frozenset({"none", "brown_conrady", "inverse_brown_conrady"})
MAX_RESIDUAL_MM: Final[float] = 5.0  # the arm-shape gate (C.2); a calibrated pose above it is refused
# DR floor on the camera pose until H8 (plan C.3): at least +-5 mm and +-1.5 deg, more when the calibration says so.
DR_MIN_TRANS_MM: Final[float] = 5.0
DR_MIN_ROT_DEG: Final[float] = 1.5

_STREAM_KEYS = frozenset({"width", "height", "fps", "fx", "fy", "cx", "cy", "distortion"})
_TOP_KEYS = frozenset({"version", "serial", "firmware", "color", "depth", "depth_to_color", "T_base_color", "table_plane"})
_EXT_KEYS = frozenset({"matrix", "method", "date", "residual_mm", "uncertainty", "notes"})
_PLANE_KEYS = frozenset({"normal_color", "height_m", "inlier_fraction", "n_frames", "date"})


class CameraNotCalibrated(RuntimeError):
    """``calib/camera.yaml`` has no pose (or only a provisional one where a calibrated one is required)."""


@dataclass(frozen=True)
class StreamIntrinsics:
    width: int
    height: int
    fps: int
    fx: float
    fy: float
    cx: float  # principal point, pixel coordinates with pixel CENTRES at integers (OpenCV / RealSense)
    cy: float
    distortion_model: str
    distortion_coeffs: tuple[float, ...]

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        """Pinhole: camera-frame points ``[..., 3]`` -> pixels ``[..., 2]`` (no clipping)."""
        p = np.asarray(p_cam, dtype=np.float64)
        z = p[..., 2]
        return np.stack([self.fx * p[..., 0] / z + self.cx, self.fy * p[..., 1] / z + self.cy], axis=-1)

    def deproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        u, v, z = (np.asarray(a, dtype=np.float64) for a in (u, v, z))
        return np.stack([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z], axis=-1)

    def as_intrinsics(self, depth_scale: float) -> Intrinsics:
        return Intrinsics(self.fx, self.fy, self.cx, self.cy, self.width, self.height, float(depth_scale))


@dataclass(frozen=True, eq=False)
class CameraPose:
    matrix: np.ndarray  # 4x4 T_base_color
    method: str
    date: str
    residual_mm: float | None
    trans_mm: float  # uncertainty
    rot_deg: float
    notes: str = ""

    @property
    def calibrated(self) -> bool:
        return self.method in CALIBRATED_METHODS


@dataclass(frozen=True, eq=False)
class TablePlane:
    normal_color: np.ndarray  # unit [3], up (towards the camera), colour frame
    height_m: float
    inlier_fraction: float
    n_frames: int
    date: str

    def tilt_deg(self) -> float:
        """Angle between the optical axis and the downward vertical."""
        return float(math.degrees(math.acos(float(np.clip(-self.normal_color[2], -1.0, 1.0)))))


@dataclass(frozen=True, eq=False)
class CameraCalib:
    serial: str
    firmware: str
    color: StreamIntrinsics
    depth: StreamIntrinsics
    depth_scale: float
    T_color_depth: np.ndarray  # 4x4: p_color = T @ p_depth
    pose: CameraPose | None
    table_plane: TablePlane | None
    version: int = CAMERA_YAML_VERSION

    # ---------------------------------------------------------------- poses
    def require_pose(self, *, calibrated: bool = False) -> CameraPose:
        if self.pose is None:
            raise CameraNotCalibrated(
                "calib/camera.yaml has no T_base_color: run `pixi run camera-calibrate --method arm-shape`"
            )
        if calibrated and not self.pose.calibrated:
            raise CameraNotCalibrated(
                f"calib/camera.yaml T_base_color is {self.pose.method!r}, not calibrated: run "
                "`pixi run camera-calibrate --method arm-shape`"
            )
        return self.pose

    @property
    def T_base_color(self) -> np.ndarray:
        return self.require_pose().matrix

    @property
    def T_base_depth(self) -> np.ndarray:
        return self.T_base_color @ self.T_color_depth

    def color_intrinsics(self) -> Intrinsics:
        """The perception ``Intrinsics`` of the colour stream (depth aligned to colour, real pipeline)."""
        return self.color.as_intrinsics(self.depth_scale)

    def dr_ranges(self) -> tuple[float, float]:
        """(translation m, rotation rad) of the camera-pose domain randomisation: the calibration uncertainty, at
        least +-5 mm and +-1.5 deg (plan C.3, until H8). A provisional pose gets the floor only: its uncertainty is
        a statement of ignorance, not a training range."""
        pose = self.require_pose()
        t, r = DR_MIN_TRANS_MM, DR_MIN_ROT_DEG
        if pose.calibrated:
            t, r = max(t, pose.trans_mm), max(r, pose.rot_deg)
        return t / 1000.0, math.radians(r)

    def with_pose(self, pose: CameraPose, table_plane: TablePlane | None = None) -> CameraCalib:
        return replace(self, pose=pose, table_plane=table_plane if table_plane is not None else self.table_plane)


# ------------------------------------------------------------------------------------------------ validation
def _keys(node: Any, expected: frozenset[str], where: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(node).__name__}")
    missing, extra = expected - set(node), set(node) - expected
    if missing:
        raise ValueError(f"{where}: missing keys {sorted(missing)}")
    if extra:
        raise ValueError(f"{where}: unknown keys {sorted(extra)}")
    return node


def _num(v: Any, where: str, lo: float = -math.inf, hi: float = math.inf) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise TypeError(f"{where}: expected a number, got {v!r}")
    f = float(v)
    if not math.isfinite(f) or not lo <= f <= hi:
        raise ValueError(f"{where}: {f} outside [{lo}, {hi}]")
    return f


def _int(v: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{where}: expected an integer, got {v!r}")
    if not lo <= v <= hi:
        raise ValueError(f"{where}: {v} outside [{lo}, {hi}]")
    return int(v)


def _str(v: Any, where: str) -> str:
    if not isinstance(v, str) or not v:
        raise TypeError(f"{where}: expected a non-empty string, got {v!r}")
    return v


def _rigid(m: Any, where: str) -> np.ndarray:
    T = np.array(m, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{where}: must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError(f"{where}: last row must be [0, 0, 0, 1]")
    r = T[:3, :3]
    if not (np.allclose(r @ r.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(r) - 1.0) < 1e-6):
        raise ValueError(f"{where}: rotation is not a proper rotation")
    return T


def _stream(node: Any, where: str, *, depth: bool) -> tuple[StreamIntrinsics, float | None]:
    keys = _STREAM_KEYS | ({"depth_scale"} if depth else frozenset())
    n = _keys(node, frozenset(keys), where)
    w = _int(n["width"], f"{where}.width", 16, 4096)
    h = _int(n["height"], f"{where}.height", 16, 4096)
    dist = _keys(n["distortion"], frozenset({"model", "coeffs"}), f"{where}.distortion")
    model = _str(dist["model"], f"{where}.distortion.model")
    if model not in DISTORTION_MODELS:
        raise ValueError(f"{where}.distortion.model {model!r} not in {sorted(DISTORTION_MODELS)}")
    coeffs = dist["coeffs"]
    if not isinstance(coeffs, list) or len(coeffs) != 5:
        raise ValueError(f"{where}.distortion.coeffs must be a list of 5 numbers")
    co = tuple(_num(c, f"{where}.distortion.coeffs") for c in coeffs)
    if any(c != 0.0 for c in co):
        raise ValueError(f"{where}: non-zero distortion {co}; every consumer is pinhole-only (module docstring)")
    s = StreamIntrinsics(
        width=w,
        height=h,
        fps=_int(n["fps"], f"{where}.fps", 1, 300),
        fx=_num(n["fx"], f"{where}.fx", 1.0, 1e5),
        fy=_num(n["fy"], f"{where}.fy", 1.0, 1e5),
        cx=_num(n["cx"], f"{where}.cx", 0.0, w - 1.0),
        cy=_num(n["cy"], f"{where}.cy", 0.0, h - 1.0),
        distortion_model=model,
        distortion_coeffs=co,
    )
    scale = _num(n["depth_scale"], f"{where}.depth_scale", 1e-6, 0.1) if depth else None
    return s, scale


def parse_camera_calib(doc: Any, where: str = "camera.yaml") -> CameraCalib:
    d = _keys(doc, _TOP_KEYS, where)
    if isinstance(d["version"], bool) or d["version"] != CAMERA_YAML_VERSION:
        raise ValueError(f"{where}: version {d['version']!r}, expected {CAMERA_YAML_VERSION}")
    color, _ = _stream(d["color"], f"{where}: color", depth=False)
    depth, scale = _stream(d["depth"], f"{where}: depth", depth=True)
    T_cd = _rigid(d["depth_to_color"], f"{where}: depth_to_color")
    if np.linalg.norm(T_cd[:3, 3]) > 0.1:
        raise ValueError(f"{where}: depth_to_color translation {T_cd[:3, 3]} m is not a D435 baseline")
    pose = None
    if d["T_base_color"] is not None:
        e = _keys(d["T_base_color"], _EXT_KEYS, f"{where}: T_base_color")
        method = _str(e["method"], f"{where}: T_base_color.method")
        if method not in METHODS:
            raise ValueError(f"{where}: T_base_color.method {method!r} not in {sorted(METHODS)}")
        unc = _keys(e["uncertainty"], frozenset({"trans_mm", "rot_deg"}), f"{where}: T_base_color.uncertainty")
        res = e["residual_mm"]
        if res is None:
            if method != "provisional":
                raise ValueError(f"{where}: a {method} pose needs residual_mm")
        else:
            res = _num(res, f"{where}: T_base_color.residual_mm", 0.0, 1e3)
            if method in CALIBRATED_METHODS and res > MAX_RESIDUAL_MM:
                raise ValueError(f"{where}: residual {res} mm exceeds the {MAX_RESIDUAL_MM} mm gate; recalibrate")
        pose = CameraPose(
            matrix=_rigid(e["matrix"], f"{where}: T_base_color.matrix"),
            method=method,
            date=_str(str(e["date"]), f"{where}: T_base_color.date"),
            residual_mm=res,
            trans_mm=_num(unc["trans_mm"], f"{where}: uncertainty.trans_mm", 0.0, 1e4),
            rot_deg=_num(unc["rot_deg"], f"{where}: uncertainty.rot_deg", 0.0, 180.0),
            notes=str(e["notes"] or ""),
        )
    plane = None
    if d["table_plane"] is not None:
        p = _keys(d["table_plane"], _PLANE_KEYS, f"{where}: table_plane")
        n = np.array(p["normal_color"], dtype=np.float64)
        if n.shape != (3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n) - 1.0) > 1e-3:
            raise ValueError(f"{where}: table_plane.normal_color must be a unit 3-vector")
        if n[2] >= 0:
            raise ValueError(f"{where}: table_plane.normal_color must point up, towards the camera (z < 0)")
        plane = TablePlane(
            normal_color=n / np.linalg.norm(n),
            height_m=_num(p["height_m"], f"{where}: table_plane.height_m", 0.05, 3.0),
            inlier_fraction=_num(p["inlier_fraction"], f"{where}: table_plane.inlier_fraction", 0.0, 1.0),
            n_frames=_int(p["n_frames"], f"{where}: table_plane.n_frames", 1, 10_000),
            date=_str(str(p["date"]), f"{where}: table_plane.date"),
        )
    return CameraCalib(
        serial=_str(str(d["serial"]), f"{where}: serial"),
        firmware=_str(str(d["firmware"]), f"{where}: firmware"),
        color=color,
        depth=depth,
        depth_scale=float(scale),
        T_color_depth=T_cd,
        pose=pose,
        table_plane=plane,
    )


def load_camera_calib(path: Path | str | None = None, *, require_calibrated: bool = False) -> CameraCalib:
    """Load and validate ``calib/camera.yaml`` (or ``path``). ``require_calibrated``: refuse a missing or
    provisional pose (:class:`CameraNotCalibrated`)."""
    p = CAMERA_YAML if path is None else Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing: run `pixi run camera-intrinsics --serial <serial> --out {p}` (plan C.1)"
        )
    calib = parse_camera_calib(yaml.safe_load(p.read_text()), str(p))
    if require_calibrated:
        calib.require_pose(calibrated=True)
    return calib


def _stream_doc(s: StreamIntrinsics) -> dict[str, Any]:
    return {
        "width": s.width,
        "height": s.height,
        "fps": s.fps,
        "fx": float(s.fx),
        "fy": float(s.fy),
        "cx": float(s.cx),
        "cy": float(s.cy),
        "distortion": {"model": s.distortion_model, "coeffs": [float(c) for c in s.distortion_coeffs]},
    }


def camera_calib_doc(c: CameraCalib) -> dict[str, Any]:
    depth = _stream_doc(c.depth)
    depth["depth_scale"] = float(c.depth_scale)
    pose = None
    if c.pose is not None:
        pose = {
            "matrix": [[float(x) for x in row] for row in c.pose.matrix],
            "method": c.pose.method,
            "date": c.pose.date,
            "residual_mm": None if c.pose.residual_mm is None else float(c.pose.residual_mm),
            "uncertainty": {"trans_mm": float(c.pose.trans_mm), "rot_deg": float(c.pose.rot_deg)},
            "notes": c.pose.notes,
        }
    plane = None
    if c.table_plane is not None:
        tp = c.table_plane
        plane = {
            "normal_color": [float(x) for x in tp.normal_color],
            "height_m": float(tp.height_m),
            "inlier_fraction": float(tp.inlier_fraction),
            "n_frames": int(tp.n_frames),
            "date": tp.date,
        }
    return {
        "version": c.version,
        "serial": c.serial,
        "firmware": c.firmware,
        "color": _stream_doc(c.color),
        "depth": depth,
        "depth_to_color": [[float(x) for x in row] for row in c.T_color_depth],
        "T_base_color": pose,
        "table_plane": plane,
    }


def save_camera_calib(c: CameraCalib, path: Path | str) -> None:
    """Validate (round trip through the strict parser), then write atomically."""
    doc = camera_calib_doc(c)
    parse_camera_calib(doc, "camera calib to save")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = "# D435 camera calibration (plan C.1, schema: perception/camera_calib.py, calib/README.md). Generated.\n"
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(header + yaml.safe_dump(doc, sort_keys=False))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ------------------------------------------------------------------------------------------------ geometry
def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def plane_constrained_pose(up_color: np.ndarray, height_m: float, table_z_m: float, x: float, y: float, yaw: float) -> np.ndarray:
    """``T_base_color`` with the table plane fixing height, roll and pitch: the camera's measured up direction
    (colour frame) maps to base +z, the camera centre sits ``height_m`` above the table (base z = ``table_z_m``),
    and the remaining three DOF are ``x``, ``y`` and ``yaw`` (rotation about base z)."""
    R0 = rotation_aligning(up_color, np.array([0.0, 0.0, 1.0]))
    T = np.eye(4)
    T[:3, :3] = rot_z(yaw) @ R0
    T[:3, 3] = (x, y, table_z_m + height_m)
    return T


def look_at_yaw(up_color: np.ndarray, view_dir_xy: np.ndarray) -> float:
    """The yaw for :func:`plane_constrained_pose` that points the optical axis' horizontal component along
    ``view_dir_xy`` (base frame)."""
    R0 = rotation_aligning(up_color, np.array([0.0, 0.0, 1.0]))
    z = R0[:, 2]
    if np.hypot(z[0], z[1]) < 1e-6:
        raise ValueError("the optical axis is vertical: yaw is not defined by a viewing direction")
    want = math.atan2(float(view_dir_xy[1]), float(view_dir_xy[0]))
    return want - math.atan2(z[1], z[0])


def provisional_pose(
    up_color: np.ndarray, height_m: float, table_z_m: float, look_at_xy: np.ndarray, view_dir_xy: np.ndarray
) -> np.ndarray:
    """A plane-constrained pose whose optical axis meets the table at ``look_at_xy`` from direction
    ``view_dir_xy``. For a PROVISIONAL ``calib/camera.yaml`` before the arm-shape calibration has run: height and
    tilt are measured, the rest is an operator's statement (recorded in ``notes``)."""
    yaw = look_at_yaw(up_color, view_dir_xy)
    T = plane_constrained_pose(up_color, height_m, table_z_m, 0.0, 0.0, yaw)
    z = T[:3, 2]
    if z[2] >= 0:
        raise ValueError("the optical axis does not point down at the table")
    s = height_m / -z[2]  # along the axis from the camera centre to the table
    T[:2, 3] = np.asarray(look_at_xy, dtype=np.float64) - s * z[:2]
    return T


def today() -> str:
    return _dt.datetime.now().astimezone().date().isoformat()


def main(argv: list[str] | None = None) -> int:
    """``python -m so101_rl.perception.camera_calib --provisional ...``: write a PROVISIONAL pose into an existing
    camera.yaml from a measured table plane plus an operator-stated look-at point and viewing direction."""
    import argparse

    from so101_rl.sim.model import TABLE_Z_M

    ap = argparse.ArgumentParser(prog="python -m so101_rl.perception.camera_calib")
    ap.add_argument("--file", type=Path, default=CAMERA_YAML)
    ap.add_argument("--provisional", action="store_true", required=True)
    ap.add_argument("--plane-normal", type=float, nargs=3, required=True, help="measured table normal, colour frame")
    ap.add_argument("--plane-height-m", type=float, required=True)
    ap.add_argument("--plane-date", required=True)
    ap.add_argument("--plane-inliers", type=float, required=True)
    ap.add_argument("--plane-frames", type=int, required=True)
    ap.add_argument("--look-at", type=float, nargs=2, required=True, help="base xy where the optical axis meets the table")
    ap.add_argument("--view-dir", type=float, nargs=2, required=True, help="base xy direction the camera looks along")
    ap.add_argument("--trans-mm", type=float, required=True)
    ap.add_argument("--rot-deg", type=float, required=True)
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    calib = load_camera_calib(a.file)
    if calib.pose is not None and calib.pose.calibrated:
        raise SystemExit(f"{a.file} holds a calibrated pose ({calib.pose.method}); refusing to replace it")
    n = np.array(a.plane_normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    up = n if n[2] < 0 else -n  # orient towards the camera
    T = provisional_pose(up, a.plane_height_m, TABLE_Z_M, np.array(a.look_at), np.array(a.view_dir))
    plane = TablePlane(up, a.plane_height_m, a.plane_inliers, a.plane_frames, a.plane_date)
    pose = CameraPose(T, "provisional", today(), None, a.trans_mm, a.rot_deg, a.notes)
    save_camera_calib(calib.with_pose(pose, plane), a.file)
    print(f"provisional T_base_color written to {a.file}: camera at {np.round(T[:3, 3], 4).tolist()}, "
          f"tilt {plane.tilt_deg():.1f} deg from vertical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

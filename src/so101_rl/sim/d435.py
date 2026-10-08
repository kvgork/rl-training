"""The real D435 in MuJoCo (camera parity unit SUC, task C.3).

``add_d435_cameras(spec, calib)`` adds two fixed world cameras built from ``calib/camera.yaml``
(``perception/camera_calib.py``):

* ``d435_color`` at ``T_base_color``, with the colour stream's intrinsics;
* ``d435_depth`` at ``T_base_color @ depth_to_color`` (the depth sensor's own pose, 14.8 mm from the colour
  sensor), with the depth stream's intrinsics.

**Intrinsics (MuJoCo 3.14, verified on the installed build).** ``MjsCamera`` has ``resolution``,
``sensor_size``, ``focal_pixel`` and ``principal_pixel`` (MJCF ``resolution``, ``sensorsize``, ``focalpixel``,
``principalpixel``); a non-zero ``sensor_size`` switches the camera from ``fovy`` to the intrinsic model. The
compiler stores ``cam_intrinsic = [fx, fy, px, py] * sensor_size / resolution`` (length units; checked here at
build time), so the sensor size itself is arbitrary. The principal point is an OFFSET FROM THE IMAGE CENTRE with
BOTH axes flipped relative to OpenCV/RealSense pixel coordinates: a pixel at OpenCV ``(u, v)`` (pixel centres at
integers) is rendered at ``u = (W - 1)/2 - px + fx X/Z`` and ``v = (H - 1)/2 - py + fy Y/Z``. Hence
:func:`mujoco_principal_pixel` ``= ((W - 1)/2 - cx, (H - 1)/2 - cy)``. Measured by rendering marker spheres (offset
(7.95, -3.88) px moved the image of the optical axis from (319.5, 239.5) to (311.5, 243.4)) and pinned by the
projection parity test (``tests/test_camera_parity.py``: 200 points, <= 0.5 px, colour and depth). Render at the
stream's resolution (or the same aspect ratio) for the intrinsics to hold.

**Frames.** RealSense optical frame: x right, y down, z forward. MuJoCo camera frame: x right, y UP, looking along
-z. So ``R_mujoco = R_realsense @ diag(1, -1, -1)`` (:func:`realsense_to_mujoco_rotation`; its own inverse), tested
in both directions against MuJoCo's ``cam_xmat``.

**No MSAA.** ``add_d435_cameras`` sets ``offsamples = 0`` (the depth parity needs it; see its docstring).

**Domain randomisation (S3 priority 1).** :func:`perturb_cameras` moves both cameras rigidly together (the D435 is
one rigid body) by a random rotation about the colour centre and a translation, within the calibration uncertainty
(``CameraCalib.dr_ranges``: at least +-5 mm and +-1.5 deg until H8). Intrinsics are not randomised (factory
calibrated).

**Back-lit scene (C.5, pixel world-model track only).** :func:`add_window_backlight` adds a bright area-like light
behind the workspace, as seen through the real window behind the arm.
"""

from __future__ import annotations

import math
from typing import Final

import mujoco
import numpy as np

from so101_rl.perception.camera_calib import CameraCalib, StreamIntrinsics

COLOR_CAM: Final[str] = "d435_color"
DEPTH_CAM: Final[str] = "d435_depth"
CAMERA_NAMES: Final[tuple[str, str]] = (COLOR_CAM, DEPTH_CAM)
_PIXEL_PITCH_M: Final[float] = 3e-6  # arbitrary sensor pixel pitch: only ratios reach the renderer (docstring)
_FLIP: Final[np.ndarray] = np.diag([1.0, -1.0, -1.0])


def realsense_to_mujoco_rotation(r_rs: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation of a RealSense/OpenCV optical frame -> the MuJoCo camera frame (and back)."""
    return np.asarray(r_rs, dtype=np.float64) @ _FLIP


mujoco_to_realsense_rotation = realsense_to_mujoco_rotation  # diag(1, -1, -1) is its own inverse


def mujoco_principal_pixel(s: StreamIntrinsics) -> tuple[float, float]:
    """MuJoCo ``principal_pixel`` for an OpenCV/RealSense principal point (module docstring)."""
    return (s.width - 1) / 2.0 - s.cx, (s.height - 1) / 2.0 - s.cy


def camera_pose_mujoco(T_base_cam: np.ndarray) -> tuple[list[float], list[float]]:
    """(pos, wxyz quat) of a MuJoCo camera at the RealSense-convention pose ``T_base_cam``."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, realsense_to_mujoco_rotation(np.asarray(T_base_cam)[:3, :3]).ravel())
    return [float(x) for x in np.asarray(T_base_cam)[:3, 3]], [float(x) for x in quat]


def _set_intrinsics(cam: mujoco.MjsCamera, s: StreamIntrinsics) -> None:
    cam.resolution = [s.width, s.height]
    cam.sensor_size = [s.width * _PIXEL_PITCH_M, s.height * _PIXEL_PITCH_M]
    cam.focal_pixel = [s.fx, s.fy]
    cam.principal_pixel = list(mujoco_principal_pixel(s))


def add_d435_cameras(spec: mujoco.MjSpec, calib: CameraCalib) -> None:
    """Add ``d435_color`` and ``d435_depth`` to ``spec.worldbody`` (module docstring). Needs a pose.

    Also turns off offscreen multisampling (``visual.quality.offsamples = 0``): with MSAA the depth image is
    resolved from ONE sample, a fraction of a pixel off the pixel centre, which put the rendered depth of grazing
    table pixels up to ~1.6 mm (p95) off the analytic depth; without it the error is < 0.02 mm (p95,
    tests/test_camera_parity.py). Neither D435 sensor antialiases either."""
    spec.visual.quality.offsamples = 0
    for name, T, s in ((COLOR_CAM, calib.T_base_color, calib.color), (DEPTH_CAM, calib.T_base_depth, calib.depth)):
        pos, quat = camera_pose_mujoco(T)
        cam = spec.worldbody.add_camera(name=name, pos=pos, quat=quat)
        _set_intrinsics(cam, s)


def check_compiled_intrinsics(model: mujoco.MjModel, calib: CameraCalib) -> None:
    """Raise if the compiled cameras do not carry ``calib``'s intrinsics (guards the MuJoCo conversion)."""
    for name, s in ((COLOR_CAM, calib.color), (DEPTH_CAM, calib.depth)):
        c = model.camera(name).id
        res = model.cam_resolution[c]
        sx, sy = model.cam_sensorsize[c]
        fx, fy, px, py = model.cam_intrinsic[c]
        got = (fx * res[0] / sx, fy * res[1] / sy, px * res[0] / sx, py * res[1] / sy)
        want = (s.fx, s.fy, *mujoco_principal_pixel(s))
        if tuple(res) != (s.width, s.height) or not np.allclose(got, want, rtol=1e-5, atol=1e-3):
            raise RuntimeError(f"camera {name}: compiled intrinsics {got} @ {tuple(res)} != {want}")


def camera_T_base(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """RealSense-convention ``T_base_cam`` of a camera in the compiled scene (after ``mj_forward``)."""
    c = model.camera(name).id
    T = np.eye(4)
    T[:3, :3] = mujoco_to_realsense_rotation(data.cam_xmat[c].reshape(3, 3))
    T[:3, 3] = data.cam_xpos[c]
    return T


def _axis_angle(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def sample_pose_perturbation(rng: np.random.Generator, trans_m: float, rot_rad: float) -> np.ndarray:
    """A 4x4 rigid perturbation ``dT``: rotation uniform in angle up to ``rot_rad`` about a random axis, translation
    uniform in the ball of radius ``trans_m``. Applied as ``T' = T @ dT`` (about the camera centre)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    t = rng.normal(size=3)
    t *= trans_m * rng.uniform() ** (1 / 3) / np.linalg.norm(t)
    dT = np.eye(4)
    dT[:3, :3] = _axis_angle(axis * rot_rad * rng.uniform())
    dT[:3, 3] = t
    return dT


def perturb_cameras(model: mujoco.MjModel, calib: CameraCalib, dT_color: np.ndarray) -> None:
    """Move ``d435_color`` / ``d435_depth`` in ``model`` (world cameras: ``cam_pos``/``cam_quat`` are world) to
    ``T_base_color @ dT_color`` and the rigidly attached depth pose. ``dT_color = eye(4)`` restores them."""
    Tc = calib.T_base_color @ dT_color
    for name, T in ((COLOR_CAM, Tc), (DEPTH_CAM, Tc @ calib.T_color_depth)):
        c = model.camera(name).id
        pos, quat = camera_pose_mujoco(T)
        model.cam_pos[c] = pos
        model.cam_quat[c] = quat


def add_window_backlight(spec: mujoco.MjSpec, calib: CameraCalib, *, distance_m: float = 1.2) -> None:
    """A bright light behind the workspace as seen from the camera (the real window behind the arm, C.5), shining
    back towards the camera. For the pixel world-model renders only; physics is unaffected."""
    T = calib.T_base_color
    fwd = T[:3, 2].copy()
    fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    pos = T[:3, 3] + distance_m * fwd
    pos[2] = T[2, 3] + 0.3
    d = T[:3, 3] - pos
    spec.worldbody.add_light(
        name="window_backlight",
        pos=[float(x) for x in pos],
        dir=[float(x) for x in d / np.linalg.norm(d)],
        diffuse=[1.6, 1.6, 1.5],
        specular=[0.6, 0.6, 0.6],
        ambient=[0.25, 0.25, 0.25],
        castshadow=False,
    )

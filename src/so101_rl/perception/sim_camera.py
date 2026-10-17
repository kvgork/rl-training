"""A fake D435 that serves MuJoCo renders of the sim scene (``perception-process --backend sim`` and the tests).

CLAUDE.md: hardware code needs a fake backend. :class:`SimRenderCamera` implements the
:class:`~so101_rl.perception.camera.Camera` protocol on the RL scene compiled with the cameras of
``calib/camera.yaml`` (``sim/d435.py`` ``d435_color``): colour + depth rendered from the colour camera, i.e. depth
aligned to colour exactly as the real pipeline delivers it (the same rendering as ``tests/test_die_convention.py``).
It reports the calibration's serial and colour intrinsics unless told otherwise (``serial=`` / ``dT_color=`` let the
tests fake a wrong camera or a camera that moved after the calibration).

The scene is the ``home`` keyframe with the die placed by :meth:`set_die` (``None`` = parked far outside the view) and
the arm joints optionally set by :meth:`set_arm` (the arm-mask tests).

H8 fakes (``calibrate_hand_eye --execute --camera-backend sim``, ``tests/test_hand_eye_execute.py``):
``tip_marker_radius_m`` adds a red sphere (the hand-eye tip sticker) to the gripper body at the FK tip point
(``gripperframe``) plus ``tip_marker_offset_m`` (gripper body frame), and ``follow=(joint_source, calibration)`` poses
the arm from the servo's present ticks before every render (``arm_mask.ServoStateReader``: read-only), so the camera
sees the arm that ``servo --backend mujoco`` is driving.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from so101_rl.perception.camera import Frame, Intrinsics
from so101_rl.perception.camera_calib import CameraCalib, load_camera_calib

_PARKED_XYZ = (2.0, 2.0, 0.5)  # outside the camera's view and clear of everything


class SimRenderCamera:
    """Rendered colour + aligned depth of the sim scene (module docstring). ``realtime`` paces :meth:`read` to the
    calibration's fps on an absolute schedule (the CLI); the tests read as fast as it renders."""

    def __init__(
        self,
        calib: CameraCalib | None = None,
        *,
        die_xyz_yaw: tuple[float, float, float, float] | None = None,
        serial: str | None = None,
        dT_color: np.ndarray | None = None,
        realtime: bool = False,
        tip_marker_radius_m: float | None = None,
        tip_marker_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        follow: tuple[Any, Any] | None = None,
    ) -> None:
        self.calib = calib if calib is not None else load_camera_calib()
        self.serial = self.calib.serial if serial is None else str(serial)
        self.device_serial: str | None = None
        self.realtime = bool(realtime)
        self._dT = None if dT_color is None else np.asarray(dT_color, dtype=np.float64)
        self._die = die_xyz_yaw
        self._arm: np.ndarray | None = None
        self._model = self._data = self._renderer = None
        self._next_ns = 0
        self.frames = 0
        self.tip_marker_radius_m = tip_marker_radius_m
        self.tip_marker_offset_m = tuple(float(v) for v in tip_marker_offset_m)
        self.follow = follow

    # ------------------------------------------------------------------ scene
    def set_die(self, die_xyz_yaw: tuple[float, float, float, float] | None) -> None:
        """Die centre (x, y, z) in the base frame and yaw (rad); ``None`` parks it outside the view."""
        self._die = die_xyz_yaw
        if self._model is not None:
            self._pose_scene()

    def set_arm(self, q: np.ndarray | None) -> None:
        """The six joint angles (rad, ``config.JOINT_NAMES`` order); ``None`` = the ``home`` keyframe."""
        self._arm = None if q is None else np.asarray(q, dtype=np.float64).reshape(6).copy()
        if self._model is not None:
            self._pose_scene()

    def die_true_xyz(self) -> np.ndarray:
        """The die body's position in the current scene (the ground truth the tests compare against)."""
        if self._model is None:
            raise RuntimeError("camera not started")
        return self._data.xpos[self._model.body("die").id].copy()

    def _pose_scene(self) -> None:
        import mujoco

        m, d = self._model, self._data
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        if self._arm is not None:
            from so101_rl.config import JOINT_NAMES

            for j, n in enumerate(JOINT_NAMES):
                d.qpos[m.jnt_qposadr[m.joint(n).id]] = self._arm[j]
        dq = m.jnt_qposadr[m.joint("die_free").id]
        if self._die is None:
            x, y, z, yaw = (*_PARKED_XYZ, 0.0)
        else:
            x, y, z, yaw = self._die
        d.qpos[dq : dq + 7] = (x, y, z, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------ Camera protocol
    def start(self) -> Intrinsics:
        if self._model is not None:
            raise RuntimeError("camera already started")
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        from so101_rl.sim.d435 import perturb_cameras
        from so101_rl.sim.model import build_spec

        c = self.calib.color
        spec = build_spec(camera=self.calib)
        spec.visual.global_.offwidth = c.width
        spec.visual.global_.offheight = c.height
        if self.tip_marker_radius_m is not None:
            frame = spec.site("gripperframe")
            pos = np.asarray(frame.pos, dtype=np.float64) + np.asarray(self.tip_marker_offset_m)
            # emissive: evenly red to its silhouette (a shaded sphere loses its dark side to the HSV threshold and
            # biases the blob centroid towards the light)
            spec.add_material(name="tip_marker_mat", rgba=[0.8, 0.04, 0.04, 1.0], emission=1.0, specular=0.0)
            spec.body("gripper").add_geom(
                name="tip_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[float(self.tip_marker_radius_m), 0, 0],
                pos=pos.tolist(), material="tip_marker_mat", contype=0, conaffinity=0, mass=1e-6,
            )  # fmt: skip
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        if self._dT is not None:
            perturb_cameras(self._model, self.calib, self._dT)
        self._renderer = mujoco.Renderer(self._model, c.height, c.width)
        self._pose_scene()
        self.device_serial = self.serial
        self._next_ns = time.monotonic_ns()
        return self.calib.color_intrinsics()

    def read(self, rgb_out: np.ndarray, depth_m_out: np.ndarray) -> Frame:
        if self._renderer is None:
            raise RuntimeError("camera not started")
        from so101_rl.sim.d435 import COLOR_CAM

        if self.realtime:
            self._next_ns += round(1e9 / self.calib.color.fps)
            delay = self._next_ns - time.monotonic_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
            else:
                self._next_ns = time.monotonic_ns()
        if self.follow is not None:
            from so101_rl.perception.arm_mask import joint_angles

            joints, calibration = self.follow
            ticks = joints.read()
            if ticks is not None:
                self.set_arm(joint_angles(ticks, calibration))
        r = self._renderer
        r.update_scene(self._data, camera=COLOR_CAM)
        rgb_out[:] = r.render()
        r.enable_depth_rendering()
        try:
            r.update_scene(self._data, camera=COLOR_CAM)
            depth_m_out[:] = r.render()
        finally:
            r.disable_depth_rendering()
        self.frames += 1
        return Frame(time.monotonic_ns(), True)

    def stop(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        self._model = self._data = None

"""Synthetic D435 frames rendered by MuJoCo from a known camera (tests of the camera parity unit, SUC)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from so101_rl.perception.camera_calib import CameraCalib, CameraPose
from so101_rl.sim.d435 import COLOR_CAM
from so101_rl.sim.model import TABLE_Z_M, build_spec


def with_pose(calib: CameraCalib, T: np.ndarray, method: str = "provisional") -> CameraCalib:
    return calib.with_pose(CameraPose(T, method, "2026-10-07", None if method == "provisional" else 1.0, 5.0, 1.5))


def perturbed(T: np.ndarray, dx=0.02, dy=-0.015, dz=0.0, dyaw_deg=6.0, dtilt_deg=0.0) -> np.ndarray:
    c, s = np.cos(np.radians(dyaw_deg)), np.sin(np.radians(dyaw_deg))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    a = np.radians(dtilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    out = T.copy()
    out[:3, :3] = Rz @ T[:3, :3] @ Rx
    out[:3, 3] = Rz @ T[:3, 3] + (dx, dy, dz)
    return out


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1.0, 0], [-np.sin(a), 0, np.cos(a)]])


def render_rig(calib: CameraCalib, qpos: np.ndarray, *, R_world: np.ndarray | None = None, table_dz_m: float = 0.0,
               n_frames: int = 5, noise_m: float = 0.0015, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(rgb, depth frames [n, H, W]) of the bare rig (no die, tray, sites or clutter) seen from ``calib``'s colour
    camera, with the table rotated by ``R_world`` about the base origin (a tilted base: table AND camera move together,
    so pass a camera pose rotated the same way) and raised by ``table_dz_m`` (a base standing higher than assumed)."""
    R = np.eye(3) if R_world is None else np.asarray(R_world, np.float64)
    spec = build_spec(camera=calib)
    tbl = spec.geom("table")
    tbl.pos = list(R @ np.array([0.0, 0.0, TABLE_Z_M + table_dz_m]))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    tbl.quat = list(quat)
    spec.visual.global_.offwidth = calib.color.width
    spec.visual.global_.offheight = calib.color.height
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qpos[model.jnt_qposadr[model.joint("die_free").id] + 2] = -1.0  # the die out of sight, under the table
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    opt.geomgroup[1] = 0  # the start-well tray
    r = mujoco.Renderer(model, calib.color.height, calib.color.width)
    try:
        r.update_scene(data, camera=COLOR_CAM, scene_option=opt)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=COLOR_CAM, scene_option=opt)
        depth = r.render().copy().astype(np.float32)
    finally:
        r.close()
    depth[depth > 3.0] = 0.0
    rng = np.random.default_rng(seed)
    frames = np.stack([depth + rng.normal(0.0, noise_m, depth.shape).astype(np.float32) for _ in range(n_frames)])
    frames[frames < 0.05] = 0.0
    return rgb, frames


def render_rgbd(calib: CameraCalib, qpos: np.ndarray, *, clutter: bool = True, noise_m: float = 0.0, seed: int = 0, camera: str = COLOR_CAM) -> tuple[np.ndarray, np.ndarray]:
    """(rgb, depth aligned to colour) of the scene from ``calib``'s colour camera. Clutter: a cup-sized cylinder and a
    box on the table near the arm, a wall-like box far away. Depth noise: Gaussian, metres."""
    spec = build_spec(camera=calib)
    if clutter:
        wb = spec.worldbody
        wb.add_geom(name="cup", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.04, 0.05, 0], pos=[0.17, -0.16, TABLE_Z_M + 0.05], rgba=[0.9, 0.9, 0.85, 1], contype=0, conaffinity=0)
        wb.add_geom(name="phone", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.08, 0.005], pos=[0.30, 0.16, TABLE_Z_M + 0.005], rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0)
        wb.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.01, 0.4], pos=[0.1, -0.6, TABLE_Z_M + 0.4], rgba=[0.6, 0.6, 0.7, 1], contype=0, conaffinity=0)
    spec.visual.global_.offwidth = calib.color.width
    spec.visual.global_.offheight = calib.color.height
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    w, h = (calib.color.width, calib.color.height) if camera == COLOR_CAM else (calib.depth.width, calib.depth.height)
    r = mujoco.Renderer(model, h, w)
    try:
        r.update_scene(data, camera=camera)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=camera)
        depth = r.render().copy().astype(np.float32)
        r.disable_depth_rendering()
    finally:
        r.close()
    if noise_m:
        depth = depth + np.random.default_rng(seed).normal(0.0, noise_m, depth.shape).astype(np.float32)
    depth[depth > 3.0] = 0.0
    return rgb, depth

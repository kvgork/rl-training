"""Camera parity (C.3, C.4): the sim D435 is the real one. Frame conversion, compiled intrinsics, projection
parity (<= 0.5 px, colour and depth), depth parity (<= 1 mm), tracker parity on the calibrated pose, camera DR."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.perception.die_tracker import HsvDepthDieTracker
from so101_rl.sim.d435 import (
    COLOR_CAM,
    DEPTH_CAM,
    add_d435_cameras,
    add_window_backlight,
    camera_T_base,
    check_compiled_intrinsics,
    mujoco_principal_pixel,
    mujoco_to_realsense_rotation,
    perturb_cameras,
    realsense_to_mujoco_rotation,
    sample_pose_perturbation,
)
from so101_rl.sim.model import DIE_EDGE_M, TABLE_Z_M, build_spec, load_model
from so101_rl.sim.sim_tracker import NO_NOISE, CameraModel, Occluders, SimDieTracker

# the synthetic scenes were laid out for the provisional camera view (see conftest.provisional_camera)
pytestmark = pytest.mark.usefixtures("provisional_camera")

CPU = torch.device("cpu")
CALIB = load_camera_calib(Path(__file__).parent / "data" / "camera_provisional_2026-10-07.yaml")  # = provisional_camera


def _renderer(model, w, h):
    try:
        return mujoco.Renderer(model, h, w)
    except Exception as exc:  # noqa: BLE001 - no OpenGL context on this host
        pytest.skip(f"no MuJoCo offscreen rendering: {exc}")


# ---- conversion ------------------------------------------------------------------------------------------------
def test_realsense_mujoco_rotation_conversion_both_ways():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        M = realsense_to_mujoco_rotation(R)
        np.testing.assert_allclose(M[:, 0], R[:, 0])  # x right in both
        np.testing.assert_allclose(M[:, 1], -R[:, 1])  # y: up (MuJoCo) vs down (RealSense)
        np.testing.assert_allclose(M[:, 2], -R[:, 2])  # MuJoCo looks along -z, RealSense along +z
        np.testing.assert_allclose(mujoco_to_realsense_rotation(M), R)


def test_scene_cameras_sit_at_the_calibrated_poses():
    model, data = load_model()
    check_compiled_intrinsics(model, CALIB)
    np.testing.assert_allclose(camera_T_base(model, data, COLOR_CAM), CALIB.T_base_color, atol=1e-9)
    np.testing.assert_allclose(camera_T_base(model, data, DEPTH_CAM), CALIB.T_base_color @ CALIB.T_color_depth, atol=1e-9)
    # optical axis of the MuJoCo camera (-z) = RealSense +z
    c = model.camera(COLOR_CAM).id
    np.testing.assert_allclose(-data.cam_xmat[c].reshape(3, 3)[:, 2], CALIB.T_base_color[:3, 2], atol=1e-9)
    assert tuple(model.cam_resolution[c]) == (640, 480)


def test_principal_pixel_convention():
    s = CALIB.color
    px, py = mujoco_principal_pixel(s)
    assert px == pytest.approx(319.5 - s.cx) and py == pytest.approx(239.5 - s.cy)


# ---- projection parity -----------------------------------------------------------------------------------------
def _marker_model(r=0.004):
    spec = mujoco.MjSpec()
    spec.stat.extent = 1.0
    spec.stat.center = [0.2, 0.0, 0.0]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
    add_d435_cameras(spec, CALIB)
    b = spec.worldbody.add_body(name="marker", mocap=True)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0])
    return spec.compile()


@pytest.mark.parametrize("cam_name,stream", [(COLOR_CAM, "color"), (DEPTH_CAM, "depth")])
def test_projection_parity_200_points(cam_name, stream):
    """Pinhole projection with the camera.yaml intrinsics vs the rendered marker sphere: <= 0.5 px."""
    model = _marker_model()
    data = mujoco.MjData(model)
    cm = CameraModel.from_calib(CALIB, stream)
    r = _renderer(model, cm.width, cm.height)
    r.enable_segmentation_rendering()
    rng = np.random.default_rng(1)
    errs = []
    try:
        while len(errs) < 200:
            p = rng.uniform([0.0, -0.30, TABLE_Z_M], [0.40, 0.30, 0.30])
            uv, z = cm.project(p)
            if not (z > 0.15 and 12 <= uv[0] <= cm.width - 13 and 12 <= uv[1] <= cm.height - 13):
                continue
            data.mocap_pos[0] = p
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=cam_name)
            seg = r.render()
            v, u = np.nonzero(seg[..., 0] == 0)
            assert len(u) >= 4, (p, uv)
            errs.append(math.hypot(u.mean() - uv[0], v.mean() - uv[1]))
    finally:
        r.close()
    errs = np.array(errs)
    print(f"{stream}: projection error mean {errs.mean():.3f} px, max {errs.max():.3f} px over {len(errs)} points")
    assert errs.max() <= 0.5


def test_projection_parity_shown_to_fail_with_the_principal_point_unconverted():
    """Passing the RealSense principal point straight to MuJoCo moves the image by ~(16.9, 7.2) px."""
    s = CALIB.color
    wrong = np.hypot(2 * (319.5 - s.cx), 2 * (239.5 - s.cy))
    assert wrong > 5.0  # the converted and unconverted conventions differ by far more than the 0.5 px bound


# ---- depth parity ----------------------------------------------------------------------------------------------
def test_depth_parity_table_and_die():
    """Rendered depth at table and die pixels = the analytic depth (mj_ray along the pixel's ray), <= 1 mm."""
    spec = build_spec(camera=CALIB)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    dq = model.jnt_qposadr[model.joint("die_free").id]
    data.qpos[dq : dq + 7] = (0.22, 0.10, TABLE_Z_M + DIE_EDGE_M / 2, 1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    table, die = model.geom("table").id, model.body("die").id
    checked = {"table": 0, "die": 0}
    for cam_name, stream in ((COLOR_CAM, "color"), (DEPTH_CAM, "depth")):
        cm = CameraModel.from_calib(CALIB, stream)
        r = _renderer(model, cm.width, cm.height)
        try:
            r.enable_depth_rendering()
            r.update_scene(data, camera=cam_name)
            depth = r.render().copy()
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
            r.update_scene(data, camera=cam_name)
            seg = r.render().copy()
        finally:
            r.close()
        R, t = cm.T_base_cam[:3, :3], cm.T_base_cam[:3, 3]
        rng = np.random.default_rng(2)
        targets = []
        vv, uu = np.nonzero(seg[..., 0] == table)
        targets += [("table", vv[i], uu[i]) for i in rng.choice(len(uu), 300, replace=False)]
        die_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == die]
        vv, uu = np.nonzero(np.isin(seg[..., 0], die_geoms))
        assert len(uu) > 50, stream
        targets += [("die", vv[i], uu[i]) for i in rng.choice(len(uu), min(60, len(uu)), replace=False)]
        for kind, v, u in targets:
            ray_c = np.array([(u - cm.cx) / cm.fx, (v - cm.cy) / cm.fy, 1.0])
            ray_b = R @ ray_c
            geomid = np.zeros(1, np.int32)
            dist = mujoco.mj_ray(model, data, t, ray_b / np.linalg.norm(ray_b), None, 1, -1, geomid)
            if dist < 0:
                continue
            z = dist / np.linalg.norm(ray_c)  # distance along the ray -> depth along the optical axis
            if abs(depth[v, u] - z) > 0.001:
                # a pixel on a silhouette edge can see the other surface; only same-geom hits count
                hit_body = model.geom_bodyid[geomid[0]]
                same = (kind == "table" and geomid[0] == table) or (kind == "die" and hit_body == die)
                assert not same, (stream, kind, u, v, depth[v, u], z)
                continue
            checked[kind] += 1
    assert checked["table"] > 500 and checked["die"] > 60, checked


# ---- tracker parity on the calibrated pose ---------------------------------------------------------------------
DIE_Z = TABLE_Z_M + DIE_EDGE_M / 2


def _configs(model, n=20):
    """``n`` (arm qpos[6], die xy): gripper-low arm poses x {die under the tip, 6-8 cm aside, 3-5 cm behind the tip
    as seen from the camera (hidden by the gripper)}."""
    rng = np.random.default_rng(5)
    away = CALIB.T_base_color[:2, 2] / np.linalg.norm(CALIB.T_base_color[:2, 2])  # horizontal viewing direction
    data = mujoco.MjData(model)
    tip_id = model.site("gripper_tip").id
    out = []
    while len(out) < n:
        ticks = np.array([rng.integers(1750, 2350), rng.integers(1300, 2200), rng.integers(2000, 3000),
                          rng.integers(1700, 2600), rng.integers(1500, 2600)])  # fmt: skip
        q = np.zeros(model.nq)
        q[:5] = (ticks - 2047) * 2 * np.pi / 4096
        q[5] = 0.3
        q[6:9] = (0.2, -0.05, DIE_Z)
        q[9] = 1.0
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        tip = data.site_xpos[tip_id].copy()
        if not (0.13 < tip[0] < 0.30 and abs(tip[1]) < 0.12 and 0.03 < tip[2] < 0.10):
            continue
        out.append((q[:6].copy(), tip[:2] + rng.uniform(-0.003, 0.003, 2)))  # under the tip
        ang = rng.uniform(0, 2 * np.pi)
        out.append((q[:6].copy(), tip[:2] + rng.uniform(0.06, 0.08) * np.array([np.cos(ang), np.sin(ang)])))
        out.append((q[:6].copy(), tip[:2] + rng.uniform(0.03, 0.05) * away))
    return out[:n]


@pytest.mark.parametrize("yaw_deg", [0.0, 45.0], ids=["yaw0", "yaw45"])
def test_tracker_parity_on_the_calibrated_pose(yaw_deg):
    """The real HsvDepthDieTracker on the sim RGB-D render (colour + depth aligned to colour, from d435_color) and
    SimDieTracker (colour + depth sensor, no noise): visibility agrees in >= 18 of 20 configurations, the centroid
    within 3 mm wherever both see the die. The die is yawed 0 and 45 deg about the table normal: the blob (hence the
    border cut) is widest at 45 deg, and the sim visibility must not depend on the yaw."""
    half = math.radians(yaw_deg) / 2
    die_quat = (math.cos(half), 0, 0, math.sin(half))
    model, data = load_model()
    r = _renderer(model, 640, 480)
    real = HsvDepthDieTracker.from_camera_calib(CALIB)
    tr = SimDieTracker(CameraModel.from_calib(CALIB), NO_NOISE, 1, device=CPU, seed=0, depth_cam=CameraModel.from_calib(CALIB, "depth"))
    occl = Occluders(model, CPU)
    body_ids = [model.body(b).id for b in occl.bodies]
    die_q = model.jnt_qposadr[model.joint("die_free").id]
    agree, log, errs = 0, [], []
    try:
        for q, xy in _configs(model):
            data.qpos[:6] = q
            data.qpos[die_q : die_q + 7] = (xy[0], xy[1], DIE_Z, *die_quat)
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=COLOR_CAM)
            rgb = r.render().copy()
            r.enable_depth_rendering()
            r.update_scene(data, camera=COLOR_CAM)
            depth = r.render().copy()
            r.disable_depth_rendering()
            det = real.detect(rgb, depth)
            die = torch.tensor(data.qpos[die_q : die_q + 3], dtype=torch.float32)[None]
            tip = torch.tensor(data.site_xpos[model.site("gripper_tip").id], dtype=torch.float32)[None]
            c, h, rot = occl.compute(
                torch.tensor(data.xpos[body_ids], dtype=torch.float32)[None],
                torch.tensor(data.xquat[body_ids], dtype=torch.float32)[None],
            )
            sim_visible = bool(tr.in_view(die)[0]) and not bool(tr.occluded(die, tip, c, h, rot)[0])
            agree += (det is not None) == sim_visible
            if det is not None and sim_visible:
                errs.append(float(np.linalg.norm(real.centroid_base(det, CALIB.T_base_color) - die[0].double().numpy())))
            cov = float(tr.covered_fraction(die, c, h, rot)[0])
            log.append((np.round(xy, 3).tolist(), det is not None, sim_visible, round(cov, 2), None if det is None else det.area_px))
    finally:
        r.close()
    print("\n".join(str(x) for x in log))
    print(f"visibility agreement {agree}/20; centroid errors (mm) {np.round(np.array(errs) * 1000, 2).tolist()}")
    assert sum(x[1] for x in log) >= 5 and sum(not x[1] for x in log) >= 3  # both verdicts exercised
    assert agree >= 18
    assert errs and max(errs) < 0.003


# ---- camera DR and the back light ------------------------------------------------------------------------------
def test_perturb_cameras_moves_both_rigidly_within_range():
    model, data = load_model()
    rng = np.random.default_rng(3)
    t_max, r_max = CALIB.dr_ranges()
    for _ in range(20):
        dT = sample_pose_perturbation(rng, t_max, r_max)
        assert np.linalg.norm(dT[:3, 3]) <= t_max + 1e-12
        assert math.degrees(math.acos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1))) <= math.degrees(r_max) + 1e-9
        perturb_cameras(model, CALIB, dT)
        mujoco.mj_forward(model, data)
        Tc = camera_T_base(model, data, COLOR_CAM)
        Td = camera_T_base(model, data, DEPTH_CAM)
        np.testing.assert_allclose(Tc, CALIB.T_base_color @ dT, atol=1e-9)
        np.testing.assert_allclose(np.linalg.inv(Tc) @ Td, CALIB.T_color_depth, atol=1e-9)  # rigid D435
    perturb_cameras(model, CALIB, np.eye(4))
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(camera_T_base(model, data, COLOR_CAM), CALIB.T_base_color, atol=1e-9)
    assert t_max >= 0.005 and r_max >= math.radians(1.5)


def test_window_backlight_brightens_the_render():
    imgs = []
    for backlit in (False, True):
        spec = build_spec(camera=CALIB)
        if backlit:
            add_window_backlight(spec, CALIB)
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        r = _renderer(model, 320, 240)
        r.update_scene(data, camera=COLOR_CAM)
        imgs.append(r.render().astype(float))
        r.close()
    assert imgs[1].mean() > imgs[0].mean() + 2.0

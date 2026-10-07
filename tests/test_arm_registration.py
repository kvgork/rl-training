"""Camera-only extrinsics (C.2): table plane + arm-shape registration on synthetic D435 frames rendered by MuJoCo
from a known camera; ``camera-calibrate`` writes calib/camera.yaml only when the gate passes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from camera_synth import perturbed, render_rgbd, with_pose

from so101_rl.config import load_calibration
from so101_rl.perception import camera_calibrate
from so101_rl.perception.arm_registration import rotation_angle_deg
from so101_rl.perception.camera_calib import load_camera_calib, save_camera_calib
from so101_rl.perception.camera_calibrate import Capture, calibrate
from so101_rl.perception.table_plane import depth_to_points, fit_table_plane
from so101_rl.servo.probe import ProbeRefused
from so101_rl.sim.arm_cloud import ArmSurface, qpos_from_ticks
from so101_rl.sim.model import TABLE_Z_M, load_model

CAL = load_calibration(Path(__file__).parent / "data" / "so101_follower.calib.json")
TICKS = np.array([2050, 1500, 2500, 2700, 2040, 2300], dtype=np.int32)


@pytest.fixture(scope="module")
def scene():
    c0 = load_camera_calib()
    T_true = perturbed(c0.T_base_color)  # 2 cm / 1.5 cm / 6 deg away from the committed pose (the seed must not matter)
    model = load_model(camera=False)[0]
    q = qpos_from_ticks(model, TICKS, CAL)
    rgb, depth = render_rgbd(with_pose(c0, T_true), q)
    frames = np.stack([depth + np.random.default_rng(k).normal(0, 0.0015, depth.shape).astype(np.float32) for k in range(5)])
    frames[frames < 0.05] = 0.0
    return c0, T_true, model, rgb, frames


def test_arm_surface_follows_the_joints():
    model = load_model(camera=False)[0]
    s = ArmSurface(model)
    q0 = qpos_from_ticks(model, TICKS, CAL)
    p0, n0 = s.points(q0)
    assert len(p0) > 3000 and np.allclose(np.linalg.norm(n0, axis=1), 1.0)
    q1 = q0.copy()
    q1[1] += 0.3
    p1, _ = s.points(q1)
    moved = np.linalg.norm(p1 - p0, axis=1) > 1e-6
    base = s.geom == s.geom[0]  # the first geom belongs to the base: it does not move
    assert moved.any() and not moved[base].any()


def test_table_plane_fit(scene):
    c0, T_true, _, _, frames = scene
    pts, _ = depth_to_points(frames[0], c0.color)
    fit = fit_table_plane(pts)
    up_true = T_true[:3, :3].T @ np.array([0, 0, 1.0])
    assert math.degrees(math.acos(min(1.0, float(fit.normal_up @ up_true)))) < 0.3
    assert fit.height_m == pytest.approx(T_true[2, 3] - TABLE_Z_M, abs=0.002)


def test_arm_shape_recovers_the_true_camera_and_passes_the_gate(scene):
    c0, T_true, model, rgb, frames = scene
    res, _plane, _obs = calibrate(model, CAL, c0, [Capture(rgb, frames, TICKS)], TABLE_Z_M, bootstrap=8)
    err_t = np.linalg.norm(res.T_base_color[:3, 3] - T_true[:3, 3]) * 1000
    err_r = rotation_angle_deg(res.T_base_color[:3, :3] @ T_true[:3, :3].T)
    print(json.dumps({k: v for k, v in res.to_json().items() if k not in ("seeds", "T_base_color")}, default=float))
    print(f"pose error {err_t:.2f} mm, {err_r:.3f} deg")
    assert res.passed, res.failures
    assert err_t < 3.0 and err_r < 0.5
    assert res.rms_mm <= 5.0 and res.tip_px_err <= 8.0 and res.coverage >= 0.5
    assert len(res.seeds) >= 3  # several seeds were refined


def test_wrong_joint_angles_fail_the_gate(scene):
    """Shown to fail: the same frames with the shoulder and elbow 0.25 rad off cannot pass."""
    c0, _, model, rgb, frames = scene
    bad = TICKS.copy()
    bad[1] += 160
    bad[2] -= 160
    res, _, _ = calibrate(model, CAL, c0, [Capture(rgb, frames, bad)], TABLE_Z_M, bootstrap=0)
    print(json.dumps({k: v for k, v in res.to_json().items() if k not in ("seeds", "T_base_color")}, default=float))
    assert not res.passed


def _fake_main(tmp_path, scene, ticks, probe_exc=None):
    c0, _, _, rgb, frames = scene
    cam_yaml = tmp_path / "camera.yaml"
    save_camera_calib(c0, cam_yaml)
    before = cam_yaml.read_text()

    def probe():
        if probe_exc:
            raise probe_exc
        return ticks

    rc = camera_calibrate.main(
        ["--method", "arm-shape", "--camera-yaml", str(cam_yaml), "--out-dir", str(tmp_path / "runs"),
         "--calibration", str(Path(__file__).parent / "data" / "so101_follower.calib.json")],
        probe=probe, camera=lambda calib, n: (rgb, frames),
    )  # fmt: skip
    return rc, cam_yaml, before


def test_camera_calibrate_writes_the_file_only_on_pass(tmp_path, scene):
    rc, cam_yaml, _before = _fake_main(tmp_path, scene, TICKS)
    assert rc == 0
    new = load_camera_calib(cam_yaml)
    assert new.pose.method == "arm-shape" and new.pose.residual_mm <= 5.0
    assert new.color == scene[0].color  # intrinsics untouched
    np.testing.assert_allclose(new.T_base_color[:3, 3], scene[1][:3, 3], atol=0.003)
    run = next((tmp_path / "runs").iterdir())
    assert {"report.json", "overlay_0.png", "scene_0.ply", "model_0.ply", "capture_0.npz"} <= {p.name for p in run.iterdir()}


def test_camera_calibrate_keeps_the_file_on_fail_and_on_refusal(tmp_path, scene):
    bad = TICKS.copy()
    bad[1] += 160
    bad[2] -= 160
    rc, cam_yaml, before = _fake_main(tmp_path, scene, bad)
    assert rc == 1 and cam_yaml.read_text() == before
    rc, cam_yaml, before = _fake_main(tmp_path / "b", scene, TICKS, probe_exc=ProbeRefused("motors [1] do not answer"))
    assert rc == 3 and cam_yaml.read_text() == before

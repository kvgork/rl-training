"""Camera-only extrinsics (C.2): table plane + arm-shape registration on synthetic D435 frames rendered by MuJoCo
from a known camera; ``camera-calibrate`` writes calib/camera.yaml only when the gate passes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from camera_synth import perturbed, render_rgbd, with_pose

from so101_rl.config import load_calibration, ticks_to_rad
from so101_rl.perception import camera_calibrate
from so101_rl.perception.arm_registration import rotation_angle_deg
from so101_rl.perception.camera_calib import load_camera_calib, save_camera_calib
from so101_rl.perception.camera_calibrate import Capture, calibrate
from so101_rl.perception.table_plane import PlaneFit, depth_to_points, fit_table_plane, refit_in_workspace
from so101_rl.servo.probe import ProbeRefused
from so101_rl.sim.arm_cloud import ArmSurface, qpos_from_ticks
from so101_rl.sim.model import TABLE_Z_M, load_model

# the synthetic scenes were laid out for the provisional camera view (see conftest.provisional_camera)
pytestmark = pytest.mark.usefixtures("provisional_camera")

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
    assert res.icp6_converged  # fixed subsample + Tukey-weighted trim: the 6-DOF ICP settles (2026-10-09)
    d = res.diagnostics
    print(json.dumps({k: v for k, v in d.items() if k != "icp_information"}, default=float))
    # the synthetic table is level in the base frame: no tilt, the table where it is assumed, no joint offsets
    assert d["plane_in_base"]["tilt_deg"] < 0.5
    assert abs(d["plane_in_base"]["table_z_at_base_origin_mm"] - TABLE_Z_M * 1000) < 3.0
    assert d["table_plane_used"] == "workspace" and d["workspace_vs_global_plane"]["angle_deg"] < 0.3
    alt = d["level_base_alternative"]
    assert abs(alt["table_z_minus_assumed_mm"]) < 3.0
    assert all(abs(v) < 1.0 for k, v in alt["offsets_deg"].items() if k != "wrist_roll"), alt["offsets_deg"]
    axes = d["icp_information"]["rotation_axes_base"]
    assert len(axes) == 3 and all(a["sigma_deg"] > 0 for a in axes)


def test_wrong_joint_angles_fail_the_gate(scene):
    """Shown to fail: the same frames with the shoulder and elbow 0.25 rad off cannot pass."""
    c0, _, model, rgb, frames = scene
    bad = TICKS.copy()
    bad[1] += 160
    bad[2] -= 160
    res, _, _ = calibrate(model, CAL, c0, [Capture(rgb, frames, bad)], TABLE_Z_M, bootstrap=0)
    print(json.dumps({k: v for k, v in res.to_json().items() if k not in ("seeds", "T_base_color", "diagnostics")}, default=float))
    assert not res.passed
    # the level-base alternative (diagnostics) finds both injected joint errors and the right table height
    err = np.degrees(ticks_to_rad(bad, CAL) - ticks_to_rad(TICKS, CAL))
    alt = res.diagnostics["level_base_alternative"]
    assert alt["offsets_deg"]["shoulder_lift"] == pytest.approx(-err[1], abs=1.0)
    assert alt["offsets_deg"]["elbow_flex"] == pytest.approx(-err[2], abs=1.0)
    assert abs(alt["table_z_minus_assumed_mm"]) < 3.0 and alt["rms_mm"] < 3.0


@pytest.mark.parametrize("joint,delta", [(1, 70), (2, -70)], ids=["shoulder+70", "elbow-70"])
def test_small_joint_error_fails_the_gate(scene, joint, delta):
    """Shown to fail: a 70-tick (~6 deg) error on one big joint, everything else perfect, must not pass. (A wrist-flex
    error of the same size is NOT reliably caught by this gate; that is a known blind spot, see ARCHITECTURE.)"""
    c0, _, model, rgb, frames = scene
    bad = TICKS.copy()
    bad[joint] += delta
    res, _, _ = calibrate(model, CAL, c0, [Capture(rgb, frames, bad)], TABLE_Z_M, bootstrap=0)
    assert not res.passed, res.to_json()["gate"]
    assert res.n_inliers >= 1000 and min(res.visible_samples) >= 500  # it failed on the fit, not on a vacuous input
    if joint == 1:
        # the level-base alternative names the culprit: shoulder_lift back by the injected error
        err_deg = math.degrees(float(ticks_to_rad(bad, CAL)[1] - ticks_to_rad(TICKS, CAL)[1]))
        got = res.diagnostics["level_base_alternative"]["offsets_deg"]["shoulder_lift"]
        assert got == pytest.approx(-err_deg, abs=1.5), (got, err_deg)


def test_report_carries_the_new_gates_on_a_passing_run(scene):
    c0, _, model, rgb, frames = scene
    res, _, _ = calibrate(model, CAL, c0, [Capture(rgb, frames, TICKS)], TABLE_Z_M, bootstrap=0)
    rep = res.to_json()
    assert res.passed, res.failures
    assert rep["gate"]["n_inliers>="] == 1000 and rep["gate"]["visible_samples_per_obs>="] == 500
    assert rep["gate"]["icp6_iterations>="] == 3
    assert rep["n_inliers"] >= 1000 and all(c >= 500 for c in rep["visible_samples"]) and rep["icp6_iterations"] >= 3


def test_tiny_overlap_fails_the_gate(scene):
    """Shown to fail: only a 12x12 px patch of the arm is left in the depth (all other arm pixels are holes): a handful
    of points cannot certify a calibration, whatever the RMS over them says."""
    c0, T_true, model, rgb, frames = scene
    pts, pix = depth_to_points(np.median(frames, axis=0), c0.color)
    p_base = pts @ T_true[:3, :3].T + T_true[:3, 3]
    arm = p_base[:, 2] > TABLE_Z_M + 0.003  # every arm point, incl. the base foot that stands on the table
    v, u = pix[arm].T
    cu, cv = int(np.median(u)), int(np.median(v))
    mask = np.zeros(frames.shape[1:], bool)
    mask[v, u] = True
    keep = np.zeros_like(mask)
    keep[cv - 6 : cv + 6, cu - 6 : cu + 6] = True
    sparse = frames.copy()
    sparse[:, mask & ~keep] = 0.0
    res, _, _ = calibrate(model, CAL, c0, [Capture(rgb, sparse, TICKS)], TABLE_Z_M, bootstrap=0)
    assert not res.passed
    assert res.n_inliers < 1000
    assert any("too little overlap" in f for f in res.failures), res.failures


def test_plane_check_is_not_vacuous_when_icp_did_not_run(scene):
    """With no 6-DOF iteration the pose is still the plane-constrained one and the plane check agrees with itself by
    construction: the gate must report that instead of passing."""
    from so101_rl.perception.arm_registration import ArmShapeRegistration
    from so101_rl.perception.camera_calibrate import observations

    c0, _, model, rgb, frames = scene
    obs, plane = observations(model, CAL, c0, [Capture(rgb, frames, TICKS)])
    reg = ArmShapeRegistration(obs, plane, c0.color, TABLE_Z_M, seed=0)
    reg.icp_6dof = lambda T, **kw: (setattr(reg, "icp6_iterations", 0), setattr(reg, "icp6_converged", False), T)[2]
    res = reg.run(bootstrap=0)
    assert not res.passed and any("6-DOF ICP ran 0" in f for f in res.failures), res.failures


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


def test_replay_from_capture_is_report_only(tmp_path, scene):
    """``--from-capture`` re-runs a saved capture offline: no probe, no camera, and calib/camera.yaml is never written
    (even on PASS: the camera may have moved since)."""
    c0, T_true, _, rgb, frames = scene
    src = tmp_path / "runs" / "20261009-000000"
    src.mkdir(parents=True)
    np.savez_compressed(src / "capture_0.npz", rgb=rgb, depth=frames, ticks=TICKS)  # what camera-calibrate saves
    cam_yaml = tmp_path / "camera.yaml"
    save_camera_calib(c0, cam_yaml)
    before = cam_yaml.read_text()

    def forbidden(*_a, **_k):
        raise AssertionError("replay touched the hardware")

    rc = camera_calibrate.main(
        ["--method", "arm-shape", "--camera-yaml", str(cam_yaml), "--out-dir", str(tmp_path / "replay"),
         "--calibration", str(Path(__file__).parent / "data" / "so101_follower.calib.json"), "--from-capture", str(src)],
        probe=forbidden, camera=forbidden, wait=forbidden,
    )  # fmt: skip
    assert rc == 0 and cam_yaml.read_text() == before
    out = next((tmp_path / "replay").iterdir())
    assert out.name.endswith("-replay")
    rep = json.loads((out / "report.json").read_text())
    assert rep["passed"] and rep["replay_of"] == str(src) and rep["ticks"] == [TICKS.tolist()]
    np.testing.assert_allclose(np.array(rep["T_base_color"])[:3, 3], T_true[:3, 3], atol=0.003)
    assert (out / "capture_0.npz").exists() and (out / "overlay_0.png").exists()
    assert camera_calibrate.main(["--method", "arm-shape", "--camera-yaml", str(cam_yaml), "--from-capture", str(tmp_path / "empty")]) == 2


def test_workspace_plane_ignores_a_surface_outside_the_workspace(scene):
    """A big tilted surface away from the arm (a person, the bed) outvotes the table in the global RANSAC; the refit on
    the workspace around the base finds the table."""
    c0 = scene[0]
    T = c0.T_base_color
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.35, 0.35, size=(30_000, 2))
    table = np.c_[xy, np.full(len(xy), TABLE_Z_M)]
    far = rng.uniform([0.45, -0.6], [0.9, 0.6], size=(60_000, 2))
    slope = math.tan(math.radians(4.0))
    other = np.c_[far, TABLE_Z_M + 0.004 + slope * (far[:, 0] - 0.45)]  # 4 deg, meets the table's band at its edge
    pb = np.concatenate([table, other])
    Ti = np.linalg.inv(T)
    pts = pb @ Ti[:3, :3].T + Ti[:3, 3]
    up_true = Ti[:3, :3] @ np.array([0.0, 0.0, 1.0])
    glob = fit_table_plane(pts)
    assert math.degrees(math.acos(min(1.0, float(glob.normal_up @ up_true)))) > 2.0  # the global fit took the slope
    first = PlaneFit(glob.normal_up, glob.height_m, glob.inlier_fraction, glob.inliers)
    ws = refit_in_workspace(pts, first, T, band_m=0.05)
    assert ws is not None
    assert math.degrees(math.acos(min(1.0, float(ws.normal_up @ up_true)))) < 0.2
    assert ws.height_m == pytest.approx(T[2, 3] - TABLE_Z_M, abs=0.001)

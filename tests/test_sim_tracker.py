"""FS6 SimDieTracker (sim track S1.2), CPU torch."""

import math
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.sim.model import TABLE_Z_M, load_model
from so101_rl.sim.sim_tracker import (
    NEVER_SEEN_AGE,
    NO_NOISE,
    CameraModel,
    DistractorModel,
    SimDieTracker,
    TrackerNoise,
    TrackerOut,
)

# the synthetic scenes were laid out for the provisional camera view (see conftest.provisional_camera)
pytestmark = pytest.mark.usefixtures("provisional_camera")

CPU = torch.device("cpu")
DIE_Z = TABLE_Z_M + 0.008
FAR = 1.0  # jaws / tip far away (no occlusion)


CALIB = load_camera_calib(Path(__file__).parent / "data" / "camera_provisional_2026-10-07.yaml")  # = provisional_camera
CAM = CameraModel.from_calib(CALIB)
DCAM = CameraModel.from_calib(CALIB, "depth")


def _tracker(noise=NO_NOISE, n=1, seed=0, **kw):
    tr = SimDieTracker(CAM, noise, n, device=CPU, seed=seed, depth_cam=DCAM, **kw)
    return tr, TrackerOut.empty(n, CPU)


def _toward_camera(die, dist):
    """A point ``dist`` m from the die along the ray to the colour camera."""
    o = torch.tensor(CAM.T_base_cam[:3, 3], dtype=torch.float32)
    d = o - die
    return die + dist * d / d.norm(dim=-1, keepdim=True)


def _away(n):
    """tip and jaw boxes far above/aside: never occluding."""
    tip = torch.tensor([[0.0, 0.3, 0.3]]).repeat(n, 1)
    centers = torch.tensor([[0.0, 0.3, 0.3]]).repeat(n, 4, 1)
    half = torch.full((n, 4, 3), 0.005)
    return tip, centers, half


def _die(x=0.20, y=-0.05, n=1):
    return torch.tensor([[x, y, DIE_Z]]).repeat(n, 1)


def _start(tr, die, n=1):
    tr.reset(torch.ones(n, dtype=torch.bool), die)


def test_camera_model_is_the_scene_d435():
    from so101_rl.sim.d435 import COLOR_CAM, DEPTH_CAM, camera_T_base

    model, data = load_model()
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(camera_T_base(model, data, COLOR_CAM), CAM.T_base_cam, atol=1e-9)
    np.testing.assert_allclose(camera_T_base(model, data, DEPTH_CAM), DCAM.T_base_cam, atol=1e-9)
    assert (CAM.fx, CAM.cx, DCAM.fx) == (CALIB.color.fx, CALIB.color.cx, CALIB.depth.fx)
    assert CameraModel.default().T_base_cam.tolist() == CAM.T_base_cam.tolist()


def test_visible_die_noise_stats():
    noise = TrackerNoise(rot_bias_max_rad=0.0, dropout_p=0.0, latency_frames=(0, 0))
    tr, out = _tracker(noise, n=1, seed=3)
    die = _die()
    _start(tr, die)
    tip, c, h = _away(1)
    xs = []
    for _ in range(10_000):
        tr.update(die, tip, c, h, 1, out)
        assert bool(out.visible[0])
        xs.append(out.xyz[0].clone())
    xs = torch.stack(xs).double()
    std = (xs @ torch.tensor(CAM.T_base_cam[:3, :3]).double()).std(dim=0).numpy()  # in the camera frame
    assert abs(std[0] / noise.sigma_xy_m - 1) < 0.1
    assert abs(std[1] / noise.sigma_xy_m - 1) < 0.1
    assert abs(std[2] / noise.sigma_z_m - 1) < 0.1  # along the optical axis
    bias = (xs.mean(dim=0) - die[0].double()).norm().item()
    assert bias <= noise.bias_max_m + 1e-3


def test_occluded_when_tip_in_front_of_die_with_the_fallback_on():
    noise = replace(NO_NOISE, occlusion_radius_m=0.02)
    tr, out = _tracker(noise)
    die = _die()
    _start(tr, die)
    tip = _toward_camera(die, 0.05)
    _, c, h = _away(1)
    assert bool(tr.occluded(die, tip, c, h)[0])
    tr.update(die, tip, c, h, 3, out)
    assert not bool(out.visible[0])
    assert not bool(_tracker()[0].occluded(die, tip, c, h)[0])  # default: off, the boxes decide


def test_occluded_by_projected_jaw_boxes():
    tr, _ = _tracker()
    die = _die()
    tip, _, _ = _away(1)
    centers = _toward_camera(die, 0.06)[:, None].repeat(1, 4, 1)  # a box between the die and the camera
    half = torch.full((1, 4, 3), 0.012)
    assert float(tr.covered_fraction(die, centers, half)[0]) > 0.9
    assert bool(tr.occluded(die, tip, centers, half)[0])
    behind = _toward_camera(die, -0.05)[:, None].repeat(1, 4, 1)  # behind the die: not occluding
    assert float(tr.covered_fraction(die, behind, half)[0]) == 0.0


def test_depth_sensor_occlusion_counts():
    """A box that hides the die from the depth sensor only (14.8 mm beside the colour sensor) still counts."""
    tr, _ = _tracker()
    die = _die()
    o_d = torch.tensor(DCAM.T_base_cam[:3, 3], dtype=torch.float32)
    p = die + 0.7 * (o_d - die)  # 70 % of the way from the die to the DEPTH sensor (the rays have parted ~1 cm)
    centers = p[:, None].repeat(1, 1, 1)
    half = torch.full((1, 1, 3), 0.004)
    color_only = SimDieTracker(CAM, NO_NOISE, 1, device=CPU, seed=0)
    assert float(tr.covered_fraction(die, centers, half)[0]) > float(color_only.covered_fraction(die, centers, half)[0])


def test_not_occluded_when_tip_away():
    tr, out = _tracker()
    die = _die()
    _start(tr, die)
    tip = die + torch.tensor([[0.0, 0.0, 0.05]])  # above the die: the oblique camera still sees it
    _, c, h = _away(1)
    assert not bool(tr.occluded(die, tip, c, h)[0])
    tr.update(die, tip, c, h, 3, out)
    assert bool(out.visible[0])
    assert torch.allclose(out.xyz, die, atol=1e-6)


@pytest.mark.parametrize("lat", [1, 2, 3])
def test_latency_frames_exact(lat):
    noise = TrackerNoise(
        sigma_xy_m=0.0, sigma_z_m=0.0, bias_max_m=0.0, rot_bias_max_rad=0.0, latency_frames=(lat, lat), dropout_p=0.0
    )
    tr, out = _tracker(noise)
    a, b = _die(0.20, -0.05), _die(0.25, 0.05)
    _start(tr, a)
    tip, c, h = _away(1)
    tr.update(a, tip, c, h, 10, out)
    assert torch.allclose(out.xyz, a)
    for k in range(1, lat + 2):
        tr.update(b, tip, c, h, 1, out)
        expect = b if k > lat else a
        assert torch.allclose(out.xyz, expect), (k, lat)


def test_reset_starts_never_seen_and_warm_up_fills():
    noise = TrackerNoise(latency_frames=(3, 3), dropout_p=0.0)
    tr, out = _tracker(noise)
    die = _die()
    _start(tr, die)
    tip, c, h = _away(1)
    tr.update(die, tip, c, h, 3, out)  # latency 3: the 3 frames emitted so far predate the reset
    assert not bool(out.visible[0]) and int(out.age_frames[0]) >= NEVER_SEEN_AGE
    assert (out.xyz == 0).all()
    tr.update(die, tip, c, h, 1, out)
    assert bool(out.visible[0]) and int(out.age_frames[0]) == 0


def test_dropout_burst_holds_last_and_ages():
    noise = TrackerNoise(
        sigma_xy_m=0.0, sigma_z_m=0.0, bias_max_m=0.0, rot_bias_max_rad=0.0, latency_frames=(0, 0),
        dropout_p=1.0, dropout_burst_frames=(4, 4),
    )  # fmt: skip
    tr, out = _tracker(noise)
    die = _die()
    _start(tr, die)
    tip, c, h = _away(1)
    tr.noise = NO_NOISE  # first sighting without dropout
    tr.update(die, tip, c, h, 1, out)
    assert bool(out.visible[0])
    tr.noise = noise
    moved = _die(0.22, -0.02)
    ages = []
    for _ in range(4):
        tr.update(moved, tip, c, h, 1, out)
        assert not bool(out.visible[0])
        assert torch.allclose(out.xyz, die)  # held
        ages.append(int(out.age_frames[0]))
    assert ages == [1, 2, 3, 4]


def test_outside_fov_invisible():
    tr, out = _tracker()
    tip, c, h = _away(1)
    for x, y in ((0.20, 0.60), (0.9, 0.0), (-0.45, 0.0)):  # behind the camera, beside the FOV
        die = _die(x, y)
        _start(tr, die)
        assert not bool(tr.in_view(die)[0]), (x, y)
        tr.update(die, tip, c, h, 3, out)
        assert not bool(out.visible[0])
    fallen = torch.tensor([[0.2, 0.0, TABLE_Z_M - 0.05]])  # below the table
    assert not bool(tr.in_view(fallen)[0])


def test_per_world_independence():
    noise = TrackerNoise(dropout_p=0.0, latency_frames=(0, 0))
    tr, out = _tracker(noise, n=3, seed=1)
    die = _die(n=3)
    tr.reset(torch.tensor([True, True, True]), die)
    tip, c, h = _away(3)
    c[1] = _toward_camera(die[1:2], 0.05)  # only world 1 occluded
    h[1] = 0.015
    tr.update(die, tip, c, h, 3, out)
    assert out.visible.tolist() == [True, False, True]
    b0 = tr.bias.clone()
    tr.reset(torch.tensor([False, True, False]), die)
    assert torch.equal(tr.bias[0], b0[0]) and torch.equal(tr.bias[2], b0[2])


# ---- semantics vs the real tracker on rendered frames: tests/test_camera_parity.py (calibrated pose) ------------


# ---- camera-pose DR (C.3) ---------------------------------------------------------------------------------------
def test_true_camera_pose_is_drawn_within_the_dr_range_and_equals_the_measurement_error():
    noise = replace(NO_NOISE, bias_max_m=0.005, rot_bias_max_rad=math.radians(1.5))
    tr, out = _tracker(noise, n=256, seed=4)
    die = _die(n=256)
    tr.reset(torch.ones(256, dtype=torch.bool), die)
    R0 = torch.tensor(CAM.T_base_cam[:3, :3], dtype=torch.float32)
    t0 = torch.tensor(CAM.T_base_cam[:3, 3], dtype=torch.float32)
    dR = tr.cam_R @ R0.T
    ang = torch.rad2deg(torch.arccos(((dR.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
    assert float(ang.max()) <= 1.5 + 1e-3 and float(ang.max()) > 1.0
    assert float((tr.cam_t - t0).norm(dim=-1).max()) <= 0.005 * 1.001 + 1e-6
    # the reported position = calibrated camera applied to what the TRUE camera measures
    tip, c, h = _away(256)
    tr.update(die, tip, c, h, 1, out)
    p_cam_true = torch.einsum("nji,nj->ni", tr.cam_R, die - tr.cam_t)  # die in the true camera frame
    expect = p_cam_true @ R0.T + t0
    assert torch.allclose(out.xyz, expect, atol=2e-5)


def test_noise_widens_to_the_calibration_uncertainty():
    from so101_rl.perception.camera_calib import CameraPose

    wide = CALIB.with_pose(CameraPose(CALIB.T_base_color, "arm-shape", "x", 2.0, 12.0, 4.0))
    n = TrackerNoise().with_camera_dr(wide)
    assert n.bias_max_m == pytest.approx(0.012) and n.rot_bias_max_rad == pytest.approx(math.radians(4.0))
    assert TrackerNoise().with_camera_dr(CALIB).bias_max_m >= 0.005  # the floor


# ---- distractors (C.5) ------------------------------------------------------------------------------------------
def test_a_larger_distractor_in_view_replaces_the_die():
    die = _die(0.22, 0.10)
    tip, c, h = _away(1)
    small, big = 40.0, 3000.0
    for area, fp in ((small, False), (big, True)):
        dm = DistractorModel(fixed_xyz=np.array([[0.25, 0.0, TABLE_Z_M + 0.01]]), fixed_area_px=np.array([area]))
        tr, out = _tracker(NO_NOISE, distractors=dm)
        _start(tr, die)
        tr.update(die, tip, c, h, 1, out)
        assert bool(out.visible[0])
        target = torch.tensor([[0.25, 0.0, TABLE_Z_M + 0.01]]) if fp else die
        assert torch.allclose(out.xyz, target, atol=1e-6), (area, out.xyz)
        assert bool(tr.false_positive[0]) == fp


def test_a_larger_distractor_off_the_table_is_ignored():
    """Plan §10 2026-10-17: like the real tracker, only blobs on the table workspace count."""
    from so101_rl.perception.die_tracker import table_workspace

    lo, hi = table_workspace()
    die = _die(0.22, 0.10)
    tip, c, h = _away(1)
    off = None
    for x in np.arange(hi[0] + 0.01, hi[0] + 0.6, 0.02):  # beyond the table, still in the image
        dm = DistractorModel(fixed_xyz=np.array([[x, 0.0, TABLE_Z_M + 0.01]]), fixed_area_px=np.array([3000.0]))
        tr, out = _tracker(NO_NOISE, distractors=dm)
        if bool(tr._in_image(torch.tensor([[x, 0.0, TABLE_Z_M + 0.01]], dtype=torch.float32))[0]):
            off = x
            break
    assert off is not None, "no off-table point in view for this camera"
    _start(tr, die)
    tr.update(die, tip, c, h, 1, out)
    assert bool(out.visible[0]) and not bool(tr.false_positive[0])
    assert torch.allclose(out.xyz, die, atol=1e-6)


def test_a_distractor_is_a_false_positive_while_the_die_is_hidden():
    dm = DistractorModel(fixed_xyz=np.array([[0.25, 0.0, TABLE_Z_M + 0.01]]), fixed_area_px=np.array([60.0]))
    tr, out = _tracker(NO_NOISE, distractors=dm)
    die = _die()
    _start(tr, die)
    tip, _, _ = _away(1)
    centers = _toward_camera(die, 0.06)[:, None].repeat(1, 4, 1)
    tr.update(die, tip, centers, torch.full((1, 4, 3), 0.012), 1, out)
    assert bool(out.visible[0]) and bool(tr.false_positive[0])
    out_of_band = DistractorModel(fixed_xyz=dm.fixed_xyz, fixed_area_px=np.array([5000.0]))
    tr, out = _tracker(NO_NOISE, distractors=out_of_band)
    _start(tr, die)
    tr.update(die, tip, centers, torch.full((1, 4, 3), 0.012), 1, out)
    assert not bool(out.visible[0])  # bigger than the tracker's band: ignored, as by the real tracker


def test_random_dr_distractors_are_drawn_per_episode():
    dm = DistractorModel(random_p=1.0, random_count=(2, 2), random_area_px=(3000.0, 3500.0))
    tr, out = _tracker(NO_NOISE, n=64, seed=2, distractors=dm)
    die = _die(n=64)
    tr.reset(torch.ones(64, dtype=torch.bool), die)
    assert int(tr._dist_on.sum()) == 128
    xyz = tr._dist_xyz
    assert float(xyz[..., 0].min()) >= 0.10 and float(xyz[..., 0].max()) <= 0.32
    tip, c, h = _away(64)
    tr.update(die, tip, c, h, 1, out)
    assert bool(tr.false_positive.any())  # a 3000 px blob beats a ~500 px die wherever it is in view
    none = DistractorModel(random_p=0.0)
    tr2, _out2 = _tracker(NO_NOISE, n=4, distractors=none)
    assert tr2._dist_xyz.shape[1] == 0


def test_measured_distractors_file_loads():
    dm = DistractorModel.from_yaml(random_p=0.3)
    assert dm.fixed_xyz.shape[1] == 3 and len(dm.fixed_area_px) == len(dm.fixed_xyz)
    assert dm.min_area_px == 40 and dm.max_area_px == 4000


def test_area_band_matches_the_real_tracker_and_applies_without_distractors():
    import yaml

    from so101_rl.perception.die_tracker import DEFAULT_HSV_YAML
    from so101_rl.sim.sim_tracker import DEFAULT_MAX_AREA_PX, DEFAULT_MIN_AREA_PX

    hsv = yaml.safe_load(DEFAULT_HSV_YAML.read_text())
    assert (DEFAULT_MIN_AREA_PX, DEFAULT_MAX_AREA_PX) == (hsv["min_area_px"], hsv["max_area_px"])
    tr, out = _tracker()
    near = _toward_camera(_die(0.22, 0.10), 0.33)  # a die a few cm from the lens: blob far above the band
    assert float(tr.die_area_px(near)[0]) > DEFAULT_MAX_AREA_PX
    _start(tr, near)
    tip, c, h = _away(1)
    tr.update(near, tip, c, h, 1, out)
    assert not bool(out.visible[0])


def test_die_cut_by_the_image_border_is_invisible_like_on_the_real_tracker():
    """The real HsvDepthDieTracker drops a blob that touches the image border; the sim tracker must call a die
    invisible while its centre is closer to the border than the projected half-blob width (not only when the centre
    leaves the image)."""
    tr, _ = _tracker()
    xs = torch.linspace(-0.3, 0.6, 91)
    ys = torch.linspace(-0.45, 0.45, 91)
    die = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    die = torch.cat([die, torch.full((len(die), 1), DIE_Z)], dim=-1)
    uv, z = tr._project(die)
    half = 0.5 * tr.die_area_px(die).sqrt()
    edge_dist = torch.minimum(
        torch.minimum(uv[:, 0], CAM.width - 1 - uv[:, 0]), torch.minimum(uv[:, 1], CAM.height - 1 - uv[:, 1])
    )
    sees = tr._in_image(die)  # centre-in-image only (the old criterion)
    table = (die[:, :2].abs() <= 0.45).all(-1)
    cut = sees & (z > 0) & (edge_dist < half - 0.5) & table
    inside = sees & (z > 0) & (edge_dist > half + 0.5) & table
    assert cut.sum() > 0 and inside.sum() > 0, "grid must contain both cases"
    vis = tr.in_view(die)
    assert not vis[cut].any()
    assert vis[inside & (die[:, :2].abs() <= 0.30).all(-1)].all()

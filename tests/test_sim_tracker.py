"""FS6 SimDieTracker (sim track S1.2), CPU torch."""

import mujoco
import numpy as np
import pytest
import torch

from so101_rl.perception.camera import Intrinsics
from so101_rl.perception.die_tracker import HsvDepthDieTracker
from so101_rl.sim.model import TABLE_Z_M, load_model
from so101_rl.sim.sim_tracker import (
    NEVER_SEEN_AGE,
    NO_NOISE,
    CameraModel,
    Occluders,
    SimDieTracker,
    TrackerNoise,
    TrackerOut,
)

CPU = torch.device("cpu")
DIE_Z = TABLE_Z_M + 0.008
FAR = 1.0  # jaws / tip far away (no occlusion)


def _tracker(noise=NO_NOISE, n=1, seed=0):
    tr = SimDieTracker(CameraModel.nominal(), noise, n, device=CPU, seed=seed)
    return tr, TrackerOut.empty(n, CPU)


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


def test_camera_matches_scene_overhead_camera():
    model, data = load_model()
    cam = CameraModel.nominal()
    c = model.camera("overhead")
    mujoco.mj_forward(model, data)
    assert np.allclose(data.cam_xpos[c.id], cam.T_base_cam[:3, 3], atol=1e-9)
    # MuJoCo camera looks along -z, y up; OpenCV z forward, y down
    R_mj = data.cam_xmat[c.id].reshape(3, 3)
    assert np.allclose(R_mj[:, 0], cam.T_base_cam[:3, 0], atol=1e-9)
    assert np.allclose(-R_mj[:, 1], cam.T_base_cam[:3, 1], atol=1e-9)
    assert np.allclose(-R_mj[:, 2], cam.T_base_cam[:3, 2], atol=1e-9)
    assert abs(model.cam_fovy[c.id] - cam.mujoco_fovy_deg()) < 1e-3


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
    std = xs.std(dim=0).numpy()
    assert abs(std[0] / noise.sigma_xy_m - 1) < 0.1
    assert abs(std[1] / noise.sigma_xy_m - 1) < 0.1
    assert abs(std[2] / noise.sigma_z_m - 1) < 0.1  # optical axis = base -z for the overhead mount
    bias = (xs.mean(dim=0) - die[0].double()).norm().item()
    assert bias <= noise.bias_max_m + 1e-3


def test_occluded_when_tip_over_die():
    tr, out = _tracker()
    die = _die()
    _start(tr, die)
    tip = die + torch.tensor([[0.0, 0.0, 0.005]])
    _, c, h = _away(1)
    assert bool(tr.occluded(die, tip, c, h)[0])
    tr.update(die, tip, c, h, 3, out)
    assert not bool(out.visible[0])


def test_occluded_by_projected_jaw_boxes():
    tr, _ = _tracker()
    die = _die()
    tip, _, _ = _away(1)
    centers = (die + torch.tensor([[0.0, 0.0, 0.04]]))[:, None].repeat(1, 4, 1)  # a box 4 cm above the die
    half = torch.full((1, 4, 3), 0.012)
    assert float(tr.covered_fraction(die, centers, half)[0]) > 0.9
    assert bool(tr.occluded(die, tip, centers, half)[0])
    below = (die + torch.tensor([[0.0, 0.0, -0.03]]))[:, None].repeat(1, 4, 1)  # behind the die: not occluding
    assert float(tr.covered_fraction(die, below, half)[0]) == 0.0


def test_not_occluded_when_tip_away():
    tr, out = _tracker()
    die = _die()
    _start(tr, die)
    tip = die + torch.tensor([[0.06, 0.0, 0.03]])
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
    for x, y in ((0.60, 0.0), (0.20, 0.30), (-0.20, 0.0)):
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
    tip[1] = die[1] + torch.tensor([0.0, 0.0, 0.005])  # only world 1 occluded
    tr.update(die, tip, c, h, 3, out)
    assert out.visible.tolist() == [True, False, True]
    b0 = tr.bias.clone()
    tr.reset(torch.tensor([False, True, False]), die)
    assert torch.equal(tr.bias[0], b0[0]) and torch.equal(tr.bias[2], b0[2])


# ---- semantics vs the real tracker on rendered frames ----------------------------------------------------------
def _configs(model):
    """20 (arm qpos[6], die xy): 10 gripper-low arm poses x {die under the tip, die 6-8 cm aside}."""
    rng = np.random.default_rng(5)
    data = mujoco.MjData(model)
    tip_id = model.site("gripper_tip").id
    out = []
    while len(out) < 20:
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
    return out


def test_semantics_match_real_tracker_on_synthetic_frames():
    model, data = load_model()
    try:
        renderer = mujoco.Renderer(model, 480, 640)
    except Exception as exc:  # noqa: BLE001 - no OpenGL context on this host
        pytest.skip(f"no MuJoCo offscreen rendering: {exc}")
    cam = CameraModel.nominal()
    real = HsvDepthDieTracker.from_yaml(
        Intrinsics(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, width=640, height=480, depth_scale=0.001)
    )
    tr, _ = _tracker()
    jaws = Occluders(model, CPU)
    body_ids = [model.body(b).id for b in jaws.bodies]
    agree, log = 0, []
    for q, xy in _configs(model):
        data.qpos[:6] = q
        die_q = model.jnt_qposadr[model.joint("die_free").id]
        data.qpos[die_q : die_q + 7] = (xy[0], xy[1], DIE_Z, 1, 0, 0, 0)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="overhead")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="overhead")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        real_visible = real.detect(rgb, depth) is not None
        die = torch.tensor(data.qpos[die_q : die_q + 3], dtype=torch.float32)[None]
        tip = torch.tensor(data.site_xpos[model.site("gripper_tip").id], dtype=torch.float32)[None]
        c, h, rot = jaws.compute(
            torch.tensor(data.xpos[body_ids], dtype=torch.float32)[None],
            torch.tensor(data.xquat[body_ids], dtype=torch.float32)[None],
        )
        sim_visible = bool(tr.in_view(die)[0]) and not bool(tr.occluded(die, tip, c, h, rot)[0])
        agree += real_visible == sim_visible
        log.append((np.round(xy, 3).tolist(), np.round(tip[0].numpy(), 3).tolist(), real_visible, sim_visible))
    renderer.close()
    print("\n".join(str(r) for r in log))
    disagreements = [r for r in log if r[2] != r[3]]
    print(f"agreement {agree}/20; disagreements (die xy, tip, real, sim): {disagreements}")
    assert sum(r[2] for r in log) >= 5 and sum(not r[2] for r in log) >= 5  # both verdicts exercised
    assert agree >= 18

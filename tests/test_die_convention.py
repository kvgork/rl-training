"""Die position convention: the real and the sim tracker report the die CENTROID (integration 2026-10-06).

``HsvDepthDieTracker`` deprojects the visible TOP face (~8 mm above the centre); ``centroid_base`` subtracts half an
edge along the table normal, so the object table publishes what the sim tracker reports (FS3 ``die_xyz``). Both
trackers look at the SAME rendered scene (the overhead camera of the sim, rendered through MuJoCo) here.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from so101_rl.env.tasks import DIE_HALF_EDGE_M
from so101_rl.perception.camera import Intrinsics
from so101_rl.perception.die_tracker import HsvDepthDieTracker, die_centroid_base
from so101_rl.perception.frames import cam_to_base
from so101_rl.sim.model import DIE_EDGE_M, TABLE_Z_M, load_model
from so101_rl.sim.sim_tracker import NO_NOISE, CameraModel, SimDieTracker, TrackerOut

CPU = torch.device("cpu")
# resting die positions in view and clear of the arm at home: start well, zone, two more on the table
SCENES = [(0.20, -0.05, 0.0), (0.22, 0.10, 0.6), (0.26, -0.10, 1.1), (0.15, 0.08, 0.3)]


@pytest.fixture(scope="module")
def rendered():
    model, data = load_model()
    try:
        renderer = mujoco.Renderer(model, 480, 640)
    except Exception as exc:  # noqa: BLE001 - no OpenGL context on this host
        pytest.skip(f"no MuJoCo offscreen rendering: {exc}")
    out = []
    dq = model.jnt_qposadr[model.joint("die_free").id]
    for x, y, yaw in SCENES:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.qpos[dq : dq + 7] = (x, y, TABLE_Z_M + DIE_EDGE_M / 2, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="overhead")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="overhead")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        out.append((data.xpos[model.body("die").id].copy(), rgb, depth))
    renderer.close()
    return out


def _real():
    cam = CameraModel.nominal()
    intr = Intrinsics(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, width=640, height=480, depth_scale=0.001)
    return HsvDepthDieTracker.from_yaml(intr), cam


def _sim(die_true: np.ndarray) -> np.ndarray:
    tr = SimDieTracker(CameraModel.nominal(), NO_NOISE, 1, device=CPU, seed=0)
    out = TrackerOut.empty(1, CPU)
    die = torch.tensor(die_true, dtype=torch.float32)[None]
    tr.reset(torch.ones(1, dtype=torch.bool), die)
    tr.prime(torch.ones(1, dtype=torch.bool), die, out)
    assert bool(out.visible[0])
    return out.xyz[0].double().numpy()


def test_the_tracker_edge_is_the_scene_die_edge():
    real, _ = _real()
    assert real.die_edge_m == pytest.approx(DIE_EDGE_M) == pytest.approx(2 * DIE_HALF_EDGE_M)


def test_real_and_sim_trackers_report_the_centroid_of_the_same_scene(rendered):
    real, cam = _real()
    for die_true, rgb, depth in rendered:
        det = real.detect(rgb, depth)
        assert det is not None, die_true
        r = real.centroid_base(det, cam.T_base_cam)
        s = _sim(die_true)
        np.testing.assert_allclose(s, die_true, atol=1e-6)  # the sim tracker reports the true centre
        # same convention: within the 1 px ~ 1 mm deprojection error of the real tracker, z exact to < 1 mm
        assert np.abs(r - s).max() < 0.003, (die_true, r, s)
        assert abs(r[2] - s[2]) < 0.001, (die_true, r, s)
        assert r[2] == pytest.approx(TABLE_Z_M + DIE_HALF_EDGE_M, abs=1e-3)


def test_without_the_correction_the_real_tracker_reports_the_top_face(rendered):
    """Shown to fail: the raw deprojection is half an edge above the sim tracker's centroid."""
    real, cam = _real()
    for die_true, rgb, depth in rendered:
        raw = cam_to_base(real.detect(rgb, depth).xyz_cam, cam.T_base_cam)
        assert raw[2] - _sim(die_true)[2] == pytest.approx(DIE_HALF_EDGE_M, abs=1e-3)


def test_centroid_helper_subtracts_half_an_edge_along_the_table_normal():
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # looking down
    T[:3, 3] = (0.2, 0.0, 0.6)
    p = die_centroid_base(np.array([0.01, -0.02, 0.5]), T, 0.016)
    np.testing.assert_allclose(p, [0.21, 0.02, 0.1 - 0.008])
    with pytest.raises(ValueError, match="die_edge_m"):
        die_centroid_base(np.zeros(3), T, 0.0)

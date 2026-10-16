"""Die position convention: the real and the sim tracker report the die CENTROID (integration 2026-10-06).

``HsvDepthDieTracker`` deprojects the point where the viewing ray enters the die (on the oblique real camera: mostly
a side face); ``centroid_base`` pushes it on along the ray by the entry depth, so the object table publishes what the
sim tracker reports (FS3 ``die_xyz``). Both trackers look at the SAME rendered scene (the sim ``d435_color``
camera of ``calib/camera.yaml``, colour + depth aligned to colour, rendered through MuJoCo) here.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from so101_rl.env.tasks import DIE_HALF_EDGE_M
from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.perception.die_tracker import HsvDepthDieTracker, die_centroid_base, die_entry_depth
from so101_rl.perception.frames import cam_to_base
from so101_rl.sim.d435 import COLOR_CAM
from so101_rl.sim.model import DIE_EDGE_M, TABLE_Z_M, load_model
from so101_rl.sim.sim_tracker import NO_NOISE, CameraModel, SimDieTracker, TrackerOut

# the synthetic scenes were laid out for the provisional camera view (see conftest.provisional_camera)
pytestmark = pytest.mark.usefixtures("provisional_camera")

CPU = torch.device("cpu")
# resting die positions clear of the arm at home: start well (hidden by its rim from the oblique camera), zone, two
# more on the table
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
        renderer.update_scene(data, camera=COLOR_CAM)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=COLOR_CAM)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        out.append((data.xpos[model.body("die").id].copy(), rgb, depth))
    renderer.close()
    return out


def _real():
    calib = load_camera_calib()
    return HsvDepthDieTracker.from_camera_calib(calib), CameraModel.from_calib(calib)


def _sim(die_true: np.ndarray) -> np.ndarray:
    tr = SimDieTracker(CameraModel.default(), NO_NOISE, 1, device=CPU, seed=0)
    out = TrackerOut.empty(1, CPU)
    die = torch.tensor(die_true, dtype=torch.float32)[None]
    tr.reset(torch.ones(1, dtype=torch.bool), die)
    tr.prime(torch.ones(1, dtype=torch.bool), die, out)
    assert bool(out.visible[0])
    return out.xyz[0].double().numpy()


def test_the_tracker_edge_is_the_scene_die_edge():
    real, _ = _real()
    assert real.die_edge_m == pytest.approx(DIE_EDGE_M) == pytest.approx(2 * DIE_HALF_EDGE_M)


def test_the_start_well_hides_the_die_from_the_oblique_camera(rendered):
    """From the real (oblique) mount a die in the start well fails the real tracker's ring test (the tray rim around
    it is nearer than the die), and the sim tracker's ring rule over the tray occluder boxes agrees (camera parity
    C.3; the 2026-10-10 gap, fixed 2026-10-16 by the sim ring rule)."""
    import torch as _t

    from so101_rl.sim.sim_tracker import Occluders

    real, cam = _real()
    die_true, rgb, depth = rendered[0]
    assert real.detect(rgb, depth) is None
    model, data = load_model()
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)  # the pose the scene was rendered at
    mujoco.mj_forward(model, data)
    occl = Occluders(model, CPU)
    ids = [model.body(b).id for b in occl.bodies]
    c, h, rot = occl.compute(_t.tensor(data.xpos[ids], dtype=_t.float32)[None], _t.tensor(data.xquat[ids], dtype=_t.float32)[None])
    tr = SimDieTracker(cam, NO_NOISE, 1, device=CPU, seed=0)
    die = _t.tensor(die_true, dtype=_t.float32)[None]
    assert bool(tr.occluded(die, _t.tensor([[0.2, 0.0, 0.2]]), c, h, rot)[0])


def test_real_and_sim_trackers_report_the_centroid_of_the_same_scene(rendered):
    real, cam = _real()
    for die_true, rgb, depth in rendered[1:]:  # [0] is the start well (hidden, test above)
        det = real.detect(rgb, depth)
        assert det is not None, die_true
        r = real.centroid_base(det, cam.T_base_cam)
        s = _sim(die_true)
        np.testing.assert_allclose(s, die_true, atol=1e-6)  # the sim tracker reports the true centre
        # same convention: within 3 mm (the entry-depth yaw assumption and the ~1 px deprojection error)
        assert np.linalg.norm(r - s) < 0.003, (die_true, r, s)


def test_without_the_correction_the_real_tracker_reports_a_surface_point(rendered):
    """Shown to fail: the raw deprojection is a surface point, more than 6 mm from the sim tracker's centroid; the
    old overhead rule (half an edge straight down) still misses by more than 3 mm on the oblique camera."""
    real, cam = _real()
    for die_true, rgb, depth in rendered[1:]:
        raw = cam_to_base(real.detect(rgb, depth).xyz_cam, cam.T_base_cam)
        s = _sim(die_true)
        assert np.linalg.norm(raw - s) > 0.006
        assert np.linalg.norm(raw - np.array([0.0, 0.0, DIE_HALF_EDGE_M]) - s) > 0.003


def test_entry_depth_overhead_and_oblique():
    assert die_entry_depth(np.array([0.0, 0.0, -1.0]), 0.016) == pytest.approx(0.008)
    th = np.radians(64.0)  # the real camera's tilt
    d = np.array([0.0, np.sin(th), -np.cos(th)])
    assert 0.008 < die_entry_depth(d, 0.016) < 0.008 / np.cos(np.radians(64.0))


def test_centroid_helper_subtracts_half_an_edge_along_the_table_normal():
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # looking down
    T[:3, 3] = (0.2, 0.0, 0.6)
    p = die_centroid_base(np.array([0.0, 0.0, 0.5]), T, 0.016)  # straight down the optical axis
    np.testing.assert_allclose(p, [0.2, 0.0, 0.1 - 0.008])
    with pytest.raises(ValueError, match="die_edge_m"):
        die_centroid_base(np.zeros(3), T, 0.0)


def test_sim_ring_constants_match_the_real_tracker():
    from so101_rl.perception import die_tracker as dt
    from so101_rl.sim import sim_tracker as st

    assert st.RING_OCCLUDED_FRACTION == dt._OCCLUDED_RING_FRACTION
    assert st.RING_PX == (dt._RING_KERNEL.shape[0] - 1) / 2
    import inspect

    assert st.RING_DEPTH_M == inspect.signature(HsvDepthDieTracker).parameters["depth_window_m"].default

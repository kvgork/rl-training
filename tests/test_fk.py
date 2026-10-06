"""Pure-numpy FK (plan task 2.2). The MuJoCo cross-check of the whole pipeline lives in 3.6 (needs U4); here the
upstream MJCF is used directly, when mujoco is importable, as an independent oracle for the same chain."""

import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from so101_rl.config import CENTER_TICK, DEFAULT_CONFIG_PATH, JOINT_NAMES, WorkspaceBox, ticks_to_rad
from so101_rl.safety.fk import KinematicChain, So101FK

KIN = DEFAULT_CONFIG_PATH.with_name("so101_kinematics.yaml")
MJCF = Path(__file__).parent.parent / "assets" / "so101" / "upstream" / "so101_new_calib.xml"
# A folded, low pose (shoulder back, elbow bent): plausible gravity-stable rest. NOT the q = 0 pose, which is the
# arm stretched out horizontally at x = 0.39 m (outside the workspace box).
REST_LIKE = np.array([0.0, -1.57, 1.57, 0.8, 0.0, 0.0])


@pytest.fixture(scope="module")
def chain():
    return KinematicChain.from_yaml(KIN)


@pytest.fixture
def fk(chain):
    return So101FK(chain)


def test_chain_loads_five_arm_joints(chain):
    assert chain.parent_T.shape == (5, 4, 4)
    assert chain.joint_names == JOINT_NAMES
    # rotation parts are orthonormal
    for t in (chain.base_T, chain.tip_T, *chain.parent_T):
        r = t[:3, :3]
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)


def test_rest_like_pose_is_plausible(fk):
    xyz = fk.gripper_xyz(REST_LIKE, np.zeros(3))
    box = WorkspaceBox()
    assert xyz[2] > 0.0  # above the table plane (z = 0 at the robot base)
    assert box.x[0] <= xyz[0] <= box.x[1]
    assert fk.in_box(xyz, box)


def test_zero_pose_value_is_reach_forward(fk):
    xyz = fk.gripper_xyz(np.zeros(6), np.zeros(3))
    np.testing.assert_allclose(xyz, [0.3914, 0.0, 0.2265], atol=1e-3)  # regression value of the extracted chain


def test_pan_rotates_about_the_pan_axis(fk, chain):
    pivot = chain.parent_T[0][:2, 3]  # the pan axis is vertical through the shoulder body origin
    a = fk.gripper_xyz(np.array([0.0, -1.0, 1.0, 0.0, 0.0, 0.0]), np.zeros(3))
    th = 0.6
    b = fk.gripper_xyz(np.array([th, -1.0, 1.0, 0.0, 0.0, 0.0]), np.zeros(3))
    assert b[2] == pytest.approx(a[2], abs=1e-12)
    da, db = a[:2] - pivot, b[:2] - pivot
    assert np.hypot(*db) == pytest.approx(np.hypot(*da), abs=1e-9)
    # the pan body is mounted upside-down in the MJCF, so +q turns clockwise seen from above: compare magnitude
    assert abs(np.arctan2(db[1], db[0]) - np.arctan2(da[1], da[0])) == pytest.approx(th, abs=1e-6)


def test_gripper_joint_does_not_move_tip(fk):
    q = REST_LIKE.copy()
    a = fk.gripper_xyz(q, np.zeros(3)).copy()
    q[5] = 1.0
    np.testing.assert_array_equal(a, fk.gripper_xyz(q, np.zeros(3)))


def test_deterministic(fk):
    rng = np.random.default_rng(1)
    q = rng.uniform(-1.5, 1.5, 6)
    first = fk.gripper_xyz(q, np.zeros(3)).copy()
    for _ in range(5):
        np.testing.assert_array_equal(first, fk.gripper_xyz(q, np.zeros(3)))
    other = So101FK(KinematicChain.from_yaml(KIN))
    np.testing.assert_array_equal(first, other.gripper_xyz(q, np.zeros(3)))


def test_allocation_free(fk):
    q = REST_LIKE.copy()
    ticks = np.full(6, CENTER_TICK, dtype=np.int32)
    out = np.zeros(3)
    fk.gripper_xyz(q, out)
    fk.gripper_xyz_from_ticks(ticks, out)  # warm up
    flt = [tracemalloc.Filter(True, "*/safety/fk.py")]

    def run():
        for _ in range(1000):
            fk.gripper_xyz(q, out)
            fk.gripper_xyz_from_ticks(ticks, out)

    tracemalloc.start()
    try:
        run()  # any one-off cache numpy/CPython keeps (a few dozen bytes) is paid here
        first = tracemalloc.take_snapshot().filter_traces(flt)
        run()
        second = tracemalloc.take_snapshot().filter_traces(flt)
    finally:
        tracemalloc.stop()
    # 1000 further calls (2000 FK evaluations) leave zero additional memory held by fk.py
    assert sum(s.size_diff for s in second.compare_to(first, "filename")) == 0


def test_ticks_to_rad_matches_config_and_drive_mode(chain, calibration):
    fk_cal = So101FK(chain, calibration)
    ticks = np.array([2047, 2500, 1500, 2047 + 512, 2047 - 1024, 2047], dtype=np.int32)
    out = np.zeros(6)
    np.testing.assert_allclose(fk_cal.ticks_to_rad(ticks, out), ticks_to_rad(ticks, calibration), atol=1e-12)
    # center tick is q = 0 -> the q = 0 pose
    np.testing.assert_allclose(
        fk_cal.gripper_xyz_from_ticks(np.full(6, CENTER_TICK, dtype=np.int32), np.zeros(3)),
        fk_cal.gripper_xyz(np.zeros(6), np.zeros(3)),
    )


def test_drive_mode_flips_sign(chain, calibration):
    from dataclasses import replace

    flipped = {n: replace(c, drive_mode=1) for n, c in calibration.items()}
    ticks = np.array([2047 + 400, 2047, 2047, 2047, 2047, 2047], dtype=np.int32)
    a = So101FK(chain, calibration).ticks_to_rad(ticks, np.zeros(6))
    b = So101FK(chain, flipped).ticks_to_rad(ticks, np.zeros(6))
    np.testing.assert_allclose(a, -b)
    np.testing.assert_allclose(b, ticks_to_rad(ticks, flipped))


@pytest.mark.parametrize(
    "xyz,inside",
    [
        ((0.08, 0.0, 0.1), True),  # x min boundary inclusive
        ((0.34, 0.0, 0.1), True),
        ((0.0799999, 0.0, 0.1), False),
        ((0.3400001, 0.0, 0.1), False),
        ((0.2, -0.22, 0.1), True),
        ((0.2, 0.2200001, 0.1), False),
        ((0.2, 0.0, -0.015), True),  # floor = table + 5 mm (plan §10 2026-10-06; was 0.010)
        ((0.2, 0.0, -0.0150001), False),
        ((0.2, 0.0, 0.28), True),
        ((0.2, 0.0, 0.2800001), False),
        ((np.nan, 0.0, 0.1), False),
        ((0.2, np.inf, 0.1), False),
    ],
)
def test_in_box_boundaries(fk, xyz, inside):
    assert fk.in_box(np.array(xyz), WorkspaceBox()) is inside


def test_non_z_axis_rejected(tmp_path):
    import yaml

    data = yaml.safe_load(KIN.read_text())
    data["joints"][1]["axis"] = [0.0, 1.0, 0.0]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="axes"):
        KinematicChain.from_yaml(p)


def test_joint_order_mismatch_rejected(tmp_path):
    import yaml

    data = yaml.safe_load(KIN.read_text())
    data["joints"][0], data["joints"][1] = data["joints"][1], data["joints"][0]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="joints"):
        KinematicChain.from_yaml(p)


def test_matches_mujoco_site_on_random_poses(fk):
    mujoco = pytest.importorskip("mujoco")
    if not MJCF.exists():
        pytest.skip("upstream MJCF not fetched")
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    rng = np.random.default_rng(0)
    out = np.zeros(3)
    for _ in range(25):
        q = rng.uniform(-1.5, 1.5, 6)
        for i, n in enumerate(JOINT_NAMES):
            data.qpos[model.joint(n).qposadr] = q[i]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(fk.gripper_xyz(q, out), data.site_xpos[site], atol=1e-9)

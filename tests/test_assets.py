import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from so101_rl.config import JOINT_NAMES, load_config

ASSET_DIR = Path(__file__).parent.parent / "assets" / "so101"
UPSTREAM = ASSET_DIR / "upstream"
MJCF = UPSTREAM / "so101_new_calib.xml"


@pytest.fixture(scope="module")
def meta():
    return json.loads((ASSET_DIR / "UPSTREAM.json").read_text())


@pytest.fixture(scope="module")
def kin():
    return yaml.safe_load(load_config().kinematics_path.read_text())


def test_upstream_sha_pinned(meta):
    assert len(meta["sha"]) == 40 and meta["files"]


def test_sha256_matches_upstream_json(meta):
    for rel, digest in meta["files"].items():
        data = (UPSTREAM / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, rel


def test_license_and_notice():
    assert "Apache License" in (ASSET_DIR / "LICENSE").read_text()
    assert "TheRobotStudio" in (ASSET_DIR / "NOTICE").read_text()


def test_mjcf_loads_with_six_joints():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert names == list(JOINT_NAMES)


def test_yaml_joint_order_matches_joint_names(kin):
    assert [j["name"] for j in kin["joints"]] == list(JOINT_NAMES)
    assert all(j["mjcf_joint"] == j["name"] for j in kin["joints"])
    assert kin["gripper_tip"]["pos"]


def _quat_to_mat(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _numpy_fk(kin, q):
    R = _quat_to_mat(kin["base"]["quat"])
    p = np.asarray(kin["base"]["pos"], dtype=float)
    for joint, qi in zip(kin["joints"][:5], q, strict=True):
        p = p + R @ np.asarray(joint["parent_body_pos"])
        R = R @ _quat_to_mat(joint["parent_body_quat"]) @ _rot(joint["axis"], qi)
    tip = kin["gripper_tip"]
    return p + R @ np.asarray(tip["pos"])


def test_yaml_fk_matches_mujoco(kin):
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, kin["gripper_tip"]["site"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-1.0, 1.0, 5)
        data.qpos[:5] = q
        mujoco.mj_forward(model, data)
        assert np.allclose(_numpy_fk(kin, q), data.site_xpos[sid], atol=1e-6)

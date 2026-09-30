import mujoco
import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES, load_config, ticks_to_rad
from so101_rl.sim.gripper_map import gripper_ticks_to_rad
from so101_rl.sim.model import (
    ASSETS,
    DIE_EDGE_M,
    FIXED_JAW_BODY,
    MOVING_JAW_BODY,
    SCENE_XML,
    TABLE_Z_M,
    ZONE_EDGE_M,
    jaw_collision_geom_ids,
    load_model,
)


@pytest.fixture(scope="module")
def sim():
    return load_model()


def test_model_compiles_and_paths_resolve_under_assets(sim):
    model, _ = sim
    assert SCENE_XML.is_relative_to(ASSETS)
    assert model.nu == 6 and model.nmesh > 0


def test_keyframes_exist(sim):
    model, _ = sim
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)}
    assert {"rest", "home"} <= names


def test_keyframes_consistent_with_config_placeholders(sim, calibration):
    """Until H5 replaces the placeholders, the keyframes must agree with config safe_state ticks (1 tick ~ 1.5 mrad)."""
    model, _ = sim
    cfg = load_config(calibration=calibration)
    for name, ticks in (("rest", cfg.safe_state.rest_ticks), ("home", cfg.safe_state.home_ticks)):
        q = model.key(name).qpos[:6]
        expected = ticks_to_rad(ticks, calibration)
        expected[5] = gripper_ticks_to_rad(float(ticks[5]), calibration)
        np.testing.assert_allclose(q, expected, atol=2e-3, err_msg=name)


def test_jaw_geoms_are_primitives_and_mesh_jaws_do_not_collide(sim):
    model, _ = sim
    ids = jaw_collision_geom_ids(model)
    assert len(ids) >= 4
    assert {int(model.geom_type[i]) for i in ids} <= {int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CAPSULE)}
    assert all(model.geom_contype[i] and model.geom_conaffinity[i] for i in ids)
    jaw_bodies = {model.body(FIXED_JAW_BODY).id, model.body(MOVING_JAW_BODY).id}
    mesh_jaw = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] in jaw_bodies and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    assert mesh_jaw
    for g in mesh_jaw:
        assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0


def test_both_jaws_have_primitive_geoms(sim):
    model, _ = sim
    owners = {int(model.geom_bodyid[i]) for i in jaw_collision_geom_ids(model)}
    assert owners == {model.body(FIXED_JAW_BODY).id, model.body(MOVING_JAW_BODY).id}


def test_rest_is_gravity_stable_in_sim(sim):
    model, data = sim
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
    q0 = data.qpos[:6].copy()
    data.ctrl[:6] = q0
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert np.all(np.degrees(np.abs(data.qpos[:6] - q0)) < 0.5)


def test_six_sts3215_actuators(sim):
    model, _ = sim
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert names == list(JOINT_NAMES)
    # class sts3215 sets kp=998.22, kv=2.731 (position actuator: gain kp, bias [0, -kp, -kv])
    assert np.allclose(model.actuator_gainprm[:, 0], 998.22)
    assert np.allclose(model.actuator_biasprm[:, 2], -2.731)


def test_die_zone_fiducial_and_sites(sim):
    model, data = sim
    assert model.body_mass[model.body("die").id] == pytest.approx(0.004, rel=0.05)
    np.testing.assert_allclose(model.geom("die_geom").size, DIE_EDGE_M / 2)
    zone = model.site("zone")
    np.testing.assert_allclose(zone.size[:2] * 2, ZONE_EDGE_M)
    assert model.site("fiducial").id >= 0
    for s in ("gripper_tip", "base"):
        assert model.site(s).id >= 0
    # tip comes from the same frame as the kinematics yaml / upstream gripperframe
    np.testing.assert_allclose(data.site("gripper_tip").xpos, data.site("gripperframe").xpos, atol=1e-9)
    np.testing.assert_allclose(data.site("base").xpos, 0.0, atol=1e-9)


def test_table_height_matches_constant(sim):
    model, _ = sim
    assert model.geom("table").pos[2] == pytest.approx(TABLE_Z_M)
    # the die rests on the table at the rest keyframe
    assert model.key("rest").qpos[8] == pytest.approx(TABLE_Z_M + DIE_EDGE_M / 2)


def test_die_settles_on_table(sim):
    model, _ = sim
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
    data.ctrl[:6] = data.qpos[:6]
    for _ in range(200):
        mujoco.mj_step(model, data)
    die_z = data.body("die").xpos[2]
    assert die_z == pytest.approx(TABLE_Z_M + DIE_EDGE_M / 2, abs=5e-4)


def test_upstream_files_untouched_by_scene():
    """The scene must include upstream, not copy it (the sha test in test_assets guards the bytes)."""
    text = SCENE_XML.read_text()
    assert 'file="upstream/so101_new_calib.xml"' in text

"""Gripper floor check (plan §10 2026-10-06 "Gripper floor check", 2026-10-07 "Gripper floor clearance").

* Property: no reachable configuration passes ``So101FK.in_box_gripper`` with any floor point below
  ``box.gripper_floor_z_m`` (points recomputed independently from the chain), and every configuration with a point
  below it fails.
* Mesh coverage: on random poses (moving jaw at its own random angle) the lowest floor point is never more than the
  generator's ``tolerance_m`` (0.5 mm) above the lowest EXACT collision-mesh vertex of the wrist, gripper and moving
  jaw (the points may be lower: the moving jaw is swept over its whole range).
* Config: the two keys are strict and range-checked.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_rl.config import DEFAULT_CONFIG_PATH, load_calibration, load_config
from so101_rl.safety.fk import KinematicChain, So101FK

DATA = Path(__file__).parent / "data"
KIN = DEFAULT_CONFIG_PATH.with_name("so101_kinematics.yaml")
_RAD = 2 * np.pi / 4096


@pytest.fixture(scope="module")
def setup():
    cal = load_calibration(DATA / "so101_follower.calib.json")
    cfg = load_config(calibration=cal)
    chain = KinematicChain.from_yaml(cfg.kinematics_path)
    return cfg, cal, chain, So101FK(chain, cal)


def _points_z(chain: KinematicChain, ticks) -> np.ndarray:
    """Every floor point's z, recomputed with plain numpy (drive_mode 0 on every joint of the test calibration)."""
    q = (np.asarray(ticks[:5], np.float64) - 2047) * _RAD
    a = chain.base_T.copy()
    frames = []
    for i in range(5):
        c, s = np.cos(q[i]), np.sin(q[i])
        rz = np.eye(4)
        rz[:2, :2] = [[c, -s], [s, c]]
        a = a @ chain.parent_T[i] @ rz
        frames.append(a)
    zw = chain.floor_wrist @ frames[3][2, :3] + frames[3][2, 3]
    zg = chain.floor_gripper @ frames[4][2, :3] + frames[4][2, 3]
    return np.concatenate([zw, zg])


_STATE: dict = {}


def _ticks_strategy():
    cal = load_calibration(DATA / "so101_follower.calib.json")
    cfg = load_config(calibration=cal)
    lo, hi = cfg.joints.min_ticks, cfg.joints.max_ticks
    return st.tuples(*[st.integers(int(lo[j]), int(hi[j])) for j in range(6)])


@settings(max_examples=3000, deadline=None)
@given(ticks=_ticks_strategy())
def test_no_reachable_configuration_passes_with_a_point_below_the_floor(ticks):
    if not _STATE:
        cal = load_calibration(DATA / "so101_follower.calib.json")
        cfg = load_config(calibration=cal)
        chain = KinematicChain.from_yaml(cfg.kinematics_path)
        _STATE.update(cfg=cfg, chain=chain, fk=So101FK(chain, cal))
    cfg, chain, fk = _STATE["cfg"], _STATE["chain"], _STATE["fk"]
    t = np.asarray(ticks, np.int32)
    xyz = np.zeros(3)
    zf = fk.tip_and_floor_z_from_ticks(t, xyz)
    z = _points_z(chain, t)
    assert abs(zf - z.min()) < 1e-12
    below = bool((z < cfg.box.gripper_floor_z_m - 1e-12).any())
    passes = fk.in_box_gripper(xyz, zf, cfg.box)
    if passes:
        assert not below and fk.in_box(xyz, cfg.box)
    if below:
        assert not passes


def test_the_property_is_not_vacuous(setup):
    """Random reachable configurations hit all three cases: pass, tip in the box but a point below the floor, out."""
    cfg, _, _chain, fk = setup
    rng = np.random.default_rng(3)
    lo, hi = cfg.joints.min_ticks, cfg.joints.max_ticks
    seen = {"pass": 0, "point_only": 0}
    xyz = np.zeros(3)
    for _ in range(20000):
        t = rng.integers(lo, hi + 1).astype(np.int32)
        zf = fk.tip_and_floor_z_from_ticks(t, xyz)
        if fk.in_box_gripper(xyz, zf, cfg.box):
            seen["pass"] += 1
        elif fk.in_box(xyz, cfg.box) and zf < cfg.box.gripper_floor_z_m:
            seen["point_only"] += 1
    print(seen)
    assert seen["pass"] > 100 and seen["point_only"] > 10


def test_floor_points_cover_the_exact_meshes(setup):
    from so101_rl.sim.gripper_map import gripper_ticks_to_rad
    from so101_rl.sim.model import load_model

    cfg, cal, _chain, fk = setup
    tol = float(yaml.safe_load(KIN.read_text())["floor_points"]["tolerance_m"])
    assert tol == pytest.approx(0.0005)
    model, data = load_model()
    geoms = []
    for b in ("wrist", "gripper", "moving_jaw_so101_v1"):
        bid = model.body(b).id
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 3:
                mid = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
                geoms.append((g, v.astype(np.float64)))
    rng = np.random.default_rng(0)
    lo, hi = cfg.joints.min_ticks, cfg.joints.max_ticks
    xyz = np.zeros(3)
    worst = -np.inf
    for _ in range(1500):
        t = rng.integers(lo, hi + 1).astype(np.int32)
        data.qpos[:] = 0
        data.qpos[:5] = (t[:5] - 2047) * _RAD
        data.qpos[5] = gripper_ticks_to_rad(int(t[5]), cal)
        mujoco.mj_kinematics(model, data)
        mesh_low = min(float((v @ data.geom_xmat[g].reshape(3, 3)[2] + data.geom_xpos[g][2]).min()) for g, v in geoms)
        worst = max(worst, fk.tip_and_floor_z_from_ticks(t, xyz) - mesh_low)
    print(f"points above the exact mesh: worst {worst * 1000:.3f} mm (tolerance {tol * 1000} mm)")
    assert worst <= tol + 1e-4  # 0.1 mm for directions between the generator's 8000 samples


def _variant(tmp_path, mutate) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["kinematics_path"] = str(KIN)
    mutate(raw)
    p = tmp_path / "so101.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.mark.parametrize("key", ["gripper_floor_clearance_m", "gripper_present_tolerance_m"])
def test_gripper_floor_keys_have_no_hidden_default(tmp_path, setup, key):
    cal = setup[1]
    with pytest.raises(ValueError, match=key):
        load_config(_variant(tmp_path, lambda r: r["box"].pop(key)), calibration=cal)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("gripper_floor_clearance_m", -0.001),
        ("gripper_floor_clearance_m", 0.05),
        ("gripper_floor_clearance_m", "0.001"),
        ("gripper_present_tolerance_m", -0.001),
        ("gripper_present_tolerance_m", 0.0025),  # would let a point press 1.5 mm into the table
        ("gripper_present_tolerance_m", float("nan")),
    ],
)
def test_gripper_floor_keys_are_range_checked(tmp_path, setup, key, bad):
    cal = setup[1]
    with pytest.raises(ValueError, match=key):
        load_config(_variant(tmp_path, lambda r: r["box"].update({key: bad})), calibration=cal)


def test_shipped_gripper_floor_values(setup):
    cfg = setup[0]
    assert cfg.box.gripper_floor_z_m == pytest.approx(cfg.safe_state.table_z_m + 0.001)
    assert cfg.box.gripper_present_tolerance_m == pytest.approx(0.002)
    assert cfg.box.z[0] == pytest.approx(cfg.safe_state.table_z_m + 0.005)  # the tip floor: table + 5 mm

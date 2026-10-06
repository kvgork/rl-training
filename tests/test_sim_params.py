"""FS7 SimParams / BatchedSimParams / calib/sim_params.yaml (sim-first track S2.1)."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from so101_rl.config import JOINT_NAMES
from so101_rl.sim.compare_real import MAX_RMS_PCT
from so101_rl.sim.model import load_model
from so101_rl.sim.sysid.params import (
    FRICTIONLOSS_RANGE,
    SERVO_GROUP_FIELDS,
    SIM_PARAMS_VERSION,
    BatchedSimParams,
    DrRanges,
    ServoGroup,
    SimParams,
    SimParamsMeta,
    apply_deadband,
    mjcf_sha256,
    read_sim_params_file,
    read_sim_params_yaml,
)

ROOT = Path(__file__).parent.parent
SHIPPED = ROOT / "calib" / "sim_params.yaml"


@pytest.fixture(scope="module")
def prior():
    return SimParams.from_mjcf(load_model()[0])


def _meta(heldout=None, version=1):
    return SimParamsMeta(version=version, date="2026-10-06", git_sha="x", mjcf_sha256="y", excitation_run="r",
                         fit_rms_pct={}, heldout_rms_pct=heldout, spread={}, backend="cpu")


def test_mjcf_roundtrip(prior):
    assert np.allclose(prior.kp, 998.22) and np.allclose(prior.kv, 2.731) and np.allclose(prior.forcerange, 3.35)
    assert prior.delay_ticks == 0 and not prior.deadband_ticks.any()
    other = replace(prior.scaled(kp=0.7, kv=1.4, frictionloss=2.0, armature=1.3, die_mass_kg=2.0),
                    forcerange=prior.forcerange * 0.9, die_friction=0.6, table_friction=0.5)
    model, _ = load_model()
    inertia0 = model.body("die").inertia.copy()
    other.apply_cpu(model)
    back = SimParams.from_mjcf(model)
    assert replace(back, delay_ticks=other.delay_ticks).close_to(other)
    np.testing.assert_allclose(model.body("die").inertia, inertia0 * 2.0)  # inertia follows the mass


def test_from_mjcf_refuses_a_deenergised_model(calibration, cfg):
    from so101_rl.sim.mujoco_bus import MujocoBus

    model, data = load_model()
    MujocoBus(model, data, calibration, cfg)  # starts limp: gains zeroed
    with pytest.raises(ValueError, match="de-energised"):
        SimParams.from_mjcf(model)


def test_yaml_roundtrip_and_version_check(prior, tmp_path):
    p = prior.scaled(kp=0.8)
    p = replace(p, delay_ticks=2, deadband_ticks=np.array([0, 1, 2, 0, 0, 3]))
    path = tmp_path / "sp.yaml"
    held = dict.fromkeys(JOINT_NAMES, 0.8)
    p.to_yaml(path, meta=_meta(held))
    q, meta = read_sim_params_yaml(path)
    assert q.close_to(p) and meta.heldout_rms_pct == held and meta.version == 1
    assert SimParams.from_yaml(path).close_to(p)

    doc = yaml.safe_load(path.read_text())
    doc["version"] = SIM_PARAMS_VERSION + 1
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match=f"schema version {SIM_PARAMS_VERSION + 1}"):
        SimParams.from_yaml(path, allow_unverified=True)


def test_schema_1_files_are_refused_with_a_regenerate_hint(prior, tmp_path):
    """Schema 2 (2026-10-06) adds servo_group, dr and meta.unidentified; a schema-1 file is not read silently."""
    assert SIM_PARAMS_VERSION == 2
    path = tmp_path / "sp.yaml"
    prior.to_yaml(path, meta=_meta(dict.fromkeys(JOINT_NAMES, 0.5)))
    doc = yaml.safe_load(path.read_text())
    doc["version"] = 1
    for k in ("servo_group", "dr"):
        doc.pop(k)
    doc["meta"].pop("unidentified")
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="schema version 1 .*--write-prior"):
        read_sim_params_yaml(path, allow_unverified=True)

@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d["params"].pop("kp"), "missing keys"),
        (lambda d: d["params"].update(extra=1), "unknown keys"),
        (lambda d: d["params"].update(kp=[1.0] * 5), "shape"),
        (lambda d: d["params"].update(kv=[-1.0] * 6), "kv"),
        (lambda d: d["params"].update(delay_ticks=7), "delay_ticks"),
        (lambda d: d["params"].update(deadband_ticks=[0.5] * 6), "integers"),
        (lambda d: d["meta"].pop("spread"), "missing keys"),
        (lambda d: d["meta"].pop("unidentified"), "missing keys"),
        (lambda d: d.update(other=1), "top-level"),
        (lambda d: d.pop("servo_group"), "top-level"),
        (lambda d: d.pop("dr"), "top-level"),
        (lambda d: d["servo_group"].update(scale=[1.1] * 6), "ratio group"),  # no longer reproduces kp, kv, armature
        (lambda d: d["servo_group"]["prior"].pop("armature"), "keys"),
        (lambda d: d["servo_group"].update(scale=[0.0] * 6), "scale"),
        (lambda d: d["dr"].update(servo_group_rel=1.0), "servo_group_rel"),
        (lambda d: d["dr"].update(frictionloss_range=[0.2, 0.1]), "frictionloss_range"),
        (lambda d: d["dr"].update(frictionloss_source=" "), "source"),
    ],
)
def test_yaml_is_strict(prior, tmp_path, mutate, match):
    path = tmp_path / "sp.yaml"
    prior.to_yaml(path, meta=_meta(dict.fromkeys(JOINT_NAMES, 1.0)))
    doc = yaml.safe_load(path.read_text())
    mutate(doc)
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises((ValueError, TypeError), match=match):
        read_sim_params_yaml(path, allow_unverified=True)


def test_unverified_refused_by_default(prior, tmp_path):
    path = tmp_path / "sp.yaml"
    prior.to_yaml(path, meta=_meta(None, version=0))
    with pytest.raises(ValueError, match="unverified"):
        SimParams.from_yaml(path)
    assert SimParams.from_yaml(path, allow_unverified=True).close_to(prior)
    # above the 1 % gate (plan §10 2026-10-06; was 5 %) on one joint, or a joint missing: also unverified
    assert MAX_RMS_PCT == 1.0
    prior.to_yaml(path, meta=_meta(dict.fromkeys(JOINT_NAMES, 0.5) | {"elbow_flex": 1.01}))
    with pytest.raises(ValueError, match="elbow_flex"):
        SimParams.from_yaml(path)
    prior.to_yaml(path, meta=_meta({n: 0.5 for n in JOINT_NAMES[:5]}))
    with pytest.raises(ValueError, match="gripper"):
        SimParams.from_yaml(path)
    prior.to_yaml(path, meta=_meta(dict.fromkeys(JOINT_NAMES, 1.0)))
    assert SimParams.from_yaml(path).close_to(prior)  # exactly at the gate passes
    prior.to_yaml(path, meta=_meta(dict.fromkeys(JOINT_NAMES, 4.9)))
    with pytest.raises(ValueError, match="unverified"):  # what the old 5 % gate accepted is refused now
        SimParams.from_yaml(path)


def test_shipped_v0_is_the_mjcf_prior(prior):
    """calib/sim_params.yaml release 0 = the MJCF prior, unverified, so every consumer must opt in."""
    with pytest.raises(ValueError, match="null"):
        SimParams.from_yaml(SHIPPED)
    p, meta = read_sim_params_yaml(SHIPPED, allow_unverified=True)
    assert p.close_to(prior)
    assert meta.version == 0 and meta.heldout_rms_pct is None and meta.backend == "mjcf-prior"
    assert len(meta.mjcf_sha256) == 64 and len(mjcf_sha256()) == 64
    assert meta.mjcf_sha256 == mjcf_sha256()  # regenerated for the current scene
    f = read_sim_params_file(SHIPPED, allow_unverified=True)
    np.testing.assert_array_equal(f.servo_group.scale, np.ones(6))  # the prior is the group's reference
    for k in SERVO_GROUP_FIELDS:
        np.testing.assert_array_equal(f.servo_group.prior[k], getattr(prior, k))
    assert f.dr == DrRanges() and f.dr.servo_group_rel == 0.40 and f.dr.frictionloss_range == FRICTIONLOSS_RANGE
    assert "frictionloss" in meta.unidentified  # never read as a precise value


def test_servo_group_roundtrip_and_semantics(prior, tmp_path):
    """kp, kv, armature = prior x scale per joint (one ratio group, plan §10 2026-10-06); the ratios are the
    prior's; the yaml records the group and the DR ranges."""
    s = np.array([0.6, 0.7, 0.8, 1.25, 1.4, 0.75])
    g = ServoGroup.identity(prior)
    fitted = replace(g.apply(prior, s), delay_ticks=3)
    np.testing.assert_allclose(fitted.kv / fitted.kp, prior.kv / prior.kp)
    np.testing.assert_allclose(fitted.armature / fitted.kp, prior.armature / prior.kp)
    np.testing.assert_array_equal(fitted.frictionloss, prior.frictionloss)
    grp = ServoGroup.from_fit(fitted, prior)
    np.testing.assert_allclose(grp.scale, s)
    path = tmp_path / "sp.yaml"
    dr = DrRanges(servo_group_rel=0.4, frictionloss_range=(0.0, 0.21), frictionloss_source="test")
    fitted.to_yaml(path, meta=_meta(dict.fromkeys(JOINT_NAMES, 0.3)), servo_group=grp, dr=dr)
    f = read_sim_params_file(path)
    assert f.params.close_to(fitted) and f.dr == dr
    np.testing.assert_allclose(f.servo_group.scale, s)
    lo, hi = f.dr.servo_group_bounds(f.servo_group.scale)
    np.testing.assert_allclose(lo, 0.6 * s) and np.testing.assert_allclose(hi, 1.4 * s)
    with pytest.raises(ValueError, match="ratio group"):  # kp alone changed: not a group member any more
        ServoGroup.from_fit(fitted.scaled(kp=1.1), prior)
    with pytest.raises(ValueError, match="ratio group"):
        fitted.to_yaml(path, meta=_meta(None), servo_group=g)  # identity around the prior does not reproduce it


def test_gainprm_biasprm_mapping(prior):
    """Direction check through the physics: doubling kp halves the gravity sag of shoulder_lift (sag ~ tau_g / kp)."""
    def sag(p: SimParams) -> float:
        model, data = load_model()
        p.apply_cpu(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_step(model, data, 510)  # 1 s
        j = model.joint("shoulder_lift")
        return abs(float(data.qpos[j.qposadr[0]] - data.ctrl[model.actuator("shoulder_lift").id]))

    s1, s2 = sag(prior), sag(prior.scaled(kp=2.0))
    assert s1 > 1e-5
    assert s2 == pytest.approx(s1 / 2.0, rel=0.15)

    model, _ = load_model()
    p = prior.scaled(kp=1.5, kv=0.5)
    p.apply_cpu(model)
    gain, bias = BatchedSimParams.broadcast(p, 2, "cpu").to_gainprm_biasprm()
    act = [model.actuator(n).id for n in JOINT_NAMES]
    np.testing.assert_allclose(gain[1, :, 0].numpy(), model.actuator_gainprm[act, 0], rtol=1e-6)
    np.testing.assert_allclose(bias[1, :, :3].numpy(), model.actuator_biasprm[act, :3], rtol=1e-6)


def test_vector_roundtrip(prior):
    names = ("kp", "kv", "delay_ticks", "die_mass_kg")
    v = prior.as_vector(names)
    assert v.shape == (6 + 6 + 1 + 1,)
    v2 = v.copy()
    v2[:6] *= 0.5
    v2[12] = 3.0
    q = SimParams.from_vector(v2, names, base=prior)
    np.testing.assert_allclose(q.kp, prior.kp * 0.5)
    assert q.delay_ticks == 3 and q.die_mass_kg == prior.die_mass_kg
    with pytest.raises(ValueError, match="does not match"):
        SimParams.from_vector(v[:-1], names, base=prior)
    with pytest.raises(KeyError):
        prior.as_vector(("nope",))


def test_arrays_are_read_only(prior):
    with pytest.raises(ValueError):
        prior.kp[0] = 1.0


def test_batched_stack_and_world(prior):
    ps = [prior, prior.scaled(kp=0.5), replace(prior, delay_ticks=4, deadband_ticks=np.full(6, 2))]
    b = BatchedSimParams.stack(ps, "cpu")
    assert b.kp.shape == (3, 6) and b.die_mass.shape == (3,) and b.delay_ticks.dtype.is_floating_point is False
    for i, p in enumerate(ps):
        assert b.world(i).close_to(p, rtol=1e-6)
    assert BatchedSimParams.broadcast(prior, 4, "cpu").world(3).close_to(prior, rtol=1e-6)


def test_apply_deadband():
    t = np.array([[0] * 6, [2] * 6, [5] * 6, [3] * 6, [-4] * 6])
    out = apply_deadband(t, 2)
    assert out[:, 0].tolist() == [0, 0, 3, 3, -2]
    np.testing.assert_array_equal(apply_deadband(t, 0), t)
    batched = apply_deadband(np.stack([t, t]), np.array([[0] * 6, [2] * 6]))
    np.testing.assert_array_equal(batched[0], t)
    np.testing.assert_array_equal(batched[1], out)


@pytest.mark.gpu
def test_batched_params_drive_warpsim(prior):
    """BatchedSimParams is the FS2 set_world_params input: the per-world kp lands in the model."""
    import torch

    from so101_rl.config import load_config
    from so101_rl.sim.warp_model import WarpSim

    model, _ = load_model()
    with WarpSim(model, load_config().sim_warp, 2) as ws:
        b = BatchedSimParams.stack([prior, prior.scaled(kp=0.3)], ws.device)
        ws.set_world_params(b)
        gain = ws._p["actuator_gainprm"][:, [model.actuator(n).id for n in JOINT_NAMES], 0]
        torch.testing.assert_close(gain.cpu(), b.kp.cpu())


def test_sysid_package_import_is_lazy():
    """Like so101_rl.sim: importing the sysid package (and its params module) loads neither mujoco, torch nor warp."""
    import subprocess
    import sys

    code = (
        "import sys, so101_rl.sim.sysid, so101_rl.sim.sysid.params as p; "
        "bad = [m for m in ('mujoco', 'torch', 'warp', 'mujoco_warp') if m in sys.modules]; "
        "assert not bad, bad; assert p.SimParams"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

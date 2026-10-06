"""Sim tray check (sim track S1.3, D-TRAY): the start-well tray returns a dropped die; without it, it does not."""

import json

import numpy as np

from so101_rl.sim.model import load_model
from so101_rl.sim.tray_check import PASS_RATE, main, run_tray_check, tray_geom_ids


def test_sim_tray_returns_die():
    rep = run_tray_check(20, seed=1)
    assert rep.returned >= 18, rep.final_dist_m
    assert rep.max_final_speed_m_s < 0.05  # settled


def test_tray_check_fails_without_tray(tmp_path, capsys):
    rep = run_tray_check(20, seed=1, with_tray=False)
    assert rep.return_rate < 0.5
    assert not rep.passed
    assert main(["--drops", "10", "--no-tray", "--out", str(tmp_path)]) == 1
    assert "FAIL" in capsys.readouterr().out
    (path,) = tmp_path.glob("*.json")
    doc = json.loads(path.read_text())
    assert doc["passed"] is False and doc["with_tray"] is False and doc["pass_rate"] == PASS_RATE


def test_tray_collides_with_die_only():
    """Tray bits: die yes; jaw primitives, arm meshes no (scene comment: the S0 parity gate sweeps the rest-pose
    jaws through the tray area)."""
    model, _ = load_model()
    tray = tray_geom_ids(model)
    assert len(tray) == 8

    def collide(a, b):
        return bool(model.geom_contype[a] & model.geom_conaffinity[b]) or bool(
            model.geom_contype[b] & model.geom_conaffinity[a]
        )

    die = model.geom("die_geom").id
    others = [g for g in range(model.ngeom) if g != die and g not in set(tray.tolist())]
    for t in tray:
        assert collide(t, die)
        assert not any(collide(t, g) for g in others if model.geom_contype[g] or model.geom_conaffinity[g])
        assert model.geom_priority[t] > model.geom_priority[die]


def test_rest_and_home_keyframes_clear_of_tray():
    import mujoco

    model, data = load_model()
    tray = set(tray_geom_ids(model).tolist())
    for key in ("rest", "home"):
        mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
        mujoco.mj_forward(model, data)
        hits = [i for i in range(data.ncon) if {int(data.contact[i].geom1), int(data.contact[i].geom2)} & tray]
        assert not hits, key
    assert np.isfinite(data.qpos).all()

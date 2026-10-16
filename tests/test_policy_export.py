"""Policy export (MVP M4, FS10): plain-dict export, load_policy callable, version / layout refusals."""

import numpy as np
import pytest
import torch

from so101_rl.rl import eval_sim, train_sim_sac
from so101_rl.rl.policy_export import EXPORT_VERSION, export_policy, load_policy
from so101_rl.rl.toy_batched_env import ToyBatchedReachEnv


@pytest.fixture
def toy_ckpt(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_sim, "build_env", lambda task, n, **kw: ToyBatchedReachEnv(n, seed=kw["seed"]))
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: 50.0)
    cfg = train_sim_sac.load_sim_sac_config({
        "task": "reach", "nworld": 16, "budget_min": 5.0, "seed": 0, "dr_max_stage": 0, "eval_every_steps": 16 * 20,
        "eval_episodes": 16, "log_every_steps": 16 * 10, "max_env_steps": 16 * 20, "batch_size": 64,
        "grad_steps_per_env_step": 1, "warmup_steps": 2, "replay_capacity": 1000, "tensorboard": False,
    })  # fmt: skip
    return train_sim_sac.run_training(cfg, tmp_path / "run").last_ckpt


def test_roundtrip(toy_ckpt, tmp_path):
    out = export_policy(toy_ckpt, tmp_path / "policy_export.pt")
    pol = load_policy(out)
    d = eval_sim.load_checkpoint(toy_ckpt)
    actor = eval_sim.actor_from_checkpoint(d)
    x = np.random.default_rng(0).normal(size=(7, pol.obs_dim)).astype(np.float32)
    a = pol(x)
    assert a.dtype == np.float32 and a.shape == (7, pol.act_dim) and np.abs(a).max() <= 1.0
    np.testing.assert_allclose(a, actor(torch.from_numpy(x)).detach().numpy(), rtol=1e-6, atol=1e-6)
    assert pol(x[0]).shape == (pol.act_dim,)
    assert pol.meta["task_name"] == "reach" and pol.meta["export_version"] == EXPORT_VERSION
    assert "calibration_sha256" in pol.meta  # None for the toy env; the sim env records the file's sha256
    raw = torch.load(out, weights_only=True)  # a plain dict: no pickled classes
    assert isinstance(raw, dict) and "actor" in raw


def test_refuses_wrong_layout_version(toy_ckpt, tmp_path):
    out = export_policy(toy_ckpt, tmp_path / "p.pt")
    d = torch.load(out, weights_only=True)
    for key, val in (("obs_layout_version", 99), ("export_version", 99), ("format", "x")):
        torch.save(dict(d, **{key: val}), tmp_path / "bad.pt")
        with pytest.raises(ValueError):
            load_policy(tmp_path / "bad.pt")


def test_refuses_bad_inputs(toy_ckpt, tmp_path):
    pol = load_policy(export_policy(toy_ckpt, tmp_path / "p.pt"))
    with pytest.raises(ValueError):
        pol(np.zeros((2, pol.obs_dim + 1), np.float32))
    with pytest.raises(ValueError):
        pol(np.full((pol.obs_dim,), np.nan, np.float32))


def test_die_export_records_its_task_and_is_never_deployed_as_reach(monkeypatch, tmp_path):
    """plans/die-to-zone-plan.md D1: a die_to_zone export says so (task_name, 29-wide input) and the deploy check
    refuses it for reach."""
    from so101_rl.env.tasks import DIE_TO_ZONE, REACH
    from so101_rl.rl.deploy_policy import Refused, check_policy
    from so101_rl.rl.toy_batched_env import ToyBatchedDieShapedEnv

    monkeypatch.setattr(eval_sim, "build_env", lambda task, n, **kw: ToyBatchedDieShapedEnv(n, seed=kw["seed"]))
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: 50.0)
    cfg = train_sim_sac.load_sim_sac_config({
        "task": "die_to_zone", "nworld": 16, "budget_min": 5.0, "seed": 0, "dr_max_stage": 0,
        "eval_every_steps": 16 * 20, "eval_episodes": 16, "log_every_steps": 16 * 10, "max_env_steps": 16 * 20,
        "batch_size": 64, "grad_steps_per_env_step": 1, "warmup_steps": 2, "replay_capacity": 1000,
        "tensorboard": False, "allow_unverified_sim_params": True,
    })  # fmt: skip
    ckpt = train_sim_sac.run_training(cfg, tmp_path / "run").last_ckpt
    pol = load_policy(export_policy(ckpt, tmp_path / "die.pt"))
    assert pol.meta["task_name"] == "die_to_zone" and pol.obs_dim == 29 and pol.meta["success_radius_mm"] is None
    with pytest.raises(Refused, match="task"):
        check_policy(pol, REACH)
    with pytest.raises(Refused, match="dims"):  # right task, but the toy actor has 2 actions, not 6
        check_policy(pol, DIE_TO_ZONE)


def test_cli(toy_ckpt, tmp_path):
    from so101_rl.rl.policy_export import main

    assert main(["--ckpt", str(toy_ckpt), "--out", str(tmp_path / "e.pt")]) == 0
    assert (tmp_path / "e.pt").exists()

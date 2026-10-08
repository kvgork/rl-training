"""Sim eval + numeric exit gate (MVP M4, sim plan S4.3 step 3). CPU: toy env through the one env factory."""

import json
from pathlib import Path

import pytest

from so101_rl.rl import eval_sim, train_sim_sac
from so101_rl.rl.eval_sim import gate_verdict, wilson_interval
from so101_rl.rl.toy_batched_env import ToyBatchedReachEnv


def test_wilson():
    lo, hi = wilson_interval(30, 30)
    assert lo == pytest.approx(0.8865, abs=1e-4) and hi == 1.0  # plan: 30/30 -> lower bound 88.6 %
    lo, hi = wilson_interval(950, 1000)
    assert lo == pytest.approx(0.9346, abs=1e-3) and hi == pytest.approx(0.9613, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_gate_fails_at_94pct_and_passes_at_97():
    ok, why = gate_verdict(940, 1000, 4)
    assert not ok and any("success" in w for w in why)
    ok, why = gate_verdict(950, 1000, 4)  # 95.0 % but the Wilson lower bound 93.46 % passes too
    assert ok, why
    assert gate_verdict(970, 1000, 4)[0]
    assert not gate_verdict(970, 1000, 3)[0]  # not at the final DR stage
    assert not gate_verdict(999, 999, 4)[0]  # fewer than 1000 episodes
    assert gate_verdict(1900, 2000, 4)[0]  # 95 % over 2000 episodes: Wilson lower bound 93.9 %


@pytest.fixture
def toy(monkeypatch):
    def build(task, nworld, *, stage, seed, calibration=None, sim_params="", device=None):
        return ToyBatchedReachEnv(nworld, seed=seed)

    monkeypatch.setattr(eval_sim, "build_env", build)
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: 50.0)


@pytest.fixture
def toy_ckpt(toy, tmp_path):
    cfg = train_sim_sac.load_sim_sac_config({
        "task": "reach", "nworld": 32, "budget_min": 5.0, "seed": 0, "dr_max_stage": 0, "eval_every_steps": 32 * 30,
        "eval_episodes": 32, "log_every_steps": 32 * 10, "max_env_steps": 32 * 30, "batch_size": 128,
        "grad_steps_per_env_step": 1, "warmup_steps": 5, "replay_capacity": 10_000, "tensorboard": False,
    })  # fmt: skip
    return train_sim_sac.run_training(cfg, tmp_path / "run").last_ckpt


def test_evaluate_writes_eval_json(toy_ckpt, tmp_path):
    res = eval_sim.evaluate(toy_ckpt, "reach", episodes=100, dr_stage=4, out_dir=tmp_path / "ev")
    doc = json.loads(res.eval_json.read_text())
    assert doc["episodes"] == res.episodes == 100
    assert doc["success_rate"] == res.success_rate and doc["wilson95"] == [res.wilson_lo, res.wilson_hi]
    assert doc["gate"]["numeric_passed"] is False  # 100 episodes are never enough
    assert doc["camera_pose_method"] == "none" and doc["ckpt_sha256"]
    assert (tmp_path / "ev" / "review_checklist.md").read_text().startswith("# Reach exit gate")
    assert res.renders == ()


def test_run_episodes_counts_each_world_once():
    """A policy that succeeds fast must not be over-counted: exactly `episodes` episodes, one per world per round."""
    env = ToyBatchedReachEnv(16, seed=0, success_radius=10.0)  # every step succeeds
    st = eval_sim.run_episodes(env, lambda x: x[:, :2] * 0, 40)
    assert st["episodes"] == 40 and st["successes"] == 40 and st["mean_length"] == 1.0


def test_cli_gate_exit_codes(toy_ckpt, tmp_path, monkeypatch):
    assert eval_sim.main(["--ckpt", str(toy_ckpt), "--episodes", "50", "--out", str(tmp_path / "a")]) == 0
    assert eval_sim.main(["--ckpt", str(toy_ckpt), "--episodes", "50", "--out", str(tmp_path / "b"), "--gate"]) == 1


def test_refuses_other_tasks_and_bad_files(toy_ckpt, tmp_path):
    with pytest.raises(ValueError):
        eval_sim.evaluate(toy_ckpt, "die_to_zone", episodes=10, dr_stage=4, out_dir=tmp_path)
    bad = tmp_path / "x.pt"
    bad.write_text("nope")
    with pytest.raises(ValueError):
        eval_sim.load_checkpoint(bad)
    with pytest.raises(ValueError):
        eval_sim.load_checkpoint(Path(__file__))

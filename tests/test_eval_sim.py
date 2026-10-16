"""Sim eval + numeric exit gate (MVP M4, sim plan S4.3 step 3). CPU: toy env through the one env factory."""

import json
from pathlib import Path

import pytest
import torch

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
    def build(task, nworld, *, stage, seed, calibration=None, sim_params="", device=None, success_radius_mm=None):
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


def test_calibration_change_since_training_is_flagged(toy_ckpt, tmp_path, monkeypatch):
    """The checkpoint carries the training env's calibration sha256; eval under another file warns and says so."""
    import torch

    d = torch.load(toy_ckpt, weights_only=True)
    assert d["calibration_sha256"] is None  # the toy env records none
    d["calibration_sha256"] = "a" * 64
    torch.save(d, toy_ckpt)
    for env_sha, changed in (("a" * 64, False), ("b" * 64, True), (None, None)):

        def build(task, nworld, *, stage, seed, calibration=None, sim_params="", device=None, success_radius_mm=None, sha=env_sha):
            env = ToyBatchedReachEnv(nworld, seed=seed)
            env.metadata = {"calibration_sha256": sha}
            return env

        monkeypatch.setattr(eval_sim, "build_env", build)
        if changed:
            with pytest.warns(UserWarning, match="calibration file changed"):
                res = eval_sim.evaluate(toy_ckpt, "reach", episodes=8, dr_stage=0, out_dir=tmp_path / "ev")
        else:
            res = eval_sim.evaluate(toy_ckpt, "reach", episodes=8, dr_stage=0, out_dir=tmp_path / "ev")
        assert json.loads(res.eval_json.read_text())["calibration_changed_since_training"] is changed


def test_success_radius_reaches_the_env_through_the_checkpoint(toy, tmp_path, monkeypatch):
    """A run's ``success_radius_mm`` (plan §10 2026-10-14) is passed to every env the trainer and ``evaluate`` build,
    and lands in the export's meta."""
    from so101_rl.rl.policy_export import export_policy

    seen: list[float | None] = []

    def build(task, nworld, *, stage, seed, calibration=None, sim_params="", device=None, success_radius_mm=None):
        seen.append(success_radius_mm)
        return ToyBatchedReachEnv(nworld, seed=seed)

    monkeypatch.setattr(eval_sim, "build_env", build)
    cfg = train_sim_sac.load_sim_sac_config({
        "task": "reach", "nworld": 32, "budget_min": 5.0, "seed": 0, "dr_max_stage": 0, "eval_every_steps": 32 * 30,
        "eval_episodes": 32, "log_every_steps": 32 * 10, "max_env_steps": 32 * 30, "batch_size": 128,
        "grad_steps_per_env_step": 1, "warmup_steps": 5, "replay_capacity": 10_000, "tensorboard": False,
        "success_radius_mm": 5.0,
    })  # fmt: skip
    ckpt = train_sim_sac.run_training(cfg, tmp_path / "run").last_ckpt
    eval_sim.evaluate(ckpt, "reach", episodes=10, dr_stage=0, out_dir=tmp_path / "ev")
    assert seen and set(seen) == {5.0}
    export_policy(ckpt, tmp_path / "p.pt")
    assert torch.load(tmp_path / "p.pt", weights_only=False)["success_radius_mm"] == 5.0


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


def test_render_env_freezes_return_and_flags_after_end():
    """_RenderEnv: no auto-reset, return/flags constant once a world has ended (video overlay = episode return)."""
    import types

    import torch

    from so101_rl.rl.eval_sim import _RenderEnv

    class E:
        nworld, device, dr_stage = 2, "cpu", 0
        task = types.SimpleNamespace(success_radius_m=0.015)
        goal = torch.zeros(2, 3)

        def __init__(self):
            self.t = 0
            self.resets = 0

        def reset(self, *a, **k):
            self.resets += 1
            return (torch.zeros(2, 4),)

        def step(self, action):
            self.t += 1
            self.reset(torch.ones(2, dtype=torch.bool))  # the real env auto-resets inside step
            success = torch.tensor([self.t >= 3, False])
            return types.SimpleNamespace(
                reward=torch.full((2,), -1.0), obs=torch.zeros(2, 4),
                terminated=success, truncated=torch.tensor([False, self.t >= 5]),
                info={"success": success, "tripped": torch.zeros(2, dtype=torch.bool)},
            )

    e = E()
    r = _RenderEnv(e)
    r.reset()
    base = e.resets
    rets = []
    for _ in range(6):
        r.step(torch.zeros(2, 1))
        rets.append(r._ret.clone())
    assert e.resets == base  # suppressed during step
    assert "reset" not in e.__dict__
    assert [float(x[0]) for x in rets] == [-1, -2, -3, -3, -3, -3]  # constant after success
    assert [float(x[1]) for x in rets] == [-1, -2, -3, -4, -5, -5]  # constant after timeout
    assert bool(r._succ[0]) and not bool(r._succ[1]) and bool(r._trunc[1]) and not bool(r._trunc[0])


# ------------------------------------------------------------------------------------------------ die_to_zone (D1)
def test_task_spec_and_reach_only_radius():
    """``die_to_zone`` is a sim task; the 2026-10-14 radius override stays reach-only (refused before any CUDA use)."""
    import dataclasses

    from so101_rl.env.tasks import DIE_TO_ZONE, REACH

    assert eval_sim.TASKS == ("reach", "die_to_zone")
    assert eval_sim.task_spec("reach") is REACH and eval_sim.task_spec("die_to_zone") is DIE_TO_ZONE
    assert eval_sim.task_spec("reach", 5.0) == dataclasses.replace(REACH, success_radius_m=0.005)
    with pytest.raises(ValueError, match="reach-only"):
        eval_sim.task_spec("die_to_zone", 5.0)
    with pytest.raises(ValueError, match="reach-only"):
        eval_sim.build_env("die_to_zone", 4, stage=0, seed=0, success_radius_mm=5.0)
    with pytest.raises(ValueError, match="not a sim task"):
        eval_sim.build_env("pick", 4, stage=0, seed=0)


@pytest.fixture
def toy_die_ckpt(monkeypatch, tmp_path):
    """A die_to_zone checkpoint trained on the die-shaped toy env (policy input 29) with the S4.6 waiver set."""
    from so101_rl.rl.toy_batched_env import ToyBatchedDieShapedEnv

    seen: list[tuple[str, float | None]] = []

    def build(task, nworld, *, stage, seed, calibration=None, sim_params="", device=None, success_radius_mm=None):
        seen.append((task, success_radius_mm))
        env = ToyBatchedDieShapedEnv(nworld, seed=seed)
        env.metadata = {"camera_pose_method": "arm-shape"}
        return env

    monkeypatch.setattr(eval_sim, "build_env", build)
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: 50.0)
    cfg = train_sim_sac.load_sim_sac_config({
        "task": "die_to_zone", "nworld": 32, "budget_min": 5.0, "seed": 0, "dr_max_stage": 0,
        "eval_every_steps": 32 * 30, "eval_episodes": 32, "log_every_steps": 32 * 10, "max_env_steps": 32 * 30,
        "batch_size": 128, "grad_steps_per_env_step": 1, "warmup_steps": 5, "replay_capacity": 10_000,
        "tensorboard": False, "allow_unverified_sim_params": True,
    })  # fmt: skip
    return train_sim_sac.run_training(cfg, tmp_path / "run").last_ckpt, seen


def test_evaluate_die_writes_eval_json_and_die_checklist(toy_die_ckpt, tmp_path):
    ckpt, seen = toy_die_ckpt
    res = eval_sim.evaluate(ckpt, "die_to_zone", episodes=40, dr_stage=4, out_dir=tmp_path / "ev")
    doc = json.loads(res.eval_json.read_text())
    assert doc["task"] == "die_to_zone" and doc["episodes"] == 40 and doc["gate"]["numeric_passed"] is False
    assert doc["camera_pose_method"] == "arm-shape"  # the die env's camera pose, not reach's "none"
    assert (tmp_path / "ev" / "review_checklist.md").read_text().startswith("# Die-to-zone exit gate")
    assert {t for t, _ in seen} == {"die_to_zone"} and {r for _, r in seen} == {None}
    with pytest.raises(ValueError, match="checkpoint task"):
        eval_sim.evaluate(ckpt, "reach", episodes=10, dr_stage=4, out_dir=tmp_path / "ev2")

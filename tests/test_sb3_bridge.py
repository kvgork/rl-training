"""``rl.sb3_bridge`` / ``rl.train_sac_her`` (sim plan S5.2): the sim actor in SB3 SAC + HER, the warm-up, rollback,
export round trip, and the fine-tuning entry point on the CPU reference rig."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import HerReplayBuffer

from so101_rl.rl.fast_sac import Actor
from so101_rl.rl.policy_export import EXPORT_VERSION, load_policy
from so101_rl.rl.sb3_bridge import (
    FinetuneSAC,
    GoalPolicyInput,
    actor_policy,
    export_sb3_actor,
    load_actor_into_sb3,
    rollback_due,  # fmt: skip
)

ROOT = Path(__file__).parent.parent
M6_POLICY = ROOT / "runs" / "lab" / "fastsac_sim-reach-20261008-130617-58ef" / "policy.pt"
HIDDEN = [16, 16]


class _GoalEnv(gym.Env):
    """A tiny goal env with the reach shapes (obs 21, goals 3, action 6): the tip moves by the action."""

    observation_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (21,), np.float32),
        "achieved_goal": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        "desired_goal": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
    })  # fmt: skip
    action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tip = np.zeros(3, np.float32)
        self.goal = self.np_random.uniform(-1, 1, 3).astype(np.float32)
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        o = np.zeros(21, np.float32)
        o[:3] = self.tip
        return {"observation": o, "achieved_goal": self.tip.copy(), "desired_goal": self.goal.copy()}

    def compute_reward(self, achieved, desired, info):
        return -np.linalg.norm(achieved - desired, axis=-1).astype(np.float32)

    def step(self, a):
        self.tip = self.tip + 0.1 * np.asarray(a[:3], np.float32)
        self.t += 1
        obs = self._obs()
        r = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {}))
        return obs, r, False, self.t >= 10, {}


def _export(path: Path) -> Path:
    from so101_rl.env.obs import OBS_LAYOUT_VERSION

    actor = Actor(24, 6, tuple(HIDDEN), (-5.0, 2.0))
    torch.save({"format": "so101_rl.policy_export", "export_version": EXPORT_VERSION,
                "obs_layout_version": OBS_LAYOUT_VERSION, "task_name": "reach", "obs_dim": 24, "act_dim": 6,
                "actor_hidden": HIDDEN, "log_std_bounds": [-5.0, 2.0],
                "actor": {k: v.detach().clone() for k, v in actor.state_dict().items()}}, path)  # fmt: skip
    return path


def _model(env, *, warmup=0, learning_starts=0):
    return FinetuneSAC(
        "MultiInputPolicy", env, policy_kwargs={"net_arch": HIDDEN, "features_extractor_class": GoalPolicyInput},
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True},
        learning_starts=learning_starts, batch_size=16, train_freq=(1, "episode"), gradient_steps=-1,
        warmup_frozen_actor=warmup, buffer_size=1000, device="cpu", seed=0,
    )  # fmt: skip


def _obs_batch(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return {"observation": rng.standard_normal((n, 21)).astype(np.float32),
            "achieved_goal": rng.standard_normal((n, 3)).astype(np.float32),
            "desired_goal": rng.standard_normal((n, 3)).astype(np.float32)}  # fmt: skip


def test_actor_copy_exact(tmp_path):
    """The SB3 actor gives the export's deterministic action on the same Dict obs (observation ++ desired_goal)."""
    pol = load_policy(_export(tmp_path / "p.pt"))
    model = _model(_GoalEnv())
    load_actor_into_sb3(model, pol)
    obs = _obs_batch()
    sb3, _ = model.predict(obs, deterministic=True)
    ours = pol(np.concatenate([obs["observation"], obs["desired_goal"]], axis=1))
    np.testing.assert_allclose(sb3, ours, atol=1e-6)


def test_actor_layout_mismatch_is_refused(tmp_path):
    pol = load_policy(_export(tmp_path / "p.pt"))
    model = FinetuneSAC("MultiInputPolicy", _GoalEnv(), buffer_size=10, device="cpu",
                        policy_kwargs={"net_arch": [32, 32], "features_extractor_class": GoalPolicyInput})  # fmt: skip
    with pytest.raises(ValueError, match="shapes differ"):
        load_actor_into_sb3(model, pol)


def test_frozen_warmup_keeps_actor_and_entropy_and_never_samples_uniformly(tmp_path, monkeypatch):
    pol = load_policy(_export(tmp_path / "p.pt"))
    model = _model(_GoalEnv(), warmup=10_000, learning_starts=20)
    load_actor_into_sb3(model, pol)
    monkeypatch.setattr(model.action_space, "sample", lambda *a, **k: pytest.fail("uniform-random warm-up action"))
    actor0 = {k: v.clone() for k, v in model.actor.state_dict().items()}
    critic0 = {k: v.clone() for k, v in model.critic.state_dict().items()}
    alpha0 = model.log_ent_coef.detach().clone()
    model.learn(total_timesteps=60)
    assert model._n_updates > 0 and model.actor_frozen
    assert all(torch.equal(actor0[k], v) for k, v in model.actor.state_dict().items())
    assert torch.equal(alpha0, model.log_ent_coef.detach())
    assert any(not torch.equal(critic0[k], v) for k, v in model.critic.state_dict().items())
    model.warmup_frozen_actor = 0  # after the warm-up the actor learns
    model.learn(total_timesteps=30, reset_num_timesteps=False)
    assert any(not torch.equal(actor0[k], v) for k, v in model.actor.state_dict().items())


def test_export_round_trip_is_deployable(tmp_path):
    pol = load_policy(_export(tmp_path / "p.pt"))
    model = _model(_GoalEnv())
    load_actor_into_sb3(model, pol)
    with torch.no_grad():
        for p in model.actor.mu.parameters():
            p.add_(0.1)  # "fine-tuned"
    out = export_sb3_actor(model, pol.meta, tmp_path / "ft.pt", finetune_steps=5)
    back = load_policy(out)
    assert back.meta["finetune_steps"] == 5 and back.meta["task_name"] == "reach"
    x = np.random.default_rng(1).standard_normal((50, 24)).astype(np.float32)
    np.testing.assert_allclose(back(x), actor_policy(model, pol.meta)(x), atol=1e-6)
    assert not np.allclose(back(x), pol(x))


@pytest.mark.parametrize(
    ("hist", "baseline", "due"),
    [([0.6, 0.7], 0.95, True), ([0.9, 0.7], 0.95, False), ([0.7, 0.9], 0.95, False), ([0.7], 0.95, False), ([0.81, 0.81], 0.95, False),
     ([0.5, 0.79, 0.79], 0.95, True)],
)  # fmt: skip
def test_rollback_rule(hist, baseline, due):
    assert rollback_due(hist, baseline, margin=0.15) is due


def test_entry_point_refusals(tmp_path, capsys):
    from so101_rl.rl import train_sac_her as t

    p = _export(tmp_path / "p.pt")
    assert t.main(["--init-from", str(p), "--total-steps", "0", "--out-root", str(tmp_path)]) == t.EXIT_REFUSED
    assert t.main(["--init-from", str(p), "--total-steps", "10", "--backend", "shm",
                   "--out-root", str(tmp_path)]) == t.EXIT_REFUSED  # fmt: skip
    assert t.main(["--init-from", str(tmp_path / "nope.pt"), "--total-steps", "10",
                   "--out-root", str(tmp_path)]) == t.EXIT_REFUSED  # fmt: skip
    assert "not built yet" in capsys.readouterr().err
    assert not any(tmp_path.glob("2*"))  # no run dir for a refusal


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_finetune_on_the_sim_rig_writes_every_artifact(tmp_path):
    """300 steps from the M6 export on the CPU reference rig: baseline, 2 evals, every artifact, a deployable export."""
    from so101_rl.rl import train_sac_her as t

    if not M6_POLICY.is_file():
        pytest.skip(f"the M6 policy export is not on this host ({M6_POLICY})")
    code = t.main(["--init-from", str(M6_POLICY), "--total-steps", "300", "--learning-starts", "100",
                   "--warmup-frozen-actor", "50", "--baseline-eval-every", "150", "--eval-episodes", "2",
                   "--out-root", str(tmp_path)])  # fmt: skip
    assert code == t.EXIT_OK
    (run,) = tmp_path.iterdir()
    assert all((run / a).is_file() for a in t.ARTIFACTS)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["total_steps"] >= 300 and summary["n_updates"] > 0 and summary["missing_artifacts"] == []
    evals = [json.loads(line) for line in (run / "evals.jsonl").read_text().splitlines()]
    assert len(evals) == 2 and all(e["n"] == 2 for e in evals)
    assert json.loads((run / "baseline.json").read_text())["n"] == 2
    assert load_policy(run / "policy_export.pt").meta["finetune_steps"] == summary["total_steps"]

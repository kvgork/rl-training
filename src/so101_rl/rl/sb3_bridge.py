"""SB3 SAC + HER fine-tuning from the sim actor (sim plan S5.2 / FS11): the bridge between the FastSAC export and SB3.

* :class:`GoalPolicyInput`: SB3's features extractor for the goal env's Dict obs = ``observation ++ desired_goal``, the
  exported actor's input (FS3 ``policy_input_dim``; SB3's default ``CombinedExtractor`` would sort the keys and add
  ``achieved_goal``, so the copied weights would see a different input).
* :func:`load_actor_into_sb3`: copies ``latent_pi`` / ``mu`` / ``log_std`` (D-ACTOR-SHAPE); the deterministic action
  is then identical. The critics stay freshly initialised (the sim critics are asymmetric: privileged inputs).
  SB3 clamps ``log_std`` to [-20, 2] instead of the export's bounds, which only changes samples below ``exp(-5)``.
* :class:`FinetuneSAC`: SB3 ``SAC`` that (1) never takes uniform-random actions (SB3 does so before
  ``learning_starts``; on the arm the warm-up must be the sim actor, stochastic) and (2) keeps the actor and the
  entropy coefficient frozen (learning rate 0) for the first ``warmup_frozen_actor`` gradient steps while the fresh
  critics learn.
* :func:`export_sb3_actor`: the fine-tuned actor in the ``policy_export`` format, so ``deploy-policy`` runs it.
* :func:`rollback_due`: the rollback rule (fine-tuned success below the baseline minus a margin on consecutive evals).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

__all__ = ["FinetuneSAC", "GoalPolicyInput", "actor_policy", "export_sb3_actor", "load_actor_into_sb3",
           "rollback_due"]  # fmt: skip


class GoalPolicyInput(BaseFeaturesExtractor):
    """``concat(obs["observation"], obs["desired_goal"])``: the exported actor's input (no parameters)."""

    def __init__(self, observation_space: spaces.Dict) -> None:
        dim = int(observation_space["observation"].shape[0] + observation_space["desired_goal"].shape[0])
        super().__init__(observation_space, features_dim=dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([observations["observation"], observations["desired_goal"]], dim=-1)


class FinetuneSAC(SAC):
    """SB3 SAC with the fine-tuning warm-up (module docstring). ``warmup_frozen_actor`` counts gradient steps."""

    def __init__(self, *args: Any, warmup_frozen_actor: int = 0, **kwargs: Any) -> None:
        self.warmup_frozen_actor = int(warmup_frozen_actor)
        super().__init__(*args, **kwargs)

    @property
    def actor_frozen(self) -> bool:
        return self._n_updates < self.warmup_frozen_actor

    def _update_learning_rate(self, optimizers: Any) -> None:
        super()._update_learning_rate(optimizers)
        if self.actor_frozen:  # Adam with lr 0 leaves the parameters exactly unchanged
            frozen = [self.actor.optimizer] + ([self.ent_coef_optimizer] if self.ent_coef_optimizer else [])
            for opt in frozen:
                for group in opt.param_groups:
                    group["lr"] = 0.0

    def _sample_action(self, learning_starts: int, action_noise: Any = None, n_envs: int = 1) -> tuple:
        # never the uniform-random warm-up actions of OffPolicyAlgorithm: always the (stochastic) actor
        return super()._sample_action(0, action_noise, n_envs)


def load_actor_into_sb3(model: SAC, policy: Any) -> None:
    """Copy a :class:`~so101_rl.rl.policy_export.LoadedPolicy`'s actor into ``model.actor`` (shapes asserted)."""
    ours = policy.actor.state_dict()
    theirs = {k: v for k, v in model.actor.state_dict().items() if not k.startswith("features_extractor.")}
    if set(ours) != set(theirs):
        raise ValueError(f"actor layouts differ: export {sorted(ours)} vs SB3 {sorted(theirs)}")
    bad = [k for k in ours if ours[k].shape != theirs[k].shape]
    if bad:
        raise ValueError(f"actor shapes differ for {bad} (net_arch must equal the export's actor_hidden)")
    model.actor.load_state_dict(ours, strict=False)
    if model.actor.features_extractor.features_dim != policy.obs_dim:
        raise ValueError(f"SB3 actor input {model.actor.features_extractor.features_dim} != export {policy.obs_dim}")


def actor_policy(model: SAC, meta: dict[str, Any]) -> Any:
    """The SB3 actor as a deployable :class:`~so101_rl.rl.policy_export.LoadedPolicy` (a CPU copy)."""
    from so101_rl.rl.fast_sac import Actor
    from so101_rl.rl.policy_export import LoadedPolicy

    actor = Actor(int(meta["obs_dim"]), int(meta["act_dim"]), tuple(meta["actor_hidden"]), tuple(meta["log_std_bounds"]))
    sd = {k: v.detach().cpu() for k, v in model.actor.state_dict().items() if not k.startswith("features_extractor.")}
    actor.load_state_dict(sd)
    actor.eval()
    return LoadedPolicy(actor, dict(meta))


def export_sb3_actor(model: SAC, meta: dict[str, Any], out: Path, **extra: Any) -> Path:
    """Write the SB3 actor as a ``policy_export`` (``meta`` = the source export's, ``extra`` added); atomic."""
    pol = actor_policy(model, meta)
    doc = {**meta, **extra, "actor": {k: v.clone() for k, v in pol.actor.state_dict().items()}}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(doc, tmp)
    tmp.replace(out)
    return out


def rollback_due(successes: Sequence[float], baseline: float, *, margin: float = 0.15, consecutive: int = 2) -> bool:
    """True when the last ``consecutive`` eval success rates are all below ``baseline - margin``."""
    tail = list(successes)[-consecutive:]
    return len(tail) == consecutive and bool(np.all(np.asarray(tail) < baseline - margin))

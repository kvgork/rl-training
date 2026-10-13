"""SB3 SAC + HER fine-tuning from the sim actor (sim plan S5.2 / FS11): the bridge between the FastSAC export and SB3.

* :class:`GoalPolicyInput`: SB3's features extractor for the goal env's Dict obs = ``observation ++ desired_goal``, the
  exported actor's input (FS3 ``policy_input_dim``; SB3's default ``CombinedExtractor`` would sort the keys and add
  ``achieved_goal``, so the copied weights would see a different input).
* :func:`load_actor_into_sb3`: copies ``latent_pi`` / ``mu`` / ``log_std`` (D-ACTOR-SHAPE); the deterministic action
  is then identical. The critics stay freshly initialised (the sim critics are asymmetric: privileged inputs).
  SB3 clamps ``log_std`` to [-20, 2] instead of the export's bounds, which only changes samples below ``exp(-5)``.
* :class:`FinetuneSAC`: SB3 ``SAC`` that (1) never takes uniform-random actions (SB3 does so before
  ``learning_starts``; on the arm the warm-up must be the sim actor, stochastic), (2) keeps the actor and the
  entropy coefficient frozen (learning rate 0) for the first ``warmup_frozen_actor`` gradient steps while the fresh
  critics learn, (3) gives the actor its own ``actor_learning_rate``, (4) can hold the entropy coefficient at
  ``fixed_ent_coef`` and (5) anchors the actor to the sim actor (plan §10 2026-10-13 "Anchored fine-tuning"): with
  ``anchor_weight`` w > 0 the actor loss is the TD3+BC form ``SAC_loss / max(mean|Q|, 1) + w * mse(mu(s), mu_sim(s))``
  (mu = the deterministic action; ``set_anchor`` gives the frozen sim actor; the floor keeps the scale bounded while
  the fresh critics' Q is near 0). w = 0 is SB3's SAC actor loss unchanged.
* :func:`export_sb3_actor`: the fine-tuned actor in the ``policy_export`` format, so ``deploy-policy`` runs it.
* :func:`rollback_due`: the rollback rule (fine-tuned success below the baseline minus a margin on consecutive evals).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import polyak_update

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
    """SB3 SAC with the fine-tuning warm-up, actor learning rate, fixed entropy and sim-actor anchor (module
    docstring). ``warmup_frozen_actor`` counts gradient steps (absolute: the actor is frozen while ``_n_updates`` is
    below it). ``actor_learning_rate`` / ``fixed_ent_coef`` None = SB3's behaviour."""

    def __init__(self, *args: Any, warmup_frozen_actor: int = 0, actor_learning_rate: float | None = None,
                 fixed_ent_coef: float | None = None, anchor_weight: float = 0.0, **kwargs: Any) -> None:  # fmt: skip
        self.warmup_frozen_actor = int(warmup_frozen_actor)
        self.actor_learning_rate = actor_learning_rate
        self.fixed_ent_coef = fixed_ent_coef
        self.anchor_weight = float(anchor_weight)
        self.anchor_actor: Any = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "anchor_actor"]  # rebuilt from the export (set_anchor)

    @property
    def actor_frozen(self) -> bool:
        return self._n_updates < self.warmup_frozen_actor

    def set_anchor(self, policy: Any) -> None:
        """The frozen anchor = a copy of ``self.actor`` holding ``policy``'s (the sim export's) actor weights."""
        anchor = copy.deepcopy(self.actor)
        _copy_actor(anchor, policy)
        anchor.requires_grad_(False)
        anchor.set_training_mode(False)
        self.anchor_actor = anchor

    def reset_actor(self, policy: Any, frozen_updates: int) -> None:
        """Actor weights <- ``policy``, a fresh actor optimizer, and the actor frozen for ``frozen_updates`` more
        gradient steps (the critics keep learning)."""
        load_actor_into_sb3(self, policy)
        self.actor.optimizer = self.policy.optimizer_class(self.actor.parameters(), lr=self.lr_schedule(1),
                                                           **self.policy.optimizer_kwargs)  # fmt: skip
        self.warmup_frozen_actor = int(self._n_updates) + int(frozen_updates)

    def rollback_actor(self, policy_state: dict[str, Any]) -> None:
        """Actor weights <- the ``actor.*`` entries of a saved ``policy.state_dict()`` and a fresh actor optimizer;
        the critics are kept (they have learned from everything since)."""
        actor = {k[len("actor."):]: v for k, v in policy_state.items() if k.startswith("actor.")}
        self.actor.load_state_dict(actor)
        self.actor.optimizer = self.policy.optimizer_class(self.actor.parameters(), lr=self.lr_schedule(1),
                                                           **self.policy.optimizer_kwargs)  # fmt: skip

    def _update_learning_rate(self, optimizers: Any) -> None:
        super()._update_learning_rate(optimizers)
        if self.actor_learning_rate is not None:
            for group in self.actor.optimizer.param_groups:
                group["lr"] = float(self.actor_learning_rate)
        frozen = []
        if self.actor_frozen:  # Adam with lr 0 leaves the parameters exactly unchanged
            frozen.append(self.actor.optimizer)
        if self.ent_coef_optimizer is not None and (self.actor_frozen or self.fixed_ent_coef is not None):
            frozen.append(self.ent_coef_optimizer)
        for opt in frozen:
            for group in opt.param_groups:
                group["lr"] = 0.0

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """SB3 2.9 ``SAC.train`` (no SDE, no n-step) with the anchored actor loss; identical to it for w = 0."""
        if self.fixed_ent_coef is not None and self.log_ent_coef is not None:
            with torch.no_grad():
                self.log_ent_coef.fill_(math.log(self.fixed_ent_coef))
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        anchored = self.anchor_weight > 0.0
        if anchored and self.anchor_actor is None:
            raise RuntimeError("anchor_weight > 0 needs set_anchor() first")
        ent_coefs, actor_losses, critic_losses, anchor_mses = [], [], [], []
        for gradient_step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = data.discounts if data.discounts is not None else self.gamma
            actions_pi, log_prob = self.actor.action_log_prob(data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_q = torch.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                next_q, _ = torch.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = data.rewards + (1 - data.dones) * discounts * next_q
            current_q = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]
            self.critic.optimizer.zero_grad()
            critic_loss.backward()  # type: ignore[union-attr]
            self.critic.optimizer.step()
            q_pi = torch.cat(self.critic(data.observations, actions_pi), dim=1)
            min_q_pi, _ = torch.min(q_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_q_pi).mean()
            if anchored:
                with torch.no_grad():
                    mu_sim = self.anchor_actor(data.observations, deterministic=True)
                mse = F.mse_loss(self.actor(data.observations, deterministic=True), mu_sim)
                q_scale = min_q_pi.abs().mean().detach().clamp(min=1.0)  # TD3+BC scale; the floor: Q ~ 0 early on
                actor_loss = actor_loss / q_scale + self.anchor_weight * mse
                anchor_mses.append(mse.item())
            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if anchor_mses:
            self.logger.record("train/anchor_mse", np.mean(anchor_mses))

    def _sample_action(self, learning_starts: int, action_noise: Any = None, n_envs: int = 1) -> tuple:
        # never the uniform-random warm-up actions of OffPolicyAlgorithm: always the (stochastic) actor
        return super()._sample_action(0, action_noise, n_envs)


def load_actor_into_sb3(model: SAC, policy: Any) -> None:
    """Copy a :class:`~so101_rl.rl.policy_export.LoadedPolicy`'s actor into ``model.actor`` (shapes asserted)."""
    _copy_actor(model.actor, policy)


def _copy_actor(actor: Any, policy: Any) -> None:
    ours = policy.actor.state_dict()
    theirs = {k: v for k, v in actor.state_dict().items() if not k.startswith("features_extractor.")}
    if set(ours) != set(theirs):
        raise ValueError(f"actor layouts differ: export {sorted(ours)} vs SB3 {sorted(theirs)}")
    bad = [k for k in ours if ours[k].shape != theirs[k].shape]
    if bad:
        raise ValueError(f"actor shapes differ for {bad} (net_arch must equal the export's actor_hidden)")
    actor.load_state_dict(ours, strict=False)
    if actor.features_extractor.features_dim != policy.obs_dim:
        raise ValueError(f"SB3 actor input {actor.features_extractor.features_dim} != export {policy.obs_dim}")


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

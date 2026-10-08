"""Minimal SAC for the batched GPU env (sim plan S4.1 / FS10, MVP M2).

- **Actor**: SB3-compatible shape (decision D-ACTOR-SHAPE): ``latent_pi`` = ``Sequential(Linear, ReLU, ...)``,
  ``mu`` and ``log_std`` linear heads, tanh-squashed Gaussian with the exact log-prob correction, actions in
  [-1, 1] (scaling to ticks happens only in the env). Its ``state_dict`` keys equal SB3 ``SACPolicy.actor``'s, so the
  sim actor copies into SB3 for real fine-tuning.
- **Critics**: ``n_critics`` asymmetric Q-nets on ``concat(obs, priv, action)`` (privileged sim state is critic-only),
  ``Linear -> LayerNorm -> ReLU`` blocks, polyak target copies.
- **Targets**: ``y = r + gamma^n (1 - terminated) (min Q'(s', a') - alpha log pi(a'|s'))``. ``truncated`` does NOT
  zero the bootstrap: a step-limit or trip end is not a terminal state (``next_obs`` is the pre-reset observation).
- **alpha**: auto-tuned toward ``target_entropy``, ``log_alpha`` clamped to ``log_alpha_bounds``.

:meth:`FastSAC.update` returns detached device tensors (no host sync); the caller reads them at log time.
``rl/`` imports nothing from ``safety/``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from so101_rl.rl.replay_gpu import Batch

__all__ = ["Actor", "Critic", "FastSAC", "SacConfig", "tanh_gaussian_log_prob"]


@dataclass
class SacConfig:
    actor_hidden: tuple[int, ...] = (256, 256, 256)  # SB3-compatible: ReLU, no LayerNorm
    critic_hidden: tuple[int, ...] = (512, 512, 512)
    critic_layernorm: bool = True
    n_critics: int = 2
    batch_size: int = 4096
    grad_steps_per_env_step: int = 8  # per VECTOR step
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.97
    tau: float = 0.005
    n_step: int = 1
    alpha_init: float = 0.02
    target_entropy: float = -3.0
    log_alpha_bounds: tuple[float, float] = (-9.2, 0.0)
    log_std_bounds: tuple[float, float] = (-5.0, 2.0)
    grad_clip_critic: float = 10.0
    warmup_env_steps: int = 20  # vector steps of uniform random actions before the policy acts
    compile: bool = False  # reserved (FS10); not used by the MVP
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SacConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("actor_hidden", "critic_hidden", "log_alpha_bounds", "log_std_bounds"):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)


def _mlp(inp: int, hidden: tuple[int, ...], *, layernorm: bool) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for h in hidden:
        layers.append(nn.Linear(inp, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        inp = h
    return layers


def tanh_gaussian_log_prob(u: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """log density of ``a = tanh(u)``, ``u ~ N(mu, exp(log_std))``, summed over the last dim (stable form:
    ``log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))``)."""
    std = log_std.exp()
    logp_u = -0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2.0 * math.pi)
    corr = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
    return (logp_u - corr).sum(dim=-1)


class Actor(nn.Module):
    """``latent_pi`` / ``mu`` / ``log_std``: the SB3 ``SACPolicy.actor`` parameter layout."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256, 256),
                 log_std_bounds: tuple[float, float] = (-5.0, 2.0)) -> None:  # fmt: skip
        super().__init__()
        self.latent_pi = nn.Sequential(*_mlp(obs_dim, tuple(hidden), layernorm=False))
        last = hidden[-1] if hidden else obs_dim
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)
        self.log_std_bounds = tuple(float(x) for x in log_std_bounds)

    def dist_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.latent_pi(obs)
        lo, hi = self.log_std_bounds
        return self.mu(h), self.log_std(h).clamp(lo, hi)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action ``tanh(mu)``."""
        return torch.tanh(self.dist_params(obs)[0])

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised ``(action, log_prob)``."""
        mu, log_std = self.dist_params(obs)
        u = mu + log_std.exp() * torch.randn_like(mu)
        return torch.tanh(u), tanh_gaussian_log_prob(u, mu, log_std)


class Critic(nn.Module):
    """``n`` independent Q-nets on ``concat(obs, priv, action)``; ``forward`` returns ``[n, B]``."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], n: int, *, layernorm: bool) -> None:
        super().__init__()
        self.qs = nn.ModuleList(
            nn.Sequential(*_mlp(in_dim, tuple(hidden), layernorm=layernorm), nn.Linear(hidden[-1], 1)) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([q(x).squeeze(-1) for q in self.qs])


class FastSAC:
    def __init__(self, obs_dim: int, priv_dim: int, act_dim: int, cfg: SacConfig, *, device: torch.device | str) -> None:
        if cfg.n_step != 1:
            raise NotImplementedError("n_step > 1 is deferred with the die task (MVP)")
        self.cfg, self.device = cfg, torch.device(device)
        self.obs_dim, self.priv_dim, self.act_dim = int(obs_dim), int(priv_dim), int(act_dim)
        torch.manual_seed(cfg.seed)
        self.actor = Actor(obs_dim, act_dim, cfg.actor_hidden, cfg.log_std_bounds).to(self.device)
        cin = obs_dim + priv_dim + act_dim
        self.critic = Critic(cin, cfg.critic_hidden, cfg.n_critics, layernorm=cfg.critic_layernorm).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(cfg.alpha_init), device=self.device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.detach().exp()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        obs = obs.to(self.device, torch.float32)
        return self.actor(obs) if deterministic else self.actor.sample(obs)[0]

    @torch.no_grad()
    def td_target(self, batch: Batch) -> torch.Tensor:
        """``r + gamma^n (1 - terminated) (min Q'(s', a') - alpha log pi(a'|s'))``; ``truncated`` is ignored on purpose
        (a step-limit / trip end bootstraps from the pre-reset ``next_obs``)."""
        a2, logp2 = self.actor.sample(batch.next_obs)
        q2 = self.critic_target(torch.cat([batch.next_obs, batch.next_priv, a2], dim=-1)).min(dim=0).values
        return batch.rew + batch.n_step_discount * (1.0 - batch.terminated) * (q2 - self.alpha * logp2)

    def update(self, batch: Batch) -> dict[str, torch.Tensor]:
        """One gradient step on the critics, the actor and alpha. Returns detached device scalars."""
        c = self.cfg
        alpha = self.alpha
        y = self.td_target(batch)
        q = self.critic(torch.cat([batch.obs, batch.priv, batch.act], dim=-1))
        critic_loss = 0.5 * ((q - y.unsqueeze(0)) ** 2).mean(dim=1).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if c.grad_clip_critic > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip_critic)
        self.critic_opt.step()

        a, logp = self.actor.sample(batch.obs)
        self.critic.requires_grad_(False)
        q_pi = self.critic(torch.cat([batch.obs, batch.priv, a], dim=-1)).min(dim=0).values
        self.critic.requires_grad_(True)
        actor_loss = (alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + c.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(*c.log_alpha_bounds)
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters(), strict=True):
                pt.lerp_(p, c.tau)
        self.updates += 1
        return {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "alpha": self.alpha,
            "entropy": -logp.detach().mean(),
            "q_mean": q.detach().mean(),
            "q_max": q.detach().abs().max(),
            "action_saturation": (a.detach().abs() > 0.99).float().mean(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg.to_dict(),
            "dims": [self.obs_dim, self.priv_dim, self.act_dim],
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if list(d["dims"]) != [self.obs_dim, self.priv_dim, self.act_dim]:
            raise ValueError(f"checkpoint dims {d['dims']} != {[self.obs_dim, self.priv_dim, self.act_dim]}")
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(d["actor_opt"])
        self.critic_opt.load_state_dict(d["critic_opt"])
        self.alpha_opt.load_state_dict(d["alpha_opt"])
        self.updates = int(d["updates"])

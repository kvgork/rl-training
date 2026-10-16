"""GPU ring replay for the batched sim learner (sim plan S4.1 / FS10, MVP M2).

One :meth:`GpuReplay.add` stores one vector step of N worlds. ``next_obs`` / ``next_priv`` must be the PRE-reset
observation for the worlds that reset on that step (``BatchedStep.final_obs`` / ``final_priv``): that is what lets the
learner bootstrap through truncation (the most common bug in batched SAC). ``terminated`` and ``truncated`` are stored
separately; only ``terminated`` stops the bootstrap.

1-step returns only (the MVP trains reach, whose reward is dense); the FS10 ``n_step_discount`` field is therefore
``gamma`` for every sample. Everything lives on ``device``; ``sample`` makes no host sync.

Demo mixing (RLPD-style, D2b of ``plans/die-to-zone-plan.md``): :func:`sample_mixed` draws a fixed share of each
batch from a second ``GpuReplay`` filled with scripted demonstrations (``rl/scripted.py`` ``collect_demos``, the
same transition format) and the rest from the online replay; with no demo replay or a zero share it is exactly
``replay.sample``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Batch", "GpuReplay", "demo_count", "sample_mixed"]


@dataclass
class Batch:
    obs: torch.Tensor  # [B, obs_dim] policy input
    priv: torch.Tensor  # [B, priv_dim] critic-only input
    act: torch.Tensor  # [B, act_dim] in [-1, 1]
    rew: torch.Tensor  # [B]
    next_obs: torch.Tensor
    next_priv: torch.Tensor
    terminated: torch.Tensor  # [B] float 0/1
    truncated: torch.Tensor  # [B] float 0/1 (bookkeeping only; does NOT zero the bootstrap)
    n_step_discount: torch.Tensor  # [B] gamma ** n (n = 1)


class GpuReplay:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        priv_dim: int,
        act_dim: int,
        *,
        device: torch.device | str,
        n_step: int = 1,
        gamma: float = 0.97,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if n_step != 1:
            raise NotImplementedError("n-step returns are deferred with the die task (MVP: n_step = 1)")
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.capacity, self.device, self.gamma = int(capacity), torch.device(device), float(gamma)
        f32 = {"dtype": torch.float32, "device": self.device}
        c = self.capacity
        self._obs = torch.zeros((c, obs_dim), **f32)
        self._priv = torch.zeros((c, priv_dim), **f32)
        self._act = torch.zeros((c, act_dim), **f32)
        self._rew = torch.zeros(c, **f32)
        self._next_obs = torch.zeros((c, obs_dim), **f32)
        self._next_priv = torch.zeros((c, priv_dim), **f32)
        self._term = torch.zeros(c, **f32)
        self._trunc = torch.zeros(c, **f32)
        self._ptr = 0
        self.size = 0

    @property
    def vram_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self._obs, self._priv, self._act, self._rew, self._next_obs, self._next_priv, self._term, self._trunc)
        )

    def add(
        self,
        obs: torch.Tensor,
        priv: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        next_priv: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Store one vector step ``[N, ...]`` (N <= capacity). Oldest entries are overwritten (ring)."""
        n = int(obs.shape[0])
        if n > self.capacity:
            raise ValueError(f"a vector step of {n} transitions does not fit a capacity of {self.capacity}")
        idx = (torch.arange(n, device=self.device) + self._ptr) % self.capacity
        for buf, x in (
            (self._obs, obs), (self._priv, priv), (self._act, act), (self._rew, rew), (self._next_obs, next_obs),
            (self._next_priv, next_priv), (self._term, terminated), (self._trunc, truncated),
        ):  # fmt: skip
            buf[idx] = x.to(device=self.device, dtype=torch.float32).reshape(buf[idx].shape)
        self._ptr = (self._ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, *, generator: torch.Generator | None = None) -> Batch:
        if self.size == 0:
            raise RuntimeError("sample from an empty replay")
        i = torch.randint(0, self.size, (int(batch_size),), device=self.device, generator=generator)
        return Batch(
            obs=self._obs[i], priv=self._priv[i], act=self._act[i], rew=self._rew[i], next_obs=self._next_obs[i],
            next_priv=self._next_priv[i], terminated=self._term[i], truncated=self._trunc[i],
            n_step_discount=torch.full((int(batch_size),), self.gamma, dtype=torch.float32, device=self.device),
        )  # fmt: skip


def demo_count(batch_size: int, demo_ratio: float) -> int:
    """Rows of a ``batch_size`` batch drawn from the demo replay: ``round(batch_size * demo_ratio)``, at least 1 for a
    positive ratio and at most ``batch_size - 1`` (the online replay always contributes)."""
    if not 0.0 <= demo_ratio < 1.0:
        raise ValueError(f"demo_ratio must be in [0, 1), got {demo_ratio}")
    if demo_ratio == 0.0:
        return 0
    return min(max(1, round(batch_size * demo_ratio)), batch_size - 1)


def sample_mixed(
    replay: GpuReplay,
    demo: GpuReplay | None,
    batch_size: int,
    demo_ratio: float,
    *,
    generator: torch.Generator | None = None,
) -> Batch:
    """One training batch: :func:`demo_count` rows from ``demo``, the rest from ``replay`` (online rows first). No
    demo replay or ``demo_ratio == 0``: ``replay.sample(batch_size, generator=generator)`` unchanged (same draws)."""
    k = demo_count(batch_size, demo_ratio) if demo is not None else 0
    if k == 0:
        return replay.sample(batch_size, generator=generator)
    a = replay.sample(batch_size - k, generator=generator)
    b = demo.sample(k, generator=generator)
    return Batch(**{f: torch.cat([getattr(a, f), getattr(b, f)], dim=0) for f in Batch.__dataclass_fields__})

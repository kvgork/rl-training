"""Torch point-mass goal env with the FS9 ``BatchedStep`` contract (sim plan S4.1; CPU-capable, no MuJoCo).

Used by the FastSAC / replay / training-loop unit tests. ``N`` worlds of a 2-D point moved by ``action * speed``
per step; the reward has the FS4 shape (``-dist/scale * (1 - success) - (1 - success) - 0.05``, every step < 0),
``terminated`` on success, ``truncated`` at ``episode_steps``, auto-reset with ``final_*`` = pre-reset copies.

``ToyBatchedDieShapedEnv``: the same env with the die task's tensor widths (observation 26 = ``DIE_OBS.dim``, goal
3), so the FS3 policy input is 29 wide as for ``die_to_zone`` (tests of the die plumbing on CPU). It also has the
batched die env's start-stage setter (``set_start_stage`` / ``start_stage``, D2a); the stage is recorded only, the
toy's dynamics do not change with it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch

__all__ = ["ToyBatchedDieShapedEnv", "ToyBatchedReachEnv", "ToyStep"]


@dataclass
class ToyStep:
    obs: torch.Tensor
    priv: torch.Tensor
    achieved: torch.Tensor
    desired: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_obs: torch.Tensor
    final_priv: torch.Tensor
    final_achieved: torch.Tensor
    info: dict[str, torch.Tensor] = field(default_factory=dict)


class ToyBatchedReachEnv:
    act_dim = 2
    obs_dim = 4  # position (2) + last action (2)
    priv_dim = 2  # position again (a stand-in for privileged state)
    randomizer = None
    dr_stage = 0

    def __init__(self, nworld: int, *, episode_steps: int = 50, success_radius: float = 0.05, speed: float = 0.15,
                 constant_reward: float | None = None, seed: int = 0, device: str = "cpu") -> None:  # fmt: skip
        self.nworld, self.episode_steps, self.success_radius, self.speed = int(nworld), episode_steps, success_radius, speed
        self.constant_reward = constant_reward  # tests: a constant per-step reward (bootstrap checks)
        self.device = torch.device(device)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        n = self.nworld
        self.pos = torch.zeros(n, 2, device=self.device)
        self.goal = torch.zeros(n, 2, device=self.device)
        self.last = torch.zeros(n, 2, device=self.device)
        self.episode_step = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.stats: dict[str, float] = {}

    def _u(self, n: int) -> torch.Tensor:
        return torch.rand(n, 2, generator=self._gen, device=self.device) * 1.6 - 0.8

    def _obs(self) -> torch.Tensor:
        return torch.cat([self.pos, self.last], dim=1)

    def reset(self, mask: torch.Tensor | None = None, *, phase_randomise: bool = True):
        n = self.nworld
        m = torch.ones(n, dtype=torch.bool, device=self.device) if mask is None else mask
        self.pos = torch.where(m[:, None], self._u(n), self.pos)
        self.goal = torch.where(m[:, None], self._u(n), self.goal)
        self.last = torch.where(m[:, None], torch.zeros_like(self.last), self.last)
        self.episode_step = torch.where(m, torch.zeros_like(self.episode_step), self.episode_step)
        return self._obs(), self.pos.clone(), self.pos.clone(), self.goal.clone()

    def step(self, action: torch.Tensor) -> ToyStep:
        a = action.to(self.device).clamp(-1.0, 1.0)
        self.pos = (self.pos + self.speed * a).clamp(-1.0, 1.0)
        self.last = a
        self.episode_step += 1
        dist = (self.pos - self.goal).norm(dim=1)
        success = dist <= self.success_radius
        if self.constant_reward is not None:
            reward = torch.full((self.nworld,), float(self.constant_reward), device=self.device)
            success = torch.zeros_like(success)
        else:
            s = success.float()
            reward = -dist / 0.3 * (1 - s) - (1 - s) - 0.05
        terminated = success
        truncated = self.episode_step >= self.episode_steps
        final_obs, final_priv, desired = self._obs(), self.pos.clone(), self.goal.clone()
        info = {"success": success, "tripped": torch.zeros_like(success), "episode_step": self.episode_step.clone(),
                "final_dist": dist}
        obs, priv, ach, _ = self.reset(terminated | truncated)
        return ToyStep(obs=obs, priv=priv, achieved=ach, desired=desired, reward=reward, terminated=terminated,
                       truncated=truncated, final_obs=final_obs, final_priv=final_priv, final_achieved=final_priv,
                       info=info)  # fmt: skip

    def close(self) -> None:
        pass


def _pad(x: torch.Tensor, width: int) -> torch.Tensor:
    return torch.cat([x, x.new_zeros(x.shape[0], width - x.shape[1])], dim=1)


class ToyBatchedDieShapedEnv:
    """:class:`ToyBatchedReachEnv` (same dynamics, reward and success) with zero-padded die-task widths: observation
    26 (``DIE_OBS.dim``), goal / achieved 3. Policy input = 29, as for ``die_to_zone`` (FS3)."""

    obs_dim = 26
    randomizer = None
    dr_stage = 0

    def __init__(self, nworld: int, **kw: object) -> None:
        self._env = ToyBatchedReachEnv(nworld, **kw)  # type: ignore[arg-type]
        self.nworld, self.device, self.episode_steps = self._env.nworld, self._env.device, self._env.episode_steps
        self.act_dim, self.priv_dim = ToyBatchedReachEnv.act_dim, ToyBatchedReachEnv.priv_dim
        self.metadata: dict[str, object] = {}
        self.stats: dict[str, float] = {}
        from so101_rl.env.tasks import DIE_TO_ZONE

        self.well_stage = DIE_TO_ZONE.die_start.well_stage
        self.start_stage = self.well_stage  # BatchedGoalEnv's default: the well

    def set_start_stage(self, k: int) -> None:
        if not 0 <= int(k) <= self.well_stage:
            raise ValueError(f"start stage {k} outside 0..{self.well_stage}")
        self.start_stage = int(k)

    @property
    def goal(self) -> torch.Tensor:
        return _pad(self._env.goal, 3)

    def reset(self, mask: torch.Tensor | None = None, *, phase_randomise: bool = True):
        obs, priv, ach, goal = self._env.reset(mask, phase_randomise=phase_randomise)
        return _pad(obs, self.obs_dim), priv, _pad(ach, 3), _pad(goal, 3)

    def step(self, action: torch.Tensor) -> ToyStep:
        st = self._env.step(action)
        return dataclasses.replace(st, obs=_pad(st.obs, self.obs_dim), final_obs=_pad(st.final_obs, self.obs_dim),
                                   achieved=_pad(st.achieved, 3), desired=_pad(st.desired, 3),
                                   final_achieved=_pad(st.final_achieved, 3))  # fmt: skip

    def close(self) -> None:
        pass

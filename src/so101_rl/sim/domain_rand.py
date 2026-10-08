"""Domain randomisation for the reach MVP (plans/mvp-plan.md M1; sim plan S3.1 / FS8, reach stages only).

A **fixed stage list** (no curriculum controller class, S3.2 is cut down): stage 0 is nominal, the last stage is the
full randomisation the reach exit gate is evaluated at. The training loop widens one stage at a time with the simple
rule :func:`next_stage`: sim success >= 70 % at the current stage (plan §5 S3).

What is randomised, **per episode per world** (decision D-DR-FREQ; physics parameters change only on reset):

- the **servo ratio group** (plan §10 2026-10-06 "Servo parameter identification"): one factor per joint
  ``g ~ U(1 - r, 1 + r)`` multiplies kp, kv and armature jointly around the ``calib/sim_params.yaml`` centre;
  ``r = stage.servo_frac * dr.servo_group_rel`` (0.40 at the last stage);
- **frictionloss** per joint, uniform over the centre widened toward ``dr.frictionloss_range`` (0 .. 0.21 N m at the
  last stage, plan §10 "Friction DR ceiling");
- **servo delay**: whole ticks added to the fitted ``delay_ticks``, uniform in ``stage.delay_extra_ticks``;
- **proprio noise**: the amplitude ``k`` of the uniform integer noise in ``[-k, k]`` ticks the env adds to the
  present ticks the POLICY sees (``q_norm`` and the FK tip of the observation; drawn per policy step by the env).
  Success and reward use the un-noised reading.

Not randomised for reach: camera pose (reach has no camera in its observation), tracker, die, backlash
(D-BACKLASH; deferred with the die task).

Imports: torch, numpy, ``sim.sysid.params`` only (no ``rl``/``env``; ``safety`` never).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch

from so101_rl.sim.sysid.params import BatchedSimParams, DrRanges, SimParams

__all__ = [
    "REACH_DR_STAGES",
    "WIDEN_THRESHOLD",
    "DomainRandomizer",
    "DrStage",
    "next_stage",
]

N_JOINTS: Final[int] = 6
WIDEN_THRESHOLD: Final[float] = 0.70  # plan §5 S3: widen only at >= 70 % sim success at the current stage


@dataclass(frozen=True)
class DrStage:
    """One row of the fixed stage list. ``*_frac`` scale the ranges recorded in ``calib/sim_params.yaml`` ``dr``."""

    index: int
    name: str
    servo_frac: float  # fraction of dr.servo_group_rel (0 = the centre, 1 = +-40 %)
    friction_frac: float  # 0 = the centre value, 1 = the whole dr.frictionloss_range
    delay_extra_ticks: tuple[int, int]  # inclusive, added to the fitted delay
    proprio_noise_ticks: int  # +-k ticks on the policy's present reading

    def __post_init__(self) -> None:
        if not (0.0 <= self.servo_frac <= 1.0 and 0.0 <= self.friction_frac <= 1.0):
            raise ValueError(f"stage {self.index}: servo_frac / friction_frac must be within [0, 1]")
        lo, hi = self.delay_extra_ticks
        if not (0 <= lo <= hi) or self.proprio_noise_ticks < 0:
            raise ValueError(f"stage {self.index}: bad delay range {self.delay_extra_ticks} or noise")

    @property
    def physics(self) -> bool:
        """True when this stage changes per-world physics parameters (a ``set_world_params`` push is needed)."""
        return self.servo_frac > 0.0 or self.friction_frac > 0.0


REACH_DR_STAGES: Final[tuple[DrStage, ...]] = (
    DrStage(0, "nominal", 0.0, 0.0, (0, 0), 0),
    DrStage(1, "proprio+delay", 0.0, 0.0, (0, 1), 1),
    DrStage(2, "servo-half", 0.5, 0.5, (0, 2), 2),
    DrStage(3, "servo-3/4", 0.75, 0.75, (0, 3), 2),
    DrStage(4, "full", 1.0, 1.0, (0, 3), 2),
)


def next_stage(stage: int, success_rate: float, max_stage: int, *, threshold: float = WIDEN_THRESHOLD) -> int:
    """The widening rule: one stage up when ``success_rate >= threshold`` and below ``max_stage``; never down."""
    if math.isfinite(success_rate) and success_rate >= threshold and stage < max_stage:
        return stage + 1
    return stage


class DomainRandomizer:
    """Per-world episode parameters for :class:`~so101_rl.env.batched_env.BatchedGoalEnv` (FS8, reach subset).

    ``params`` (FS7 ``BatchedSimParams``, on ``device``) always holds every world's CURRENT episode parameters;
    :meth:`sample_episode` redraws the rows in ``mask`` (no host sync). ``proprio_noise`` int64[N] is the per-world
    noise amplitude of the current episode.
    """

    def __init__(
        self,
        centre: SimParams,
        dr: DrRanges,
        nworld: int,
        *,
        stages: Sequence[DrStage] = REACH_DR_STAGES,
        stage: int = 0,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        if not stages or [s.index for s in stages] != list(range(len(stages))):
            raise ValueError("stages must be indexed 0..K-1 in order")
        self.stages = tuple(stages)
        self.centre, self.dr = centre, dr
        self.nworld = int(nworld)
        self.device = torch.device(device)
        self.max_stage = len(self.stages) - 1
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))
        self.params = BatchedSimParams.broadcast(centre, self.nworld, self.device)
        self.group_scale = torch.ones(self.nworld, N_JOINTS, device=self.device)
        self.proprio_noise = torch.zeros(self.nworld, dtype=torch.int64, device=self.device)
        f32 = {"dtype": torch.float32, "device": self.device}
        self._kp = torch.tensor(np.asarray(centre.kp), **f32)
        self._kv = torch.tensor(np.asarray(centre.kv), **f32)
        self._arm = torch.tensor(np.asarray(centre.armature), **f32)
        self._fl = torch.tensor(np.asarray(centre.frictionloss), **f32)
        self._delay0 = int(centre.delay_ticks)
        self.stage = 0
        self.set_stage(stage)

    # ------------------------------------------------------------------ stages
    @property
    def current(self) -> DrStage:
        return self.stages[self.stage]

    @property
    def max_delay_ticks(self) -> int:
        """The largest total delay any stage can draw (the env sizes its delay FIFO with it)."""
        return self._delay0 + max(s.delay_extra_ticks[1] for s in self.stages)

    def set_stage(self, k: int) -> None:
        if not 0 <= int(k) <= self.max_stage:
            raise ValueError(f"DR stage {k} outside 0..{self.max_stage}")
        self.stage = int(k)

    def ranges(self, k: int | None = None) -> dict[str, tuple[float, float]]:
        """The sampling ranges of stage ``k`` (default: the current one): group factor, frictionloss, delay, noise."""
        s = self.stages[self.stage if k is None else int(k)]
        rel = s.servo_frac * self.dr.servo_group_rel
        flo, fhi = self.dr.frictionloss_range
        c = self._fl.cpu().numpy()
        return {
            "servo_group": (1.0 - rel, 1.0 + rel),
            "frictionloss": (float((c - s.friction_frac * np.maximum(c - flo, 0.0)).min()),
                             float((c + s.friction_frac * np.maximum(fhi - c, 0.0)).max())),
            "delay_ticks": (float(self._delay0 + s.delay_extra_ticks[0]), float(self._delay0 + s.delay_extra_ticks[1])),
            "proprio_noise_ticks": (float(-s.proprio_noise_ticks), float(s.proprio_noise_ticks)),
        }  # fmt: skip

    # ------------------------------------------------------------------ sampling
    def _u(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=self._gen, device=self.device)

    def sample_episode(self, mask: torch.Tensor) -> BatchedSimParams:
        """Redraw the episode parameters of the worlds in ``mask`` (bool[N]) at the current stage; returns
        :attr:`params` (all worlds). The draws for every world are made and then masked, so the random stream does
        not depend on which worlds reset."""
        s, n, p = self.current, self.nworld, self.params
        m = mask.to(device=self.device, dtype=torch.bool)
        m6 = m[:, None]
        rel = s.servo_frac * self.dr.servo_group_rel
        g = 1.0 + rel * (2.0 * self._u((n, N_JOINTS)) - 1.0)
        self.group_scale = torch.where(m6, g, self.group_scale)
        p.kp.copy_(torch.where(m6, self._kp * g, p.kp))
        p.kv.copy_(torch.where(m6, self._kv * g, p.kv))
        p.armature.copy_(torch.where(m6, self._arm * g, p.armature))
        flo, fhi = self.dr.frictionloss_range
        lo = self._fl - s.friction_frac * (self._fl - flo).clamp_min(0.0)
        hi = self._fl + s.friction_frac * (fhi - self._fl).clamp_min(0.0)
        fl = lo + (hi - lo) * self._u((n, N_JOINTS))
        p.frictionloss.copy_(torch.where(m6, fl, p.frictionloss))
        dlo, dhi = s.delay_extra_ticks
        d = self._delay0 + torch.randint(dlo, dhi + 1, (n,), generator=self._gen, device=self.device)
        p.delay_ticks.copy_(torch.where(m, d, p.delay_ticks))
        self.proprio_noise.copy_(torch.where(m, torch.full_like(self.proprio_noise, s.proprio_noise_ticks), self.proprio_noise))
        return p

    def dr_ratios(self) -> torch.Tensor:
        """float32[N, 30]: (kp, kv, damping, frictionloss, armature) x 6 of the current episodes divided by the centre
        (the FS3 privileged ``dr_params`` block; frictionloss relative to its centre value)."""
        p = self.params
        cols = []
        for name, c in (("kp", self._kp), ("kv", self._kv), ("damping", None), ("frictionloss", self._fl), ("armature", self._arm)):
            x = getattr(p, name)
            if c is None:
                c = torch.tensor(np.asarray(getattr(self.centre, name)), dtype=torch.float32, device=self.device)
            cols.append(x / torch.where(c > 0, c, torch.ones_like(c)))
        return torch.cat(cols, dim=1)

    def describe(self) -> dict[str, Any]:
        """For checkpoints, eval.json and logs."""
        s = self.current
        return {
            "stage": self.stage,
            "max_stage": self.max_stage,
            "name": s.name,
            "ranges": {k: list(v) for k, v in self.ranges().items()},
            "servo_group_rel_full": self.dr.servo_group_rel,
            "frictionloss_range_full": list(self.dr.frictionloss_range),
        }

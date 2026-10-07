"""Frozen task specifications: one reward and one success definition for sim and real (FS4 of
``plans/sim-track-implementation-plan.md``; plan §10 2026-10-05 "Die success definition", D-REWARD, and the
2026-10-06 rows "Reward structure", "Die success height", "Die must be settled").

- ``reward_*`` / ``success_*`` take the **tracker-reported** die position (held last-known while invisible) as
  ``achieved``, so sim and real compute the same function of the same inputs. The simulator's true die position is
  critic-only (``obs.PRIV_OBS``) and never enters these functions.
- The configurable part (reward terms and weights, the die height bound and the settle distance) is the
  ``tasks:`` section of ``so101.yaml`` (:func:`task_from_config`). ``REACH`` / ``DIE_TO_ZONE`` are built from the
  shipped yaml at import; every :class:`TaskSpec` runs :func:`~so101_rl.env.reward.validate_reward_spec` when it is
  constructed, so no task with a reward that can be positive can exist.
- **Reward** (``env/reward.py``): a sum of named, weighted, non-positive terms, each a function of ONE step's
  ``(achieved, desired, visible, info)`` (the instantaneous predicate, never the debounce or settle state), so SB3
  HER's vectorised ``compute_reward`` relabels it. Default for both tasks:
  ``-dist/0.30 * (1 - predicate) - 1 * (1 - predicate)``: 0 on success steps, <= -1 on every other step.
- **Success predicate** (per step): reach ``|tip - goal| <= success_radius_m``. Die: visible AND the centroid's xy
  within the zone square around ``desired`` (``|dx|, |dy| <= zone.half_edge_m``) AND the centroid at most
  ``zone.max_height_m`` (16 mm) above ``desired`` (for the real goal ``zone.goal_xyz()`` = the zone centre at resting
  height; 16 mm leaves room for D435 depth noise while a die held ~2 cm up fails). ``visible=False`` is UNKNOWN,
  never SUCCESS (L11). Everything is relative to ``desired``, so HER-relabelled goals (an achieved die position)
  are consistent: achieved == desired always satisfies the geometric part.
- **Termination** (:func:`debounce_np` / :func:`debounce_torch`): ``success_debounce_steps`` consecutive predicate
  steps (die 2, reach 1). The die task also needs the die **settled**: the tracker position moved at most
  ``settle_m`` (3 mm) between consecutive counted steps, else the count restarts at 1. A slide through the zone
  (0.05 m/s = 5 mm per 10 Hz step) or a nudge out of it never terminates.
- **Episode end** (both envs): ``terminated`` = debounced success, where the predicate only counts on a step with a
  good read and no trip (a trip or stale read resets the debounce count); ``truncated`` = step limit OR trip OR
  stalled servo, independently of ``terminated`` (on the step-limit step both may be true; a learner zeroes the
  bootstrap on ``terminated`` only).
- **Action mapping** (:func:`action_to_delta_np` / :func:`action_to_delta_torch`, the F7 rule, the only
  [-1, 1] -> ticks scaling): ``delta = float32(rint(float64(clip(float32(a), -1, 1)) * max_delta_ticks))``. ±inf
  clips to ±1, NaN passes through unchanged so the servo's sanitizer trips on it; rint is half-to-even in float64.
- SB3 HER: ``compute_reward(achieved, desired, info)`` must take the visibility of the STORED transition from
  ``info`` (:func:`visible_from_info`, key ``"tracker_visible"``); relabelling the goal never changes it, and it
  never defaults to visible. Terms that read ``info`` (``clamp``, ``action_l2``) take the stored transition's
  ``info`` the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from so101_rl.config.loader import load_tasks_config
from so101_rl.config.schema import RewardSpec, TasksConfig, WorkspaceBox

from .obs import DIE_OBS, REACH_OBS, PolicyObsLayout
from .reward import evaluate_np, evaluate_torch, terms_np, terms_torch, validate_reward_spec

if TYPE_CHECKING:
    import torch

DIE_HALF_EDGE_M: Final[float] = 0.008  # 16 mm die (sim/model.py DIE_EDGE_M / 2; test-enforced)


@dataclass(frozen=True, kw_only=True)
class GoalBounds:
    """Where reach goals are sampled, base frame, metres. Shared by the numpy (real) and torch (sim) samplers."""

    radius_max_m: float = 0.30  # from the base
    box: WorkspaceBox  # cfg.box, shrunk by margin_m on every face
    margin_m: float = 0.02
    z_min_m: float = 0.03  # kinematic floor; the H-track may raise it

    @property
    def lo(self) -> np.ndarray:
        return np.array(
            [self.box.x[0] + self.margin_m, self.box.y[0] + self.margin_m, max(self.box.z[0] + self.margin_m, self.z_min_m)]
        )

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.box.x[1] - self.margin_m, self.box.y[1] - self.margin_m, self.box.z[1] - self.margin_m])

    def contains_np(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, np.float64)
        inside = np.all((xyz >= self.lo) & (xyz <= self.hi), axis=-1)
        return inside & (np.linalg.norm(xyz, axis=-1) <= self.radius_max_m)

    def sample_np(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` goals uniform in the shrunk box intersected with the radius ball (rejection sampling).

        No FK-reachability filter: ``env/goals.py`` (SR1) adds it on top of these bounds.
        """
        lo, hi = self.lo, self.hi
        out = np.empty((0, 3))
        while len(out) < n:
            cand = rng.uniform(lo, hi, size=(max(2 * (n - len(out)), 64), 3))
            out = np.concatenate([out, cand[self.contains_np(cand)]])
        return out[:n]


@dataclass(frozen=True, kw_only=True)
class ZoneSpec:
    center_xy_m: tuple[float, float]
    half_edge_m: float = 0.030
    z_table_m: float = -0.02
    max_height_m: float  # centroid at most this far above ``desired`` (the resting height) counts (config: 16 mm)

    def goal_xyz(self) -> np.ndarray:
        """``desired_goal`` of the die task: the zone centre at the height of a die resting on the table."""
        return np.array([self.center_xy_m[0], self.center_xy_m[1], self.z_table_m + DIE_HALF_EDGE_M])


@dataclass(frozen=True)
class DieStartSpec:
    """Die start distribution (the tray's start well). Placeholder until H9 measures the real tray."""

    well_center_xy_m: tuple[float, float]
    sigma_xy_m: float
    yaw: Literal["uniform"] = "uniform"
    measured_on_hardware: bool = False


@dataclass(frozen=True)
class TaskSpec:
    name: Literal["reach", "die_to_zone"]
    obs: PolicyObsLayout
    episode_steps: int  # 50 / 100
    gamma: float  # 0.97 / 0.98 (training hint, not an env property)
    success_radius_m: float  # reach: 0.015 (Euclidean); die: = zone half edge (square, L-inf in xy)
    success_debounce_steps: int  # reach 1; die 2 (consecutive steps with the predicate true)
    settle_m: float | None  # die: max tracker displacement between consecutive counted steps (0.003); reach: None
    reward: RewardSpec  # config tasks.<name>.reward; validated on construction (env/reward.py)
    goal_bounds: GoalBounds
    zone: ZoneSpec | None
    die_start: DieStartSpec | None
    start_pose: Literal["home", "random_in_box"]
    reset_settle_steps: int = 0  # die task: physics steps after placing the die (0 = place at rest)

    def __post_init__(self) -> None:
        validate_reward_spec(self.reward)
        if self.success_debounce_steps < 1:
            raise ValueError(f"success_debounce_steps must be >= 1, got {self.success_debounce_steps}")
        if self.settle_m is not None and (self.success_debounce_steps < 2 or not self.settle_m > 0):
            raise ValueError("settle_m needs success_debounce_steps >= 2 and a positive distance")


_ZONE_CENTER_XY_M: Final[tuple[float, float]] = (0.22, 0.10)  # scene site `zone` (test-enforced)
_ZONE_HALF_EDGE_M: Final[float] = 0.030


def task_from_config(name: str, tasks: TasksConfig) -> TaskSpec:
    """Build the FS4 task ``name`` with the configurable part from ``tasks`` (``cfg.tasks`` or
    ``load_tasks_config()``). Raises :class:`~so101_rl.env.reward.RewardSpecError` for an invalid reward."""
    if name == "reach":
        tc = tasks.reach
        return TaskSpec(
            name="reach",
            obs=REACH_OBS,
            episode_steps=50,
            gamma=0.97,
            success_radius_m=0.015,
            success_debounce_steps=1,
            settle_m=None,
            reward=tc.reward,
            goal_bounds=GoalBounds(box=WorkspaceBox()),  # schema defaults == so101.yaml box (test-enforced)
            zone=None,
            die_start=None,
            start_pose="home",
        )
    if name == "die_to_zone":
        tc = tasks.die_to_zone
        if tc.max_height_m is None or tc.settle_mm is None:
            raise ValueError("tasks.die_to_zone needs max_height_m and settle_mm")
        zone = ZoneSpec(
            center_xy_m=_ZONE_CENTER_XY_M, half_edge_m=_ZONE_HALF_EDGE_M, z_table_m=-0.02, max_height_m=tc.max_height_m
        )
        return TaskSpec(
            name="die_to_zone",
            obs=DIE_OBS,
            episode_steps=100,
            gamma=0.98,
            success_radius_m=zone.half_edge_m,
            success_debounce_steps=2,
            settle_m=tc.settle_mm / 1000.0,
            reward=tc.reward,
            goal_bounds=GoalBounds(box=WorkspaceBox()),
            zone=zone,
            # NEEDS H9: placeholder start well
            die_start=DieStartSpec(well_center_xy_m=(0.20, -0.05), sigma_xy_m=0.005, measured_on_hardware=False),
            start_pose="home",
            reset_settle_steps=0,
        )
    raise ValueError(f"unknown task {name!r}")


_SHIPPED: Final[TasksConfig] = load_tasks_config()
REACH: Final[TaskSpec] = task_from_config("reach", _SHIPPED)
DIE_TO_ZONE: Final[TaskSpec] = task_from_config("die_to_zone", _SHIPPED)


def _need_visible(spec: TaskSpec, visible: object) -> None:
    if spec.zone is not None and visible is None:
        raise ValueError(f"task {spec.name!r} uses the tracker: pass visible= (None is not 'visible')")


# ---------------------------------------------------------------- numpy (real arm, SB3 HER)


def success_np(
    achieved: np.ndarray, desired: np.ndarray, spec: TaskSpec, *, visible: np.ndarray | None = None
) -> np.ndarray:
    """Per-step success predicate, bool[...]. ``visible=False`` never succeeds (UNKNOWN, L11)."""
    _need_visible(spec, visible)
    achieved = np.asarray(achieved, np.float64)
    desired = np.asarray(desired, np.float64)
    if spec.zone is None:
        ok = np.linalg.norm(achieved - desired, axis=-1) <= spec.success_radius_m
    else:
        d = np.abs(achieved[..., :2] - desired[..., :2])
        ok = np.all(d <= spec.zone.half_edge_m, axis=-1)
        ok = ok & (achieved[..., 2] - desired[..., 2] <= spec.zone.max_height_m)
    if visible is not None:
        ok = ok & np.asarray(visible, bool)
    return ok


def reward_np(
    achieved: np.ndarray,
    desired: np.ndarray,
    spec: TaskSpec,
    *,
    visible: np.ndarray | None = None,
    info: object = None,
) -> np.ndarray:
    """Vectorised step reward, float32[...] (SB3 HER ``compute_reward``): ``spec.reward`` evaluated on the
    distance and the instantaneous :func:`success_np` predicate. Always <= 0 (``validate_reward_spec``).
    ``info``: needed only by the ``clamp`` / ``action_l2`` terms (one mapping or HER's sequence of mappings)."""
    dist = np.linalg.norm(np.asarray(achieved, np.float64) - np.asarray(desired, np.float64), axis=-1)
    return evaluate_np(spec.reward, dist=dist, success=success_np(achieved, desired, spec, visible=visible), info=info)


def reward_terms_np(
    achieved: np.ndarray, desired: np.ndarray, spec: TaskSpec, *, visible: np.ndarray | None = None, info: object = None
) -> dict[str, np.ndarray]:
    """Per-term weighted values (float64[...]) whose sum is :func:`reward_np` (logging / dashboards)."""
    dist = np.linalg.norm(np.asarray(achieved, np.float64) - np.asarray(desired, np.float64), axis=-1)
    return terms_np(spec.reward, dist=dist, success=success_np(achieved, desired, spec, visible=visible), info=info)


def visible_from_info(info: object, spec: TaskSpec) -> np.ndarray | None:
    """Tracker visibility for SB3 HER's ``compute_reward(achieved, desired, info)``.

    ``info`` is one dict or a sequence/array of dicts (HER passes the stored transitions' infos). Reach: None (no
    tracker). Die: bool array from ``info["tracker_visible"]``; a missing key is an error, never "visible".
    """
    if spec.zone is None:
        return None
    if isinstance(info, dict):
        return np.asarray(bool(info["tracker_visible"]))
    return np.array([bool(i["tracker_visible"]) for i in info], dtype=bool)


def _need_positions(spec: TaskSpec, achieved: object, prev_achieved: object) -> None:
    if spec.settle_m is not None and (achieved is None or prev_achieved is None):
        raise ValueError(
            f"task {spec.name!r} needs the die settled: pass achieved= and prev_achieved= (tracker positions)"
        )


def debounce_np(
    predicate: np.ndarray,
    count: np.ndarray,
    spec: TaskSpec,
    *,
    achieved: np.ndarray | None = None,
    prev_achieved: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(success, new_count)``: consecutive-step counter; success once it reaches ``success_debounce_steps``.

    With ``spec.settle_m`` (die task) a step only extends the run if the tracker position moved at most
    ``settle_m`` since the previous step (``|achieved - prev_achieved|``); otherwise it starts a new run at 1. So
    "2 consecutive visible in-zone steps that moved <= 3 mm between them". Termination only; the reward stays
    instantaneous.
    """
    _need_positions(spec, achieved, prev_achieved)
    pred = np.asarray(predicate, bool)
    count = np.asarray(count)
    extend = count > 0
    if spec.settle_m is not None:
        moved = np.linalg.norm(np.asarray(achieved, np.float64) - np.asarray(prev_achieved, np.float64), axis=-1)
        extend = extend & (moved <= spec.settle_m)
    new = np.where(pred, np.where(extend, count + 1, 1), 0)
    return new >= spec.success_debounce_steps, new


# ---------------------------------------------------------------- torch (batched sim)


def success_torch(
    achieved: torch.Tensor, desired: torch.Tensor, spec: TaskSpec, *, visible: torch.Tensor | None = None
) -> torch.Tensor:
    """Same predicate as :func:`success_np` for tensors (bool[...])."""
    import torch

    _need_visible(spec, visible)
    a, d = achieved.to(torch.float64), desired.to(torch.float64)
    if spec.zone is None:
        ok = (a - d).norm(dim=-1) <= spec.success_radius_m
    else:
        ok = ((a[..., :2] - d[..., :2]).abs() <= spec.zone.half_edge_m).all(dim=-1)
        ok = ok & (a[..., 2] - d[..., 2] <= spec.zone.max_height_m)
    if visible is not None:
        ok = ok & visible.to(torch.bool)
    return ok


def reward_torch(
    achieved: torch.Tensor,
    desired: torch.Tensor,
    spec: TaskSpec,
    *,
    visible: torch.Tensor | None = None,
    info: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Same reward as :func:`reward_np` for tensors (float32[...]); ``info`` is a mapping of tensors."""
    import torch

    dist = (achieved.to(torch.float64) - desired.to(torch.float64)).norm(dim=-1)
    pred = success_torch(achieved, desired, spec, visible=visible)
    return evaluate_torch(spec.reward, dist=dist, success=pred, info=info)


def reward_terms_torch(
    achieved: torch.Tensor,
    desired: torch.Tensor,
    spec: TaskSpec,
    *,
    visible: torch.Tensor | None = None,
    info: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Tensor twin of :func:`reward_terms_np`."""
    import torch

    dist = (achieved.to(torch.float64) - desired.to(torch.float64)).norm(dim=-1)
    pred = success_torch(achieved, desired, spec, visible=visible)
    return terms_torch(spec.reward, dist=dist, success=pred, info=info)


def debounce_torch(
    predicate: torch.Tensor,
    count: torch.Tensor,
    spec: TaskSpec,
    *,
    achieved: torch.Tensor | None = None,
    prev_achieved: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor twin of :func:`debounce_np` (same settle rule; no host sync)."""
    import torch

    _need_positions(spec, achieved, prev_achieved)
    pred = predicate.to(torch.bool)
    extend = count > 0
    if spec.settle_m is not None:
        moved = (achieved.to(torch.float64) - prev_achieved.to(torch.float64)).norm(dim=-1)
        extend = extend & (moved <= spec.settle_m)
    new = torch.where(pred, torch.where(extend, count + 1, torch.ones_like(count)), torch.zeros_like(count))
    return new >= spec.success_debounce_steps, new


def action_to_delta_np(action: np.ndarray, max_delta_ticks: np.ndarray) -> np.ndarray:
    """F7 action rule (real env): ``float32(rint(float64(clip(float32(a), -1, 1)) * max_delta_ticks))``.

    ``max_delta_ticks`` is used in float64. NaN stays NaN (``np.clip`` propagates it) so the servo sanitizer trips;
    ±inf clip to ±1 like any other out-of-range value.
    """
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return np.rint(a.astype(np.float64) * np.asarray(max_delta_ticks, np.float64)).astype(np.float32)


def action_to_delta_torch(action: torch.Tensor, max_delta_ticks: torch.Tensor) -> torch.Tensor:
    """Tensor twin of :func:`action_to_delta_np` (bit-identical: float32 clamp, float64 product, half-to-even
    round, float32 result). ``max_delta_ticks`` float64 [6]. ``torch.clamp`` propagates NaN like ``np.clip``."""
    import torch

    a = action.to(torch.float32).clamp(-1.0, 1.0)
    return torch.round(a.to(torch.float64) * max_delta_ticks.to(torch.float64)).to(torch.float32)

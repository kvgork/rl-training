"""Frozen task specifications: one reward and one success definition for sim and real (FS4 of
``plans/sim-track-implementation-plan.md``; plan §10 2026-10-05 "Die success definition", D-REWARD).

- ``reward_*`` / ``success_*`` take the **tracker-reported** die position (held last-known while invisible) as
  ``achieved``, so sim and real compute the same function of the same inputs. The simulator's true die position is
  critic-only (``obs.PRIV_OBS``) and never enters these functions.
- Die success needs the die **visible** (``visible=False`` is UNKNOWN, never SUCCESS, L11). The predicate is
  per step; ``TaskSpec.success_debounce_steps`` consecutive true steps (:func:`debounce_np` /
  :func:`debounce_torch`) make the episode ``terminated``. Die: 2 consecutive steps; reach: 1.
- Reward per step: ``-dist(achieved, desired) / dense_scale + success_bonus * predicate`` (the instantaneous
  predicate, so SB3 HER's vectorised ``compute_reward`` relabels it without a debounce state).
- Reach predicate: ``|tip - goal| <= success_radius_m``. Die predicate: visible AND the centroid's xy within the
  zone square around ``desired`` (``|dx|, |dy| <= zone.half_edge_m``) AND the centroid at most
  ``zone.max_height_m - DIE_HALF_EDGE_M`` above ``desired``'s height. For the real goal (``zone.goal_xyz()``: the
  zone centre at resting height) that means "on the table, at most one die edge up": a die held in the air above
  the zone is not in the zone. Everything is relative to ``desired``, so HER-relabelled goals (an achieved die
  position) are consistent: achieved == desired always satisfies the geometric part.
- SB3 HER: ``compute_reward(achieved, desired, info)`` must take the visibility of the STORED transition from
  ``info`` (:func:`visible_from_info`, key ``"tracker_visible"``); relabelling the goal never changes it, and it
  never defaults to visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from so101_rl.config.schema import WorkspaceBox

from .obs import DIE_OBS, REACH_OBS, PolicyObsLayout

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


@dataclass(frozen=True)
class ZoneSpec:
    center_xy_m: tuple[float, float]
    half_edge_m: float = 0.030
    z_table_m: float = -0.02
    max_height_m: float = 2 * DIE_HALF_EDGE_M  # centroid at most this far above the table counts as "in the zone"

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
    success_bonus: float  # 1.0
    dense_scale: float  # reward = -dist / dense_scale (+ bonus)
    goal_bounds: GoalBounds
    zone: ZoneSpec | None
    die_start: DieStartSpec | None
    start_pose: Literal["home", "random_in_box"]
    reset_settle_steps: int = 0  # die task: physics steps after placing the die (0 = place at rest)


REACH: Final[TaskSpec] = TaskSpec(
    name="reach",
    obs=REACH_OBS,
    episode_steps=50,
    gamma=0.97,
    success_radius_m=0.015,
    success_debounce_steps=1,
    success_bonus=1.0,
    dense_scale=0.30,
    goal_bounds=GoalBounds(box=WorkspaceBox()),  # schema defaults == so101.yaml box (test-enforced)
    zone=None,
    die_start=None,
    start_pose="home",
)
_ZONE: Final[ZoneSpec] = ZoneSpec(center_xy_m=(0.22, 0.10), half_edge_m=0.030, z_table_m=-0.02)  # scene site `zone`
DIE_TO_ZONE: Final[TaskSpec] = TaskSpec(
    name="die_to_zone",
    obs=DIE_OBS,
    episode_steps=100,
    gamma=0.98,
    success_radius_m=_ZONE.half_edge_m,
    success_debounce_steps=2,
    success_bonus=1.0,
    dense_scale=0.30,
    goal_bounds=GoalBounds(box=WorkspaceBox()),
    zone=_ZONE,
    die_start=DieStartSpec(well_center_xy_m=(0.20, -0.05), sigma_xy_m=0.005, measured_on_hardware=False),  # NEEDS H9
    start_pose="home",
    reset_settle_steps=0,
)


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
        ok = ok & (achieved[..., 2] - desired[..., 2] <= spec.zone.max_height_m - DIE_HALF_EDGE_M)
    if visible is not None:
        ok = ok & np.asarray(visible, bool)
    return ok


def reward_np(
    achieved: np.ndarray, desired: np.ndarray, spec: TaskSpec, *, visible: np.ndarray | None = None
) -> np.ndarray:
    """Vectorised reward, float32[...] (SB3 HER ``compute_reward``)."""
    dist = np.linalg.norm(np.asarray(achieved, np.float64) - np.asarray(desired, np.float64), axis=-1)
    bonus = spec.success_bonus * success_np(achieved, desired, spec, visible=visible)
    return (-dist / spec.dense_scale + bonus).astype(np.float32)


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


def debounce_np(predicate: np.ndarray, count: np.ndarray, spec: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
    """``(success, new_count)``: consecutive-step counter; success once it reaches ``success_debounce_steps``."""
    new = np.where(np.asarray(predicate, bool), np.asarray(count) + 1, 0)
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
        ok = ok & (a[..., 2] - d[..., 2] <= spec.zone.max_height_m - DIE_HALF_EDGE_M)
    if visible is not None:
        ok = ok & visible.to(torch.bool)
    return ok


def reward_torch(
    achieved: torch.Tensor, desired: torch.Tensor, spec: TaskSpec, *, visible: torch.Tensor | None = None
) -> torch.Tensor:
    """Same reward as :func:`reward_np` for tensors (float32[...])."""
    import torch

    dist = (achieved.to(torch.float64) - desired.to(torch.float64)).norm(dim=-1)
    bonus = spec.success_bonus * success_torch(achieved, desired, spec, visible=visible).to(torch.float64)
    return (-dist / spec.dense_scale + bonus).to(torch.float32)


def debounce_torch(predicate: torch.Tensor, count: torch.Tensor, spec: TaskSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor twin of :func:`debounce_np`."""
    import torch

    new = torch.where(predicate, count + 1, torch.zeros_like(count))
    return new >= spec.success_debounce_steps, new

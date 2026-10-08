"""Modular, non-positive reward (FS4; plan §10 2026-10-06 "Reward structure").

The step reward is ``sum(weight * term)`` over the task's configured terms (``tasks.<name>.reward`` in
``so101.yaml``, parsed strictly by ``config.loader.parse_tasks`` into a :class:`~so101_rl.config.RewardSpec`).
Every term is a pure, instantaneous function of one step's ``(achieved, desired, visible, info)`` — no history, no
debounce state — so SB3 HER's vectorised ``compute_reward`` can relabel it. ``env/tasks.py`` turns
``(achieved, desired, visible)`` into the distance and the instantaneous success predicate and calls
:func:`evaluate_np` / :func:`evaluate_torch`; this module never sees a TaskSpec.

Terms (unweighted value; every one is <= 0 by construction):

==============  ===============  ==========================================================================
name            range            value
==============  ===============  ==========================================================================
``distance``    (-inf, 0]        ``-dist / scale_m``; with ``zero_on_success`` 0 on steps where the
                                 success predicate holds
``success``     [-1, 0]          ``-(1 - predicate)``: -1 until the (instantaneous) predicate holds
``clamp``       [-1, 0]          ``-info["action_clamped"]``: the safety chain clamped this step's action
``action_l2``   (-inf, 0]        ``-mean(info["applied_action"] ** 2)`` over the last axis (6 joints, [-1, 1])
``time``        {-1}             constant ``-1`` on every step, including in-zone steps (never zeroed on success)
==============  ===============  ==========================================================================

Default (both tasks): ``-dist/0.30 * (1 - pred) - 1 * (1 - pred) - 0.05``: -0.05 on success steps, <= -1.05
otherwise.

Why non-positive (the SU0 review): the original ``-dist/scale + bonus * predicate`` paid up to +1 per step, so a
policy earned more by NOT finishing (dip in and out of the zone, or hide the die every other step to keep the
episode alive) than by finishing. The ``time`` term closes the last tie: a die kept moving INSIDE the zone (the
success predicate holds, the settle debounce never does) would earn 0 per step, the same as the terminal state.
With ``time`` every non-terminal step is <= -w_time < 0, so the terminal state (value 0) is strictly the best place
to be and the fastest settled finish is optimal.
:func:`validate_reward_spec` proves that for any configured combination; ``TaskSpec`` runs it on construction.

Trip penalty (plan §10 2026-10-08): a step that trips NAN / RANGE / BOX_PRESENT / STUCK is terminal and pays the
task's fixed ``trip_penalty`` instead of these terms (``env/tasks.py`` ``step_reward_*``). :func:`validate_trip_penalty`
proves it is < 0 and at most ``episode_steps`` x :func:`worst_step_reward` (the undiscounted worst trip-free rest of
an episode), so tripping never beats any trip-free behaviour, for any discount.

``info`` (numpy): a mapping, or a sequence of mappings (HER passes the stored transitions' infos); ``info``
(torch): a mapping of tensors (the batched env). A term that needs an ``info`` key raises ``KeyError`` when it is
missing: it never defaults. Terms with weight 0 are skipped (and need no ``info``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from so101_rl.config.schema import REWARD_TERM_PARAMS, RewardSpec, RewardTermSpec

if TYPE_CHECKING:
    import torch

__all__ = [
    "REWARD_TERMS",
    "RewardBounds",
    "RewardSpecError",
    "RewardTerm",
    "evaluate_np",
    "evaluate_torch",
    "terms_np",
    "terms_torch",
    "validate_reward_spec",
    "validate_trip_penalty",
    "worst_step_reward",
]


class RewardSpecError(ValueError):
    """The configured reward could pay a positive step reward, or does not make finishing strictly best."""


@dataclass(frozen=True)
class StepNp:
    dist: np.ndarray  # float64[...], |achieved - desired|
    success: np.ndarray  # bool[...], the instantaneous success predicate (visible included)
    info: Any  # mapping | sequence of mappings | None


@dataclass(frozen=True)
class StepT:
    dist: torch.Tensor  # float64[...]
    success: torch.Tensor  # bool[...]
    info: Mapping[str, torch.Tensor] | None


def _info_np(info: Any, key: str) -> np.ndarray:
    if info is None:
        raise KeyError(f"reward term needs info[{key!r}] and no info was passed")
    if isinstance(info, Mapping):
        return np.asarray(info[key], np.float64)
    return np.asarray([np.asarray(i[key], np.float64) for i in info])


def _info_t(info: Mapping[str, torch.Tensor] | None, key: str) -> torch.Tensor:
    import torch

    if info is None:
        raise KeyError(f"reward term needs info[{key!r}] and no info was passed")
    return info[key].to(torch.float64)


def _distance_np(t: RewardTermSpec, s: StepNp) -> np.ndarray:
    v = -s.dist / t.scale_m
    return np.where(s.success, 0.0, v) if t.zero_on_success else v


def _distance_t(t: RewardTermSpec, s: StepT) -> torch.Tensor:
    import torch

    v = -s.dist / t.scale_m
    return torch.where(s.success, torch.zeros_like(v), v) if t.zero_on_success else v


def _time_t(t: RewardTermSpec, s: StepT) -> torch.Tensor:
    import torch

    return torch.full_like(s.dist, -1.0)


@dataclass(frozen=True)
class RewardTerm:
    """A term's implementations and the range of its unweighted value (``sup <= 0`` is what the validator uses)."""

    name: str
    inf: float
    sup: float
    np_fn: Callable[[RewardTermSpec, StepNp], np.ndarray]
    torch_fn: Callable[[RewardTermSpec, StepT], Any]


REWARD_TERMS: Final[Mapping[str, RewardTerm]] = {
    t.name: t
    for t in (
        RewardTerm("distance", -math.inf, 0.0, _distance_np, _distance_t),
        RewardTerm(
            "success",
            -1.0,
            0.0,
            lambda t, s: s.success.astype(np.float64) - 1.0,
            lambda t, s: s.success.to(s.dist.dtype) - 1.0,
        ),
        RewardTerm(
            "clamp",
            -1.0,
            0.0,
            lambda t, s: -(_info_np(s.info, "action_clamped") != 0).astype(np.float64),
            lambda t, s: -(_info_t(s.info, "action_clamped") != 0).to(s.dist.dtype),
        ),
        RewardTerm(
            "action_l2",
            -math.inf,
            0.0,
            lambda t, s: -np.mean(np.square(_info_np(s.info, "applied_action")), axis=-1),
            lambda t, s: -_info_t(s.info, "applied_action").square().mean(dim=-1),
        ),
        RewardTerm("time", -1.0, -1.0, lambda t, s: np.full(np.shape(s.dist), -1.0), _time_t),
    )
}
if set(REWARD_TERMS) != set(REWARD_TERM_PARAMS):  # the loader's known-term table and the implementations
    raise RuntimeError(f"REWARD_TERMS {sorted(REWARD_TERMS)} != config REWARD_TERM_PARAMS {sorted(REWARD_TERM_PARAMS)}")


@dataclass(frozen=True)
class RewardBounds:
    """What :func:`validate_reward_spec` proved: every step reward <= ``max_step`` (= -w_time < 0) and every step
    whose success predicate is false <= ``max_unsuccessful_step`` (= -(w_success + w_time))."""

    max_step: float
    max_unsuccessful_step: float


def validate_reward_spec(spec: RewardSpec) -> RewardBounds:
    """Prove that every step of ``spec`` earns a strictly negative reward and unsuccessful steps even less.

    Construction rules (each checked here, so the proof holds for ANY configured combination):

    1. every term name is a registered term, at most once;
    2. every term's value is <= 0 (``REWARD_TERMS[name].sup <= 0``);
    3. every weight is finite and >= 0, so ``sum(w_i * term_i) <= sum(w_i * sup_i) <= 0``;
    4. ``success`` is present with weight > 0: a step without the predicate earns <= -w_success < 0, a strict gap
       to the terminal state (0), so finishing beats dipping in and out of the zone or hiding the die;
    5. ``distance`` has ``scale_m > 0``;
    6. ``time`` is present with weight > 0 (it is the only term not zero on success steps): every step, in-zone
       steps included, earns <= -w_time < 0, so a policy that never terminates is strictly worse than finishing.

    "Non-terminal" steps are a subset of all steps: the bound covers them. Returns the proved bounds; raises :class:`RewardSpecError`.
    """
    names = [t.name for t in spec.terms]
    if not names:
        raise RewardSpecError("reward has no terms")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise RewardSpecError(f"reward terms listed twice: {dup}")
    max_step = 0.0
    for t in spec.terms:
        term = REWARD_TERMS.get(t.name)
        if term is None:
            raise RewardSpecError(f"unknown reward term {t.name!r} (known: {sorted(REWARD_TERMS)})")
        if not (math.isfinite(t.weight) and t.weight >= 0.0):
            raise RewardSpecError(f"reward term {t.name!r}: weight must be finite and >= 0, got {t.weight}")
        if term.sup > 0.0:
            raise RewardSpecError(f"reward term {t.name!r} can be positive (sup {term.sup}): not allowed")
        if t.name == "distance" and not (t.scale_m is not None and t.scale_m > 0.0):
            raise RewardSpecError(f"reward term 'distance': scale_m must be > 0, got {t.scale_m}")
        max_step += t.weight * term.sup
    s = spec.term("success")
    if s is None or not s.weight > 0.0:
        raise RewardSpecError(
            "reward needs the 'success' term with weight > 0: otherwise an unsuccessful step can earn 0 and "
            "never finishing is as good as finishing"
        )
    tm = spec.term("time")
    if tm is None or not tm.weight > 0.0:
        raise RewardSpecError(
            "reward needs the 'time' term with weight > 0: otherwise a die kept moving inside the zone earns 0 per "
            "step, a tie with finishing"
        )
    max_step = min(max_step, -tm.weight)  # == -w_time: every other registered term has sup 0
    if max_step >= 0.0:  # unreachable with the rules above; kept as the stated invariant
        raise RewardSpecError(f"reward can be non-negative on a non-terminal step: max step reward {max_step}")
    return RewardBounds(max_step=max_step, max_unsuccessful_step=-(s.weight + tm.weight))


def worst_step_reward(spec: RewardSpec, *, max_dist_m: float) -> float:
    """The lowest step reward ``spec`` can pay when ``|achieved - desired| <= max_dist_m``: every term at its
    infimum, ``distance`` at ``-max_dist_m / scale_m`` and ``action_l2`` at -1 (both envs clip ``applied_action`` to
    [-1, 1])."""
    total = 0.0
    for t in spec.terms:
        if t.name == "distance":
            v = -max_dist_m / float(t.scale_m)  # type: ignore[arg-type]  # validate_reward_spec: scale_m > 0
        elif t.name == "action_l2":
            v = -1.0
        else:
            v = REWARD_TERMS[t.name].inf
        if not math.isfinite(v):
            raise RewardSpecError(f"reward term {t.name!r} has no finite worst value: add it to worst_step_reward")
        total += t.weight * v
    return total


def validate_trip_penalty(spec: RewardSpec, trip_penalty: float, *, episode_steps: int, max_dist_m: float) -> float:
    """Prove that a trip is worse than any trip-free outcome (plan §10 2026-10-08 "Trip penalty").

    A trip step is terminal and pays ``trip_penalty`` instead of the step reward. The rest of a trip-free episode
    has at most ``episode_steps`` steps, each paying at least :func:`worst_step_reward`, so
    ``trip_penalty <= episode_steps * worst_step_reward`` makes tripping strictly no better than the worst possible
    continuation, for any discount ``gamma <= 1`` (the bound is undiscounted, so it does not depend on the learner).
    Also ``trip_penalty < 0`` (rewards stay non-positive). Returns the bound; raises :class:`RewardSpecError`.
    """
    if not (math.isfinite(trip_penalty) and trip_penalty < 0.0):
        raise RewardSpecError(f"trip_penalty must be finite and < 0, got {trip_penalty}")
    bound = episode_steps * worst_step_reward(spec, max_dist_m=max_dist_m)
    if trip_penalty > bound:
        raise RewardSpecError(
            f"trip_penalty {trip_penalty} is not worse than the worst trip-free return {bound:.2f} "
            f"({episode_steps} steps x worst step {bound / episode_steps:.3f} at |achieved - desired| <= {max_dist_m:.3f} m)"
        )
    return bound


def _active(spec: RewardSpec) -> Sequence[RewardTermSpec]:
    return [t for t in spec.terms if t.weight != 0.0]


def terms_np(spec: RewardSpec, *, dist: np.ndarray, success: np.ndarray, info: Any = None) -> dict[str, np.ndarray]:
    """Weighted value of every non-zero-weight term, float64[...] (for logging; their sum is the reward)."""
    step = StepNp(np.asarray(dist, np.float64), np.asarray(success, bool), info)
    return {t.name: t.weight * REWARD_TERMS[t.name].np_fn(t, step) for t in _active(spec)}


def evaluate_np(spec: RewardSpec, *, dist: np.ndarray, success: np.ndarray, info: Any = None) -> np.ndarray:
    """Step reward, float32[...]: the sum of :func:`terms_np`."""
    total = np.zeros(np.shape(dist), np.float64)
    for v in terms_np(spec, dist=dist, success=success, info=info).values():
        total = total + v
    return total.astype(np.float32)


def terms_torch(
    spec: RewardSpec, *, dist: torch.Tensor, success: torch.Tensor, info: Mapping[str, torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    """Tensor twin of :func:`terms_np` (float64[...])."""
    import torch

    step = StepT(dist.to(torch.float64), success.to(torch.bool), info)
    return {t.name: t.weight * REWARD_TERMS[t.name].torch_fn(t, step) for t in _active(spec)}


def evaluate_torch(
    spec: RewardSpec, *, dist: torch.Tensor, success: torch.Tensor, info: Mapping[str, torch.Tensor] | None = None
) -> torch.Tensor:
    """Tensor twin of :func:`evaluate_np` (float32[...])."""
    import torch

    total = torch.zeros(dist.shape, dtype=torch.float64, device=dist.device)
    for v in terms_torch(spec, dist=dist, success=success, info=info).values():
        total = total + v
    return total.to(torch.float32)

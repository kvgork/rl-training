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
==============  ===============  ==========================================================================

Default (both tasks): ``-dist/0.30 * (1 - pred) - 1 * (1 - pred)``: 0 on success steps, <= -1 otherwise.

Why non-positive (the SU0 review): the original ``-dist/scale + bonus * predicate`` paid up to +1 per step, so a
policy earned more by NOT finishing (dip in and out of the zone, or hide the die every other step to keep the
episode alive) than by finishing. With every non-terminal step <= 0 and every unsuccessful step <= -w_success < 0,
the terminal state (value 0) is the best place to be and the fastest settled finish is optimal.
:func:`validate_reward_spec` proves that for any configured combination; ``TaskSpec`` runs it on construction.

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
    )
}
if set(REWARD_TERMS) != set(REWARD_TERM_PARAMS):  # the loader's known-term table and the implementations
    raise RuntimeError(f"REWARD_TERMS {sorted(REWARD_TERMS)} != config REWARD_TERM_PARAMS {sorted(REWARD_TERM_PARAMS)}")


@dataclass(frozen=True)
class RewardBounds:
    """What :func:`validate_reward_spec` proved: every step reward <= ``max_step`` (= 0) and every step whose
    success predicate is false <= ``max_unsuccessful_step`` (< 0)."""

    max_step: float
    max_unsuccessful_step: float


def validate_reward_spec(spec: RewardSpec) -> RewardBounds:
    """Prove that no step of ``spec`` earns a positive reward and that unsuccessful steps are strictly negative.

    Construction rules (each checked here, so the proof holds for ANY configured combination):

    1. every term name is a registered term, at most once;
    2. every term's value is <= 0 (``REWARD_TERMS[name].sup <= 0``);
    3. every weight is finite and >= 0, so ``sum(w_i * term_i) <= sum(w_i * sup_i) <= 0``;
    4. ``success`` is present with weight > 0: a step without the predicate earns <= -w_success < 0, a strict gap
       to the terminal state (0), so finishing beats dipping in and out of the zone or hiding the die;
    5. ``distance`` has ``scale_m > 0``.

    Since the success predicate needs the die visible and the debounce terminates the episode, "non-terminal" steps
    are a subset of all steps: the bound covers them. Returns the proved bounds; raises :class:`RewardSpecError`.
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
    if max_step > 0.0:  # unreachable with the rules above; kept as the stated invariant
        raise RewardSpecError(f"reward can be positive: max step reward {max_step}")
    return RewardBounds(max_step=max_step, max_unsuccessful_step=-s.weight)


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

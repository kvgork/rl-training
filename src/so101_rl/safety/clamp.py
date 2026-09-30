"""Pure-numpy clamps (plan F5). Every function writes into caller-owned ``out`` arrays.

Conventions
-----------
* ``mask`` arguments are OVERWRITTEN with "this clamp modified joint i" (``out != target``). The pipeline ORs the
  per-stage masks into ``Verdict.clamped_mask``. Because the mask is computed as ``out != target``, ``out`` must
  NOT alias ``target`` when a mask is wanted.
* Targets are int32 ticks that have already passed ``clamp_joint_limits`` (so differences cannot overflow int32).
* **Anti-windup** is a property of how the *caller stores* the result, not of these functions: the value the
  caller keeps as its next ``prev_target`` / ``goal`` is the CLAMPED one (``out``), never the raw request. A joint
  that is blocked therefore never accumulates an unbounded error that would unwind as a violent jump later.
* No per-call array allocation: only ufuncs with ``out=`` are used (numpy may still create tiny scalars).
"""

from __future__ import annotations

import numpy as np

from so101_rl.config import N_JOINTS, TICKS_PER_REV, JointLimits

from .types import TripCause

_MAX_TICK = TICKS_PER_REV - 1


def _symmetric_clip_inplace(out: np.ndarray, bound: np.ndarray) -> None:
    """``out = clip(out, -bound, +bound)`` without allocating: max(x, -b) == -min(-x, b)."""
    np.minimum(out, bound, out=out)
    np.negative(out, out=out)
    np.minimum(out, bound, out=out)
    np.negative(out, out=out)


def sanitize_delta(delta: np.ndarray, max_delta: np.ndarray, out: np.ndarray) -> TripCause:
    """Policy delta -> bounded delta.

    Any NaN/+-inf anywhere in ``delta`` returns ``TripCause.NAN`` and zeroes the WHOLE ``out`` (a corrupt
    vector is not partially trusted). Otherwise every component is clipped to ``[-max_delta, +max_delta]``
    (huge finite values included) and ``TripCause.NONE`` is returned. ``out`` must be a float array.
    """
    # delta - delta == 0 for every finite value (whatever its magnitude) and NaN for NaN/inf.
    with np.errstate(invalid="ignore", over="ignore"):
        np.subtract(delta, delta, out=out)
        finite = out.sum() == 0.0
        if not finite:
            out[...] = 0.0
            return TripCause.NAN
        # A huge negative float64 may overflow to -inf in out's dtype here; the next three ops still map it
        # to exactly -max_delta (max(x, -b) == -min(-x, b)), so the overflow is benign.
        np.minimum(delta, max_delta, out=out)
        np.negative(out, out=out)
        np.minimum(out, max_delta, out=out)
        np.negative(out, out=out)
    return TripCause.NONE


def clamp_joint_limits(target: np.ndarray, limits: JointLimits, out: np.ndarray, mask: np.ndarray) -> None:
    """``out = clip(target, limits.min_ticks, limits.max_ticks)``; ``mask`` = joints that were moved."""
    np.clip(target, limits.min_ticks, limits.max_ticks, out=out)
    np.not_equal(out, target, out=mask)


def _clamp_around(
    target: np.ndarray, center: np.ndarray, bound: np.ndarray, out: np.ndarray, mask: np.ndarray
) -> None:
    np.subtract(target, center, out=out)
    _symmetric_clip_inplace(out, bound)
    np.add(out, center, out=out)
    np.not_equal(out, target, out=mask)


def clamp_velocity(
    target: np.ndarray, prev_target: np.ndarray, vmax: np.ndarray, out: np.ndarray, mask: np.ndarray
) -> None:
    """Per-30-Hz-tick velocity cap: ``|out - prev_target| <= vmax``."""
    _clamp_around(target, prev_target, vmax, out, mask)


def clamp_tracking(target: np.ndarray, present: np.ndarray, E: np.ndarray, out: np.ndarray, mask: np.ndarray) -> None:
    """Tracking-error cap: ``|out - present| <= E``.

    E must exceed ``vmax x (2-3 ticks + read latency)`` or the loop limit-cycles (real-time-control guidance); the
    caller should log the fraction of ticks this clamp is active.
    """
    _clamp_around(target, present, E, out, mask)


def check_range(goal: np.ndarray) -> bool:
    """True iff ``goal`` is an INTEGER array of shape ``(N_JOINTS,)`` with every value in ``0..4095``.

    This is the last call before ``Bus.write_goal`` (ARCHITECTURE section 6.3): a float dtype (so NaN/inf can never
    reach the serial encoder), a wrong shape, or any out-of-range value is rejected BEFORE serialisation.
    """
    if not isinstance(goal, np.ndarray) or goal.dtype.kind not in "iu" or goal.shape != (N_JOINTS,):
        return False
    return bool(goal.min() >= 0 and goal.max() <= _MAX_TICK)

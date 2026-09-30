"""Bounded safe-state sequence (plan F5, ARCHITECTURE section 6.4).

``RAMP_HOME -> REST -> TORQUE_OFF``, driven one servo tick at a time by ``next()``; the whole sequence is bounded
by ``cfg.bound_s`` measured on the injected clock from ``start()``. If it cannot finish in time (bus failing,
arm blocked, gravity fighting the ramp) it falls through to torque-off IN PLACE (``timed_out=True``): a dropped
arm is preferred to an unbounded energised one, and the hardware deadman is the next layer.

Ramp semantics are a port of ``ramped_home`` in ``arm_motor_writer.py``: the target moves toward the goal pose by
at most ``ramp_ticks_per_tick`` per tick (the caller ticks at ``servo_hz``), starting from where the arm IS.
When ``tracking_error_ticks`` is given the ramp target never leads the present position by more than E (same
clamp as in the normal pipeline): a blocked joint stalls the ramp rather than winding the target up, and the
bound then ends the sequence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from so101_rl.config import JOINT_NAMES, N_JOINTS, SafeStateConfig
from so101_rl.servo.timing import Clock

from . import clamp

SETTLE_TICKS = 25  # |present - phase goal| at which a phase counts as reached (module parameter, not a limit)
_NS = 1_000_000_000


class Phase(Enum):
    RAMP_HOME = "ramp_home"
    REST = "rest"
    TORQUE_OFF = "torque_off"
    DONE = "done"


@dataclass(frozen=True)
class SafeStep:
    write_target: bool  # True: ``target_out`` was filled and must be written this tick
    torque_off_ids: tuple[int, ...]  # motor ids to torque off NOW, in order (empty = none)
    done: bool  # sequence finished (normally or timed out); nothing more to do
    timed_out: bool  # finished because ``bound_s`` expired (torque off in place)


_FINISHED = SafeStep(False, (), True, False)
_FINISHED_TIMED_OUT = SafeStep(False, (), True, True)


def ramp_toward(current: np.ndarray, goal: np.ndarray, step: int, out: np.ndarray) -> None:
    """``out = current`` moved toward ``goal`` by at most ``step`` ticks per joint (int32 arrays, no allocation)."""
    np.subtract(goal, current, out=out)
    np.minimum(out, step, out=out)
    np.negative(out, out=out)
    np.minimum(out, step, out=out)
    np.negative(out, out=out)
    np.add(out, current, out=out)


class SafeStateSequencer:
    """RAMP_HOME -> REST -> TORQUE_OFF, bounded by ``cfg.bound_s``; falls through to torque-off in place."""

    def __init__(
        self,
        cfg: SafeStateConfig,
        clock: Clock,
        *,
        tracking_error_ticks: np.ndarray | None = None,
        motor_ids: Mapping[str, int] | None = None,
    ) -> None:
        if cfg.home_ticks is None or cfg.rest_ticks is None:
            raise ValueError("SafeStateConfig.home_ticks and rest_ticks are required")
        self._cfg = cfg
        self._clock = clock
        self._bound_ns = int(cfg.bound_s * _NS)
        self._step = int(cfg.ramp_ticks_per_tick)
        self._home = np.asarray(cfg.home_ticks, dtype=np.int32)
        self._rest = np.asarray(cfg.rest_ticks, dtype=np.int32)
        self._E = None if tracking_error_ticks is None else np.asarray(tracking_error_ticks, dtype=np.int32)
        ids = motor_ids or {n: i + 1 for i, n in enumerate(JOINT_NAMES)}
        self._off_ids = tuple(int(ids[n]) for n in cfg.torque_off_order)
        self._target = np.zeros(N_JOINTS, dtype=np.int32)
        self._scratch = np.zeros(N_JOINTS, dtype=np.int32)
        self._mask = np.zeros(N_JOINTS, dtype=bool)
        self._t0_ns = 0
        self.phase = Phase.DONE  # inactive until start()
        self.timed_out = False

    @property
    def active(self) -> bool:
        return self.phase is not Phase.DONE

    def start(self, present_ticks: np.ndarray) -> None:
        """Begin (or restart) the sequence. The ramp starts from the present position, never from a stale target."""
        self._target[...] = present_ticks
        self._t0_ns = self._clock.monotonic_ns()
        self.phase = Phase.RAMP_HOME
        self.timed_out = False

    def next(self, present_ticks: np.ndarray, target_out: np.ndarray) -> SafeStep:
        if self.phase is Phase.DONE:
            return _FINISHED_TIMED_OUT if self.timed_out else _FINISHED
        if self._clock.monotonic_ns() - self._t0_ns >= self._bound_ns:
            self.phase = Phase.DONE
            self.timed_out = True
            return SafeStep(False, self._off_ids, True, True)

        if self.phase is Phase.TORQUE_OFF:
            self.phase = Phase.DONE
            return SafeStep(False, self._off_ids, True, False)

        goal = self._home if self.phase is Phase.RAMP_HOME else self._rest
        ramp_toward(self._target, goal, self._step, self._scratch)
        if self._E is not None:
            clamp.clamp_tracking(self._scratch, present_ticks, self._E, self._target, self._mask)
        else:
            self._target[...] = self._scratch
        target_out[...] = self._target

        at_goal = bool((self._target == goal).all())
        settled = bool((np.abs(present_ticks - goal) <= SETTLE_TICKS).all())
        if at_goal and settled:
            self.phase = Phase.REST if self.phase is Phase.RAMP_HOME else Phase.TORQUE_OFF
        return SafeStep(True, (), False, False)

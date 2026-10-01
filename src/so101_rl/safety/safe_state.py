"""Bounded safe-state sequence (plan F5, ARCHITECTURE section 6.4).

``RAMP_HOME -> REST -> TORQUE_OFF``, driven one servo tick at a time by ``next()``; the whole sequence is bounded
by ``cfg.bound_s`` measured on the injected clock from ``start()``. If it cannot finish in time (bus failing,
arm blocked, gravity fighting the ramp, the transit zone holding every joint) it falls through to torque-off IN
PLACE (``timed_out=True``): a dropped arm is preferred to an unbounded energised one, and the hardware deadman is
the next layer. The loader requires ``bound_s`` to cover a ramp from any joint limit to home plus the exact
home -> rest transit (review B5).

Ramp semantics are a port of ``ramped_home`` in ``arm_motor_writer.py``: the target moves toward the goal pose by
at most ``ramp_ticks_per_tick`` per tick (the caller ticks at ``servo_hz``), starting from where the arm IS, and
every tick goes through the transit-zone planner (``safety/transit.py``: lift first, never below the table
floor). The ramp target never leads the present position by more than E (``tracking_error_ticks``, same clamp as
in the normal pipeline): a blocked joint stalls the ramp rather than winding the target up, and the bound then
ends the sequence.

Constructor arguments ``tracking_error_ticks``, ``motor_ids`` and ``fk`` are REQUIRED (review S2): a default
silently dropped the tracking clamp and torqued off motor ids 1-6 instead of the calibration ids.

:meth:`SafeStateSequencer.run` is the blocking driver for process exits (``ServoCore.shutdown``). It calls
``tick_hook(ok)`` once per servo tick, ``ok`` = that tick's block read succeeded AND its bus write was accepted,
so the caller can keep the deadman heartbeat alive (review B2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from so101_rl.config import N_JOINTS, SafeStateConfig
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.servo.timing import Clock

from . import clamp
from .fk import So101FK
from .transit import TransitPlanner, ramp_toward

__all__ = ["SETTLE_TICKS", "Phase", "SafeStateSequencer", "SafeStep", "TickHook", "ramp_toward"]

_TORQUE_OFF_ATTEMPTS = 3
SETTLE_TICKS = 25  # |present - phase goal| at which a phase counts as reached (module parameter, not a limit)
_NS = 1_000_000_000

TickHook = Callable[[bool], None]


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


def _torque_off_retry(bus: Bus, ids: list[int]) -> bool:
    for _ in range(_TORQUE_OFF_ATTEMPTS):
        try:
            if bus.set_torque(False, ids):
                return True
        except Exception:  # noqa: BLE001, S112 - keep trying; run() raises if every attempt failed
            continue
    return False


class SafeStateSequencer:
    """RAMP_HOME -> REST -> TORQUE_OFF, bounded by ``cfg.bound_s``; falls through to torque-off in place."""

    def __init__(
        self,
        cfg: SafeStateConfig,
        clock: Clock,
        *,
        tracking_error_ticks: np.ndarray,
        motor_ids: Mapping[str, int],
        fk: So101FK,
    ) -> None:
        if cfg.home_ticks is None or cfg.rest_ticks is None:
            raise ValueError("SafeStateConfig.home_ticks and rest_ticks are required")
        if tracking_error_ticks is None or motor_ids is None or fk is None:
            raise TypeError("SafeStateSequencer needs tracking_error_ticks, motor_ids and fk (review S2)")
        missing = [n for n in cfg.torque_off_order if n not in motor_ids]
        if missing:
            raise ValueError(f"motor_ids is missing joints {missing}")
        self._cfg = cfg
        self._clock = clock
        self._bound_ns = int(cfg.bound_s * _NS)
        self._home = np.asarray(cfg.home_ticks, dtype=np.int32)
        self._rest = np.asarray(cfg.rest_ticks, dtype=np.int32)
        self._E = np.asarray(tracking_error_ticks, dtype=np.int32)
        if self._E.shape != (N_JOINTS,) or bool((self._E <= 0).any()):
            raise ValueError(f"tracking_error_ticks must be {N_JOINTS} positive values, got {self._E.tolist()}")
        self._off_ids = tuple(int(motor_ids[n]) for n in cfg.torque_off_order)
        self._planner = TransitPlanner(fk, cfg)
        self._target = np.zeros(N_JOINTS, dtype=np.int32)
        self._scratch = np.zeros(N_JOINTS, dtype=np.int32)
        self._clamped = np.zeros(N_JOINTS, dtype=np.int32)
        self._mask = np.zeros(N_JOINTS, dtype=bool)
        self._t0_ns = 0
        self.phase = Phase.DONE  # inactive until start()
        self.timed_out = False

    @property
    def active(self) -> bool:
        return self.phase is not Phase.DONE

    @property
    def torque_off_ids(self) -> tuple[int, ...]:
        return self._off_ids

    def start(self, present_ticks: np.ndarray) -> None:
        """Begin (or restart) the sequence. The ramp starts from the present position, never from a stale target."""
        self._target[...] = present_ticks
        self._t0_ns = self._clock.monotonic_ns()
        self.phase = Phase.RAMP_HOME
        self.timed_out = False

    def next(self, present_ticks: np.ndarray, target_out: np.ndarray, *, block_ok: bool = True) -> SafeStep:
        """One servo tick. With ``block_ok=False`` (this tick's read failed, ``present_ticks`` is stale) the ramp
        makes no progress: the previous target is re-issued and only the time bound advances."""
        if self.phase is Phase.DONE:
            return _FINISHED_TIMED_OUT if self.timed_out else _FINISHED
        if self._clock.monotonic_ns() - self._t0_ns >= self._bound_ns:
            self.phase = Phase.DONE
            self.timed_out = True
            return SafeStep(False, self._off_ids, True, True)

        if self.phase is Phase.TORQUE_OFF:
            self.phase = Phase.DONE
            return SafeStep(False, self._off_ids, True, False)

        if not block_ok:
            target_out[...] = self._target
            return SafeStep(True, (), False, False)

        goal = self._home if self.phase is Phase.RAMP_HOME else self._rest
        self._planner.next_target(self._target, goal, self._scratch)
        clamp.clamp_tracking(self._scratch, present_ticks, self._E, self._clamped, self._mask)
        # The tracking clamp moves joints independently; never let it take the tip below the table floor.
        if self._planner.acceptable(self._target, self._clamped):
            self._target[...] = self._clamped
        target_out[...] = self._target

        at_goal = bool((self._target == goal).all())
        settled = bool((np.abs(present_ticks - goal) <= SETTLE_TICKS).all())
        if at_goal and settled:
            self.phase = Phase.REST if self.phase is Phase.RAMP_HOME else Phase.TORQUE_OFF
        return SafeStep(True, (), False, False)

    def run(self, bus: Bus, block: BlockRead, *, servo_hz: float, tick_hook: TickHook | None = None) -> SafeStep:
        """Blocking driver: read -> ``next`` -> write (or torque off) -> ``tick_hook(ok)`` -> sleep, at ``servo_hz``,
        until the sequence is done. A torque-off that is refused 3 times raises ``RuntimeError`` (the caller must
        not believe the arm is limp; the deadman is the next layer). If the sequence is not active it is started from the first successful read
        (the bound still counts from entry; no good read within ``bound_s`` -> torque off in place). Returns the
        final step (``timed_out`` tells whether the bound ended it)."""
        period_ns = int(_NS / servo_hz)
        target = np.zeros(N_JOINTS, dtype=np.int32)
        t_entry = self._clock.monotonic_ns()
        next_tick = t_entry
        started = self.active
        while True:
            got = bool(bus.read_block(block)) and block.ok
            if not started:
                if got:
                    self.start(block.position)
                    self._t0_ns = t_entry
                    started = True
                elif self._clock.monotonic_ns() - t_entry >= self._bound_ns:
                    ok = _torque_off_retry(bus, list(self._off_ids))
                    if tick_hook is not None:
                        tick_hook(False)
                    if not ok:
                        raise RuntimeError("safe state: no good read within bound_s and torque off failed")
                    self.phase = Phase.DONE
                    self.timed_out = True
                    return SafeStep(False, self._off_ids, True, True)
                else:
                    if tick_hook is not None:
                        tick_hook(False)
                    next_tick += period_ns
                    self._clock.sleep_until(next_tick)
                    continue
            step = self.next(block.position, target, block_ok=got)
            wrote = False
            if step.write_target:
                wrote = clamp.check_range(target) and bool(bus.write_goal(target))
            if step.torque_off_ids:
                wrote = _torque_off_retry(bus, list(step.torque_off_ids))
                if not wrote:
                    if tick_hook is not None:
                        tick_hook(False)
                    raise RuntimeError(
                        f"safe state: torque off of motors {step.torque_off_ids} failed {_TORQUE_OFF_ATTEMPTS} times; "
                        "the deadman is the next layer"
                    )
            if tick_hook is not None:
                tick_hook(got and wrote)
            if step.done:
                return step
            next_tick += period_ns
            self._clock.sleep_until(next_tick)

"""Safety state machine (plan F5, real-time-control-specialist section 5, ARCHITECTURE section 6).

States: ``BOOT, CHECK, IDLE_TORQUE_OFF, RUN, HOLD, COOLDOWN, SAFE_STOPPING, TORQUE_OFF, LATCHED``.
``CHECK`` and ``IDLE_TORQUE_OFF`` are pre-energise states owned by the servo core's start-up; the machine leaves
``BOOT``/``CHECK``/``IDLE_TORQUE_OFF`` only through :meth:`mark_energized` (called by ``ServoCore.start`` after
``reenergize`` completed).

Transition table (first matching row wins, rows are evaluated top to bottom every ``step``)

=================  ==================================================  =================  ================
from               condition                                           to                 resume
=================  ==================================================  =================  ================
LATCHED            always                                              LATCHED (absorbing) physical reset
any                ``hb_ok`` false, MCU latched, HEARTBEAT_FAIL,       LATCHED            human + physical
                   DEADMAN or BUS_FAIL in trips, or
                   ``max_bus_failures`` consecutive ``block_ok`` false
TORQUE_OFF         always                                              TORQUE_OFF          new process / human
SAFE_STOPPING      ``notify_safe_done()`` (sequencer finished or       TORQUE_OFF
                   timed out)
RUN/HOLD/COOLDOWN  NAN, STUCK, OOD, RANGE, BOX_PRESENT in trips       SAFE_STOPPING       fresh process
RUN/HOLD/COOLDOWN  PROTECTION_BIT, CALIB_MISMATCH, USER_STOP in       SAFE_STOPPING       human
                   trips, or ``stop_requested``
BOOT/CHECK/IDLE    any of the two rows above                           TORQUE_OFF
RUN/HOLD           THERMAL in trips (<= ``max_trips_per_hour`` in      COOLDOWN            automatic
                   the last hour)
RUN/HOLD           THERMAL in trips, trip number > max per hour        SAFE_STOPPING       human
COOLDOWN           ``may_resume_thermal`` and THERMAL not in trips     RUN (resync)        -
RUN/HOLD           ``consecutive_misses >= max_consecutive_misses``    SAFE_STOPPING       fresh process
RUN                ``cmd_age_ns > stale_cmd_ms``                       HOLD                -
HOLD               ``cmd_fresh`` and age <= ``stale_cmd_ms``           RUN (resync)        -
=================  ==================================================  =================  ================

``TRACKING``, ``JOINT_LIMIT`` and ``BOX`` in ``trips`` are clamps, never state-machine transitions.

``resync_target`` is true on the step that enters RUN (first step after ``mark_energized``, HOLD -> RUN,
COOLDOWN -> RUN): the pipeline must then ``resync(present)`` so no stale target is resumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from so101_rl.config import So101Config
from so101_rl.servo.timing import Clock

from .types import RESUME_BY_CAUSE, ResumePolicy, SmState, TripCause

if TYPE_CHECKING:  # heartbeat.py belongs to the servo package (U1); duck-typed at runtime
    from so101_rl.servo.heartbeat import McuStatus

_NS = 1_000_000_000
_HOUR_NS = 3600 * _NS
_RING = 32

_LINK_CAUSES = TripCause.HEARTBEAT_FAIL | TripCause.DEADMAN | TripCause.BUS_FAIL
_FRESH_PROCESS_CAUSES = TripCause.NAN | TripCause.STUCK | TripCause.OOD | TripCause.RANGE | TripCause.BOX_PRESENT
_HUMAN_CAUSES = TripCause.PROTECTION_BIT | TripCause.CALIB_MISMATCH | TripCause.USER_STOP
_AUTO_CAUSES = TripCause.THERMAL | TripCause.STALE_CMD
_PRE_RUN = (SmState.BOOT, SmState.CHECK, SmState.IDLE_TORQUE_OFF)
_POLICY_RANK = {ResumePolicy.AUTO: 0, ResumePolicy.FRESH_PROCESS: 1, ResumePolicy.HUMAN_PHYSICAL_RESET: 2}


@dataclass
class SmInputs:
    block_ok: bool = True
    cmd_fresh: bool = False  # a NEW command (new seq) arrived since the previous step
    cmd_age_ns: int = 0  # servo-clock age of the newest command
    consecutive_misses: int = 0  # consecutive 100 ms policy windows without a new seq (counted by the core)
    trips: TripCause = TripCause.NONE  # from SafetyPipeline.apply / thermal / stuck
    hb_ok: bool = True
    mcu: McuStatus | None = None
    stop_requested: bool = False
    may_resume_thermal: bool = False


@dataclass
class SmOutputs:
    state: SmState = SmState.BOOT
    use_policy: bool = False  # integrate the policy command this tick (RUN and a good block read only)
    run_safe_sequence: bool = False  # drive SafeStateSequencer this tick (SAFE_STOPPING)
    torque_off_now: bool = False  # LATCHED entry (best effort) and every tick of TORQUE_OFF; idempotent
    resync_target: bool = False  # goal = prev_target = present, this tick


class SafetyStateMachine:
    def __init__(self, cfg: So101Config, clock: Clock) -> None:
        self._rates = cfg.rates
        self._max_trips_hour = cfg.thermal.max_trips_per_hour
        self._stale_ns = int(cfg.rates.stale_cmd_ms * 1_000_000)
        self._clock = clock
        self.state = SmState.BOOT
        self.fault_bits = TripCause.NONE
        self._bus_failures = 0
        self._resync_pending = False
        self._thermal_ns = np.zeros(_RING, dtype=np.int64)
        self._n_thermal = 0
        self._human_required = False  # the (max+1)-th thermal trip in an hour

    # -- external events ---------------------------------------------------------------------------------------
    def mark_energized(self) -> None:
        """``reenergize`` completed: leave the pre-run states for RUN (with ``resync_target`` on the next step)."""
        if self.state in _PRE_RUN:
            self.state = SmState.RUN
            self.fault_bits &= ~_AUTO_CAUSES
            self._resync_pending = True

    def notify_safe_done(self) -> None:
        """The safe-state sequence finished (normally or timed out): SAFE_STOPPING -> TORQUE_OFF."""
        if self.state is SmState.SAFE_STOPPING:
            self.state = SmState.TORQUE_OFF

    def resume_policy(self) -> ResumePolicy | None:
        """Strictest resume policy among the active fault bits (None when no fault is active)."""
        worst: ResumePolicy | None = None
        for cause, policy in RESUME_BY_CAUSE.items():
            if self.fault_bits & cause and (worst is None or _POLICY_RANK[policy] > _POLICY_RANK[worst]):
                worst = policy
        if worst is ResumePolicy.AUTO and self.state in (SmState.SAFE_STOPPING, SmState.TORQUE_OFF):
            worst = ResumePolicy.FRESH_PROCESS  # RESUME_BY_CAUSE: stale command after SAFE_STOPPING
        if self._human_required or self.state is SmState.LATCHED:
            worst = ResumePolicy.HUMAN_PHYSICAL_RESET
        return worst

    def thermal_trips_last_hour(self) -> int:
        n = min(self._n_thermal, _RING)
        return int((self._thermal_ns[:n] > self._clock.monotonic_ns() - _HOUR_NS).sum())

    # -- the step ----------------------------------------------------------------------------------------------
    def step(self, inp: SmInputs, out: SmOutputs) -> SmOutputs:
        resync = self._resync_pending
        self._resync_pending = False
        torque_off_now = False

        if self.state is SmState.LATCHED:
            return self._emit(out, resync=False, torque_off_now=False)

        # bus failures: a single one holds the target (use_policy false); N in a row is a link fault
        self._bus_failures = 0 if inp.block_ok else self._bus_failures + 1
        link = inp.trips & _LINK_CAUSES
        if not inp.hb_ok:
            link |= TripCause.HEARTBEAT_FAIL
        if inp.mcu is not None and inp.mcu.latched:
            link |= TripCause.DEADMAN
        if self._bus_failures >= self._rates.max_bus_failures:
            link |= TripCause.BUS_FAIL
        if link:
            self.fault_bits |= link
            self.state = SmState.LATCHED
            return self._emit(out, resync=False, torque_off_now=True)

        if self.state is SmState.TORQUE_OFF:
            self.fault_bits |= inp.trips & (_FRESH_PROCESS_CAUSES | _HUMAN_CAUSES | TripCause.THERMAL)
            return self._emit(out, resync=False, torque_off_now=True)

        if self.state is SmState.SAFE_STOPPING:
            self.fault_bits |= inp.trips & (_FRESH_PROCESS_CAUSES | _HUMAN_CAUSES | TripCause.THERMAL)
            return self._emit(out, resync=False, torque_off_now=False)

        # ---- RUN / HOLD / COOLDOWN / pre-run ----------------------------------------------------------------
        stop = inp.trips & (_FRESH_PROCESS_CAUSES | _HUMAN_CAUSES)
        if inp.stop_requested:
            stop |= TripCause.USER_STOP
        if stop:
            self.fault_bits |= stop
            if self.state in _PRE_RUN:
                self.state = SmState.TORQUE_OFF
                torque_off_now = True
            else:
                self.state = SmState.SAFE_STOPPING
            return self._emit(out, resync=False, torque_off_now=torque_off_now)

        if self.state in (SmState.RUN, SmState.HOLD) and inp.trips & TripCause.THERMAL:
            self.fault_bits |= TripCause.THERMAL
            self._record_thermal_trip()
            if self.thermal_trips_last_hour() > self._max_trips_hour:
                self._human_required = True
                self.state = SmState.SAFE_STOPPING  # human: no automatic resume (resume_policy() says so)
            else:
                self.state = SmState.COOLDOWN
            return self._emit(out, resync=False, torque_off_now=False)

        if self.state is SmState.COOLDOWN:
            if inp.may_resume_thermal and not inp.trips & TripCause.THERMAL:
                self._enter_run()
                return self._emit(out, resync=True, torque_off_now=False)
            return self._emit(out, resync=False, torque_off_now=False)

        if self.state in (SmState.RUN, SmState.HOLD):
            if inp.consecutive_misses >= self._rates.max_consecutive_misses:
                self.fault_bits |= TripCause.STALE_CMD
                self.state = SmState.SAFE_STOPPING  # after this only a fresh process resumes
                return self._emit(out, resync=False, torque_off_now=False)
            if self.state is SmState.RUN:
                if inp.cmd_age_ns > self._stale_ns:
                    self.fault_bits |= TripCause.STALE_CMD
                    self.state = SmState.HOLD
            elif inp.cmd_fresh and inp.cmd_age_ns <= self._stale_ns:
                self._enter_run()
                resync = True
        return self._emit(out, resync=resync, torque_off_now=False, block_ok=inp.block_ok)

    # -- helpers -----------------------------------------------------------------------------------------------
    def _enter_run(self) -> None:
        self.state = SmState.RUN
        self.fault_bits &= ~_AUTO_CAUSES

    def _record_thermal_trip(self) -> None:
        self._thermal_ns[self._n_thermal % _RING] = self._clock.monotonic_ns()
        self._n_thermal += 1

    def _emit(self, out: SmOutputs, *, resync: bool, torque_off_now: bool, block_ok: bool = True) -> SmOutputs:
        out.state = self.state
        out.use_policy = self.state is SmState.RUN and block_ok
        out.run_safe_sequence = self.state is SmState.SAFE_STOPPING
        out.torque_off_now = torque_off_now or self.state is SmState.TORQUE_OFF
        out.resync_target = resync and self.state is SmState.RUN
        return out

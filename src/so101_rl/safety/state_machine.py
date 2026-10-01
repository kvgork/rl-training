"""Safety state machine (plan F5, real-time-control-specialist section 5, ARCHITECTURE section 6).

States: ``BOOT, CHECK, IDLE_TORQUE_OFF, RUN, HOLD, COOLDOWN, SAFE_STOPPING, TORQUE_OFF, LATCHED``.
``CHECK`` and ``IDLE_TORQUE_OFF`` are pre-energise states owned by the servo core's start-up; the machine leaves
``BOOT``/``CHECK``/``IDLE_TORQUE_OFF`` for RUN only through :meth:`mark_energized` (called by the core after
``reenergize`` completed). The machine itself enters ``CHECK`` only for the automatic thermal resume.

Transition table (first matching row wins, rows are evaluated top to bottom every ``step``)

=================  ===================================================  =================  ================
from               condition                                            to                 resume
=================  ===================================================  =================  ================
LATCHED            always (``torque_off_now`` every tick)               LATCHED (absorbing) physical reset
any                ``hb_ok`` false, MCU latched, HEARTBEAT_FAIL,        LATCHED            human + physical
                   DEADMAN or BUS_FAIL in trips, or
                   ``max_bus_failures`` consecutive ``block_ok`` false
TORQUE_OFF         thermal-only fault, ``may_resume_thermal``,          CHECK              automatic
                   THERMAL not in trips, ``block_ok``                   (``reenergize_requested``)
TORQUE_OFF         otherwise                                            TORQUE_OFF          new process / human
SAFE_STOPPING      ``notify_safe_done()`` (sequencer finished or        TORQUE_OFF
                   timed out)
RUN/HOLD           NAN, STUCK, OOD, RANGE, BOX_PRESENT in trips        SAFE_STOPPING       fresh process
RUN/HOLD           PROTECTION_BIT, CALIB_MISMATCH, USER_STOP in        SAFE_STOPPING       human
                   trips, or ``stop_requested``
BOOT/CHECK/IDLE    any of the two rows above                            TORQUE_OFF
CHECK              THERMAL in trips (latched again before energising)   TORQUE_OFF          automatic
RUN/HOLD           THERMAL in trips (<= ``max_trips_per_hour`` in the   SAFE_STOPPING       automatic
                   last hour): to rest, then torque off
RUN/HOLD           THERMAL in trips, trip number > max per hour         SAFE_STOPPING       human
RUN/HOLD           ``consecutive_misses >= max_consecutive_misses``     SAFE_STOPPING       fresh process
RUN                ``cmd_age_ns > stale_cmd_ms``                        HOLD                -
HOLD               ``cmd_fresh``, age <= ``stale_cmd_ms``, ``block_ok``  RUN (resync)        -
CHECK              :meth:`mark_energized`                               RUN (resync)        -
CHECK              :meth:`notify_reenergize_failed`                     TORQUE_OFF          human
=================  ===================================================  =================  ================

Thermal (plan section 10, 2026-10-01): a thermal trip never holds pose while energised. It runs the safe-state
sequence to rest and turns torque off. Once ``may_resume_thermal`` (the monitor's cool-down dwell) holds, the
machine goes TORQUE_OFF -> CHECK and sets ``reenergize_requested`` for that one step; the core then runs the FULL
``reenergize`` and calls :meth:`mark_energized` (CHECK -> RUN, resync) or :meth:`notify_reenergize_failed`
(-> TORQUE_OFF, human). At most ``max_trips_per_hour`` automatic resumes per trailing hour: the next trip needs a
human. ``COOLDOWN`` stays in :class:`SmState` (frozen F5 enum) but is no longer entered.

This machine is the ONE source of the thermal trip count behind that limit (:meth:`thermal_trips_last_hour`,
review S10). ``ThermalMonitor.trips_last_hour`` counts monitor latch events for diagnostics and the soak report.

``TRACKING``, ``JOINT_LIMIT`` and ``BOX`` in ``trips`` are clamps, never state-machine transitions.

``resync_target`` is true on the step that enters RUN (the first good-read step after ``mark_energized``, and
HOLD -> RUN): the pipeline must then ``resync(present)`` so no stale target is resumed. Every resume requires
``block_ok`` (review S6): a pending resync waits for a step with a good block read, and ``use_policy`` is false
until then.
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
    trips: TripCause = TripCause.NONE  # from SafetyPipeline.apply / monitor_only (thermal) / stuck
    hb_ok: bool = True
    mcu: McuStatus | None = None
    stop_requested: bool = False
    may_resume_thermal: bool = False


@dataclass
class SmOutputs:
    state: SmState = SmState.BOOT
    use_policy: bool = False  # integrate the policy command this tick (RUN and a good block read only)
    run_safe_sequence: bool = False  # drive SafeStateSequencer this tick (SAFE_STOPPING)
    torque_off_now: bool = False  # every tick of LATCHED and TORQUE_OFF, and pre-run stops; idempotent
    resync_target: bool = False  # goal = prev_target = present, this tick
    reenergize_requested: bool = False  # TORQUE_OFF -> CHECK on this step: run the full reenergize (thermal resume)


class SafetyStateMachine:
    def __init__(self, cfg: So101Config, clock: Clock) -> None:
        self._rates = cfg.rates
        self._max_trips_hour = int(cfg.thermal.max_trips_per_hour)
        self._stale_ns = int(cfg.rates.stale_cmd_ms * 1_000_000)
        self._clock = clock
        self.state = SmState.BOOT
        self.fault_bits = TripCause.NONE
        self._bus_failures = 0
        self._resync_pending = False
        # The last max+1 thermal trip times: exactly enough to tell whether a trip is the (max+1)-th in an hour.
        self._thermal_ns = np.zeros(self._max_trips_hour + 1, dtype=np.int64)
        self._n_thermal = 0
        self._human_required = False  # the (max+1)-th thermal trip in an hour, or a failed automatic re-energise
        self._fresh_required = False  # a non-thermal stop (N stale commands) happened: no automatic resume

    # -- external events ---------------------------------------------------------------------------------------
    def mark_energized(self) -> None:
        """``reenergize`` completed: leave the pre-run states for RUN (``resync_target`` on the next good step)."""
        if self.state in _PRE_RUN:
            self.state = SmState.RUN
            self.fault_bits &= ~_AUTO_CAUSES
            self._resync_pending = True

    def notify_reenergize_failed(self) -> None:
        """The automatic (thermal) ``reenergize`` raised: CHECK -> TORQUE_OFF, and only a human resumes."""
        if self.state is SmState.CHECK:
            self.state = SmState.TORQUE_OFF
            self._human_required = True

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
        if (
            worst is ResumePolicy.AUTO
            and self.state in (SmState.SAFE_STOPPING, SmState.TORQUE_OFF)
            and not self._thermal_only()
        ):
            worst = ResumePolicy.FRESH_PROCESS  # RESUME_BY_CAUSE: stale command after SAFE_STOPPING
        if self._human_required or self.state is SmState.LATCHED:
            worst = ResumePolicy.HUMAN_PHYSICAL_RESET
        return worst

    def thermal_trips_last_hour(self) -> int:
        n = min(self._n_thermal, len(self._thermal_ns))
        return int((self._thermal_ns[:n] > self._clock.monotonic_ns() - _HOUR_NS).sum())

    # -- the step ----------------------------------------------------------------------------------------------
    def step(self, inp: SmInputs, out: SmOutputs) -> SmOutputs:
        if self.state is SmState.LATCHED:
            return self._emit(out, resync=False, torque_off_now=True)  # every tick (review S4); idempotent

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
            self._resync_pending = False
            return self._emit(out, resync=False, torque_off_now=True)

        if self.state is SmState.TORQUE_OFF:
            self.fault_bits |= inp.trips & (_FRESH_PROCESS_CAUSES | _HUMAN_CAUSES | TripCause.THERMAL)
            if self._thermal_only() and inp.may_resume_thermal and not inp.trips & TripCause.THERMAL and inp.block_ok:
                self.state = SmState.CHECK  # the core runs the FULL reenergize, then mark_energized()
                return self._emit(out, resync=False, torque_off_now=True, reenergize=True)
            return self._emit(out, resync=False, torque_off_now=True)

        if self.state is SmState.SAFE_STOPPING:
            self.fault_bits |= inp.trips & (_FRESH_PROCESS_CAUSES | _HUMAN_CAUSES | TripCause.THERMAL)
            return self._emit(out, resync=False, torque_off_now=False)

        # ---- RUN / HOLD / pre-run --------------------------------------------------------------------------
        stop = inp.trips & (_FRESH_PROCESS_CAUSES | _HUMAN_CAUSES)
        if inp.stop_requested:
            stop |= TripCause.USER_STOP
        if stop:
            self.fault_bits |= stop
            self._resync_pending = False
            if self.state in _PRE_RUN:
                self.state = SmState.TORQUE_OFF
                return self._emit(out, resync=False, torque_off_now=True)
            self.state = SmState.SAFE_STOPPING
            return self._emit(out, resync=False, torque_off_now=False)

        if self.state is SmState.CHECK and inp.trips & TripCause.THERMAL:
            self.fault_bits |= TripCause.THERMAL  # latched again before energising: wait in TORQUE_OFF again
            self.state = SmState.TORQUE_OFF
            return self._emit(out, resync=False, torque_off_now=True)

        if self.state in (SmState.RUN, SmState.HOLD) and inp.trips & TripCause.THERMAL:
            # plan section 10 (2026-10-01): to rest and torque off, never hold pose while energised
            self.fault_bits |= TripCause.THERMAL
            self._record_thermal_trip()
            if self.thermal_trips_last_hour() > self._max_trips_hour:
                self._human_required = True  # resume_policy() says HUMAN; TORQUE_OFF never auto-resumes
            self.state = SmState.SAFE_STOPPING
            self._resync_pending = False
            return self._emit(out, resync=False, torque_off_now=False)

        resync = False
        if self.state in (SmState.RUN, SmState.HOLD):
            if inp.consecutive_misses >= self._rates.max_consecutive_misses:
                self.fault_bits |= TripCause.STALE_CMD
                self._fresh_required = True  # a later THERMAL must not turn this stop into an automatic resume
                self.state = SmState.SAFE_STOPPING  # after this only a fresh process resumes
                self._resync_pending = False
                return self._emit(out, resync=False, torque_off_now=False)
            if self.state is SmState.RUN:
                # a pending resync (mark_energized) is delivered only on a good block read (review S6)
                if self._resync_pending and inp.block_ok:
                    self._resync_pending = False
                    resync = True
                if inp.cmd_age_ns > self._stale_ns:
                    self.fault_bits |= TripCause.STALE_CMD
                    self.state = SmState.HOLD
            elif inp.cmd_fresh and inp.cmd_age_ns <= self._stale_ns and inp.block_ok:
                self._enter_run()
                self._resync_pending = False
                resync = True
        return self._emit(out, resync=resync, torque_off_now=False, block_ok=inp.block_ok)

    # -- helpers -----------------------------------------------------------------------------------------------
    def _thermal_only(self) -> bool:
        """The only active faults are THERMAL (plus STALE_CMD from a HOLD before it) and no human is required."""
        return (
            not self._human_required
            and not self._fresh_required
            and bool(self.fault_bits & TripCause.THERMAL)
            and not self.fault_bits & ~(TripCause.THERMAL | TripCause.STALE_CMD)
        )

    def _enter_run(self) -> None:
        self.state = SmState.RUN
        self.fault_bits &= ~_AUTO_CAUSES

    def _record_thermal_trip(self) -> None:
        self._thermal_ns[self._n_thermal % len(self._thermal_ns)] = self._clock.monotonic_ns()
        self._n_thermal += 1

    def _emit(
        self, out: SmOutputs, *, resync: bool, torque_off_now: bool, block_ok: bool = True, reenergize: bool = False
    ) -> SmOutputs:
        out.state = self.state
        out.use_policy = self.state is SmState.RUN and block_ok and not self._resync_pending
        out.run_safe_sequence = self.state is SmState.SAFE_STOPPING
        out.torque_off_now = torque_off_now or self.state in (SmState.TORQUE_OFF, SmState.LATCHED)
        out.resync_target = resync and self.state is SmState.RUN
        out.reenergize_requested = reenergize
        return out

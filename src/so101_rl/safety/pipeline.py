"""Per-tick safety pipeline (plan F5). The order is fixed by ARCHITECTURE section 6.3:

    new_command (10 Hz):  sanitize_delta -> goal += delta -> clamp_joint_limits(goal)
    apply (30 Hz):        interpolate -> joint limits -> per-tick velocity -> tracking (anti-windup)
                          -> Cartesian box (FK of the target) -> thermal -> protection bits -> stuck -> check_range

State kept between calls (all int32[6], preallocated):

* ``goal``        the interpolation end point of the current 10 Hz command. **Anti-windup:** it is pulled back with
                  every clamp, so it never drifts more than E away from ``present``.
* ``prev_target`` what was written last tick (the start of the next interpolation step and the velocity-clamp
                  reference; the value a BOX violation reverts to).

``resync(present)`` (called when the state machine enters RUN, ``resync_target``) sets both to the present position.

Semantics worth knowing
-----------------------
* BOX on the *commanded* target is a CLAMP: the target reverts to ``prev_target`` (and the goal with it), reported in
  ``clamped_mask`` and ``TripCause.BOX``. BOX_PRESENT (the *present* position is more than
  ``box.present_tolerance_m`` outside the box) is a TRIP: the state machine stops the arm (fresh process). The
  tolerance (review S3) keeps a millimetre of sag at a face from ending the run: commands are clamped to the
  exact box, the present position only trips beyond the tolerance band.
* ``clamped_mask`` is the OR of every stage that modified a joint this tick, plus the command-level clamps of
  ``new_command`` (sanitize/limits) for the tick that follows. ``TRACKING``, ``JOINT_LIMIT`` and ``BOX`` are reported
  in ``Verdict.trips`` for information only; the state machine ignores them.
* ``applied_delta_ticks`` = (goal after this tick's clamps) - (goal before the current command): the post-clamp
  goal change that is LOGGED (L4). On the final tick of a command it is the command's complete applied delta.
* ``apply`` only runs in RUN. In every other state the core must call ``monitor_only`` each tick so the thermal
  monitor keeps receiving temperatures (cooldown dwell, stale-temperature detection).
* ``check_range`` is the last call. If it fails the target is deliberately left as-is (out of range) and
  ``TripCause.RANGE`` is set, so the core's own pre-write ``check_range`` rejects it a second time.
"""

from __future__ import annotations

import numpy as np

from so101_rl.config import N_JOINTS, So101Config

from . import clamp
from .fk import So101FK
from .stuck import StuckActionMonitor
from .thermal import ThermalMonitor
from .types import TripCause, Verdict


class SafetyPipeline:
    def __init__(self, cfg: So101Config, fk: So101FK, thermal: ThermalMonitor, stuck: StuckActionMonitor) -> None:
        self._cfg = cfg
        self._fk = fk
        self._thermal = thermal
        self._stuck = stuck
        n = N_JOINTS
        self.goal = np.zeros(n, dtype=np.int32)
        self.prev_target = np.zeros(n, dtype=np.int32)
        self._goal_at_cmd_start = np.zeros(n, dtype=np.int32)
        self._synced = False
        # scratch
        self._delta = np.zeros(n, dtype=np.float32)
        self._f = np.zeros(n, dtype=np.float64)
        self._t0 = np.zeros(n, dtype=np.int32)
        self._t1 = np.zeros(n, dtype=np.int32)
        self._g = np.zeros(n, dtype=np.int32)
        self._mask = np.zeros(n, dtype=bool)
        self._cmd_mask = np.zeros(n, dtype=bool)
        self._cmd_trips = TripCause.NONE  # NAN / JOINT_LIMIT raised by new_command, reported by the next apply
        self._xyz = np.zeros(3, dtype=np.float64)

    # -- state ---------------------------------------------------------------------------------------------
    def resync(self, present_ticks: np.ndarray) -> None:
        """goal = prev_target = present (every entry to RUN). Also forgets the stuck-monitor history."""
        self.goal[...] = present_ticks
        self.prev_target[...] = present_ticks
        self._goal_at_cmd_start[...] = present_ticks
        self._cmd_mask[...] = False
        self._cmd_trips = TripCause.NONE
        self._delta[...] = 0.0
        self._synced = True
        self._stuck.reset()

    # -- 10 Hz: a new policy command -----------------------------------------------------------------------
    def new_command(self, delta_ticks: np.ndarray, v: Verdict) -> TripCause:
        """sanitize_delta -> goal = goal + delta -> clamp_joint_limits(goal). Returns NAN for a non-finite delta
        (the goal is then left untouched and ``v.applied_delta_ticks`` is zeroed)."""
        trip = clamp.sanitize_delta(delta_ticks, self._cfg.action.max_delta_ticks, self._delta)
        self._goal_at_cmd_start[...] = self.goal
        if trip:
            v.applied_delta_ticks[...] = 0.0
            self._cmd_trips |= trip
            return trip
        np.rint(self._delta, out=self._delta)
        np.add(self.goal, self._delta, out=self._f)
        np.clip(self._f, -(2**30), 2**30, out=self._f)  # keep the float->int cast defined; limits do the real work
        self._t0[...] = self._f  # float64 -> int32 (truncation is exact: values are integral)
        clamp.clamp_joint_limits(self._t0, self._cfg.joints, self._g, self._cmd_mask)
        self.goal[...] = self._g
        np.subtract(self.goal, self._goal_at_cmd_start, out=v.applied_delta_ticks, casting="unsafe")
        if self._cmd_mask.any():
            self._cmd_trips |= TripCause.JOINT_LIMIT
        return TripCause.NONE

    # -- 30 Hz: one tick -----------------------------------------------------------------------------------
    def apply(self, block, ticks_remaining: int, v: Verdict) -> Verdict:
        cfg = self._cfg
        present = block.position
        if not self._synced:
            if not block.ok:
                # Never resync to a stale read (review follow-up to S6): emit an out-of-range target that the
                # core's pre-write check_range rejects, and report RANGE.
                v.trips = TripCause.RANGE
                v.target_ticks[...] = -1
                return v
            self.resync(present)
        v.trips = self._cmd_trips  # command-level results (NAN, JOINT_LIMIT) are reported once
        self._cmd_trips = TripCause.NONE
        mask = v.clamped_mask
        mask[...] = self._cmd_mask
        self._cmd_mask[...] = False
        m = self._mask

        # 1. interpolate prev_target -> goal; the last remaining tick lands exactly on the goal
        rem = max(int(ticks_remaining), 1)
        np.subtract(self.goal, self.prev_target, out=self._f)
        np.divide(self._f, rem, out=self._f)
        np.rint(self._f, out=self._f)
        np.add(self.prev_target, self._f, out=self._t0, casting="unsafe")

        # 2. joint limits
        clamp.clamp_joint_limits(self._t0, cfg.joints, self._t1, m)
        self._note(mask, m, v, TripCause.JOINT_LIMIT)

        # 3. per-tick velocity
        clamp.clamp_velocity(self._t1, self.prev_target, cfg.action.max_ticks_per_tick, self._t0, m)
        mask |= m

        # 4. tracking error |target - present| <= E; anti-windup pulls the GOAL back to the same band
        clamp.clamp_tracking(self._t0, present, cfg.action.tracking_error_ticks, self._t1, m)
        self._note(mask, m, v, TripCause.TRACKING)
        clamp.clamp_tracking(self.goal, present, cfg.action.tracking_error_ticks, self._g, m)
        self.goal[...] = self._g
        target = self._t1

        # 5. Cartesian box through FK. Present outside -> trip. Commanded target outside -> revert (clamp).
        fk = self._fk
        fk.gripper_xyz_from_ticks(present, self._xyz)
        if not fk.in_box(self._xyz, cfg.box, margin=cfg.box.present_tolerance_m):
            v.trips |= TripCause.BOX_PRESENT
            target = self.prev_target  # hold what was last written; the state machine takes over
        else:
            fk.gripper_xyz_from_ticks(target, self._xyz)
            if not fk.in_box(self._xyz, cfg.box):
                np.not_equal(target, self.prev_target, out=m)
                self._note(mask, m, v, TripCause.BOX)
                target = self.prev_target
                self.goal[...] = self.prev_target  # anti-windup: the stored goal is the clamped one

        # 6./7. thermal trip and firmware protection bits (one monitor, two causes)
        v.trips |= self._thermal.update(block.temperature_c, block.status, block.ok)

        # results
        v.target_ticks[...] = target
        v.goal_ticks[...] = self.goal
        np.subtract(self.goal, self._goal_at_cmd_start, out=v.applied_delta_ticks, casting="unsafe")

        # 8. stuck-action monitor. It watches the sanitized COMMANDED delta: with anti-windup the post-clamp delta of a
        #    blocked joint decays to zero, which would hide exactly the "same command, arm not moving" symptom.
        v.trips |= self._stuck.update(self._delta, block.moving)

        # 9. last: the range check that the core repeats immediately before write_goal
        if not clamp.check_range(v.target_ticks):
            v.trips |= TripCause.RANGE

        self.prev_target[...] = v.target_ticks
        return v

    def monitor_only(self, block, v: Verdict) -> Verdict:
        """Thermal + protection-bit monitoring without touching any target: the core calls this every tick in
        states where ``apply`` is not run (HOLD, SAFE_STOPPING, TORQUE_OFF including the thermal dwell), because
        the thermal latch can only clear, and ``may_resume`` can only advance, while ``ThermalMonitor.update``
        keeps being fed."""
        v.trips = self._thermal.update(block.temperature_c, block.status, block.ok)
        return v

    @staticmethod
    def _note(mask: np.ndarray, stage_mask: np.ndarray, v: Verdict, cause: TripCause) -> None:
        mask |= stage_mask
        if stage_mask.any():
            v.trips |= cause

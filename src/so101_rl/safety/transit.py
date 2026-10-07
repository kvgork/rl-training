"""Transit zone between the rest pose and home (review B1).

The box check that guards RUN does not run on the ramps that leave or return to rest: those ramps are outside RUN,
and the rest pose rests on the table (its tip at z = -0.012 m has been inside the box since the floor moved to
table + 5 mm, plan §10 2026-10-06, but a pose near rest can still sag below that floor). Instead they are checked
against the TABLE, independently of the box: every ramp tick of ``reenergize`` (rest -> home) and of the
:class:`~so101_rl.safety.safe_state.SafeStateSequencer` (-> home -> rest) goes through
:meth:`TransitPlanner.next_target`:

1. Every joint moves toward its goal by at most ``ramp_ticks_per_tick`` (``ramp_toward``).
2. **Lift first.** While the tip of the current target is inside the transit zone (below
   ``table_z + clearance + transit_zone_m``), a joint whose own step would LOWER the tip is held as long as
   any other joint is still LIFTING it. In practice the shoulder and elbow lift the arm off the table before the
   wrist folds down, and on the way back the wrist and elbow finish before the last descent.
3. **Table floor.** A candidate whose tip is below ``table_z + clearance`` is accepted only if it does not lower
   the tip further. Otherwise the lowering joints are held; if that is still not enough, every joint is held.

A held ramp makes no progress, so the caller's time bound (the sequencer's ``bound_s``, reenergize's ramp
budget) then ends it with torque off. The loader replays exactly this planner tick by tick from rest to home and
from home to rest (``safety/config_check.py``), so a shipped configuration never relies on that fall-back.

The planner checks the TARGET. The present position lags it by the servo's tracking error, which the tracking
clamp (applied by the callers after this step) bounds.
"""

from __future__ import annotations

import numpy as np

from so101_rl.config import N_JOINTS, SafeStateConfig

from .fk import So101FK

_DZ_EPS_M = 1e-6  # |dz| below this counts as "does not change the tip height" (gripper, wrist roll)


def ramp_toward(current: np.ndarray, goal: np.ndarray, step: int, out: np.ndarray) -> None:
    """``out = current`` moved toward ``goal`` by at most ``step`` ticks per joint (int32 arrays, no allocation)."""
    np.subtract(goal, current, out=out)
    np.minimum(out, step, out=out)
    np.negative(out, out=out)
    np.minimum(out, step, out=out)
    np.negative(out, out=out)
    np.add(out, current, out=out)


class TransitPlanner:
    """One ramp tick toward a goal with the lift-first ordering and the table-floor check. Not thread-safe."""

    def __init__(self, fk: So101FK, cfg: SafeStateConfig) -> None:
        self._fk = fk
        self.step_ticks = int(cfg.ramp_ticks_per_tick)
        self.floor_z = float(cfg.table_z_m + cfg.table_clearance_m)
        self.zone_top_z = float(self.floor_z + cfg.transit_zone_m)
        self._cand = np.zeros(N_JOINTS, dtype=np.int32)
        self._trial = np.zeros(N_JOINTS, dtype=np.int32)
        self._dz = np.zeros(N_JOINTS, dtype=np.float64)
        self._xyz = np.zeros(3, dtype=np.float64)

    def tip_z(self, ticks: np.ndarray) -> float:
        return float(self._fk.gripper_xyz_from_ticks(ticks, self._xyz)[2])

    def acceptable(self, prev: np.ndarray, new: np.ndarray) -> bool:
        """The floor rule on its own: ``new`` is above the floor, or does not lower the tip below ``prev``."""
        z_new = self.tip_z(new)
        return z_new >= self.floor_z or z_new >= self.tip_z(prev)

    def next_target(self, current: np.ndarray, goal: np.ndarray, out: np.ndarray) -> None:
        """``out`` = the next ramp target from ``current`` toward ``goal`` (int32[6]; ``out`` may alias neither)."""
        cand = self._cand
        ramp_toward(current, goal, self.step_ticks, cand)
        if bool((cand == current).all()):
            out[...] = current
            return
        z0 = self.tip_z(current)
        self._per_joint_dz(current, cand, z0)
        lowering = self._dz < -_DZ_EPS_M
        if z0 < self.zone_top_z and bool((self._dz > _DZ_EPS_M).any()):
            cand[lowering] = current[lowering]  # lift first
        z1 = self.tip_z(cand)
        if z1 < self.floor_z and z1 < z0:
            cand[lowering] = current[lowering]
            z1 = self.tip_z(cand)
            if z1 < self.floor_z and z1 < z0:
                cand[...] = current  # hold everything; the caller's time bound ends the ramp
        out[...] = cand

    def _per_joint_dz(self, current: np.ndarray, cand: np.ndarray, z0: float) -> None:
        """``dz[i]`` = tip height change if ONLY joint i took its step (0 for joints that do not move)."""
        trial = self._trial
        trial[...] = current
        for i in range(N_JOINTS):
            if cand[i] == current[i]:
                self._dz[i] = 0.0
                continue
            trial[i] = cand[i]
            self._dz[i] = self.tip_z(trial) - z0
            trial[i] = current[i]


def simulate_transit(
    planner: TransitPlanner, start: np.ndarray, goal: np.ndarray, max_ticks: int
) -> tuple[int, float, bool]:
    """Replay the planner with a perfectly following arm: ``(ticks, min tip z along the path, reached goal)``."""
    cur = np.asarray(start, dtype=np.int32).copy()
    goal = np.asarray(goal, dtype=np.int32)
    nxt = np.empty_like(cur)
    z_min = planner.tip_z(cur)
    for n in range(max_ticks):
        if bool((cur == goal).all()):
            return n, z_min, True
        planner.next_target(cur, goal, nxt)
        if bool((nxt == cur).all()):
            return n, z_min, False  # stalled
        cur[...] = nxt
        z_min = min(z_min, planner.tip_z(cur))
    return max_ticks, z_min, bool((cur == goal).all())

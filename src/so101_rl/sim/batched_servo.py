"""Batched servo clamp chain: ``SafetyPipeline.new_command`` / ``apply`` for N worlds in torch (FS5, sim track S1.1).

Decision D-CHAIN (plan §10 2026-10-05): the batched sim RE-IMPLEMENTS the clamp chain instead of vectorising
``safety/pipeline.py``; ``safety/`` stays small, allocation-free and policy-independent. The two are pinned together
by the frozen three-layer parity test (``tests/test_batched_parity.py``, Layer A: bit-identical chains on the same
present-tick stream). Only ``safety.types.TripCause`` and ``safety.fk.KinematicChain`` are imported from ``safety``
(constants only, decision D-DEPS); no clamp code.

Semantics (identical to ``safety/pipeline.py``, which is the source of truth; differences are bugs)
---------------------------------------------------------------------------------------------------
``new_command(delta)`` (10 Hz), per world:

1. ``sanitize_delta``: any non-finite entry zeroes the WHOLE row and flags ``TripCause.NAN`` (reported by the next
   ``apply``); the goal of that world is left untouched. Otherwise every component is clipped to
   ``[-max_delta_ticks, +max_delta_ticks]`` and cast to float32 (the pipeline's ``_delta`` buffer is float32).
2. ``rint`` (round half to even, on the float32 value, like ``np.rint`` on the float32 buffer).
3. ``goal += delta`` in float64, clipped to ``±2**30``, cast to integer.
4. joint-limit clip of the goal; the joints it moved form the command-level mask (``_cmd_mask``) that is ORed into
   the NEXT apply's ``clamped_mask``, with ``TripCause.JOINT_LIMIT``.
5. ``goal_at_cmd_start`` = the goal before the command (also for a NaN row).

``apply(present, ticks_remaining)`` (30 Hz), per world, in the pipeline's order:

1. interpolate: ``prev + rint((goal - prev) / max(rem, 1))`` in float64 (the last remaining tick lands on the goal)
2. joint limits (``JOINT_LIMIT``)
3. per-tick velocity cap ``|target - prev| <= max_ticks_per_tick`` (mask only, no trip bit: like the pipeline)
4. tracking ``|target - present| <= E`` (``TRACKING``); anti-windup: the GOAL is clipped to the same band (its mask
   is not reported, like the pipeline)
5. box via FK (``So101FK.in_box_gripper``: the tip box, AND the lowest of the chain's ``floor_wrist`` /
   ``floor_gripper`` points at or above ``box.gripper_floor_z_m``, plan §10 2026-10-06 "Gripper floor check" /
   2026-10-07 "Gripper floor clearance"): present outside (tip box grown by ``present_tolerance_m``, point floor
   lowered by ``gripper_present_tolerance_m``) -> ``BOX_PRESENT`` and ``target = prev_target`` (the target box test
   is skipped); else the target outside the exact box / point floor -> ``target = goal = prev_target`` with mask
   ``target != prev_target`` and ``BOX`` iff that mask is non-empty
6. thermal / protection bits: NOT emulated (decision D-TRIPS)
7. ``applied_delta_ticks = goal - goal_at_cmd_start`` (float32; the LOGGED action, L4)
8. stuck-action monitor (``safety/stuck.py`` ``StuckActionMonitor``, emulated since plan §10 2026-10-08 "STUCK
   emulated in batched sim"), only when ``apply`` gets the tick's ``moving`` flags: it watches the sanitized,
   rounded COMMANDED delta of the latest ``new_command`` (zero after a NaN row or a resync) and counts consecutive
   ticks on which that delta is non-zero (``max|delta| > STUCK_EPSILON_TICKS``), equal to the delta of the first
   counted tick (every component within ``STUCK_EPSILON_TICKS``) and no joint is ``moving``; ``STUCK`` once the
   count reaches ``STUCK_REPEAT_TICKS`` (90 ticks = 3 s at 30 Hz). Any change, a zero delta, any motion or a resync
   resets the count.
9. ``RANGE`` if any target is outside 0..4095 (the target is left as-is, pipeline semantics)

``prev_target`` = the target of this tick. All tick state is int64 (the pipeline's int32 values fit). Nothing in
``new_command`` / ``apply`` synchronises with the host; ``clamp_active_fraction`` does (one copy) when read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from so101_rl.config import CENTER_TICK, JOINT_NAMES, N_JOINTS, TICKS_PER_REV, MotorCalibration, So101Config
from so101_rl.safety.fk import N_ARM_JOINTS, KinematicChain
from so101_rl.safety.types import TripCause

__all__ = ["CHAIN_TRIP_BITS", "STAGES", "STUCK_EPSILON_TICKS", "STUCK_REPEAT_TICKS", "BatchedServoChain", "ChainOut"]

_RAD_PER_TICK: Final[float] = 2.0 * np.pi / TICKS_PER_REV
_MAX_TICK: Final[int] = TICKS_PER_REV - 1
_GOAL_CLIP: Final[float] = float(2**30)

# The trip bits this chain can set (decision D-TRIPS: THERMAL / PROTECTION_BIT are not emulated; STUCK is, since
# plan §10 2026-10-08, when ``apply`` is given the ``moving`` flags).
CHAIN_TRIP_BITS: Final[int] = int(
    TripCause.NAN
    | TripCause.JOINT_LIMIT
    | TripCause.TRACKING
    | TripCause.BOX
    | TripCause.BOX_PRESENT
    | TripCause.RANGE
    | TripCause.STUCK
)
# ``StuckActionMonitor()`` defaults (the servo builds it with them, ``servo/core.py``). Copied, not imported: ``sim/``
# imports only TripCause / KinematicChain from ``safety`` (D-DEPS); tests/test_batched_servo.py pins the equality.
STUCK_REPEAT_TICKS: Final[int] = 90
STUCK_EPSILON_TICKS: Final[float] = 0.5
# Stages counted by ``clamp_active_fraction`` (fraction of world-ticks on which the stage modified any joint).
STAGES: Final[tuple[str, ...]] = ("joint_limit", "velocity", "tracking", "box")

_NAN, _JL, _TRK = int(TripCause.NAN), int(TripCause.JOINT_LIMIT), int(TripCause.TRACKING)
_BOX, _BOXP, _RANGE = int(TripCause.BOX), int(TripCause.BOX_PRESENT), int(TripCause.RANGE)
_STUCK = int(TripCause.STUCK)


@dataclass
class ChainOut:
    """Preallocated result of one :meth:`BatchedServoChain.apply` (filled in place)."""

    target_ticks: torch.Tensor  # int64[N,6]  what is written this tick
    goal_ticks: torch.Tensor  # int64[N,6]
    applied_delta_ticks: torch.Tensor  # float32[N,6] goal - goal_at_cmd_start (the LOGGED action)
    clamped_mask: torch.Tensor  # bool[N,6]
    trips: torch.Tensor  # int64[N] bitmask of TripCause

    @classmethod
    def empty(cls, nworld: int, device: torch.device | str) -> ChainOut:
        z = torch.zeros((nworld, N_JOINTS), dtype=torch.int64, device=device)
        return cls(
            target_ticks=z,
            goal_ticks=z.clone(),
            applied_delta_ticks=torch.zeros((nworld, N_JOINTS), dtype=torch.float32, device=device),
            clamped_mask=torch.zeros((nworld, N_JOINTS), dtype=torch.bool, device=device),
            trips=torch.zeros(nworld, dtype=torch.int64, device=device),
        )


def _clip_around(x: torch.Tensor, center: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    """``center + clip(x - center, -bound, bound)`` (integer exact)."""
    return center + torch.maximum(torch.minimum(x - center, bound), -bound)


class BatchedServoChain:
    """torch re-implementation of SafetyPipeline.new_command/apply for N worlds, int64 tick math, same rounding."""

    def __init__(
        self,
        cfg: So101Config,
        calibration: dict[str, MotorCalibration],
        chain: KinematicChain,
        nworld: int,
        *,
        device: torch.device | str,
    ) -> None:
        if nworld < 1:
            raise ValueError(f"nworld must be >= 1, got {nworld}")
        self.cfg = cfg
        self.nworld = int(nworld)
        self.device = torch.device(device)
        dev = self.device

        def i64(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.asarray(a, dtype=np.int64), device=dev)

        self._jl_min = i64(cfg.joints.min_ticks)
        self._jl_max = i64(cfg.joints.max_ticks)
        self._max_delta = torch.as_tensor(np.asarray(cfg.action.max_delta_ticks, np.float64), device=dev)
        self.max_ticks_per_tick = i64(cfg.action.max_ticks_per_tick)  # public: the Layer A "shown to fail" twin
        self._E = i64(cfg.action.tracking_error_ticks)
        box = cfg.box
        m = float(box.present_tolerance_m)
        # Same float64 arithmetic as So101FK.in_box_gripper (bounds computed as Python floats): the tip box, and the
        # lowest gripper point against the gripper floor (exact for targets, lowered by its tolerance for present).
        self._gfloor = float(box.gripper_floor_z_m)
        self._gfloor_p = float(box.gripper_floor_z_m) - float(box.gripper_present_tolerance_m)
        self._box_lo = torch.tensor([box.x[0], box.y[0], box.z[0]], dtype=torch.float64, device=dev)
        self._box_hi = torch.tensor([box.x[1], box.y[1], box.z[1]], dtype=torch.float64, device=dev)
        self._boxp_lo = torch.tensor([box.x[0] - m, box.y[0] - m, box.z[0] - m], dtype=torch.float64, device=dev)
        self._boxp_hi = torch.tensor([box.x[1] + m, box.y[1] + m, box.z[1] + m], dtype=torch.float64, device=dev)

        # FK constants (So101FK: q = (ticks - CENTER) * sign * rad_per_tick; a = base; a = a @ P_i @ Rz(q_i))
        self._sign = torch.tensor(
            [-1.0 if calibration[n].drive_mode else 1.0 for n in JOINT_NAMES], dtype=torch.float64, device=dev
        )
        self._base_T = torch.as_tensor(chain.base_T, dtype=torch.float64, device=dev)
        self._parent_T = torch.as_tensor(chain.parent_T, dtype=torch.float64, device=dev)
        self._tip_pos = torch.as_tensor(chain.tip_T[:3, 3], dtype=torch.float64, device=dev)
        self._pw = torch.as_tensor(chain.floor_wrist.T.copy(), dtype=torch.float64, device=dev)  # [3, Kw]
        self._pg = torch.as_tensor(chain.floor_gripper.T.copy(), dtype=torch.float64, device=dev)  # [3, Kg]

        n = self.nworld
        z = torch.zeros((n, N_JOINTS), dtype=torch.int64, device=dev)
        self.goal = z.clone()
        self.prev_target = z.clone()
        self._goal_at_cmd_start = z.clone()
        self._cmd_mask = torch.zeros((n, N_JOINTS), dtype=torch.bool, device=dev)
        self._cmd_trips = torch.zeros(n, dtype=torch.int64, device=dev)
        # stuck-action monitor state: the pipeline's sanitized+rounded float32 ``_delta``, the reference delta and count
        self._delta = torch.zeros((n, N_JOINTS), dtype=torch.float32, device=dev)
        self._stuck_ref = torch.zeros((n, N_JOINTS), dtype=torch.float32, device=dev)
        self.stuck_count = torch.zeros(n, dtype=torch.int64, device=dev)
        self._xyz = torch.zeros((n, 3), dtype=torch.float64, device=dev)
        self._zlow = torch.zeros(n, dtype=torch.float64, device=dev)
        self._counts = torch.zeros(len(STAGES), dtype=torch.int64, device=dev)
        self._world_ticks = 0

    # ------------------------------------------------------------------ state
    def resync(self, present_ticks: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """goal = prev_target = present for the worlds in ``mask`` (all if None); forgets their pending command
        results and the stuck-monitor history (``resync_target`` semantics: every entry to RUN, every episode
        reset)."""
        p = present_ticks.to(device=self.device, dtype=torch.int64)
        if mask is None:
            for t in (self.goal, self.prev_target, self._goal_at_cmd_start):
                t.copy_(p)
            self._cmd_mask.zero_()
            self._cmd_trips.zero_()
            self._delta.zero_()
            self.stuck_count.zero_()
            return
        m = mask.to(device=self.device, dtype=torch.bool)
        m2 = m[:, None]
        for t in (self.goal, self.prev_target, self._goal_at_cmd_start):
            t.copy_(torch.where(m2, p, t))
        self._cmd_mask.masked_fill_(m2, False)
        self._cmd_trips.masked_fill_(m, 0)
        self._delta.masked_fill_(m2, 0.0)
        self.stuck_count.masked_fill_(m, 0)

    # ------------------------------------------------------------------ 10 Hz
    def new_command(self, delta_ticks: torch.Tensor) -> torch.Tensor:
        """float[N,6] -> nan_mask bool[N]. NaN/inf rows: delta zeroed, goal untouched, ``TripCause.NAN`` reported by
        the next :meth:`apply` (``sanitize_delta`` semantics)."""
        d = delta_ticks.to(device=self.device)
        if d.shape != (self.nworld, N_JOINTS):
            raise ValueError(f"delta_ticks must be [{self.nworld}, {N_JOINTS}], got {tuple(d.shape)}")
        d = d.to(torch.float64)
        nan = ~torch.isfinite(d).all(dim=1)
        self._goal_at_cmd_start.copy_(self.goal)
        # sanitize: clip in float64, store float32 (the pipeline's _delta), round half-to-even in float32
        clipped = torch.maximum(torch.minimum(torch.nan_to_num(d, nan=0.0), self._max_delta), -self._max_delta)
        clipped = torch.where(nan[:, None], torch.zeros_like(clipped), clipped)
        delta32 = torch.round(clipped.to(torch.float32))
        self._delta.copy_(delta32)  # the pipeline's _delta: zeroed for a NaN row (sanitize_delta), rounded otherwise
        delta = delta32.to(torch.float64)
        g = (self.goal.to(torch.float64) + delta).clamp(-_GOAL_CLIP, _GOAL_CLIP).to(torch.int64)
        lim = torch.maximum(torch.minimum(g, self._jl_max), self._jl_min)
        cmd_mask = (lim != g) & ~nan[:, None]
        self.goal.copy_(torch.where(nan[:, None], self.goal, lim))
        self._cmd_mask.copy_(torch.where(nan[:, None], self._cmd_mask, cmd_mask))
        trips = torch.where(nan, _NAN, torch.where(cmd_mask.any(dim=1), _JL, 0))
        self._cmd_trips.bitwise_or_(trips)
        return nan

    # ------------------------------------------------------------------ 30 Hz
    def apply(
        self, present_ticks: torch.Tensor, ticks_remaining: int, out: ChainOut, moving: torch.Tensor | None = None
    ) -> ChainOut:
        """One servo tick of every world. ``moving`` (bool/int [N,6], the Moving flags read with ``present``) enables
        the stuck-action stage; without it the stage is skipped (its history is left as it is)."""
        present = present_ticks.to(device=self.device, dtype=torch.int64)
        prev = self.prev_target
        trips = self._cmd_trips.clone()
        self._cmd_trips.zero_()
        mask = self._cmd_mask.clone()
        self._cmd_mask.zero_()

        # 1. interpolate prev_target -> goal
        rem = max(int(ticks_remaining), 1)
        step = torch.round((self.goal - prev).to(torch.float64) / rem)
        t0 = (prev.to(torch.float64) + step).to(torch.int64)
        # 2. joint limits
        t1 = torch.maximum(torch.minimum(t0, self._jl_max), self._jl_min)
        m_jl = t1 != t0
        trips |= torch.where(m_jl.any(dim=1), _JL, 0)
        # 3. per-tick velocity
        t2 = _clip_around(t1, prev, self.max_ticks_per_tick)
        m_vel = t2 != t1
        # 4. tracking (+ anti-windup on the goal)
        t3 = _clip_around(t2, present, self._E)
        m_trk = t3 != t2
        trips |= torch.where(m_trk.any(dim=1), _TRK, 0)
        goal = _clip_around(self.goal, present, self._E)
        # 5. box through FK (x/y/top on the tip, floor on the lowest gripper point)
        xyz_p, zl_p = self.fk_tip_and_floor_z(present, self._xyz, self._zlow)
        present_in = self._in_box(xyz_p, zl_p, self._boxp_lo, self._boxp_hi, self._gfloor_p)
        xyz_t, zl_t = self.fk_tip_and_floor_z(t3, self._xyz, self._zlow)
        target_in = self._in_box(xyz_t, zl_t, self._box_lo, self._box_hi, self._gfloor)
        box_present = ~present_in
        box_revert = present_in & ~target_in
        m_box = (t3 != prev) & box_revert[:, None]
        trips |= torch.where(box_present, _BOXP, 0)
        trips |= torch.where(m_box.any(dim=1), _BOX, 0)
        hold = (box_present | box_revert)[:, None]
        target = torch.where(hold, prev, t3)
        goal = torch.where(box_revert[:, None], prev, goal)
        self.goal.copy_(goal)
        mask |= m_jl | m_vel | m_trk | m_box

        # results
        out.target_ticks.copy_(target)
        out.goal_ticks.copy_(self.goal)
        out.applied_delta_ticks.copy_((self.goal - self._goal_at_cmd_start).to(torch.float32))
        out.clamped_mask.copy_(mask)
        # 8. stuck-action monitor on the commanded delta (StuckActionMonitor.update, vectorised)
        if moving is not None:
            trips |= self._stuck_update(moving)
        # 9. range check last (target left as-is)
        bad = ((target < 0) | (target > _MAX_TICK)).any(dim=1)
        trips |= torch.where(bad, _RANGE, 0)
        out.trips.copy_(trips)
        prev.copy_(target)

        self._counts += torch.stack([m_jl.any(1).sum(), m_vel.any(1).sum(), m_trk.any(1).sum(), m_box.any(1).sum()])
        self._world_ticks += self.nworld
        return out

    def _stuck_update(self, moving: torch.Tensor) -> torch.Tensor:
        eps = STUCK_EPSILON_TICKS
        d = self._delta
        still = ~moving.to(device=self.device, dtype=torch.bool).any(dim=1)
        active = (d.abs().amax(dim=1) > eps) & still
        cnt = self.stuck_count
        changed = (cnt > 0) & ((d - self._stuck_ref).abs().amax(dim=1) > eps)
        cnt = torch.where(changed, 0, cnt)
        self._stuck_ref.copy_(torch.where((active & (cnt == 0))[:, None], d, self._stuck_ref))
        cnt = torch.where(active, cnt + 1, 0)
        self.stuck_count.copy_(cnt)
        return torch.where(cnt >= STUCK_REPEAT_TICKS, _STUCK, 0)

    # ------------------------------------------------------------------ FK
    @staticmethod
    def _in_box(
        xyz: torch.Tensor, zlow: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, floor: float
    ) -> torch.Tensor:
        """``So101FK.in_box_gripper`` for every world: tip in [lo, hi], lowest gripper point >= ``floor``."""
        return ((xyz >= lo) & (xyz <= hi)).all(dim=1) & (zlow >= floor)

    def fk_tip(self, ticks: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Fixed-jaw tip (``gripperframe``) in the base frame, metres, for every world: batched ``So101FK`` (5 Rz)
        from the same ``KinematicChain``. ``ticks`` int[N,6] (only the 5 arm joints are used); fills ``out [N,3]``."""
        out.copy_(self._fk(ticks)[0])
        return out

    def fk_tip_and_floor_z(
        self, ticks: torch.Tensor, out: torch.Tensor, zlow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``So101FK.tip_and_floor_z`` for every world: tip into ``out [N,3]``, the lowest z of every gripper-floor
        point into ``zlow [N]`` (same elementwise float64 arithmetic)."""
        xyz, aw, ag = self._fk(ticks)
        out.copy_(xyz)

        def pts_min(frame: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            r = frame[:, 2, :]  # [N, 4]
            z = p[0] * r[:, 0:1]
            z = z + p[1] * r[:, 1:2]
            z = z + p[2] * r[:, 2:3]
            z = z + r[:, 3:4]
            return z.amin(dim=1)

        zl = torch.minimum(pts_min(aw, self._pw), pts_min(ag, self._pg))
        zlow.copy_(zl)
        return out, zlow

    def _fk(self, ticks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(tip xyz [N,3], wrist frame [N,4,4], gripper frame [N,4,4])."""
        q = (ticks[:, :N_ARM_JOINTS].to(torch.float64) - CENTER_TICK) * self._sign[:N_ARM_JOINTS] * _RAD_PER_TICK
        n = q.shape[0]
        c, s = torch.cos(q), torch.sin(q)
        a = self._base_T.expand(n, 4, 4)
        rz = torch.zeros((n, 4, 4), dtype=torch.float64, device=q.device)
        rz[:, 2, 2] = 1.0
        rz[:, 3, 3] = 1.0
        aw = a
        for i in range(N_ARM_JOINTS):
            b = torch.matmul(a, self._parent_T[i])
            rz[:, 0, 0] = c[:, i]
            rz[:, 0, 1] = -s[:, i]
            rz[:, 1, 0] = s[:, i]
            rz[:, 1, 1] = c[:, i]
            a = torch.matmul(b, rz)
            if i == N_ARM_JOINTS - 2:
                aw = a  # the wrist body frame
        xyz = torch.matmul(a[:, :3, :3], self._tip_pos) + a[:, :3, 3]
        return xyz, aw, a

    # ------------------------------------------------------------------ stats
    @property
    def clamp_active_fraction(self) -> dict[str, float]:
        """Running fraction of world-ticks on which each stage modified at least one joint (one host sync)."""
        if self._world_ticks == 0:
            return dict.fromkeys(STAGES, 0.0)
        counts = self._counts.cpu().tolist()
        return {k: c / self._world_ticks for k, c in zip(STAGES, counts, strict=True)}

    def reset_stats(self) -> None:
        self._counts.zero_()
        self._world_ticks = 0

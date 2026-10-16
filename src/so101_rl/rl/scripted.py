"""Scripted die controller for ``die_to_zone`` (attempt 2, D2b of ``plans/die-to-zone-plan.md``).

A privileged, sim-only pick-and-place controller whose output is exactly the policy's action: normalised per-joint
deltas in [-1, 1] that the env scales by ``cfg.action.max_delta_ticks`` (``tasks.action_to_delta_*``, the F7 rule).
Its episodes seed a FastSAC demo replay (``train_sim_sac`` ``demo_ratio`` / ``demo_episodes``, RLPD-style mixing),
because random SAC exploration never lifts the die out of the start well (``plans/reviews/2026-10-16-s4-die-run.md``).
It is never deployed: it reads the TRUE die pose, which the policy never sees.

**State machine** per world (:class:`Phase`), on the grasp point ``G`` = the point between the jaw tips
(:data:`GRASP_POINT_GRIPPER`, gripper body frame, on the wrist_roll axis):

``OPEN_ABOVE`` open the gripper and bring ``G`` above the die (``pregrasp_h`` over the table, clear of the 26 mm tray
rim), jaws turned to the die's faces -> ``DESCEND`` to ``grasp_h`` -> ``CLOSE`` (gripper command to its closed limit,
held ``close_steps``) -> ``LIFT`` to ``carry_h`` (the die's bottom above the rim) -> ``CARRY`` above the place point ->
``LOWER`` to ``place_h`` -> ``RELEASE`` (open) -> ``RETREAT`` (straight up, then to the joint-space ``retreat_ticks``
pose that leaves the zone in the camera's view) -> ``DONE`` (zero actions: the die settles and is seen). A lost die
(``G`` more than ``lost_m`` from the die while carrying, or an empty closed gripper) restarts at ``OPEN_ABOVE``.

**Motion**: every policy step a setpoint for ``G`` is chosen -- the phase's waypoint itself in ``OPEN_ABOVE`` (a
joint-space line from wherever the arm is), else a point at most ``slow_step_m`` (vertical moves) / ``step_m``
(``CARRY``: a straight line, so a carried die never dips into the rim) ahead of ``G`` of the CURRENT commanded goal
ticks -- then ``ik_iters`` damped-least-squares iterations (:func:`dls_ik`, finite-difference Jacobian of the 5-joint FK
of ``safety/fk.py``'s chain, warm-started at the command) solve for the joint angles that put ``G`` on it with the
gripper pointing straight down (its z axis = world +z: a 3-component residual, weight ``w_axis``) and, while grasping,
the jaw axis at the die's yaw modulo 90 deg (the 5-DoF arm cannot hold a full orientation: position + "down" + one
yaw). The wrist_flex limit (2960) lets the gripper point straight down only up to ``G`` ~50 mm over the table, hence
the low pre-grasp and carry heights (near the base it tilts a little; a stall rule then descends from there). The IK
target is clipped to ``cfg.joints`` (already the intersection with the calibration range) less
``joint_margin_ticks``, and the arm delta to the command is scaled UNIFORMLY so its largest component is at most
``max_delta_ticks`` (the direction is kept). The gripper moves on its own cap. Heights never go below ``grasp_h`` (jaw
bottoms 4 mm above the table, lowest gripper floor point ~2.5 mm), so the box clamp's table floor is never commanded.

**Inputs** (privileged, batched torch; :meth:`ScriptedDieController.act`): the commanded goal ticks (the servo
chain's ``goal``), the present ticks, the die position and quaternion. :func:`env_state` reads them from a
:class:`~so101_rl.env.batched_env.BatchedGoalEnv` (CUDA or the CPU stand-in): ``env.chain.goal``,
``env.q_to_ticks(env.ws.qpos[:, env._q_adr])`` and ``env.ws.qpos[:, env._die_q : env._die_q + 7]``. The numpy
single-env path is :meth:`ScriptedDieController.act_np` (one world, same code). From ``safety/`` only the kinematic
chain constants are imported (like ``sim/batched_servo.py``).

**Env drivers**: :class:`ScriptedDiePolicy` (the controller bound to an env), :func:`demo_policy` (the demo factory of
``train_sim_sac``), :func:`collect_demos` (successful episodes into a demo ``GpuReplay`` in the training format) and
:func:`run_scripted_episodes` (evaluation with failure modes; ``scripts/scripted_die_eval.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final

import numpy as np
import torch

from so101_rl.config import CENTER_TICK, JOINT_NAMES, N_JOINTS, TICKS_PER_REV, MotorCalibration, So101Config
from so101_rl.env.tasks import TERMINAL_TRIPS
from so101_rl.safety.fk import N_ARM_JOINTS, KinematicChain

__all__ = [
    "GRASP_POINT_GRIPPER",
    "Phase",
    "ScriptedDieConfig",
    "ScriptedDieController",
    "ScriptedDiePolicy",
    "collect_demos",
    "demo_policy",
    "dls_ik",
    "env_state",
    "gripper_fk",
    "run_scripted_episodes",
]

_RAD_PER_TICK: Final[float] = 2.0 * math.pi / TICKS_PER_REV
GRIPPER: Final[int] = N_JOINTS - 1
# Between the jaw tips, gripper body frame (sim/model.py _JAW_BOXES: fixed-jaw tip box centre (-0.013, 0, -0.095),
# half x 5 mm -> inner face at x = -0.008; a 16 mm die pressed against it is centred at x = 0): 4 mm off the fixed
# jaw so the open jaws come down around the die without landing on it, the moving jaw then slides it over.
GRASP_POINT_GRIPPER: Final[tuple[float, float, float]] = (0.004, 0.0, -0.094)


class Phase(IntEnum):
    OPEN_ABOVE = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    CARRY = 4
    LOWER = 5
    RELEASE = 6
    RETREAT = 7
    DONE = 8


@dataclass(frozen=True)
class ScriptedDieConfig:
    """Heights are of ``G`` above the table top (``table_z_m``); the place point is in the zone, base frame."""

    zone_xy_m: tuple[float, float] = (0.15, -0.13)
    # placed 12 mm off the zone centre in -x and in -y: the zone's tray-side corner (x >= 0.15, y >= -0.125) lies
    # in the placeholder tray rim's camera shadow (plan §10 2026-10-16); still 18 mm inside the zone edges on both axes
    place_offset_xy_m: tuple[float, float] = (-0.012, -0.012)
    table_z_m: float = -0.0045
    grasp_h: float = 0.0145  # jaw-box bottoms ~4 mm above the table; the die (16 mm) is held over 4..16 mm
    # the wrist_flex limit (2960) only lets the gripper point straight down with G up to ~50 mm over the table
    pregrasp_h: float = 0.040  # jaw bottoms ~30 mm up: above the 26 mm rim
    carry_h: float = 0.050  # a held die's bottom ~35 mm up (rim 26 mm)
    place_h: float = 0.017  # release with the die ~2.5 mm above the table
    retreat_h: float = 0.080
    step_m: float = 0.030  # Cartesian setpoint step per policy step while carrying (straight line over the rim)
    slow_step_m: float = 0.012  # descend / lift / lower / retreat-up
    # gripper ticks relative to its closed limit (cfg.joints.min_ticks[5] = calibration range_min, 2010 on the arm;
    # the command goes to that limit and the die stops the jaw, the tracking clamp sets the squeeze)
    gripper_open_above_closed: int = 290  # ~32 mm between the jaw tips (a 16 mm die, a 22.6 mm diagonal)
    gripper_empty_above_closed: int = 65  # present at most this far above closed after closing: the jaws are empty
    close_steps: int = 4
    stall_steps: int = 12  # OPEN_ABOVE this long with the command at rest: descend from where the arm is
    release_steps: int = 3
    tol_xy_m: float = 0.006
    tol_z_m: float = 0.004
    yaw_tol_rad: float = 0.08
    lost_m: float = 0.025
    retreat_ticks: tuple[int, ...] = (2047, 960, 2700, 2748, 2047)  # arm joints: safe_state.home_ticks (the zone in view)
    joint_margin_ticks: int = 20
    ik_iters: int = 10
    ik_damping: float = 0.02
    w_axis: float = 0.02  # residual weights (metres per unit): gripper z axis vs world +z, jaw yaw (rad)
    w_yaw: float = 0.03
    fd_eps_rad: float = 1e-4


def gripper_fk(chain_T: tuple[torch.Tensor, torch.Tensor], q: torch.Tensor) -> torch.Tensor:
    """Gripper body frame ``[..., 4, 4]`` in the base frame for arm angles ``q [..., 5]`` (rad): ``base @ prod_i
    (P_i @ Rz(q_i))``, the chain of ``safety/fk.py`` (float64)."""
    base_T, parent_T = chain_T
    c, s = torch.cos(q), torch.sin(q)
    a = base_T.expand(*q.shape[:-1], 4, 4)
    for i in range(N_ARM_JOINTS):
        rz = torch.zeros((*q.shape[:-1], 4, 4), dtype=q.dtype, device=q.device)
        rz[..., 0, 0], rz[..., 0, 1], rz[..., 1, 0], rz[..., 1, 1] = c[..., i], -s[..., i], s[..., i], c[..., i]
        rz[..., 2, 2] = 1.0
        rz[..., 3, 3] = 1.0
        a = a @ parent_T[i] @ rz
    return a


def _wrap(x: torch.Tensor, period: float) -> torch.Tensor:
    """``x`` wrapped into ``[-period/2, period/2)``."""
    return torch.remainder(x + period / 2, period) - period / 2


def _residual(T: torch.Tensor, g_local: torch.Tensor, sp: torch.Tensor, yaw_t: torch.Tensor, w_yaw: torch.Tensor,
              w_axis: float) -> torch.Tensor:  # fmt: skip
    """[..., 7]: G - setpoint (3), w_axis * (gripper z - world z) (3), w_yaw * jaw-yaw error mod 90 deg (1)."""
    R, p = T[..., :3, :3], T[..., :3, 3]
    g = (R @ g_local.unsqueeze(-1)).squeeze(-1) + p
    z = R[..., :, 2] - torch.tensor([0.0, 0.0, 1.0], dtype=T.dtype, device=T.device)
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    e_yaw = _wrap(yaw - yaw_t, math.pi / 2)
    return torch.cat([g - sp, w_axis * z, (w_yaw * e_yaw).unsqueeze(-1)], dim=-1)


def dls_ik(
    chain_T: tuple[torch.Tensor, torch.Tensor],
    q0: torch.Tensor,
    sp: torch.Tensor,
    *,
    g_local: torch.Tensor,
    yaw_target: torch.Tensor,
    w_yaw: torch.Tensor,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
    iters: int = 8,
    damping: float = 0.02,
    w_axis: float = 0.05,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Damped-least-squares IK from ``q0 [N,5]`` (rad) for the grasp point ``g_local`` (gripper frame) at ``sp [N,3]``
    with the gripper pointing down and (weight ``w_yaw [N]``, 0 = free) the jaw axis at ``yaw_target [N]`` mod 90 deg.
    Finite-difference Jacobian; every iterate clipped to ``[q_lo, q_hi]``. Returns ``(q [N,5], residual [N,7])``."""
    q = q0.clone()
    eye = torch.eye(N_ARM_JOINTS, dtype=q.dtype, device=q.device)
    for _ in range(iters):
        qs = torch.cat([q.unsqueeze(1), q.unsqueeze(1) + eps * eye.unsqueeze(0)], dim=1)  # [N, 6, 5]
        T = gripper_fk(chain_T, qs)
        r = _residual(T, g_local, sp.unsqueeze(1), yaw_target.unsqueeze(1), w_yaw.unsqueeze(1), w_axis)  # [N, 6, 7]
        r0 = r[:, 0]
        J = (r[:, 1:] - r0.unsqueeze(1)).transpose(1, 2) / eps  # [N, 7, 5]
        # dq = -(J^T J + lambda^2 I)^-1 J^T r
        A = J.transpose(1, 2) @ J + (damping**2) * eye
        dq = -torch.linalg.solve(A, (J.transpose(1, 2) @ r0.unsqueeze(-1))).squeeze(-1)
        q = torch.maximum(torch.minimum(q + dq, q_hi), q_lo)
    T = gripper_fk(chain_T, q)
    r = _residual(T, g_local, sp, yaw_target, w_yaw, w_axis)
    return q, r


def die_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Yaw [N] of the die's most horizontal body axis (a cube: any face pair can be gripped), from ``quat [N,4]``
    wxyz."""
    w, x, y, z = quat.unbind(-1)
    # rotation matrix columns (body x, y, z axes in the world)
    cols = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], dim=-1),
        torch.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], dim=-1),
        torch.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=1)  # [N, 3 axes, 3]  # fmt: skip
    k = cols[..., 2].abs().argmin(dim=1)
    ax = cols[torch.arange(cols.shape[0], device=cols.device), k]
    return torch.atan2(ax[:, 1], ax[:, 0])


class ScriptedDieController:
    """Batched scripted die controller (module docstring). ``n`` worlds on ``device``."""

    def __init__(
        self,
        cfg: So101Config,
        n: int,
        *,
        drive_signs: Sequence[float],
        device: torch.device | str = "cpu",
        params: ScriptedDieConfig | None = None,
        chain: KinematicChain | None = None,
    ) -> None:
        """``cfg``: joint limits (already the intersection with the calibration range), ``max_delta_ticks``, the
        table height and the kinematics; ``drive_signs`` [6]: -1 for a calibration ``drive_mode`` joint, else +1
        (:meth:`from_calibration`, :meth:`for_env`)."""
        self.p = p = params if params is not None else ScriptedDieConfig(table_z_m=float(cfg.safe_state.table_z_m))
        self.n, self.device = int(n), torch.device(device)
        dev, f64 = self.device, torch.float64
        chain = chain if chain is not None else KinematicChain.from_yaml(cfg.kinematics_path)
        self._chain_T = (torch.as_tensor(chain.base_T, dtype=f64, device=dev),
                         torch.as_tensor(chain.parent_T, dtype=f64, device=dev))  # fmt: skip
        self._sign = torch.as_tensor(np.asarray(drive_signs, np.float64), dtype=f64, device=dev)
        if self._sign.shape != (N_JOINTS,) or not bool((self._sign.abs() == 1).all()):
            raise ValueError(f"drive_signs must be {N_JOINTS} values of +-1, got {list(drive_signs)}")
        lo = np.asarray(cfg.joints.min_ticks, np.int64) + p.joint_margin_ticks
        hi = np.asarray(cfg.joints.max_ticks, np.int64) - p.joint_margin_ticks
        closed = int(cfg.joints.min_ticks[GRIPPER])
        self.gripper_closed = closed
        self.gripper_open = closed + p.gripper_open_above_closed
        self.gripper_empty = closed + p.gripper_empty_above_closed
        self.tick_lo = torch.as_tensor(lo, dtype=torch.int64, device=dev)
        self.tick_hi = torch.as_tensor(hi, dtype=torch.int64, device=dev)
        qa = self._ticks_to_rad(torch.stack([self.tick_lo, self.tick_hi])[:, :N_ARM_JOINTS])
        self._q_lo, self._q_hi = torch.minimum(qa[0], qa[1]), torch.maximum(qa[0], qa[1])
        self.max_delta = torch.as_tensor(np.asarray(cfg.action.max_delta_ticks, np.float64), dtype=f64, device=dev)
        self._g_local = torch.tensor(GRASP_POINT_GRIPPER, dtype=f64, device=dev)
        self._retreat = torch.tensor(p.retreat_ticks, dtype=torch.int64, device=dev)
        z = p.table_z_m
        self._place_xy = torch.tensor([p.zone_xy_m[0] + p.place_offset_xy_m[0], p.zone_xy_m[1] + p.place_offset_xy_m[1]],
                                      dtype=f64, device=dev)  # fmt: skip
        self._h = {k: z + getattr(p, k) for k in ("grasp_h", "pregrasp_h", "carry_h", "place_h", "retreat_h")}
        self.phase = torch.zeros(self.n, dtype=torch.int64, device=dev)
        self.phase_steps = torch.zeros(self.n, dtype=torch.int64, device=dev)
        self.regrasps = torch.zeros(self.n, dtype=torch.int64, device=dev)
        self.yaw_target = torch.zeros(self.n, dtype=f64, device=dev)
        self.last_residual = torch.zeros((self.n, 7), dtype=f64, device=dev)

    @classmethod
    def from_calibration(cls, cfg: So101Config, calibration: dict[str, MotorCalibration], n: int,
                         **kw: Any) -> ScriptedDieController:  # fmt: skip
        """Drive signs from a LeRobot calibration (``drive_mode``)."""
        return cls(cfg, n, drive_signs=[-1.0 if calibration[j].drive_mode else 1.0 for j in JOINT_NAMES], **kw)

    @classmethod
    def for_env(cls, env: Any, **kw: Any) -> ScriptedDieController:
        """A controller for every world of a ``BatchedGoalEnv`` (its config, drive signs and device)."""
        return cls(env.cfg, int(env.nworld), drive_signs=env._sign.cpu().tolist(), device=env.device, **kw)

    # ------------------------------------------------------------------ conversions
    def _ticks_to_rad(self, t: torch.Tensor) -> torch.Tensor:
        return (t.to(torch.float64) - CENTER_TICK) * self._sign[: t.shape[-1]] * _RAD_PER_TICK

    def _rad_to_ticks(self, q: torch.Tensor) -> torch.Tensor:
        return torch.round(q / (self._sign[: q.shape[-1]] * _RAD_PER_TICK) + CENTER_TICK).to(torch.int64)

    def grasp_point(self, ticks: torch.Tensor) -> torch.Tensor:
        """``G`` [N,3] (base frame) for arm ticks ``[N, >=5]``."""
        T = gripper_fk(self._chain_T, self._ticks_to_rad(ticks[:, :N_ARM_JOINTS]))
        return T[:, :3, :3] @ self._g_local + T[:, :3, 3]

    # ------------------------------------------------------------------ episode
    def reset(self, mask: torch.Tensor | None = None) -> None:
        """Start the worlds in ``mask`` (all if None) at ``OPEN_ABOVE``."""
        m = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask.to(self.device, torch.bool)
        self.phase.masked_fill_(m, int(Phase.OPEN_ABOVE))
        self.phase_steps.masked_fill_(m, 0)
        self.regrasps.masked_fill_(m, 0)

    # ------------------------------------------------------------------ control
    @torch.no_grad()
    def act(self, goal_ticks: torch.Tensor, present_ticks: torch.Tensor, die_pos: torch.Tensor,
            die_quat: torch.Tensor) -> torch.Tensor:  # fmt: skip
        """One policy step: ``float32 [N,6]`` actions in [-1, 1] (module docstring)."""
        p, dev, f64 = self.p, self.device, torch.float64
        goal = goal_ticks.to(dev, torch.int64)
        present = present_ticks.to(dev, torch.int64)
        die = die_pos.to(dev, f64)
        g_now = self.grasp_point(goal)  # G of the command
        g_pres = self.grasp_point(present)  # G of the arm
        ph = self.phase
        h = self._h

        # ---- phase transitions (on the present state)
        grip = present[:, GRIPPER]
        d_xy = (g_pres[:, :2] - die[:, :2]).norm(dim=1)
        near_die = (g_pres - die).norm(dim=1) < p.lost_m
        yaw_now = self._jaw_yaw(present)
        yaw_ok = _wrap(yaw_now - self.yaw_target, math.pi / 2).abs() < p.yaw_tol_rad
        place = torch.cat([self._place_xy.expand(self.n, 2), torch.full((self.n, 1), h["place_h"], dtype=f64, device=dev)], 1)
        nxt = ph.clone()

        def to(cond: torch.Tensor, frm: Phase, new: Phase) -> None:
            nxt.copy_(torch.where((ph == int(frm)) & cond, int(new), nxt))

        self.yaw_target = torch.where(ph == int(Phase.OPEN_ABOVE), die_yaw(die_quat.to(dev, f64)), self.yaw_target)
        above = (d_xy < p.tol_xy_m) & ((g_pres[:, 2] - h["pregrasp_h"]).abs() < 2 * p.tol_z_m)
        # near the base the wrist_flex limit can keep G a few mm off the pregrasp point: once the command is at rest
        # for stall_steps, descend from there (DESCEND re-centres on the die)
        at_rest = (goal[:, :N_ARM_JOINTS] - present[:, :N_ARM_JOINTS]).abs().amax(dim=1) <= 15
        stalled = (self.phase_steps >= p.stall_steps) & at_rest & (d_xy < 2 * p.tol_xy_m)
        to(((above & yaw_ok) | stalled) & (grip >= self.gripper_open - 30), Phase.OPEN_ABOVE, Phase.DESCEND)
        to((g_pres[:, 2] - h["grasp_h"]).abs() < p.tol_z_m, Phase.DESCEND, Phase.CLOSE)
        to(d_xy > 2 * p.tol_xy_m, Phase.DESCEND, Phase.OPEN_ABOVE)  # the die moved away: line up again
        closed = self.phase_steps >= p.close_steps
        to(closed & (grip > self.gripper_empty), Phase.CLOSE, Phase.LIFT)
        to(closed & (grip <= self.gripper_empty), Phase.CLOSE, Phase.OPEN_ABOVE)  # missed: nothing in the jaws
        to((g_pres[:, 2] - h["carry_h"]).abs() < 2 * p.tol_z_m, Phase.LIFT, Phase.CARRY)
        to((g_pres[:, :2] - place[:, :2]).norm(dim=1) < p.tol_xy_m, Phase.CARRY, Phase.LOWER)
        to((g_pres[:, 2] - h["place_h"]).abs() < p.tol_z_m, Phase.LOWER, Phase.RELEASE)
        carrying = (ph == int(Phase.LIFT)) | (ph == int(Phase.CARRY)) | (ph == int(Phase.LOWER))
        lost = carrying & (~near_die | (grip <= self.gripper_empty))
        nxt = torch.where(lost, int(Phase.OPEN_ABOVE), nxt)
        self.regrasps += lost.to(torch.int64)
        to(self.phase_steps >= p.release_steps, Phase.RELEASE, Phase.RETREAT)
        at_retreat = (present[:, :N_ARM_JOINTS] - self._retreat).abs().amax(dim=1) <= 30
        to(at_retreat, Phase.RETREAT, Phase.DONE)
        self.phase_steps = torch.where(nxt == ph, self.phase_steps + 1, torch.zeros_like(self.phase_steps))
        self.phase = ph = nxt

        # ---- waypoint of G per phase
        die_xy = die[:, :2]
        wp = torch.cat([die_xy, torch.full((self.n, 1), h["pregrasp_h"], dtype=f64, device=dev)], 1)
        wp = torch.where((ph == int(Phase.DESCEND))[:, None], torch.cat([die_xy, torch.full_like(wp[:, :1], h["grasp_h"])], 1),
                         wp)  # fmt: skip
        wp = torch.where((ph == int(Phase.CLOSE))[:, None], g_now, wp)  # hold still while the jaws close
        lift_xy = torch.where((ph == int(Phase.LIFT))[:, None], g_now[:, :2], die_xy)
        wp = torch.where((ph == int(Phase.LIFT))[:, None], torch.cat([lift_xy, torch.full_like(wp[:, :1], h["carry_h"])], 1), wp)
        wp = torch.where((ph == int(Phase.CARRY))[:, None],
                         torch.cat([place[:, :2], torch.full_like(wp[:, :1], h["carry_h"])], 1), wp)  # fmt: skip
        wp = torch.where((ph == int(Phase.LOWER))[:, None] | (ph == int(Phase.RELEASE))[:, None], place, wp)
        up = torch.cat([g_now[:, :2], torch.full_like(wp[:, :1], h["retreat_h"])], 1)
        wp = torch.where((ph == int(Phase.RETREAT))[:, None], up, wp)
        # free motion (OPEN_ABOVE): the IK target is the waypoint itself, reached along a joint-space line; carrying
        # and the vertical moves: a Cartesian setpoint at most step_m / slow_step_m ahead of the command
        slow = (ph == int(Phase.DESCEND)) | (ph == int(Phase.LOWER)) | (ph == int(Phase.LIFT)) | (ph == int(Phase.CLOSE)) \
            | (ph == int(Phase.RETREAT))
        step = torch.where(slow, p.slow_step_m, torch.where(ph == int(Phase.CARRY), p.step_m, math.inf)).to(f64)
        d = wp - g_now
        dn = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        sp = g_now + d * torch.clamp(step[:, None] / dn, max=1.0)
        # never below the grasp height (the table floor is never commanded)
        sp[:, 2] = sp[:, 2].clamp_min(h["grasp_h"])

        # ---- IK
        w_yaw = torch.where((ph <= int(Phase.CLOSE)), p.w_yaw, 0.0).to(f64)
        yaw_cmd = torch.where(w_yaw > 0, self.yaw_target, self._jaw_yaw(goal))
        q0 = self._ticks_to_rad(goal[:, :N_ARM_JOINTS])
        q, res = dls_ik(self._chain_T, q0, sp, g_local=self._g_local, yaw_target=yaw_cmd, w_yaw=w_yaw,
                        q_lo=self._q_lo, q_hi=self._q_hi, iters=p.ik_iters, damping=p.ik_damping, w_axis=p.w_axis,
                        eps=p.fd_eps_rad)  # fmt: skip
        self.last_residual = res
        tgt = self._rad_to_ticks(q)
        # retreat: once up, the joint-space retreat pose (and DONE holds it)
        go_home = ((ph == int(Phase.RETREAT)) & (g_pres[:, 2] > h["retreat_h"] - 0.02)) | (ph == int(Phase.DONE))
        tgt = torch.where(go_home[:, None], self._retreat.expand(self.n, -1), tgt)
        tgt = torch.maximum(torch.minimum(tgt, self.tick_hi[:N_ARM_JOINTS]), self.tick_lo[:N_ARM_JOINTS])
        delta = (tgt - goal[:, :N_ARM_JOINTS]).to(f64)
        md = self.max_delta[:N_ARM_JOINTS]
        scale = torch.clamp((md / delta.abs().clamp_min(1e-9)).amin(dim=1, keepdim=True), max=1.0)
        delta = torch.round(delta * scale)

        # ---- gripper
        closing = (ph == int(Phase.CLOSE)) | (ph == int(Phase.LIFT)) | (ph == int(Phase.CARRY)) | (ph == int(Phase.LOWER))
        g_tgt = torch.where(closing, self.gripper_closed, self.gripper_open)
        g_tgt = torch.where(ph == int(Phase.DONE), goal[:, GRIPPER], g_tgt)
        dg = (g_tgt - goal[:, GRIPPER]).to(f64).clamp(-self.max_delta[GRIPPER], self.max_delta[GRIPPER])
        out = torch.cat([delta, dg[:, None]], dim=1) / self.max_delta
        out = torch.where((ph == int(Phase.DONE))[:, None] & (delta.abs().amax(1, keepdim=True) == 0),
                          torch.zeros_like(out), out)  # fmt: skip
        return out.clamp(-1.0, 1.0).to(torch.float32)

    def _jaw_yaw(self, ticks: torch.Tensor) -> torch.Tensor:
        T = gripper_fk(self._chain_T, self._ticks_to_rad(ticks[:, :N_ARM_JOINTS]))
        return torch.atan2(T[:, 1, 0], T[:, 0, 0])

    # ------------------------------------------------------------------ numpy single env
    def act_np(self, goal_ticks: np.ndarray, present_ticks: np.ndarray, die_pos: np.ndarray,
               die_quat: np.ndarray) -> np.ndarray:  # fmt: skip
        """:meth:`act` for ONE world (``n == 1``) with numpy inputs ``[6] [6] [3] [4]``; returns ``float32 [6]``."""
        if self.n != 1:
            raise ValueError("act_np drives a single world (n == 1)")

        def t(x: Any, dt: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(np.asarray(x), dtype=dt, device=self.device).reshape(1, -1)

        a = self.act(t(goal_ticks, torch.int64), t(present_ticks, torch.int64), t(die_pos, torch.float64),
                     t(die_quat, torch.float64))  # fmt: skip
        return a[0].cpu().numpy()


def env_state(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The controller's privileged inputs from a ``BatchedGoalEnv``: ``(goal ticks, present ticks, die pos, die
    quat)`` = ``env.chain.goal``, ``env.q_to_ticks(env.ws.qpos[:, env._q_adr])`` and the die free joint's
    ``qpos``."""
    qpos = env.ws.qpos
    return (env.chain.goal, env.q_to_ticks(qpos[:, env._q_adr]), qpos[:, env._die_q : env._die_q + 3],
            qpos[:, env._die_q + 3 : env._die_q + 7])  # fmt: skip


# ------------------------------------------------------------------------------------------------ env drivers
class ScriptedDiePolicy:
    """The controller bound to a ``BatchedGoalEnv``: ``policy(env)`` -> actions from the env's privileged state
    (:func:`env_state`); ``reset(mask)`` restarts the state machine of the worlds that reset."""

    def __init__(self, env: Any, **kw: Any) -> None:
        self.ctl = ScriptedDieController.for_env(env, **kw)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        self.ctl.reset(mask)

    def __call__(self, env: Any) -> torch.Tensor:
        return self.ctl.act(*env_state(env))


def demo_policy(env: Any) -> Any:
    """The demo source of ``train_sim_sac`` (``demo_episodes``): the scripted die policy for ``env``. One factory, so
    the CPU trainer tests swap in a toy expert (as ``eval_sim.build_env`` is swapped for the toy env)."""
    return ScriptedDiePolicy(env)


@torch.no_grad()
def collect_demos(
    env: Any,
    replay: Any,
    episodes: int,
    *,
    policy: Any | None = None,
    success_only: bool = True,
) -> dict[str, Any]:
    """Run ``policy`` (default :func:`demo_policy`) for ``episodes`` episodes and add the transitions of the kept
    ones (the successful ones, unless ``success_only=False``) to ``replay`` in exactly the training loop's format:
    ``(obs ++ goal, priv, action, reward, final_obs ++ desired, final_priv, terminated, truncated)``
    (``train_sim_sac``: the pre-reset ``final_*`` so truncation bootstraps). Each episode is stored as one block in
    time order.

    Rounds of one episode per world, like ``eval_sim.run_episodes`` (each world's FIRST episode of a round counts, the
    last round only its first ``episodes mod nworld`` worlds), so the sample is not biased towards short episodes.
    Every round restarts the env with a MASKED reset of every world (``phase_randomise=False``), which leaves the
    first-full-reset phase randomisation to the training loop. Returns ``{"episodes", "kept", "successes",
    "transitions", "success_rate"}``."""
    n, dev = int(env.nworld), torch.device(env.device)
    horizon = int(getattr(getattr(env, "task", None), "episode_steps", getattr(env, "episode_steps", 100)))
    pol = policy if policy is not None else demo_policy(env)
    every = torch.ones(n, dtype=torch.bool, device=dev)
    f32 = {"dtype": torch.float32, "device": dev}
    w_idx = torch.arange(n, device=dev)
    t_idx = torch.arange(horizon, device=dev)[None, :]
    keys = ("obs", "priv", "act", "rew", "next_obs", "next_priv", "term", "trunc")
    ended = kept = succ = trans = 0
    left = int(episodes)
    while left > 0:
        k = min(n, left)
        obs, priv = (x.clone() for x in env.reset(every, phase_randomise=False)[:2])
        pol.reset(None)
        d_in, d_priv, d_act = obs.shape[1] + int(env.goal.shape[1]), priv.shape[1], int(env.act_dim)
        buf = {key: torch.zeros((n, horizon, w), **f32) for key, w in
               (("obs", d_in), ("priv", d_priv), ("act", d_act), ("next_obs", d_in), ("next_priv", d_priv))}  # fmt: skip
        buf.update({key: torch.zeros((n, horizon), **f32) for key in ("rew", "term", "trunc")})
        length = torch.zeros(n, dtype=torch.int64, device=dev)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        success = torch.zeros(n, dtype=torch.bool, device=dev)
        for t in range(horizon):
            obs_in = torch.cat([obs, env.goal], dim=1)
            action = pol(env)
            st = env.step(action)
            live = ~done  # rows of the worlds still in their first episode of this round
            for key, x in (("obs", obs_in), ("priv", priv), ("act", action), ("next_obs", torch.cat([st.final_obs, st.desired], 1)),
                           ("next_priv", st.final_priv), ("rew", st.reward), ("term", st.terminated), ("trunc", st.truncated)):
                buf[key][w_idx[live], t] = x.to(**f32)[live]  # fmt: skip
            end = live & (st.terminated | st.truncated)
            length = torch.where(end, t + 1, length)
            success |= end & st.info["success"].to(torch.bool)
            done |= end
            pol.reset(end)
            obs, priv = st.obs, st.priv
            if bool(done.all()):
                break
        counted = done & (w_idx < k)
        keep = counted & (success | (not success_only))
        sel = (t_idx < length[:, None]) & keep[:, None]  # [N, T]: world-major, time order inside an episode
        if bool(sel.any()):
            replay.add(*(buf[key][sel] for key in keys))
        ended += int(counted.sum())
        succ += int((counted & success).sum())
        kept += int(keep.sum())
        trans += int(sel.sum())
        left -= k
    return {"episodes": ended, "kept": kept, "successes": succ, "transitions": trans,
            "success_rate": succ / ended if ended else 0.0}  # fmt: skip


@torch.no_grad()
def run_scripted_episodes(env: Any, episodes: int, *, policy: Any | None = None) -> dict[str, Any]:
    """Evaluate the scripted policy like ``eval_sim.run_episodes`` (rounds of one episode per world from a full
    reset without phase randomisation, each world's FIRST episode counted), with the controller's diagnostics: the
    phase each failed episode ended in, the phase each success fired in, regrasps, trips, final tracker visibility."""
    n, dev = int(env.nworld), torch.device(env.device)
    horizon = int(env.task.episode_steps)
    pol = policy if policy is not None else ScriptedDiePolicy(env)
    rows: list[dict[str, Any]] = []
    left = int(episodes)
    while left > 0:
        k = min(n, left)
        env.reset(phase_randomise=False)
        pol.reset(None)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        i64 = {"dtype": torch.int64, "device": dev}
        rec = {"success": torch.zeros(n, dtype=torch.bool, device=dev), "length": torch.zeros(n, **i64),
               "phase": torch.full((n,), -1, **i64), "trips": torch.zeros(n, **i64),
               "visible": torch.zeros(n, dtype=torch.bool, device=dev), "err": torch.zeros(n, device=dev),
               "regrasps": torch.zeros(n, **i64)}  # fmt: skip
        for t in range(horizon):
            a = pol(env)
            ph, rg = pol.ctl.phase.clone(), pol.ctl.regrasps.clone()
            st = env.step(a)
            end = ~done & (st.terminated | st.truncated)
            rec["success"] |= end & st.info["success"]
            rec["length"] = torch.where(end, t + 1, rec["length"])
            rec["phase"] = torch.where(end, ph, rec["phase"])
            rec["trips"] = torch.where(end, st.info["trips"].to(torch.int64), rec["trips"])
            rec["visible"] = torch.where(end, st.info["tracker_visible"], rec["visible"])
            rec["err"] = torch.where(end, (st.final_achieved - st.desired).norm(dim=1).float(), rec["err"])
            rec["regrasps"] = torch.where(end, rg, rec["regrasps"])
            done |= end
            if bool(done.all()):
                break
        r = {key: v[:k].cpu().numpy() for key, v in rec.items()}
        rows += [{key: r[key][i].item() for key in r} for i in range(k)]
        left -= k
    s = np.array([r["success"] for r in rows], bool)
    fails: dict[str, int] = {}
    succ_ph: dict[str, int] = {}
    for r in rows:
        d = succ_ph if r["success"] else fails
        name = Phase(r["phase"]).name
        d[name] = d.get(name, 0) + 1
    return {
        "episodes": len(rows), "successes": int(s.sum()), "success_rate": float(s.mean()) if rows else 0.0,
        "mean_length_success": float(np.mean([r["length"] for r in rows if r["success"]])) if s.any() else None,
        "fail_end_phase": fails, "success_phase": succ_ph,
        "trip_episodes": sum(1 for r in rows if r["trips"] & TERMINAL_TRIPS),
        "regrasp_episodes": sum(1 for r in rows if r["regrasps"] > 0),
        "fail_visible_at_end": sum(1 for r in rows if not r["success"] and r["visible"]),
        "final_err_m_p50": float(np.percentile([r["err"] for r in rows], 50)) if rows else None,
    }  # fmt: skip


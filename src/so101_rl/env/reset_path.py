"""The reset path of :meth:`ShmBackend.reset <so101_rl.env.backend.ShmBackend.reset>`: lift, then return (plan §10,
2026-10-08 "Deploy reset: lift then return").

A plain joint-space ramp from wherever an episode ended to the start pose can push the tip into a safety-box wall:
the servo's box clamp then reverts every target, the arm does not move, the same delta keeps arriving and the
stuck-action monitor trips STUCK (seen in M7: an episode that ended with the tip on the ``y`` wall). So the reset
path has two parts, both executed as ordinary RUN deltas through the servo's full safety pipeline (nothing here
bypasses or re-implements a safety check; it only chooses the targets):

1. **Lift** (Cartesian, straight line): from the start tip ``p`` to ``p_lift`` = ``p`` with each of ``x`` / ``y``
   clipped into the box shrunk by ``wall_margin_m`` and ``z`` raised to the lift height (the start pose's tip
   height, at least ``wall_margin_m`` above the floor; never above ``z_max - wall_margin_m``). Each coordinate only
   moves AWAY from its nearer wall (or stays), so along this line the tip's distance to every face it is close to
   never decreases. Solved by damped least squares on shoulder_pan / shoulder_lift / elbow_flex (the wrist and the
   gripper hold), one IK solve every ``step_m``; the lift stops early (best effort) at the last waypoint that
   solved, stays in the box (tip and gripper floor, :meth:`So101FK.in_box_gripper`) and inside the joint limits.
2. **Return** to ``q0`` from the last lift waypoint: the tip on a straight line to the start pose's tip (both
   points inside the shrunk box, so the line is too), else the straight joint-space line (:func:`plan_reset_path`).

:func:`plan_reset_path` returns the waypoints (ticks); :func:`densify` turns them into per-policy-tick targets (at
most ``max_ticks_per_tick`` per joint, every joint on the straight line between waypoints). Pure numpy, no I/O: the
same path on the real arm and on ``servo --backend mujoco|fake``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from so101_rl.config import N_JOINTS, So101Config
from so101_rl.safety.fk import So101FK

__all__ = [
    "IK_JOINTS", "LIFT_STEP_M", "WALL_MARGIN_M", "densify", "lift_height", "lift_point", "path_in_box", "plan_lift",
    "plan_reset_path",
]  # fmt: skip

WALL_MARGIN_M: Final[float] = 0.03  # the lift ends at least this far inside every box face (where it can)
LIFT_STEP_M: Final[float] = 0.005  # Cartesian spacing of the lift waypoints
IK_JOINTS: Final[tuple[int, ...]] = (0, 1, 2)  # shoulder_pan, shoulder_lift, elbow_flex
_IK_ITERS: Final[int] = 8
_IK_TOL_M: Final[float] = 0.001
_IK_DAMPING: Final[float] = 1e-12  # (m/tick)^2: damped least squares, J ~ 1e-4 m/tick
_FD_TICKS: Final[float] = 4.0  # central-difference half step of the Jacobian


def lift_height(fk: So101FK, cfg: So101Config, start_ticks: np.ndarray, q0_ticks: np.ndarray, *,
                wall_margin_m: float = WALL_MARGIN_M) -> float:  # fmt: skip
    """``z`` of the lift: max(start tip z, start-pose tip z), clipped into [floor + margin, ceiling - margin]."""
    xyz = np.empty(3)
    z_start = float(fk.gripper_xyz_from_ticks(np.asarray(start_ticks, np.float64), xyz)[2])
    z_q0 = float(fk.gripper_xyz_from_ticks(np.asarray(q0_ticks, np.float64), xyz)[2])
    lo, hi = cfg.box.z[0] + wall_margin_m, cfg.box.z[1] - wall_margin_m
    return float(np.clip(max(z_start, z_q0), lo, hi))


def lift_point(p: np.ndarray, cfg: So101Config, z_lift: float, *, wall_margin_m: float = WALL_MARGIN_M) -> np.ndarray:
    """``p`` with ``x`` / ``y`` clipped into the box shrunk by ``wall_margin_m`` and ``z`` = ``z_lift``."""
    b, m = cfg.box, wall_margin_m
    return np.array([np.clip(p[0], b.x[0] + m, b.x[1] - m), np.clip(p[1], b.y[0] + m, b.y[1] - m), z_lift])


def _tip(fk: So101FK, q: np.ndarray, out: np.ndarray) -> np.ndarray:
    return fk.gripper_xyz_from_ticks(q, out)


def _jacobian(fk: So101FK, q: np.ndarray) -> np.ndarray:
    """d tip / d ticks of the IK joints (3 x len(IK_JOINTS)), central differences."""
    jac = np.empty((3, len(IK_JOINTS)))
    a, b = np.empty(3), np.empty(3)
    for c, j in enumerate(IK_JOINTS):
        qp, qm = q.copy(), q.copy()
        qp[j] += _FD_TICKS
        qm[j] -= _FD_TICKS
        jac[:, c] = (_tip(fk, qp, a) - _tip(fk, qm, b)) / (2 * _FD_TICKS)
    return jac


def _solve(fk: So101FK, q: np.ndarray, p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray | None:
    """Ticks near ``q`` whose tip is within ``_IK_TOL_M`` of ``p`` (IK joints only, inside the limits), else None."""
    q = q.astype(np.float64).copy()
    xyz = np.empty(3)
    idx = list(IK_JOINTS)
    for _ in range(_IK_ITERS):
        err = p - _tip(fk, q, xyz)
        if float(np.linalg.norm(err)) <= _IK_TOL_M:
            return q
        jac = _jacobian(fk, q)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + _IK_DAMPING * np.eye(3), err)
        q[idx] = np.clip(q[idx] + dq, lo[idx], hi[idx])
    return q if float(np.linalg.norm(p - _tip(fk, q, xyz))) <= _IK_TOL_M else None


def _in_box(fk: So101FK, cfg: So101Config, q: np.ndarray) -> bool:
    xyz = np.empty(3)
    z_low = fk.tip_and_floor_z_from_ticks(np.asarray(q, np.float64), xyz)
    return fk.in_box_gripper(xyz, z_low, cfg.box)


def path_in_box(fk: So101FK, cfg: So101Config, start_ticks: np.ndarray, targets: np.ndarray) -> bool:
    """Every target of ``targets`` and the two servo-tick interpolation points before it (a RUN command is spread
    over 3 servo ticks) inside the box: tip and gripper floor, :meth:`So101FK.in_box_gripper` with zero margins
    (the pipeline's target test)."""
    prev = np.asarray(start_ticks, np.float64)
    for t in np.asarray(targets, np.float64):
        for f in (1 / 3, 2 / 3, 1.0):
            if not _in_box(fk, cfg, prev + (t - prev) * f):
                return False
        prev = t
    return True


def _cartesian_return(fk: So101FK, q_lift: np.ndarray, q0: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      step_m: float) -> list[np.ndarray] | None:  # fmt: skip
    """The tip on the straight line FK(q_lift) -> FK(q0) while the other joints move linearly to ``q0``'s (IK on
    the arm joints), then ``q0``; None if an IK step fails."""
    xyz = np.empty(3)
    pa = _tip(fk, q_lift.astype(np.float64), xyz).copy()
    pb = _tip(fk, q0.astype(np.float64), xyz).copy()
    n = max(1, int(np.ceil(float(np.linalg.norm(pb - pa)) / step_m)))
    other = np.ones(N_JOINTS, bool)
    other[list(IK_JOINTS)] = False
    q = q_lift.astype(np.float64)
    out: list[np.ndarray] = []
    for k in range(1, n + 1):
        s = k / n
        q = np.where(other, q_lift + (q0 - q_lift) * s, q)
        sol = _solve(fk, q, pa + (pb - pa) * s, lo, hi)
        if sol is None:
            return None
        out.append(np.rint(sol).astype(np.int64))
        q = sol
    out.append(q0)
    return out


def plan_lift(fk: So101FK, cfg: So101Config, start_ticks: np.ndarray, q0_ticks: np.ndarray, *,
              wall_margin_m: float = WALL_MARGIN_M, step_m: float = LIFT_STEP_M) -> list[np.ndarray]:  # fmt: skip
    """The lift waypoints (ticks, may be empty): the tip on the straight line FK(start) -> :func:`lift_point`, one
    IK solve every ``step_m``, stopping at the first waypoint that does not solve or leaves the box."""
    start = np.asarray(start_ticks, np.int64)
    lo = cfg.joints.min_ticks.astype(np.float64)
    hi = cfg.joints.max_ticks.astype(np.float64)
    p0 = _tip(fk, start.astype(np.float64), np.empty(3)).copy()
    z1 = lift_height(fk, cfg, start, q0_ticks, wall_margin_m=wall_margin_m)
    p1 = lift_point(p0, cfg, z1, wall_margin_m=wall_margin_m)
    n = int(np.ceil(float(np.linalg.norm(p1 - p0)) / step_m))
    lift: list[np.ndarray] = []
    q = start.astype(np.float64)
    for k in range(1, n + 1):
        sol = _solve(fk, q, p0 + (p1 - p0) * (k / n), lo, hi)
        if sol is None:
            break
        qi = np.rint(sol).astype(np.int64)
        if not _in_box(fk, cfg, qi):
            break
        lift.append(qi)
        q = sol
    return lift


def plan_reset_path(fk: So101FK, cfg: So101Config, start_ticks: np.ndarray, q0_ticks: np.ndarray, *,
                    wall_margin_m: float = WALL_MARGIN_M, step_m: float = LIFT_STEP_M) -> np.ndarray:  # fmt: skip
    """Waypoints (K, 6) int64 ticks after ``start_ticks``: :func:`plan_lift`, then the return to ``q0_ticks``
    (last). The return is the tip on a straight line to FK(q0) (IK on the arm joints, the other joints moving
    linearly) if that solves and stays in the box (:func:`path_in_box` on the densified targets), else the straight
    joint line to ``q0`` (if that leaves the box, the servo's box clamp holds the arm and a STUCK trip halts the
    reset, which is safe)."""
    start = np.asarray(start_ticks, np.int64)
    q0 = np.asarray(q0_ticks, np.int64)
    lo = cfg.joints.min_ticks.astype(np.float64)
    hi = cfg.joints.max_ticks.astype(np.float64)
    lift = plan_lift(fk, cfg, start, q0, wall_margin_m=wall_margin_m, step_m=step_m)
    q_lift = lift[-1] if lift else start
    ret = _cartesian_return(fk, q_lift, q0, lo, hi, step_m)
    if ret is None or not path_in_box(fk, cfg, q_lift, densify(q_lift, np.stack(ret), cfg.action.max_ticks_per_tick)):
        ret = [q0]
    return np.stack([*lift, *ret])


def densify(start_ticks: np.ndarray, waypoints: np.ndarray, max_step: np.ndarray) -> np.ndarray:
    """Per-policy-tick targets (M, 6) int64 along straight joint-space lines start -> waypoints[0] -> ...: each
    segment in ``ceil(max_j |delta_j| / max_step_j)`` equal sub-steps, ending exactly on every waypoint."""
    step = np.maximum(np.asarray(max_step, np.float64), 1.0)
    out: list[np.ndarray] = []
    prev = np.asarray(start_ticks, np.float64)
    for w in np.asarray(waypoints, np.int64):
        d = w.astype(np.float64) - prev
        n = max(1, int(np.ceil(float(np.max(np.abs(d) / step)))))
        for i in range(1, n + 1):
            out.append(np.rint(prev + d * (i / n)).astype(np.int64))
        prev = w.astype(np.float64)
    return np.stack(out) if out else np.empty((0, N_JOINTS), np.int64)

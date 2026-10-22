"""Phase 4 feasibility gate on the real arm (plan Phase 4 "[v2] Feasibility gate"; ``plans/die-to-zone-plan.md`` D4).

``pixi run -e cpu feasibility-gate --attended [--accept-no-watchdog] [--protocol all|reach|grasp-height|scripted]``

The plan: "measure the real arm's minimum graspable height and its reach to the start well and the zone before
training. Scripted IK seeds must reach 30 % or more success on the real arm before they go into the buffer." Three
small protocols, run in this order by ``--protocol all`` (each alone with ``--protocol <name>``):

1. **reach** (no die needed): the 3 mm reach policy (``rl/tip_control.py``, as ``point_tip`` / ``hand-eye``) drives
   the tip to a point ``--reach-height`` (4 cm) above the centre of every start region (S0, S1, both S2 rectangles),
   the zone centre, the scripted place point and (``--tray``) the well, holds it 1 s with zero-delta RUN commands and
   records the reach error (FK of the present ticks vs the point). Plus a kinematic check without motion: the
   scripted controller's IK for the grasp point at the pre-grasp and the grasp height over each point (position
   residual, gripper-axis tilt).
2. **grasp-height** (minimum graspable height): the operator places the die at the ``--grasp-stage`` centre; per height
   in ``--grasp-heights`` (default 22, 18, 14.5, 12 mm of the grasp point over the table, highest first) the scripted
   controller grasps the die at that height, lifts it to the carry height, puts it back down where it was and
   retreats. Reported: per height lifted or not, trips, clamped steps; ``lowest_lifted_grasp_h_m``. A trip (e.g. STUCK
   from a jaw pressed into the table) ends the sweep (halt rule below), so the lowest height is the last one that lifted
   before it.
3. **scripted** (the 30 % gate): ``--trials`` (20) scripted pick-and-place trials from the start regions ``--starts``
   (S0, S1, S2 in turn; ``--tray`` adds the well, a placeholder until H9). Success is the env's own die predicate
   (``So101GoalEnv`` / ``env/tasks.py``: die visible, in the zone, settled, debounced). **PASS** when every planned
   trial ran (no halt) and successes / trials >= ``--pass-rate`` (0.30, the plan); the Wilson 95 % interval is
   reported beside it; FAIL when complete and below; INCOMPLETE after a halt.

**Every die trial** (grasp-height and scripted): the arm goes home and holds (HOLD keep-alive); the target is printed
(stage, x, y) and written to ``<run dir>/die_target.json``; the operator places the die roughly there, any yaw, then
presses Enter or creates ``<run dir>/die_ready`` (deploy-policy's ``--manual-die-reset`` gate,
``deploy_policy.DieResetGate``; ``<run dir>/die_reset_waiting`` holds the trial number). The ACTUAL position comes
from perception: the object table is read for ``--window-s`` (1 s) with the env's rule (die visible in a detection no
older than one policy period + two camera frames, fiducial OK); the detections within 5 mm of their median are
averaged (a stray frame does not count); fewer than half the reads among them stops the run (``die_not_seen``). Then one env episode (``So101GoalEnv`` with the die task and the
object table, through the servo shm and its full safety chain, 10 Hz) with actions from
``ScriptedDieController.act_np`` (scaled by ``--speed-scale``).

**What the controller sees** (it was written for privileged sim state; here it is fed perception):

- the die position = the window average, frozen for the attempt (the gripper hides and the arm mask may drop the die
  near the jaws); re-read from the latest visible detection when an attempt restarts (a missed grasp or a die lost
  while carrying);
- while carrying (LIFT / CARRY / LOWER) the die position fed is the grasp point of the present ticks, so the
  controller's "lost" rule only fires on an empty gripper (perception may lose a held die, and the camera's 1-2 cm
  error must not look like a drop);
- **yaw is unknown** (the HSV tracker reports none). The grasp point is the jaw MIDPOINT (``JAW_MIDPOINT_GRIPPER``:
  16 mm to each open jaw tip, so a 16 mm die fits at any yaw, 4.7 mm clearance per side at 45 deg) and the jaw yaw
  target is the die's bearing from the base (``--yaw bearing``; or ``--yaw fixed --yaw-deg D``), turned by 45 deg on
  every restarted attempt; at most ``--max-attempts`` (3) attempts per trial. Closing the jaws squares the die.
  (A yaw from the blob's ``minAreaRect`` would need a new object-table field and the perception process: deferred.)

**Recorded per trial** (``trials.jsonl``): success, ``grasp_success`` (the gripper held something through LIFT /
CARRY and the die rose >= 8 mm in perception or moved >= 20 mm from its start while held, or ended in the zone after a
hold), ``pushed`` (zone success without a grasp), ``knocked`` (the die moved > 10 mm before the first close), the place
error (final die to the place point and to the zone centre), attempts, trips, clamped steps, steps, visible fraction,
the controller's last phase.

**Halt rules** (the run stops; the last command is HOLD; a human must look): any trip or stalled servo; perception
stale (object-table record > 1 s old) or the camera-moved fiducial not OK, checked before every trial and every step;
the die invisible on > 50 % of a trial's steps (``tracker_unknown``); the die not seen before a grasp
(``die_not_seen``); a failed reset.

**Safety:** this tool sends only shm commands through an ALREADY-RUNNING servo (``ShmBackend``; every action passes
the servo's clamp / stuck / box chain like any policy's); the servo is classified exactly like deploy-policy's
(``deploy_policy.resolve_servo``: the real arm needs ``--attended``, a real ``--no-mcu`` servo also
``--accept-no-watchdog``); start order as deploy-policy (perception first, then this tool, then the servo). The
scripted controller is not a trained policy, so deploy-policy's v0 / unverified ``sim_params`` refusal (for TRAINED
die exports) does not apply; heights below ``MIN_GRASP_H_M`` are refused.

**Outputs** ``<out-root>/<ts>/``: ``trials.jsonl`` (one line per reach point / trial, appended as it ends) and
``summary.json`` (per protocol, the verdict, the run's servo / config / calibration, halt reason), plus a printed table.

Exit codes: 0 done (whatever the verdict), 1 halted, 2 refused, 130 interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import signal
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "DEFAULT_GRASP_HEIGHTS_M",
    "MIN_GRASP_H_M",
    "PASS_RATE",
    "DieSample",
    "build_parser",
    "gate_verdict",
    "main",
    "reach_points",
    "read_die",
    "run_die_trial",
    "sample_die",
    "trial_halt",
    "trial_positions",
    "yaw_quat",
]

PASS_RATE: Final[float] = 0.30  # plan Phase 4: scripted seeds >= 30 % on the real arm
DEFAULT_TRIALS: Final[int] = 20
DEFAULT_GRASP_HEIGHTS_M: Final[tuple[float, ...]] = (0.022, 0.018, 0.0145, 0.012)
# the jaw-box bottoms sit ~10.5 mm below the grasp point (scripted.py: grasp_h 14.5 mm -> 4 mm): below 11 mm they would
# be commanded into the table
MIN_GRASP_H_M: Final[float] = 0.011
REACH_HEIGHT_M: Final[float] = 0.040  # the reach points' tip height over the table (point_tip marked the zone at 4 cm)
WINDOW_S: Final[float] = 1.0
WINDOW_MIN_VISIBLE: Final[float] = 0.5
WINDOW_MAX_SPREAD_M: Final[float] = 0.005
MAX_STEPS: Final[int] = 200  # 20 s at 10 Hz (the sim controller succeeds in ~50 steps)
DONE_STEPS: Final[int] = 20  # the controller back home this long without a success: the trial ends
# OPEN_ABOVE this long without lining up (seen in sim with a die estimate < ~0.13 m from the base axis, where the
# wrist_flex limit keeps the gripper from pointing down at the pre-grasp height): the trial ends
OPEN_ABOVE_TIMEOUT_STEPS: Final[int] = 60
EPILOGUE_MAX_STEPS: Final[int] = 40  # open + retreat home after a trial that did not end at home
MAX_ATTEMPTS: Final[int] = 3
LIFT_MIN_M: Final[float] = 0.008
MOVED_WHILE_HELD_M: Final[float] = 0.020
KNOCK_M: Final[float] = 0.010
UNKNOWN_HALT_FRACTION: Final[float] = 0.5
PERCEPTION_WAIT_S: Final[float] = 10.0  # retry a missing / stale object table this long before refusing
STAGE_NAMES: Final[tuple[str, ...]] = ("S0", "S1", "S2", "S3")
EXIT_OK, EXIT_HALTED, EXIT_REFUSED, EXIT_INTERRUPTED = 0, 1, 2, 130
TARGET_FILE: Final[str] = "die_target.json"


def _log(msg: str) -> None:
    print(f"feasibility-gate: {msg}", file=sys.stderr, flush=True)


class Halt(RuntimeError):
    """A halt rule fired: stop the run (``reason`` goes into ``summary.json``)."""

    def __init__(self, reason: str, msg: str) -> None:
        super().__init__(msg)
        self.reason = reason


# ------------------------------------------------------------------------------------------------ positions


def _stage_index(name: str, task: Any) -> int:
    key = name.strip().upper()
    if key in STAGE_NAMES[: task.die_start.n_stages]:
        return STAGE_NAMES.index(key)
    for k in range(task.die_start.n_stages):
        if task.die_start.stage_name(k) == name.strip():
            return k
    raise ValueError(f"unknown start stage {name!r} (S0..S{task.die_start.well_stage} or a stage name)")


def _shrunk(rect: tuple[float, float, float, float], m: float) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = rect
    return x0 + m, x1 - m, y0 + m, y1 - m


def trial_positions(task: Any, stages: Sequence[int], n: int, seed: int) -> list[dict[str, Any]]:
    """Per trial ``i``: stage ``stages[i % len]``; the target xy uniform in one of the stage's rectangles (picked by
    area) shrunk by the die's worst-case footprint half extent (any yaw fits), rounded to 1 mm; the well stage is the
    well centre. Depends only on the seed, the task and the stage list."""
    from so101_rl.env.tasks import DIE_MAX_FOOTPRINT_HALF_M

    rng = np.random.default_rng(seed)
    ds = task.die_start
    out = []
    for i in range(n):
        k = int(stages[i % len(stages)])
        if k == ds.well_stage:
            xy = np.asarray(ds.well_center_xy_m, float)
        else:
            reg = ds.table_stages[k]
            a = reg.areas
            r = reg.rects_xy_m[int(rng.choice(len(a), p=a / a.sum()))]
            x0, x1, y0, y1 = _shrunk(r, DIE_MAX_FOOTPRINT_HALF_M)
            xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        out.append({"stage": STAGE_NAMES[k], "stage_name": ds.stage_name(k), "xy": [round(float(v), 3) for v in xy]})
    return out


def stage_centre(task: Any, k: int) -> np.ndarray:
    """The centre of stage ``k``'s first rectangle (the well: its centre)."""
    ds = task.die_start
    if k == ds.well_stage:
        return np.asarray(ds.well_center_xy_m, float)
    x0, x1, y0, y1 = ds.table_stages[k].rects_xy_m[0]
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def reach_points(task: Any, place_xy: Sequence[float], *, tray: bool, table_z: float,
                 height: float = REACH_HEIGHT_M) -> list[tuple[str, np.ndarray]]:  # fmt: skip
    """``(name, xyz)``: the centre of every open-table start rectangle, the zone centre, the place point and (``tray``)
    the well, all ``height`` above the table. The zone first (marked on the rig with this policy, plan §10
    2026-10-16), the start regions last (S0 lies nearest the base)."""
    z = table_z + height
    pts: list[tuple[str, np.ndarray]] = [("zone", np.array([*task.zone.center_xy_m, z])),
                                         ("place", np.array([place_xy[0], place_xy[1], z]))]  # fmt: skip
    ds = task.die_start
    if tray:
        pts.append(("well", np.array([*ds.well_center_xy_m, z])))
    for k in reversed(range(len(ds.table_stages))):
        rects = ds.table_stages[k].rects_xy_m
        for j, (x0, x1, y0, y1) in enumerate(rects):
            name = STAGE_NAMES[k] + (f"[{j}]" if len(rects) > 1 else "")
            pts.append((name, np.array([(x0 + x1) / 2, (y0 + y1) / 2, z])))
    return pts


# ------------------------------------------------------------------------------------------------ perception


@dataclasses.dataclass
class DieSample:
    """The object table over a window: the mean of the visible, fresh, fiducial-OK die detections."""

    reads: int = 0
    visible: int = 0
    inliers: int = 0  # visible reads within WINDOW_MAX_SPREAD_M of their median (the average is over these)
    stale_reads: int = 0
    fiducial_bad_reads: int = 0
    xyz: list[float] | None = None
    spread_m: float | None = None
    max_record_age_s: float = 0.0

    @property
    def visible_fraction(self) -> float:
        return self.visible / self.reads if self.reads else 0.0

    def ok(self, min_visible: float = WINDOW_MIN_VISIBLE) -> bool:
        """At least ``min_visible`` of the reads are visible detections within 5 mm of their median."""
        return self.xyz is not None and self.reads > 0 and self.inliers / self.reads >= min_visible


def read_die(table: Any, rec: np.ndarray, now_ns: int, max_age_ns: int) -> tuple[str, np.ndarray | None, float]:
    """One object-table read with the env's rule (``So101GoalEnv._update_die``): ``("visible", xyz, record age)`` for
    a die detection reported visible, no older than ``max_age_ns`` (and at most one policy period "in the future"),
    with the fiducial OK; else ``"invisible"`` / ``"fiducial"`` / ``"torn"`` (and ``None``)."""
    from so101_rl.env.goal_env import DIE_LABEL

    if not table.read(rec):
        return "torn", None, math.inf
    r = rec[0]
    age = (now_ns - int(r["t_mono_ns"])) / 1e9
    if not bool(r["fiducial_ok"]):
        return "fiducial", None, age
    for o in r["objects"][: min(int(r["n"]), len(r["objects"]))]:
        if bytes(o["label"]).rstrip(b"\0") != DIE_LABEL:
            continue
        xyz = np.asarray(o["xyz_base"], np.float64)
        dt = now_ns - int(o["t_mono_ns"])
        if bool(o["visible"]) and -max_age_ns <= dt <= max_age_ns and np.all(np.isfinite(xyz)):
            return "visible", xyz, age
        break
    return "invisible", None, age


def die_max_age_ns(policy_hz: float) -> int:
    """The env's freshness bound: one policy period plus two camera frames."""
    from so101_rl.env.goal_env import CAMERA_HZ

    return round(1e9 / policy_hz) + 2 * round(1e9 / CAMERA_HZ)


def sample_die(table: Any, *, seconds: float, policy_hz: float = 10.0, period_s: float = 1 / 30,
               stale_s: float = 1.0) -> DieSample:  # fmt: skip
    """Read the object table every ``period_s`` for ``seconds`` (the caller keeps the servo alive) and average the
    visible detections (:func:`read_die`)."""
    from so101_rl.perception.object_table import new_object_table

    rec = new_object_table()
    s = DieSample()
    pts: list[np.ndarray] = []
    max_age = die_max_age_ns(policy_hz)
    t_end = time.monotonic() + seconds
    while True:
        status, xyz, age = read_die(table, rec, time.monotonic_ns(), max_age)
        s.reads += 1
        if math.isfinite(age):
            s.max_record_age_s = max(s.max_record_age_s, age)
        if status == "torn":  # the writer was mid-write (or never wrote): no evidence either way
            s.reads -= 1
            if time.monotonic() >= t_end:
                break
            time.sleep(0.001)
            continue
        s.stale_reads += int(age > stale_s)
        s.fiducial_bad_reads += int(status == "fiducial")
        if status == "visible" and xyz is not None and age <= stale_s:
            pts.append(xyz)
        if time.monotonic() >= t_end:
            break
        time.sleep(period_s)
    s.visible = len(pts)
    if pts:  # robust to a few outlier reads (a frame from before the die was put down, a blob flicker)
        p = np.asarray(pts)
        med = np.median(p, axis=0)
        inl = p[np.linalg.norm(p - med, axis=1) <= WINDOW_MAX_SPREAD_M]
        s.inliers = len(inl)
        if len(inl):
            mean = inl.mean(axis=0)
            s.xyz = [float(v) for v in mean]
            s.spread_m = float(np.linalg.norm(inl - mean, axis=1).max())
    return s


# ------------------------------------------------------------------------------------------------ the controller


def yaw_guess(xy: Sequence[float], attempt: int, mode: str = "bearing", fixed_deg: float = 0.0) -> float:
    """The jaw yaw target for an unknown die yaw: the die's bearing from the base (or a fixed angle), turned 45 deg per
    restarted attempt (the controller grips mod 90 deg)."""
    base = math.atan2(float(xy[1]), float(xy[0])) if mode == "bearing" else math.radians(fixed_deg)
    return base + attempt * math.pi / 4


def yaw_quat(yaw: float) -> np.ndarray:
    """wxyz of a die resting flat at ``yaw`` (``scripted.die_yaw`` returns ``yaw`` for it)."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def controller_params(*, grasp_h: float | None = None, place_xy: Sequence[float] | None = None) -> Any:
    """The scripted controller's parameters for the real arm: the jaw-midpoint grasp point (unknown yaw), optionally a
    grasp height and a place point (the grasp-height protocol puts the die back where it was)."""
    from so101_rl.rl.scripted import JAW_MIDPOINT_GRIPPER, ScriptedDieConfig

    p = ScriptedDieConfig(grasp_point_gripper=JAW_MIDPOINT_GRIPPER)
    if grasp_h is not None:
        p = dataclasses.replace(p, grasp_h=float(grasp_h))
    if place_xy is not None:
        p = dataclasses.replace(p, zone_xy_m=(float(place_xy[0]), float(place_xy[1])), place_offset_xy_m=(0.0, 0.0))
    p = dataclasses.replace(p, place_h=max(p.place_h, p.grasp_h))  # the controller never goes below grasp_h
    return p


def place_point(params: Any) -> np.ndarray:
    return np.asarray(params.zone_xy_m, float) + np.asarray(params.place_offset_xy_m, float)


def ik_check(ctl: Any, xy: Sequence[float], heights: dict[str, float], start_ticks: Sequence[int]) -> dict[str, Any]:
    """No motion: the controller's damped-least-squares IK from ``start_ticks`` for the grasp point at each height over
    ``xy`` (gripper pointing down, 60 iterations): position residual (mm) and gripper-axis tilt (deg)."""
    import torch

    from so101_rl.rl.scripted import dls_ik, gripper_fk

    f64 = torch.float64
    q0 = ctl._ticks_to_rad(torch.as_tensor(np.array(start_ticks, np.int64)[None, :5]))
    out = {}
    for name, h in heights.items():
        sp = torch.tensor([[float(xy[0]), float(xy[1]), ctl.p.table_z_m + h]], dtype=f64)
        q, res = dls_ik(ctl._chain_T, q0, sp, g_local=ctl._g_local, yaw_target=torch.zeros(1, dtype=f64),
                        w_yaw=torch.zeros(1, dtype=f64), q_lo=ctl._q_lo, q_hi=ctl._q_hi, iters=60,
                        damping=ctl.p.ik_damping, w_axis=ctl.p.w_axis, eps=ctl.p.fd_eps_rad)  # fmt: skip
        T = gripper_fk(ctl._chain_T, q)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(T[0, 2, 2])))))
        out[name] = {"h_m": h, "pos_residual_mm": round(float(res[0, :3].norm()) * 1e3, 2), "axis_tilt_deg": round(tilt, 1)}
    return out


@dataclasses.dataclass
class TrialSettings:
    max_steps: int = MAX_STEPS
    speed: float = 1.0
    yaw_mode: str = "bearing"
    yaw_deg: float = 0.0
    max_attempts: int = MAX_ATTEMPTS
    done_steps: int = DONE_STEPS
    open_above_timeout: int = OPEN_ABOVE_TIMEOUT_STEPS
    stale_s: float = 1.0


def _gp(ctl: Any, ticks: np.ndarray) -> np.ndarray:
    import torch

    return ctl.grasp_point(torch.as_tensor(np.asarray(ticks, np.int64)[None]))[0].cpu().numpy()


def run_die_trial(env: Any, ctl: Any, die0: Sequence[float], st: TrialSettings, *, table: Any = None,
                  zone_success: bool = True) -> dict[str, Any]:  # fmt: skip
    """One scripted episode on ``env`` (a ``So101GoalEnv`` with the die task and the object table) from the die at
    ``die0`` (perception's window average). Returns the trial record (module docstring); a perception failure inside
    the trial sets ``halt`` / ``halt_reason`` (the caller stops). ``table``: the object table, checked every step for a
    stale record or a failed fiducial. ``zone_success`` False (grasp-height protocol): the env's zone success is
    recorded but the trial runs until the controller is done."""
    from so101_rl.perception.object_table import new_object_table
    from so101_rl.rl.scripted import Phase
    from so101_rl.safety.types import SmState

    carry = (int(Phase.LIFT), int(Phase.CARRY), int(Phase.LOWER))
    start = np.asarray(die0, np.float64).copy()
    est = start.copy()
    obs, _ = env.reset()
    ctl.reset()
    rec_buf = new_object_table()
    max_age = die_max_age_ns(float(env.cfg.rates.policy_hz))
    steps = visible_steps = clamped = done_for = 0
    attempts = 1
    closed_once = held = False
    max_rise = max_moved_held = max_moved_before = 0.0
    success = tripped = stalled = False
    trips = 0
    sm = int(SmState.RUN)
    halt: tuple[str, str] | None = None
    last_vis: np.ndarray | None = None
    success_die: np.ndarray | None = None
    success_phase: int | None = None
    phase = int(ctl.phase[0])
    end_phase = phase
    end_reason = "max_steps"
    while steps < st.max_steps:
        rec = env._st[0]
        goal = rec["goal_ticks"].astype(np.int64)
        present = rec["present_ticks"].astype(np.int64)
        die_in = _gp(ctl, present) if phase in carry else est
        quat = yaw_quat(yaw_guess(est[:2], attempts - 1, st.yaw_mode, st.yaw_deg))
        a = ctl.act_np(goal, present, die_in, quat) * np.float32(st.speed)
        new_phase = int(ctl.phase[0])
        if new_phase == int(Phase.OPEN_ABOVE) and phase != int(Phase.OPEN_ABOVE):  # a missed / lost grasp: retry
            attempts += 1
            if last_vis is not None:
                est = last_vis.copy()
        phase = new_phase
        closed_once = closed_once or phase == int(Phase.CLOSE)
        if attempts > st.max_attempts:
            end_phase, end_reason = phase, "attempts_exhausted"
            break
        if phase == int(Phase.OPEN_ABOVE) and int(ctl.phase_steps[0]) >= st.open_above_timeout:
            end_phase, end_reason = phase, "open_above_timeout"
            break
        obs, _r, _term, truncated, info = env.step(a)
        steps += 1
        end_phase = phase
        clamped += int(info["action_clamped"])
        vis = bool(info["tracker_visible"])
        visible_steps += int(vis)
        die_now = np.asarray(obs["achieved_goal"], np.float64)
        if vis:
            last_vis = die_now.copy()
            if not closed_once:
                max_moved_before = max(max_moved_before, float(np.linalg.norm(die_now[:2] - start[:2])))
        grip = int(info["present_ticks"][5])
        if phase in carry and grip > ctl.gripper_empty:
            held = held or phase in (int(Phase.CARRY), int(Phase.LOWER))
            if vis:
                max_rise = max(max_rise, float(die_now[2] - start[2]))
                max_moved_held = max(max_moved_held, float(np.linalg.norm(die_now[:2] - start[:2])))
        sm = int(info["sm_state"])
        if "trip" in info:
            trips, tripped = int(info["trip"]), True
        stalled = stalled or bool(info.get("servo_stalled", False))
        if tripped or stalled:
            end_reason = "trip" if tripped else "stall"
            break
        if table is not None:
            for _ in range(3):  # a torn seqlock read (the writer mid-write) is retried, never taken as stale
                status, _xyz, age = read_die(table, rec_buf, time.monotonic_ns(), max_age)
                if status != "torn":
                    break
            if status == "torn":
                pass  # three torn reads in a row: no evidence this step (the next step reads again)
            elif status == "fiducial":
                halt = ("perception_stale", f"step {steps}: the camera-moved fiducial is not OK")
            elif age > st.stale_s:
                halt = ("perception_stale", f"step {steps}: the object table record is {age:.1f} s old")
            if halt is not None:
                end_reason = "halt"
                break
        if zone_success and info["is_success"] and not success:
            success = True
            success_die = die_now.copy()
            success_phase = phase
        done_for = done_for + 1 if phase == int(Phase.DONE) else 0
        # after a success during the release / retreat the controller finishes its retreat (the jaws may still stand
        # around the die, and the next home reset closes the gripper); in any other phase (e.g. a dropped die that
        # settled in the zone while the controller lines up a new grasp) the trial ends at once, as the env's episode
        if success and phase not in (int(Phase.RELEASE), int(Phase.RETREAT)):
            end_reason = "success"
            break
        if done_for >= st.done_steps or truncated:
            end_reason = "success" if success else "done" if done_for >= st.done_steps else "max_steps"
            break
    tripped = tripped or (not success and sm != int(SmState.RUN))
    seen = getattr(env, "_die_seen_ns", None) is not None
    final = np.asarray(obs["achieved_goal"], np.float64) if seen else None
    # epilogue (the result above is final): unless the controller is home, a trip / stall or a halt ended the trial,
    # open the jaws and retreat (RETREAT: straight up with the gripper opening, then home), so the next home reset
    # never closes the gripper on a die still between the jaws (seen in sim: ResetFailed, gripper blocked)
    epilogue_steps = 0
    if end_phase != int(Phase.DONE) and not (tripped or stalled) and halt is None:
        ctl.phase[:] = int(Phase.RETREAT)
        ctl.phase_steps[:] = 0
        while epilogue_steps < EPILOGUE_MAX_STEPS and int(ctl.phase[0]) != int(Phase.DONE):
            rec = env._st[0]
            a = ctl.act_np(rec["goal_ticks"].astype(np.int64), rec["present_ticks"].astype(np.int64),
                           _gp(ctl, rec["present_ticks"]), yaw_quat(0.0)) * np.float32(st.speed)  # fmt: skip
            _o, _r, _t, _tr, info = env.step(a)
            epilogue_steps += 1
            if "trip" in info or info.get("servo_stalled", False):
                trips, tripped = int(info.get("trip", 0)), "trip" in info
                stalled = stalled or bool(info.get("servo_stalled", False))
                sm = int(info["sm_state"])
                break
    place = place_point(ctl.p)
    zone_c = np.asarray(env.task.zone.center_xy_m, float)
    grasp_success = held and (max_rise >= LIFT_MIN_M or max_moved_held >= MOVED_WHILE_HELD_M or success)
    out: dict[str, Any] = {
        "die_start_xyz": [round(float(v), 4) for v in start],
        "success": success,
        "grasp_success": bool(grasp_success),
        "held": bool(held),
        "pushed": bool(success and not grasp_success),
        "knocked": bool(max_moved_before > KNOCK_M),
        "die_moved_before_close_m": round(max_moved_before, 4),
        "die_rise_while_held_m": round(max_rise, 4),
        "die_moved_while_held_m": round(max_moved_held, 4),
        "final_die_xyz": None if final is None else [round(float(v), 4) for v in final],
        "place_error_m": None if final is None else round(float(np.linalg.norm(final[:2] - place)), 4),
        "zone_error_m": None if final is None else round(float(np.linalg.norm(final[:2] - zone_c)), 4),
        "final_in_zone": None if final is None else bool(np.abs(final[:2] - zone_c).max() <= env.task.zone.half_edge_m),
        "success_die_xyz": None if success_die is None else [round(float(v), 4) for v in success_die],
        "success_phase": None if success_phase is None else Phase(success_phase).name,
        "attempts": min(attempts, st.max_attempts),
        "attempts_exhausted": attempts > st.max_attempts,
        "end_phase": Phase(end_phase).name,
        "end_reason": end_reason,
        "steps": steps,
        "visible_steps": visible_steps,
        "visible_fraction": visible_steps / steps if steps else 0.0,
        "clamped_steps": clamped,
        "trips": trips,
        "tripped": bool(tripped),
        "servo_stalled": bool(stalled),
        "sm_state": sm,
        "epilogue_steps": epilogue_steps,
    }
    if halt is not None:
        out["halt_reason"], out["halt"] = halt
    return out


def trial_halt(tr: dict[str, Any]) -> Halt | None:
    """The halt rule a finished die trial triggers, if any: a trip / stall, a perception failure inside the trial,
    the die UNKNOWN on more than half of its steps."""
    if tr["tripped"] or tr["servo_stalled"]:
        return Halt("trip" if tr["tripped"] else "stall",
                    f"trial {tr['trial']}: servo left RUN (sm_state {tr['sm_state']}, fault_bits {tr['trips']:#x})"
                    if tr["tripped"] else f"trial {tr['trial']}: servo stalled")  # fmt: skip
    if "halt_reason" in tr:
        return Halt(tr["halt_reason"], f"trial {tr['trial']}: {tr['halt']}")
    unknown = tr["steps"] - tr["visible_steps"]
    if tr["steps"] and unknown > UNKNOWN_HALT_FRACTION * tr["steps"]:
        return Halt("tracker_unknown", f"trial {tr['trial']}: the die was UNKNOWN on {unknown}/{tr['steps']} steps "
                    "(> 50 %): a human must check the die, the camera and perception")  # fmt: skip
    return None


def _raise_halt(tr: dict[str, Any]) -> None:
    h = trial_halt(tr)
    if h is not None:
        raise h


# ------------------------------------------------------------------------------------------------ verdicts


def _rate(successes: int, n: int) -> dict[str, Any]:
    from so101_rl.rl.eval_sim import wilson_interval

    lo, hi = wilson_interval(successes, n)
    return {"n": n, "successes": successes, "success_rate": successes / n if n else None, "wilson95": [lo, hi]}


def gate_verdict(trials: Sequence[dict[str, Any]], planned: int, *, halted: bool,
                 pass_rate: float = PASS_RATE) -> dict[str, Any]:  # fmt: skip
    """The scripted-seed gate: PASS when all ``planned`` trials ran without a halt and the success rate >= ``pass_rate``
    (the plan's 30 %, a point estimate; the Wilson 95 % interval is reported beside it); FAIL when complete and below;
    INCOMPLETE otherwise."""
    n = len(trials)
    s = sum(bool(t["success"]) for t in trials)
    row = _rate(s, n)
    complete = n >= planned and not halted
    status = ("PASS" if s >= pass_rate * n else "FAIL") if complete and n else "INCOMPLETE"
    row.update({"planned": planned, "pass_rate": pass_rate, "complete": complete, "status": status})
    return row


def summarize_scripted(trials: Sequence[dict[str, Any]], planned: int, *, halted: bool,
                       pass_rate: float = PASS_RATE) -> dict[str, Any]:  # fmt: skip
    doc = gate_verdict(trials, planned, halted=halted, pass_rate=pass_rate)
    by: dict[str, list[dict[str, Any]]] = {}
    for t in trials:
        by.setdefault(t["stage"], []).append(t)
    doc["per_stage"] = {k: _rate(sum(bool(t["success"]) for t in v), len(v)) for k, v in sorted(by.items())}
    errs = [t["zone_error_m"] for t in trials if t.get("zone_error_m") is not None]
    doc.update({
        "grasp_successes": sum(bool(t["grasp_success"]) for t in trials),
        "pushed": sum(bool(t["pushed"]) for t in trials),
        "left_zone_after_success": sum(bool(t["success"]) and t.get("final_in_zone") is False for t in trials),
        "knocked": sum(bool(t["knocked"]) for t in trials),
        "trip_trials": sum(bool(t["tripped"]) or bool(t["servo_stalled"]) for t in trials),
        "visible_fraction_mean": float(np.mean([t["visible_fraction"] for t in trials])) if trials else None,
        "zone_error_m_p50": float(np.median(errs)) if errs else None,
        "mean_steps": float(np.mean([t["steps"] for t in trials])) if trials else None,
    })  # fmt: skip
    return doc


def summarize_grasp_heights(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lifted = [t["grasp_h_m"] for t in trials if t["grasp_success"] and not t["tripped"]]
    return {"trials": len(trials), "lowest_lifted_grasp_h_m": min(lifted) if lifted else None,
            "per_height": [{k: t[k] for k in ("grasp_h_m", "grasp_success", "die_rise_while_held_m", "tripped",
                                              "trips", "clamped_steps", "end_phase")} for t in trials]}  # fmt: skip


def summarize_reach(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    errs = [p["reach_error_m"] for p in points if p.get("reach_error_m") is not None]
    return {"points": len(points), "successes": sum(bool(p["success"]) for p in points),
            "reach_error_m_max": max(errs) if errs else None,
            "reach_error_m_p50": float(np.median(errs)) if errs else None}  # fmt: skip


def _table(summary: dict[str, Any]) -> str:
    lines = []
    r = summary.get("reach")
    if r:
        lines.append("reach (tip 4 cm above each point; IK: grasp point at grasp height, residual / tilt):")
        for p in r["points_detail"]:
            ik = p.get("ik", {}).get("grasp", {})
            e = "n/a" if p.get("reach_error_m") is None else f"{p['reach_error_m'] * 1e3:5.1f} mm"
            lines.append(f"  {p['name']:>8} {e:>9} {'ok ' if p['success'] else 'MISS'} ik {ik.get('pos_residual_mm', '?')} mm "
                         f"/ {ik.get('axis_tilt_deg', '?')} deg")  # fmt: skip
    g = summary.get("grasp_height")
    if g:
        lines.append("grasp height sweep (grasp point over the table):")
        for t in g["per_height"]:
            lines.append(f"  {t['grasp_h_m'] * 1e3:5.1f} mm  {'LIFTED' if t['grasp_success'] else 'no lift':8} "
                         f"rise {t['die_rise_while_held_m'] * 1e3:5.1f} mm  clamped {t['clamped_steps']:3d}"
                         + (f"  TRIP {t['trips']:#x}" if t["tripped"] else ""))  # fmt: skip
        low = g["lowest_lifted_grasp_h_m"]
        lines.append(f"  lowest lifted: {'none' if low is None else f'{low * 1e3:.1f} mm'}")
    s = summary.get("scripted")
    if s:
        lo, hi = s["wilson95"]
        rate = "n/a" if s["success_rate"] is None else f"{s['success_rate']:.3f}"
        lines.append(f"scripted: {s['successes']}/{s['n']} (planned {s['planned']}) rate {rate} wilson95 [{lo:.3f}, "
                     f"{hi:.3f}]  grasped {s['grasp_successes']} pushed {s['pushed']} knocked {s['knocked']} "
                     f"left-zone-after-success {s['left_zone_after_success']} trips {s['trip_trials']}")  # fmt: skip
        for k, v in s["per_stage"].items():
            lines.append(f"  {k}: {v['successes']}/{v['n']}")
        lines.append(f"  gate (>= {s['pass_rate']:.0%} of {s['planned']}): {s['status']}")
    if summary.get("halted"):
        lines.append(f"HALTED ({summary['halt_reason']}): {summary['halted']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    from so101_rl.rl.tip_control import DEFAULT_REACH_POLICY

    ap = argparse.ArgumentParser(prog="feasibility-gate", description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", default="all", help="all, or a comma list of reach, grasp-height, scripted")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="scripted trials (20)")
    ap.add_argument("--starts", default="S0,S1,S2", help="scripted start stages, in turn (S0,S1,S2; S3 = the well)")
    ap.add_argument("--tray", action="store_true", help="add the well (placeholder until H9) to the starts and the "
                    "reach points")  # fmt: skip
    ap.add_argument("--seed", type=int, default=7, help="the trial positions")
    ap.add_argument("--speed-scale", type=float, default=1.0, help="scales the controller's actions (0 < s <= 1)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS, help="per die trial (10 Hz)")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS, help="grasp attempts per die trial")
    ap.add_argument("--yaw", choices=("bearing", "fixed"), default="bearing", help="jaw yaw for the unknown die yaw")
    ap.add_argument("--yaw-deg", type=float, default=0.0, help="--yaw fixed: the jaw yaw (deg)")
    ap.add_argument("--grasp-heights", default=",".join(f"{h:g}" for h in DEFAULT_GRASP_HEIGHTS_M),
                    help="grasp-height sweep: grasp point over the table (m), highest first")
    ap.add_argument("--grasp-stage", default="S0", help="grasp-height sweep: the die at this stage's centre")
    ap.add_argument("--grasp-repeats", type=int, default=1, help="trials per grasp height")
    ap.add_argument("--reach-policy", type=Path, default=DEFAULT_REACH_POLICY)
    ap.add_argument("--reach-height", type=float, default=REACH_HEIGHT_M, help="reach points above the table (m)")
    ap.add_argument("--window-s", type=float, default=WINDOW_S, help="perception average before each grasp")
    ap.add_argument("--pass-rate", type=float, default=PASS_RATE)
    ap.add_argument("--object-table", default=None, help="perception object table shm (default: the config's "
                    "object_table_shm, else so101_objects)")  # fmt: skip
    ap.add_argument("--config", type=Path, default=None, help="so101.yaml (default: the running servo's)")
    ap.add_argument("--calibration", type=Path, default=None, help="default: the running servo's")
    ap.add_argument("--shm-name", default=None, help="the running servo's shm segment (default: from config)")
    ap.add_argument("--attended", action="store_true", help="Tier 0: a human is at the e-stop (required on the arm)")
    ap.add_argument("--accept-no-watchdog", action="store_true", help="with --attended: allow a real --no-mcu servo")
    ap.add_argument("--out-root", type=Path, default=Path("runs/feasibility"))
    ap.add_argument("--wait-servo-s", type=float, default=120.0)
    return ap


PROTOCOLS: Final[tuple[str, ...]] = ("reach", "grasp-height", "scripted")


def _protocols(arg: str) -> list[str]:
    names = list(PROTOCOLS) if arg.strip() == "all" else [p.strip() for p in arg.split(",") if p.strip()]
    bad = [p for p in names if p not in PROTOCOLS]
    if bad or not names:
        raise ValueError(f"--protocol: unknown {bad or arg!r} (all, or a comma list of {', '.join(PROTOCOLS)})")
    return [p for p in PROTOCOLS if p in names]  # the fixed order


def _open_table(dp: Any, name: str, wait_s: float | None = None) -> Any:
    """``deploy_policy.open_object_table``, retried for up to ``wait_s`` (a perception writer started together with
    the servo, e.g. the sim twin, may not have written yet); the last refusal is raised."""
    t_end = time.monotonic() + (PERCEPTION_WAIT_S if wait_s is None else wait_s)
    while True:
        try:
            return dp.open_object_table(name)
        except dp.Refused:
            if time.monotonic() >= t_end:
                raise
            time.sleep(0.2)


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stopping = [False]

    def _on_signal(_signum: int, _frame: Any) -> None:
        if stopping[0]:
            return
        stopping[0] = True
        raise KeyboardInterrupt

    old = {s: signal.signal(s, _on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        return _main(args, stopping)
    except KeyboardInterrupt:
        _log("interrupted before the attach: no command was sent")
        return EXIT_INTERRUPTED
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def _main(args: argparse.Namespace, stopping: list[bool]) -> int:
    import torch

    from so101_rl.config.loader import DEFAULT_CONFIG_PATH
    from so101_rl.env.backend import ResetFailed, ServoNotRunning, ServoStateStale, ShmBackend
    from so101_rl.env.goal_env import So101GoalEnv
    from so101_rl.env.tasks import DIE_TO_ZONE
    from so101_rl.rl import deploy_policy as dp
    from so101_rl.rl import tip_control as tc
    from so101_rl.rl.scripted import ScriptedDieController

    torch.set_num_threads(1)
    task = DIE_TO_ZONE
    try:
        protocols = _protocols(args.protocol)
        if not 0 < args.speed_scale <= 1:
            raise ValueError("--speed-scale must be in (0, 1]")
        if args.trials < 1 or args.max_steps < 1 or args.max_attempts < 1 or args.grasp_repeats < 1:
            raise ValueError("--trials, --max-steps, --max-attempts and --grasp-repeats must be >= 1")
        heights = [float(h) for h in str(args.grasp_heights).split(",") if h.strip()]
        if "grasp-height" in protocols and (not heights or min(heights) < MIN_GRASP_H_M):
            raise ValueError(f"--grasp-heights {args.grasp_heights!r}: every height must be >= {MIN_GRASP_H_M} m (the "
                             "jaws would be commanded into the table)")  # fmt: skip
        starts = [_stage_index(s, task) for s in str(args.starts).split(",") if s.strip()]
        if args.tray and task.die_start.well_stage not in starts:
            starts.append(task.die_start.well_stage)
        if not starts:
            raise ValueError("--starts: no stage")
        grasp_stage = _stage_index(args.grasp_stage, task)
        reach_policy = reach_task = None
        if "reach" in protocols:  # torch + the policy BEFORE the servo is looked up (deploy-policy's order)
            try:
                reach_policy, reach_task = tc.load_reach_policy(args.reach_policy)
            except (ValueError, KeyError, RuntimeError, OSError) as exc:
                raise dp.Refused(f"cannot use the reach policy {args.reach_policy}: {exc}") from None
        shm_name = args.shm_name or str(dp._yaml_key(args.config or DEFAULT_CONFIG_PATH, "shm_name"))
        rs = dp.resolve_servo(shm_name, wait_s=args.wait_servo_s, attended=args.attended,
                              accept_no_watchdog=args.accept_no_watchdog, config=args.config,
                              calibration=args.calibration, log=_log)  # fmt: skip
        table_name = args.object_table or dp.object_table_name(rs.config_path)
        table = _open_table(dp, table_name)
    except dp.Refused as exc:
        _log(f"refused: {exc}")
        return EXIT_REFUSED
    except (ValueError, OSError, KeyError, TypeError) as exc:
        _log(f"refused: {type(exc).__name__}: {exc}")
        return EXIT_REFUSED

    servo, cfg = rs.servo, rs.cfg
    out = args.out_root / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{os.urandom(2).hex()}"
    meta: dict[str, Any] = {
        "tool": "feasibility-gate",
        "plan": "Phase 4 feasibility gate (plan §10 2026-10-21 'Phase 4 feasibility-gate runner')",
        "protocols": protocols,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "servo": {"pid": servo.pid, "backend": servo.backend, "real": servo.is_real, "mcu": servo.mcu,
                  "no_watchdog_accepted": bool(servo.is_real and servo.no_mcu), "shm_name": shm_name,
                  "cmdline": servo.cmdline},
        "config": str(Path(rs.config_path).resolve()),
        "calibration": str(rs.calibration_path),
        "calibration_sha256": rs.calibration_sha256,
        "tick_log": None if servo.tick_log is None else str(servo.tick_log),
        "object_table": table_name,
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
    }  # fmt: skip
    results: dict[str, list[dict[str, Any]]] = {p: [] for p in protocols}
    halt: str | None = None
    halt_reason: str | None = None
    backend = None
    plan_doc: dict[str, Any] = {}

    def _doc(code: int | None) -> dict[str, Any]:
        doc = dict(meta)
        doc["trial_plan"] = plan_doc
        if "reach" in results:
            doc["reach"] = {**summarize_reach(results["reach"]), "points_detail": results["reach"]}
        if "grasp-height" in results:
            doc["grasp_height"] = summarize_grasp_heights(results["grasp-height"])
        if "scripted" in results:
            doc["scripted"] = summarize_scripted(results["scripted"], args.trials, halted=halt is not None,
                                                 pass_rate=args.pass_rate)  # fmt: skip
        doc.update({"finished": None if code is None else datetime.now().astimezone().isoformat(timespec="seconds"),
                    "halted": halt, "halt_reason": halt_reason, "exit_code": code,
                    "policy_overruns": int(backend.overruns) if backend is not None else 0})  # fmt: skip
        return doc

    # ATTACH NOW (deploy-policy's invariant: from the attach until RUN a HOLD at least every 100 ms)
    keepalive = contextlib.ExitStack()
    try:
        backend = ShmBackend(cfg, calibration=rs.calibration)
    except (FileNotFoundError, RuntimeError) as exc:
        _log(f"refused: cannot attach to the servo shm {shm_name!r}: {exc}")
        table.close()
        return EXIT_REFUSED
    keepalive.enter_context(dp.hold_keepalive(backend))
    try:
        # the trial length; the trip penalty scales with it (TaskSpec validation; the reward is not used here)
        die_task = dataclasses.replace(task, episode_steps=args.max_steps,
                                       trip_penalty=task.trip_penalty * max(1.0, args.max_steps / task.episode_steps))
        die_env = So101GoalEnv(backend, cfg, die_task, object_table=table)
        tip = None
        if reach_policy is not None:
            goals = dp.GoalList(np.zeros((1, 2, 3)))
            reach_env = So101GoalEnv(backend, cfg, reach_task, goal_sampler=goals)
            tip = tc.TipController(reach_env, reach_policy, goals, backend, reach_task, float(cfg.rates.policy_hz))

        def make_ctl(params: Any) -> Any:
            return ScriptedDieController.from_calibration(cfg, rs.calibration, 1, params=params)

        base_params = controller_params()
        ctl = make_ctl(base_params)
        place_xy = place_point(ctl.p)
        positions = trial_positions(task, starts, args.trials, args.seed) if "scripted" in protocols else []
        g_xy = stage_centre(task, grasp_stage)
        height_ctls = ([(h, make_ctl(controller_params(grasp_h=h, place_xy=g_xy))) for h in heights
                        for _ in range(args.grasp_repeats)] if "grasp-height" in protocols else [])  # fmt: skip
        rpts = reach_points(task, place_xy, tray=args.tray, table_z=float(ctl.p.table_z_m), height=args.reach_height)
        if tip is not None:
            outside = [n for n, p in rpts if not bool(reach_task.goal_bounds.contains_np(p[None])[0])]
            if outside:
                raise dp.Refused(f"reach points {outside} lie outside the reach goal bounds (--reach-height)")
        plan_doc.update({
            "scripted_positions": positions, "grasp_heights_m": heights, "grasp_xy": [round(float(v), 3) for v in g_xy],
            "reach_points": [{"name": n, "xyz": [round(float(v), 4) for v in p]} for n, p in rpts] if tip else [],
            "place_xy": [round(float(v), 4) for v in place_xy], "controller": dataclasses.asdict(ctl.p),
        })  # fmt: skip
        out.mkdir(parents=True, exist_ok=False)
        _write_json(out / "summary.json", _doc(None))
        _log(f"servo pid {servo.pid} --backend {servo.backend} ({'REAL ARM, attended' if servo.is_real else 'sim'}); "
             f"out {out}")
    except (dp.Refused, ValueError, OSError, KeyError, TypeError, KeyboardInterrupt) as exc:
        stopping[0] = True
        keepalive.close()
        dp._hold(backend)
        with contextlib.suppress(Exception):
            backend.close()
        table.close()
        if isinstance(exc, KeyboardInterrupt):
            _log("interrupted during the set-up after the attach: only HOLD was sent")
            return EXIT_INTERRUPTED
        _log(f"refused: {type(exc).__name__}: {exc}")
        return EXIT_REFUSED

    st = TrialSettings(max_steps=args.max_steps, speed=args.speed_scale, yaw_mode=args.yaw, yaw_deg=args.yaw_deg,
                       max_attempts=args.max_attempts)  # fmt: skip
    gate = dp.DieResetGate(out, _log)
    home = np.asarray(cfg.safe_state.home_ticks)
    counter = [0]
    code = EXIT_OK

    def die_trial(f: Any, protocol: str, c: Any, target: dict[str, Any], *, zone_success: bool) -> dict[str, Any]:
        k = counter[0]
        counter[0] += 1
        backend.reset(home)  # arm home, then held while the operator places the die
        with dp.hold_keepalive(backend):
            _write_json(out / TARGET_FILE, {"trial": k, "protocol": protocol, **target})
            print(f"trial {k} ({protocol}, {target.get('stage', '')} {target.get('stage_name', '')}): place the die at "
                  f"x = {target['xy'][0]:.3f}, y = {target['xy'][1]:.3f} m (any yaw)", flush=True)  # fmt: skip
            gate.wait(k)
            try:
                dp.check_object_table(table, table_name)
            except dp.Refused as exc:
                raise Halt("perception_stale", f"before trial {k}: {exc}") from None
            s = sample_die(table, seconds=args.window_s, policy_hz=float(cfg.rates.policy_hz))
        if not s.ok():
            raise Halt("die_not_seen", f"trial {k}: the die was visible on {s.visible}/{s.reads} reads, {s.inliers} "
                       "within 5 mm of their median, before the grasp: check the die, the camera and perception")  # fmt: skip
        tr = run_die_trial(die_env, c, s.xyz, st, table=table, zone_success=zone_success)
        tr = {"trial": k, "protocol": protocol, **target, "window": dataclasses.asdict(s), **tr}
        tr["placement_error_m"] = round(float(np.linalg.norm(np.asarray(s.xyz[:2]) - np.asarray(target["xy"]))), 4)
        f.write(json.dumps(tr) + "\n")
        f.flush()
        return tr

    try:
        with (out / "trials.jsonl").open("a") as f:
            keepalive.close()
            dp.wait_for_run(backend, args.wait_servo_s, _log)
            try:
                if "reach" in protocols and tip is not None:
                    for name, xyz in rpts:
                        ik = ik_check(ctl, xyz[:2], {"pregrasp": ctl.p.pregrasp_h, "grasp": ctl.p.grasp_h}, home)
                        try:
                            ep = tip.reach(xyz)
                            info = tip.hold(1.0)
                        except tc.ServoLeftRun as exc:
                            row = {"protocol": "reach", "name": name, "xyz": xyz.round(4).tolist(), "success": False,
                                   "reach_error_m": None, "tripped": True, "ik": ik}  # fmt: skip
                            results["reach"].append(row)
                            f.write(json.dumps(row) + "\n")
                            raise Halt("trip", f"reach to {name}: {exc}") from None
                        tip_xyz = backend.fk().gripper_xyz_from_ticks(np.asarray(info["present_ticks"]), np.empty(3))
                        err = float(np.linalg.norm(tip_xyz - xyz))
                        row = {"protocol": "reach", "name": name, "xyz": xyz.round(4).tolist(),
                               "success": bool(ep["success"]), "episode_error_m": ep["final_error_m"],
                               "reach_error_m": round(err, 5), "steps": ep["steps"], "tripped": False, "ik": ik}  # fmt: skip
                        results["reach"].append(row)
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        print(f"reach {name:>8}: error {err * 1e3:.1f} mm ({'success' if ep['success'] else 'miss'}, "
                              f"{ep['steps']} steps)", flush=True)  # fmt: skip
                for h, c in height_ctls:
                    tgt = {"stage": STAGE_NAMES[grasp_stage], "stage_name": task.die_start.stage_name(grasp_stage),
                           "xy": [round(float(v), 3) for v in g_xy], "grasp_h_m": h}  # fmt: skip
                    tr = die_trial(f, "grasp-height", c, tgt, zone_success=False)
                    results["grasp-height"].append(tr)
                    print(f"grasp height {h * 1e3:.1f} mm: {'LIFTED' if tr['grasp_success'] else 'no lift'} (rise "
                          f"{tr['die_rise_while_held_m'] * 1e3:.1f} mm, {tr['steps']} steps)", flush=True)  # fmt: skip
                    _raise_halt(tr)
                for pos in positions:
                    tr = die_trial(f, "scripted", ctl, pos, zone_success=True)
                    results["scripted"].append(tr)
                    e = "n/a" if tr["zone_error_m"] is None else f"{tr['zone_error_m'] * 1e3:.0f} mm"
                    print(f"trial {tr['trial']} {pos['stage']}: {'SUCCESS' if tr['success'] else 'fail   '} grasp "
                          f"{'yes' if tr['grasp_success'] else 'no '} zone err {e} visible {tr['visible_fraction']:.2f} "
                          f"steps {tr['steps']} attempts {tr['attempts']}" + (f" TRIP {tr['trips']:#x}" if tr["tripped"]
                                                                              else ""), flush=True)  # fmt: skip
                    _raise_halt(tr)
            except Halt as exc:
                halt, halt_reason = str(exc), exc.reason
            except (ServoNotRunning, ResetFailed, ServoStateStale) as exc:
                halt = f"{type(exc).__name__}: {exc}"
                halt_reason = "reset_failed" if isinstance(exc, ResetFailed) else "servo_not_running"
        if halt is not None:
            _log(f"stopped: {halt}")
            code = EXIT_HALTED
    except (ServoNotRunning, ServoStateStale) as exc:
        halt, halt_reason, code = f"{type(exc).__name__}: {exc}", "servo_not_running", EXIT_HALTED
        _log(f"stopped: {halt}")
    except KeyboardInterrupt:
        halt, halt_reason, code = "interrupted (Ctrl-C / SIGTERM)", "interrupted", EXIT_INTERRUPTED
        _log("interrupted: HOLD sent, no further commands; the servo safe-stops the arm to rest 0.5 s later")
    except Exception as exc:  # noqa: BLE001 - still HOLD and write the partial summary
        halt, halt_reason, code = f"error: {exc!r}", "error", EXIT_HALTED
        _log(f"stopped by an error: {exc!r}")
    finally:
        stopping[0] = True
        keepalive.close()
        if backend is not None:
            dp._hold(backend)  # the last command is always HOLD
        with contextlib.suppress(Exception):
            backend.close()
        table.close()
        (out / TARGET_FILE).unlink(missing_ok=True)
    summary = _doc(code)
    _write_json(out / "summary.json", summary)
    print(_table(summary))
    print(f"run dir {out}  tick log {meta['tick_log']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

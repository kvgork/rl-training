"""Feasibility (L1) and predicate (L2) gates for reach and die_to_zone, before any reward tuning (sim plan S4.2, MVP M3).

``pixi run sim-gates --task reach [--out runs/gates/reach-<ts>.json]`` prints a PASS/FAIL table and exits 0 only
when every row passes. The GPU rows (scripted controller, do-nothing, random policy) need CUDA; without it they FAIL
(the gate is not proven), so the CLI exits 1.

L1 (feasibility):
- ``fk_coverage``: FK of 2e5 random in-limit joint vectors (limits shrunk by a margin) covers every one of 1e4 goals
  drawn by the env's goal sampler within 0.01 m.
- ``travel_budget``: ``max_delta x episode_steps >= 2 x`` the joint distance (ticks, per joint) from the start pose to
  the closest-to-start joint vector reaching each goal, for every goal (worst case reported).
- ``start_not_success``: no start pose (home +- start noise) already lies within the success radius of its goal.
- ``scripted_reach`` (GPU): a damped-least-squares Jacobian controller in tick space, acting through the real clamp
  chain on the batched env with stage-1 proprio noise, succeeds in >= 95 % of 1024 worlds within the horizon.

L2 (predicate):
- ``adversarial``: the reach predicate table (on the goal, inside/outside the 15 mm ball by 0.1 mm, NaN tip / goal)
  gives success exactly for the genuine rows.
- ``independent_predicate``: a separately written numpy predicate agrees with ``tasks.success_np`` on 1e5 states
  (half of them within 1 mm of the radius).
- ``do_nothing`` (GPU): zero actions never succeed (1024 worlds x 1 episode).
- ``random_policy`` (GPU): uniform random actions succeed on < 0.1 % of episode-steps (a looser predicate or a
  trivial goal distribution would show here).

Each gate has a shown-to-fail twin in ``tests/test_gates.py`` (goal radius 0.40 m without the reachability filter
fails ``fk_coverage``; a 30 mm predicate fails ``adversarial`` and ``independent_predicate``).

``pixi run sim-gates --task die_to_zone [--nworld K]`` runs the die gates (plans/die-to-zone-plan.md D1). The die
predicate (``env/tasks.py``): visible AND ``|dx|, |dy| <= zone.half_edge_m`` (a square, L-inf in xy) AND
``z - z_goal <= zone.max_height_m``; termination needs ``success_debounce_steps`` consecutive predicate steps with
the die settled (moved <= ``settle_m`` between them).

- ``zone_in_view`` (L1, CPU): every point of a 7 x 7 lattice over the zone square (corners included) at the resting
  die height is in the calibrated D435's view (``calib/camera.yaml``, ``SimDieTracker.in_view``: colour and depth
  image with the half-blob border margin). A zone the camera cannot see makes the task unsolvable (the old zone at
  (0.22, 0.10) was entirely out of view; plan §10 2026-10-16 "Die zone moved into the camera view").
- ``start_not_success`` (L1, CPU): no die start lies in the zone. Worst case over the whole start set (the batched
  env clips the die centre to the flat floor of the start well, ``batched_env._WELL_HALF_M``: the well square's
  point nearest the zone centre) plus 1e4 sampled starts (the env's draw: N(centre, sigma), clipped).
- Start curriculum (D2a, plan §10 2026-10-16): ``start_not_success`` and ``do_nothing`` run once per start stage of
  ``DIE_TO_ZONE.die_start``. The well stage (S3, the target) keeps the row names; the open-table stages S0..S2 add
  ``_S<k>`` (worst case: each footprint rectangle shrunk by the smallest footprint half extent, its point nearest
  the zone centre; samples: the env's rectangle-by-area, uniform-centre draw). The GPU rows stay at the well.
- ``adversarial`` (L2, CPU): the die predicate table (zone centre, 0.1 mm inside / outside an edge, the inside
  corner (a Euclidean predicate fails it), 0.1 mm under / over the height bound, held 20 mm up, invisible, NaN).
- ``independent_predicate`` (L2, CPU): a separately written predicate agrees with ``tasks.success_np`` on 1e5
  states (goals anywhere, as after HER relabelling; die within 1 mm of an edge or the height bound; 10 % invisible).
- ``independent_settle`` (L2, CPU): a separately written debounce / settle loop agrees with ``tasks.debounce_np``
  on random predicate / displacement sequences (moves straddling ``settle_m``).
- ``do_nothing`` (L2, CPU): CPU MuJoCo, the arm holding its start pose (home +- start noise) for one episode with
  the die placed as the env places it; the TRUE die position, read every policy step as an always-visible tracker
  (more permissive than the sim tracker), never terminates as a success (``K`` worlds, default 16).
- ``do_nothing_batched`` and ``random_policy`` (GPU): as the reach rows, on the batched die env with the simulated
  tracker. Without CUDA they FAIL (not run), so the CLI exits 1, as for reach.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "GATE_TASKS",
    "GateRow",
    "independent_die_settle",
    "independent_die_success",
    "independent_reach_success",
    "main",
    "run_die_gates",
    "run_reach_gates",
]

GATE_TASKS: tuple[str, ...] = ("reach", "die_to_zone")

Predicate = Callable[..., np.ndarray]


@dataclass(frozen=True)
class GateRow:
    level: str
    name: str
    passed: bool
    value: str
    threshold: str


def independent_reach_success(achieved: np.ndarray, desired: np.ndarray, radius: float) -> np.ndarray:
    """Second, separately written reach predicate (L2): explicit sum of squares, NaN never succeeds."""
    a = np.asarray(achieved, dtype=np.float64)
    d = np.asarray(desired, dtype=np.float64)
    sq = np.zeros(a.shape[:-1])
    for i in range(a.shape[-1]):
        sq = sq + (a[..., i] - d[..., i]) ** 2
    return np.where(np.isnan(sq), False, sq <= radius * radius)


# ------------------------------------------------------------------------------------------------ L1 (CPU)
def _setup(calibration: str | None) -> tuple[Any, Any, Any]:
    from so101_rl.config import load_calibration, load_config
    from so101_rl.safety.fk import KinematicChain, So101FK

    cal_path = load_config().calibration_path if calibration is None else Path(calibration)
    cal = load_calibration(cal_path)
    cfg = load_config(calibration=cal)
    return cfg, cal, So101FK(KinematicChain.from_yaml(cfg.kinematics_path), cal)


def _joint_samples(cfg: Any, n: int, rng: np.random.Generator, margin_frac: float = 0.02) -> np.ndarray:
    lo = np.asarray(cfg.joints.min_ticks, np.int64)
    hi = np.asarray(cfg.joints.max_ticks, np.int64)
    m = ((hi - lo) * margin_frac).astype(np.int64)
    return rng.integers(lo + m, hi - m + 1, size=(n, lo.size))


def _home_ticks(cal: Any) -> np.ndarray:
    """Present ticks of the scene's ``home`` keyframe (``MujocoBus.q_to_ticks`` semantics)."""
    from so101_rl.config import rad_to_ticks
    from so101_rl.sim.gripper_map import gripper_rad_to_ticks
    from so101_rl.sim.model import load_model

    q = np.asarray(load_model()[0].key("home").qpos[:6], np.float64)
    t = rad_to_ticks(q, cal)
    t[5] = gripper_rad_to_ticks(float(q[5]), cal)
    return np.clip(t, 0, 4095).astype(np.int64)


def _ik(fk: Any, start: np.ndarray, goals: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, iters: int = 80,
        h: float = 10.0, lam: float = 1e-9, rest: np.ndarray | None = None, k_null: float = 0.2,
        ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Damped least squares in tick space on pan / lift / elbow / wrist flex, clipped to [lo, hi] every iteration;
    with ``rest`` the 1-D null space pulls toward it (the minimum-travel solution from the start pose).
    Returns (int ticks [N, 6], final tip error [N] m)."""
    from so101_rl.env.goals import fk_tip_batch

    t = start.astype(np.float64).copy()
    eye = lam * np.eye(3)
    for _ in range(iters):
        ti = np.rint(t).astype(np.int64)
        err = goals - fk_tip_batch(fk, ti)
        jac = np.empty((len(t), 3, 4))
        for j in range(4):
            tp, tm = ti.copy(), ti.copy()
            tp[:, j] += int(h)
            tm[:, j] -= int(h)
            jac[:, :, j] = (fk_tip_batch(fk, np.clip(tp, lo, hi)) - fk_tip_batch(fk, np.clip(tm, lo, hi))) / (2 * h)
        jjt_inv = np.linalg.inv(jac @ jac.transpose(0, 2, 1) + eye)
        jpinv = jac.transpose(0, 2, 1) @ jjt_inv  # [N, 4, 3]
        dq = np.einsum("nij,nj->ni", jpinv, err)
        if rest is not None:
            null = np.eye(4)[None] - jpinv @ jac
            dq += np.einsum("nij,nj->ni", null, k_null * (rest[..., :4] - t[:, :4]))
        t[:, :4] = np.clip(t[:, :4] + dq, lo[:4], hi[:4])
    ti = np.rint(t).astype(np.int64)
    return ti, np.linalg.norm(goals - fk_tip_batch(fk, ti), axis=1)


def exclude_near_start(goals: np.ndarray, alt: np.ndarray, start_tip: np.ndarray, success_radius_m: float) -> np.ndarray:
    """The envs' reset rule, applied per row through the one definition ``env/goals.redraw_near_start_np``: a goal
    within ``GOAL_MIN_START_FACTOR`` success radii of its start tip takes the second draw ``alt`` (no trivial
    episodes; L2 do-nothing = 0)."""
    from so101_rl.env.goals import redraw_near_start_np

    return np.stack([redraw_near_start_np(g, t, success_radius_m, lambda a=a: a)
                     for g, a, t in zip(goals, alt, np.broadcast_to(start_tip, goals.shape), strict=True)])  # fmt: skip


def l1_reach_cpu(
    cfg: Any,
    cal: Any,
    fk: Any,
    *,
    sample_goals: Callable[[int, np.random.Generator], np.ndarray],
    success_radius_m: float,
    episode_steps: int,
    start_noise_ticks: int = 30,
    home_ticks: np.ndarray | None = None,
    n_joint: int = 200_000,
    n_goal: int = 10_000,
    tol_m: float = 0.01,
    seed: int = 0,
) -> list[GateRow]:
    """``fk_coverage``: nearest of ``n_joint`` random in-limit FK samples, then IK refinement (from home, else from the
    nearest sample) inside the shrunk limits; ``travel_budget``: from home to the IK solution reached from home."""
    from so101_rl.env.goals import fk_tip_batch

    rng = np.random.default_rng(seed)
    joints = _joint_samples(cfg, n_joint, rng)
    xyz = torch.as_tensor(fk_tip_batch(fk, joints))
    goals = np.asarray(sample_goals(n_goal, rng), np.float64)
    g = torch.as_tensor(goals)
    home = np.asarray(home_ticks if home_ticks is not None else _home_ticks(cal), np.int64)
    lo0, hi0 = np.asarray(cfg.joints.min_ticks, np.int64), np.asarray(cfg.joints.max_ticks, np.int64)
    m = ((hi0 - lo0) * 0.02).astype(np.int64)
    lo, hi = lo0 + m, hi0 - m
    near_idx = torch.empty(len(goals), dtype=torch.int64)
    for i in range(0, len(goals), 256):
        near_idx[i : i + 256] = torch.cdist(g[i : i + 256], xyz).argmin(dim=1)
    h0 = np.broadcast_to(np.clip(home, lo, hi), (len(goals), 6))
    t_home, e_home = _ik(fk, h0, goals, lo, hi, rest=h0.astype(np.float64))
    t_near, e_near = _ik(fk, joints[near_idx.numpy()], goals, lo, hi, rest=h0.astype(np.float64))
    err = np.minimum(e_home, e_near)
    cover = float((err <= tol_m).mean())
    sol = np.where((e_home <= tol_m)[:, None], t_home, t_near)
    maxd = np.asarray(cfg.action.max_delta_ticks, np.float64)
    steps = (np.abs(sol[:, :5] - home[None, :5]) / maxd[:5]).max(axis=1)
    worst = float(steps[err <= tol_m].max()) if (err <= tol_m).any() else math.inf
    rows = [
        GateRow("L1", "fk_coverage", cover >= 1.0, f"{cover * 100:.3f} % of {len(goals)} goals reached within {tol_m} m "
                f"inside the joint limits (max {err.max() * 1000:.1f} mm)", "100 %"),
        GateRow("L1", "travel_budget", 2.0 * worst <= episode_steps,
                f"worst goal needs {worst:.1f} full-speed steps from home (x2 = {2 * worst:.1f})",
                f"<= episode_steps {episode_steps}"),
    ]
    # start poses: home +- noise on the 5 arm joints; after the env's near-start exclusion, any start already inside
    # the success ball of its goal is a trivial episode
    off = rng.integers(-start_noise_ticks, start_noise_ticks + 1, size=(len(goals), 6))
    off[:, 5] = 0
    start_tip = fk_tip_batch(fk, np.clip(home[None, :] + off, lo0, hi0))
    final = exclude_near_start(goals, np.asarray(sample_goals(len(goals), rng), np.float64), start_tip,
                               success_radius_m)  # fmt: skip
    d0 = np.linalg.norm(start_tip - final, axis=1)
    frac0 = float((d0 <= success_radius_m).mean())
    rows.append(GateRow("L1", "start_not_success", frac0 == 0.0,
                        f"{frac0 * 100:.3f} % of starts within {success_radius_m * 1000:.0f} mm of the goal "
                        f"(min {d0.min() * 1000:.1f} mm)", "0 %"))  # fmt: skip
    return rows


# ------------------------------------------------------------------------------------------------ L2 (CPU)
def l2_reach_cpu(spec: Any, *, predicate: Predicate | None = None, n: int = 100_000, seed: int = 0) -> list[GateRow]:
    from so101_rl.env.tasks import success_np

    pred = predicate or (lambda a, d: success_np(a, d, spec))
    r = spec.success_radius_m
    goal = np.array([0.20, 0.0, 0.15])
    unit = np.array([0.0, 0.6, 0.8])
    table = [  # (case, achieved, genuine success)
        ("tip on the goal", goal, True),
        ("tip 0.1 mm inside the radius", goal + unit * (r - 1e-4), True),
        ("tip 0.1 mm outside the radius", goal + unit * (r + 1e-4), False),
        ("tip 2 x radius away", goal + unit * 2 * r, False),
        ("tip NaN", np.full(3, np.nan), False),
    ]
    bad = []
    with np.errstate(invalid="ignore"):
        for case, ach, want in table:
            got = bool(np.asarray(pred(ach[None], goal[None]))[0])
            if got != want:
                bad.append(f"{case}: got {got}")
        if bool(np.asarray(pred(goal[None], np.full((1, 3), np.nan)))[0]):
            bad.append("goal NaN: got True")
    rows = [GateRow("L2", "adversarial", not bad, "; ".join(bad) or f"{len(table) + 1} cases as expected", "all cases")]
    rng = np.random.default_rng(seed)
    d = rng.uniform(-0.3, 0.3, size=(n, 3))
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    half = n // 2
    a = d.copy()
    a[:half] = d[:half] + dirs[:half] * rng.uniform(r - 1e-3, r + 1e-3, size=(half, 1))  # near the boundary
    a[half:] = d[half:] + dirs[half:] * rng.uniform(0, 0.1, size=(n - half, 1))
    agree = float((np.asarray(pred(a, d)) == independent_reach_success(a, d, r)).mean())
    rows.append(GateRow("L2", "independent_predicate", agree == 1.0, f"agreement {agree * 100:.4f} % on {n} states", "100 %"))
    return rows


# ------------------------------------------------------------------------------------------------ GPU rows
def _scripted_action(env: Any, obs: torch.Tensor, *, lam: float = 1e-9, h: int = 20) -> torch.Tensor:
    """Damped-least-squares IK step in tick space from the (noisy) observation only: present ticks from ``q_norm``,
    tip from ``tip_xyz``, Jacobian of the batched FK by central differences on pan / lift / elbow / wrist flex."""
    from so101_rl.config.schema import CENTER_TICK

    lay = env.task.obs
    ticks = torch.round(obs[:, lay.q_norm].double() * 2048.0 + CENTER_TICK).to(torch.int64)
    tip = obs[:, lay.tip_xyz].double()
    err = env.goal.double() - tip
    n = ticks.shape[0]
    jac = torch.zeros(n, 3, 4, dtype=torch.float64, device=ticks.device)
    buf_p = torch.zeros(n, 3, dtype=torch.float64, device=ticks.device)
    buf_m = torch.zeros_like(buf_p)
    for j in range(4):
        tp, tm = ticks.clone(), ticks.clone()
        tp[:, j] += h
        tm[:, j] -= h
        jac[:, :, j] = (env.chain.fk_tip(tp, buf_p) - env.chain.fk_tip(tm, buf_m)) / (2.0 * h)
    jjt = jac @ jac.transpose(1, 2) + lam * torch.eye(3, dtype=torch.float64, device=ticks.device)
    dq = (jac.transpose(1, 2) @ torch.linalg.solve(jjt, err[:, :, None])).squeeze(-1)  # ticks
    act = torch.zeros(n, 6, dtype=torch.float32, device=ticks.device)
    act[:, :4] = (dq / env._max_delta64[:4]).clamp(-1.0, 1.0).float()
    return act


def _episodes(env: Any, act_fn: Callable[[torch.Tensor], torch.Tensor]) -> tuple[float, float]:
    """One episode per world from a full reset: (episode success fraction, success fraction of episode-steps)."""
    obs = env.reset(phase_randomise=False)[0].clone()
    n, horizon = env.nworld, env.task.episode_steps
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    succ = torch.zeros(n, dtype=torch.bool, device=env.device)
    steps = torch.zeros((), device=env.device)
    hits = torch.zeros((), device=env.device)
    for _ in range(horizon):
        st = env.step(act_fn(obs))
        live = ~done
        steps += live.sum()
        hits += (live & st.info["success"]).sum()
        succ |= live & st.info["success"]  # terminated also covers a terminal trip
        done |= st.terminated | st.truncated
        obs = st.obs
    return float(succ.float().mean()), float(hits / steps)


def l_reach_gpu(nworld: int = 1024, *, seed: int = 0, calibration: str | None = None) -> list[GateRow]:
    from so101_rl.rl.eval_sim import build_env

    rows: list[GateRow] = []
    env = build_env("reach", nworld, stage=1, seed=seed, calibration=calibration)  # stage 1: proprio noise on
    try:
        rate, _ = _episodes(env, lambda o: _scripted_action(env, o))
        rows.append(GateRow("L1", "scripted_reach", rate >= 0.95,
                            f"{rate * 100:.2f} % of {nworld} episodes (DR stage 1, proprio noise)", ">= 95 %"))  # fmt: skip
        env.randomizer.set_stage(0)
        rate0, _ = _episodes(env, lambda o: torch.zeros(o.shape[0], 6, device=o.device))
        rows.append(GateRow("L2", "do_nothing", rate0 == 0.0, f"{rate0 * 100:.3f} % episodes succeed", "0 %"))
        gen = torch.Generator(device=env.device)
        gen.manual_seed(seed)
        _, step_rate = _episodes(env, lambda o: torch.rand(o.shape[0], 6, generator=gen, device=o.device) * 2 - 1)
        rows.append(GateRow("L2", "random_policy", step_rate < 1e-3,
                            f"{step_rate * 100:.4f} % of episode-steps succeed", "< 0.1 %"))  # fmt: skip
    finally:
        env.close()
    return rows


def run_reach_gates(*, gpu: bool | None = None, calibration: str | None = None, seed: int = 0) -> list[GateRow]:
    from so101_rl.env.goals import BoxGoalSampler
    from so101_rl.env.tasks import REACH

    cfg, cal, fk = _setup(calibration)
    sampler = BoxGoalSampler(REACH.goal_bounds, fk=fk, limits=cfg.joints, seed=0)
    rows = l1_reach_cpu(cfg, cal, fk, sample_goals=sampler.sample_n, success_radius_m=REACH.success_radius_m,
                        episode_steps=REACH.episode_steps, seed=seed)  # fmt: skip
    rows += l2_reach_cpu(REACH, seed=seed)
    use_gpu = torch.cuda.is_available() if gpu is None else gpu
    if use_gpu:
        rows += l_reach_gpu(seed=seed, calibration=calibration)
    else:
        rows += [GateRow(lvl, name, False, "not run: CUDA unavailable", "needs the batched env")
                 for lvl, name in (("L1", "scripted_reach"), ("L2", "do_nothing"), ("L2", "random_policy"))]  # fmt: skip
    return rows


# ------------------------------------------------------------------------------------------------ die_to_zone
def independent_die_success(achieved: np.ndarray, desired: np.ndarray, visible: np.ndarray, half_edge: float,
                            max_height: float) -> np.ndarray:  # fmt: skip
    """Second, separately written die predicate (L2): per-axis comparisons, explicit NaN rejection, visible only."""
    a = np.asarray(achieved, dtype=np.float64)
    d = np.asarray(desired, dtype=np.float64)
    finite = ~(np.isnan(a).any(axis=-1) | np.isnan(d).any(axis=-1))
    in_x = np.abs(a[..., 0] - d[..., 0]) <= half_edge
    in_y = np.abs(a[..., 1] - d[..., 1]) <= half_edge
    low = (a[..., 2] - d[..., 2]) <= max_height
    return finite & in_x & in_y & low & np.asarray(visible, dtype=bool)


def independent_die_settle(pred: np.ndarray, moved: np.ndarray, need: int, settle_m: float) -> np.ndarray:
    """Second, separately written debounce / settle rule (L2) for ONE sequence: ``pred[t]`` the step predicate,
    ``moved[t]`` the tracker displacement since step ``t - 1``. A run of predicate steps continues only while each
    step moved at most ``settle_m``; success from the ``need``-th step of a run on. Returns bool[T]."""
    out = np.zeros(len(pred), dtype=bool)
    run = 0
    for t in range(len(pred)):
        if not pred[t]:
            run = 0
        elif run > 0 and moved[t] <= settle_m:
            run += 1
        else:
            run = 1
        out[t] = run >= need
    return out


def _well_half_m() -> float:
    from so101_rl.env.batched_env import _WELL_HALF_M  # the one value the env clips the die start with

    return float(_WELL_HALF_M)


def _start_stage(spec: Any, stage: int | None) -> int:
    ds = spec.die_start
    k = ds.well_stage if stage is None else int(stage)
    ds.stage_name(k)  # validates the range
    return k


def _stage_label(spec: Any, k: int) -> str:
    return f"S{k} {spec.die_start.stage_name(k)}"


def _row_name(name: str, spec: Any, k: int) -> str:
    """The well stage (the target start) keeps the D1 row names; the curriculum's table stages get ``_S<k>``."""
    return name if k == spec.die_start.well_stage else f"{name}_S{k}"


def _die_starts(spec: Any, n: int, rng: np.random.Generator, well_half_m: float, stage: int | None = None,
                ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The batched env's die placement (``BatchedGoalEnv._place_die``) in numpy for start stage ``stage`` (None = the
    well, the env's default): die centre xyz [n, 3] and the yaw [n] its clipping assumed (the die must be placed with
    that yaw, or its footprint can overlap the funnel / leave its rectangle). Same rule, not the same random
    stream."""
    from so101_rl.env.tasks import DIE_HALF_EDGE_M

    ds = spec.die_start
    k = _start_stage(spec, stage)
    yaw = rng.uniform(0.0, 2 * math.pi, size=n)
    reach = DIE_HALF_EDGE_M * (np.abs(np.cos(yaw)) + np.abs(np.sin(yaw)))  # footprint half extent
    if k == ds.well_stage:
        slack = np.clip(well_half_m - reach, 0.0, None)[:, None]
        xy = np.clip(rng.normal(size=(n, 2)) * ds.sigma_xy_m, -slack, slack) + np.asarray(ds.well_center_xy_m)
    else:
        reg = ds.table_stages[k]
        rects = np.asarray(reg.rects_xy_m, np.float64)
        r = rects[rng.choice(len(rects), size=n, p=reg.areas / reg.areas.sum())]
        lo = r[:, [0, 2]] + reach[:, None]
        hi = r[:, [1, 3]] - reach[:, None]
        xy = lo + rng.uniform(size=(n, 2)) * (hi - lo)
    return np.column_stack([xy, np.full(n, spec.zone.z_table_m + DIE_HALF_EDGE_M)]), yaw


def l1_die_cpu(spec: Any, *, n: int = 10_000, seed: int = 0, well_half_m: float | None = None,
               stage: int | None = None) -> list[GateRow]:  # fmt: skip
    """``start_not_success`` for start stage ``stage`` (None = the well): the worst case over the stage's start set,
    then sampled starts (see the module docstring)."""
    from so101_rl.env.tasks import DIE_HALF_EDGE_M, success_np

    zone, ds = spec.zone, spec.die_start
    if zone is None or ds is None:
        raise ValueError(f"task {spec.name!r} has no zone / die start")
    k = _start_stage(spec, stage)
    wh = _well_half_m() if well_half_m is None else float(well_half_m)
    goal = zone.goal_xyz()
    if k == ds.well_stage:  # the clipped centre square of the well floor
        c = np.asarray(ds.well_center_xy_m, np.float64)
        boxes = [(c - wh, c + wh)]
    else:  # every rectangle shrunk by the smallest footprint half extent (yaw 0): the union of all centres
        boxes = [(np.array([x0, y0]) + DIE_HALF_EDGE_M, np.array([x1, y1]) - DIE_HALF_EDGE_M)
                 for x0, x1, y0, y1 in ds.table_stages[k].rects_xy_m]  # fmt: skip
    nearest = np.array([np.append(np.clip(goal[:2], lo, hi), zone.z_table_m + DIE_HALF_EDGE_M) for lo, hi in boxes])
    starts = np.vstack([nearest, _die_starts(spec, n, np.random.default_rng(seed), wh, k)[0]])
    hit = success_np(starts, np.broadcast_to(goal, starts.shape), spec, visible=np.ones(len(starts), bool))
    margin = float((np.abs(starts[:, :2] - goal[:2]).max(axis=1) - zone.half_edge_m).min())
    return [GateRow("L1", _row_name("start_not_success", spec, k), not bool(hit.any()),
                    f"{_stage_label(spec, k)}: {int(hit.sum())} of {len(starts)} starts (worst case + {n} sampled) in "
                    f"the zone; nearest start {margin * 1000:.1f} mm outside the zone edge", "0")]  # fmt: skip


def zone_in_view_cpu(spec: Any, *, camera: Any = None, grid: int = 7) -> list[GateRow]:
    """``zone_in_view`` (L1, CPU): every point of a ``grid`` x ``grid`` lattice over the zone square (corners
    included) at the resting die height is in the calibrated camera's view by the simulated tracker's rule
    (``SimDieTracker.in_view``: colour and depth image, half-blob border margin; nominal pose, no DR), else no success
    can ever be seen there (plan §10 2026-10-16 "Die zone moved into the camera view"). ``camera``: a
    ``CameraCalib`` or a path (default ``calib/camera.yaml``)."""
    from so101_rl.perception.camera_calib import CameraCalib, load_camera_calib
    from so101_rl.sim.sim_tracker import NO_NOISE, CameraModel, SimDieTracker

    zone = spec.zone
    if zone is None:
        raise ValueError(f"task {spec.name!r} has no zone")
    calib = camera if isinstance(camera, CameraCalib) else load_camera_calib(camera)
    goal = zone.goal_xyz()
    s = np.linspace(-zone.half_edge_m, zone.half_edge_m, int(grid))
    dx, dy = (a.ravel() for a in np.meshgrid(s, s, indexing="ij"))
    pts = np.column_stack([goal[0] + dx, goal[1] + dy, np.full(dx.size, goal[2])])
    color = CameraModel.from_calib(calib, "color")
    tr = SimDieTracker(color, NO_NOISE, len(pts), device="cpu", seed=0, depth_cam=CameraModel.from_calib(calib, "depth"))
    ok = tr.in_view(torch.as_tensor(pts, dtype=torch.float32)).numpy()
    uv, _ = color.project(pts)
    out = pts[~ok][:, :2]
    worst = f"; out of view e.g. ({out[0][0]:.3f}, {out[0][1]:.3f})" if len(out) else ""
    return [GateRow("L1", "zone_in_view", bool(ok.all()),
                    f"{int(ok.sum())} of {len(pts)} zone points (centre ({goal[0]:.3f}, {goal[1]:.3f}), "
                    f"{int(grid)}x{int(grid)} incl. corners) in view; colour px u {uv[:, 0].min():.0f}..{uv[:, 0].max():.0f} "
                    f"v {uv[:, 1].min():.0f}..{uv[:, 1].max():.0f} of {color.width}x{color.height}{worst}",
                    "all")]  # fmt: skip


def l2_die_cpu(spec: Any, *, predicate: Predicate | None = None, debounce: Callable[..., Any] | None = None,
               n: int = 100_000, seed: int = 0) -> list[GateRow]:  # fmt: skip
    """``adversarial``, ``independent_predicate`` and ``independent_settle``. ``predicate(a, d, visible)`` and
    ``debounce(pred, count, achieved, prev_achieved) -> (success, count)`` replace the task's (shown-to-fail twins)."""
    from so101_rl.env.tasks import debounce_np, success_np

    zone = spec.zone
    if zone is None or spec.settle_m is None:
        raise ValueError(f"task {spec.name!r} is not a die task")
    pred = predicate or (lambda a, d, v: success_np(a, d, spec, visible=v))
    h, mh, eps = zone.half_edge_m, zone.max_height_m, 1e-4
    goal = zone.goal_xyz()
    ex, ey, ez = np.eye(3)
    table = [  # (case, achieved, visible, genuine success)
        ("die on the zone centre", goal, True, True),
        ("0.1 mm inside the +x edge", goal + ex * (h - eps), True, True),
        ("0.1 mm outside the +x edge", goal + ex * (h + eps), True, False),
        ("0.1 mm inside the -y edge", goal - ey * (h - eps), True, True),
        ("0.1 mm outside the -y edge", goal - ey * (h + eps), True, False),
        ("inside corner (L-inf, not Euclidean)", goal + (ex + ey) * (h - eps), True, True),
        ("outside corner", goal + (ex + ey) * (h + eps), True, False),
        ("0.1 mm under the height bound", goal + ez * (mh - eps), True, True),
        ("0.1 mm over the height bound", goal + ez * (mh + eps), True, False),
        ("held 20 mm up over the centre", goal + ez * 0.020, True, False),
        ("on the centre but invisible", goal, False, False),
        ("die NaN", np.full(3, np.nan), True, False),
    ]
    bad = []
    with np.errstate(invalid="ignore"):
        for case, ach, vis, want in table:
            got = bool(np.asarray(pred(ach[None], goal[None], np.array([vis])))[0])
            if got != want:
                bad.append(f"{case}: got {got}")
        if bool(np.asarray(pred(goal[None], np.full((1, 3), np.nan), np.array([True])))[0]):
            bad.append("goal NaN: got True")
    rows = [GateRow("L2", "adversarial", not bad, "; ".join(bad) or f"{len(table) + 1} cases as expected", "all cases")]

    rng = np.random.default_rng(seed)
    d = np.column_stack([rng.uniform(0.05, 0.35, n), rng.uniform(-0.2, 0.2, n), rng.uniform(-0.01, 0.1, n)])
    a = d.copy()
    a[:, :2] += rng.uniform(-2 * h, 2 * h, size=(n, 2))
    a[:, 2] += rng.uniform(-0.01, 2 * mh, n)
    q = n // 4  # a quarter each within 1 mm of the x edge, the y edge and the height bound
    a[:q, 0] = d[:q, 0] + rng.choice([-1.0, 1.0], q) * rng.uniform(h - 1e-3, h + 1e-3, q)
    a[q : 2 * q, 1] = d[q : 2 * q, 1] + rng.choice([-1.0, 1.0], q) * rng.uniform(h - 1e-3, h + 1e-3, q)
    a[2 * q : 3 * q, 2] = d[2 * q : 3 * q, 2] + rng.uniform(mh - 1e-3, mh + 1e-3, q)
    vis = rng.random(n) >= 0.1
    agree = float((np.asarray(pred(a, d, vis)) == independent_die_success(a, d, vis, h, mh)).mean())
    rows.append(GateRow("L2", "independent_predicate", agree == 1.0, f"agreement {agree * 100:.4f} % on {n} states",
                        "100 %"))  # fmt: skip

    deb = debounce or (lambda p, c, ach, prev: debounce_np(p, c, spec, achieved=ach, prev_achieved=prev))
    n_seq, horizon, s = 2000, 30, spec.settle_m
    p_seq = rng.random((n_seq, horizon)) < 0.7
    step = rng.uniform(0.5 * s, 1.5 * s, (n_seq, horizon)) * (rng.random((n_seq, horizon)) < 0.5)  # 0 or ~settle_m
    dirs = rng.normal(size=(n_seq, horizon, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    pos = np.cumsum(dirs * step[..., None], axis=1)
    count = np.zeros(n_seq, dtype=np.int64)
    got = np.zeros((n_seq, horizon), dtype=bool)
    prev = pos[:, 0]
    for t in range(horizon):
        got[:, t], count = deb(p_seq[:, t], count, pos[:, t], prev)
        prev = pos[:, t]
    moved = np.linalg.norm(np.diff(pos, axis=1, prepend=pos[:, :1]), axis=-1)
    want = np.stack([independent_die_settle(p_seq[i], moved[i], spec.success_debounce_steps, s) for i in range(n_seq)])
    agree = float((got == want).mean())
    rows.append(GateRow("L2", "independent_settle", agree == 1.0,
                        f"agreement {agree * 100:.4f} % on {n_seq} x {horizon} steps", "100 %"))  # fmt: skip
    return rows


def l2_die_do_nothing_cpu(cfg: Any, cal: Any, spec: Any, *, nworld: int = 16, seed: int = 0,
                          start_noise_ticks: int = 30, well_half_m: float | None = None,
                          stage: int | None = None) -> list[GateRow]:  # fmt: skip
    """``do_nothing`` on CPU MuJoCo (see the module docstring) for start stage ``stage`` (None = the well): every
    world holds its start pose for one episode."""
    import mujoco

    from so101_rl.config import JOINT_NAMES, ticks_to_rad
    from so101_rl.env.tasks import debounce_np, success_np
    from so101_rl.sim.model import load_model

    m = load_model()[0]
    data = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    wh = _well_half_m() if well_half_m is None else float(well_half_m)
    q_adr = [int(m.jnt_qposadr[m.joint(j).id]) for j in JOINT_NAMES]
    act = [m.actuator(j).id for j in JOINT_NAMES]
    die_q = int(m.jnt_qposadr[m.joint("die_free").id])
    die_b = m.body("die").id
    steps_per_tick = max(1, round((1.0 / cfg.rates.servo_hz) / m.opt.timestep))
    home = _home_ticks(cal)
    lo, hi = np.asarray(cfg.joints.min_ticks, np.int64), np.asarray(cfg.joints.max_ticks, np.int64)
    goal = spec.zone.goal_xyz()
    k = _start_stage(spec, stage)
    starts, yaws = _die_starts(spec, nworld, rng, wh, k)
    wins, max_move = 0, 0.0
    for w in range(nworld):
        mujoco.mj_resetDataKeyframe(m, data, m.key("home").id)
        off = rng.integers(-start_noise_ticks, start_noise_ticks + 1, size=6)
        off[5] = 0
        q_arm = ticks_to_rad(np.clip(home + off, lo, hi), cal)
        for i in range(5):  # the arm joints; the gripper stays at the keyframe
            data.qpos[q_adr[i]] = q_arm[i]
            data.ctrl[act[i]] = q_arm[i]
        yaw = float(yaws[w])  # the yaw the start was clipped for (a fresh yaw pushed the die into the funnel)
        data.qpos[die_q : die_q + 3] = starts[w]
        data.qpos[die_q + 3 : die_q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        data.qvel[:] = 0.0
        mujoco.mj_forward(m, data)
        start = data.xpos[die_b].copy()
        prev, count, won = start, np.zeros(1, np.int64), False
        for _ in range(spec.episode_steps):
            mujoco.mj_step(m, data, nstep=3 * steps_per_tick)  # one 10 Hz policy step = three servo ticks
            die = data.xpos[die_b].copy()
            p = success_np(die[None], goal[None], spec, visible=np.ones(1, bool))
            done, count = debounce_np(p, count, spec, achieved=die[None], prev_achieved=prev[None])
            won |= bool(done[0])
            prev = die
        wins += int(won)
        max_move = max(max_move, float(np.linalg.norm(data.xpos[die_b] - start)))
    return [GateRow("L2", _row_name("do_nothing", spec, k), wins == 0,
                    f"{_stage_label(spec, k)}: {wins} of {nworld} CPU episodes succeed (perfect always-visible "
                    f"tracker); die moved at most {max_move * 1000:.2f} mm", "0")]  # fmt: skip


def l_die_gpu(nworld: int = 1024, *, seed: int = 0, calibration: str | None = None) -> list[GateRow]:
    from so101_rl.rl.eval_sim import build_env

    rows: list[GateRow] = []
    env = build_env("die_to_zone", nworld, stage=0, seed=seed, calibration=calibration)
    try:
        rate0, _ = _episodes(env, lambda o: torch.zeros(o.shape[0], 6, device=o.device))
        rows.append(GateRow("L2", "do_nothing_batched", rate0 == 0.0, f"{rate0 * 100:.3f} % episodes succeed", "0 %"))
        gen = torch.Generator(device=env.device)
        gen.manual_seed(seed)
        _, step_rate = _episodes(env, lambda o: torch.rand(o.shape[0], 6, generator=gen, device=o.device) * 2 - 1)
        rows.append(GateRow("L2", "random_policy", step_rate < 1e-3,
                            f"{step_rate * 100:.4f} % of episode-steps succeed", "< 0.1 %"))  # fmt: skip
    finally:
        env.close()
    return rows


def run_die_gates(*, gpu: bool | None = None, calibration: str | None = None, seed: int = 0,
                  nworld_cpu: int = 16) -> list[GateRow]:  # fmt: skip
    from so101_rl.env.tasks import DIE_TO_ZONE

    cfg, cal, _fk = _setup(calibration)
    stages = range(DIE_TO_ZONE.die_start.n_stages)  # the start curriculum S0..S3 (S3 = the well, the target)
    rows = zone_in_view_cpu(DIE_TO_ZONE)
    rows += [r for k in stages for r in l1_die_cpu(DIE_TO_ZONE, seed=seed, stage=k)]
    rows += l2_die_cpu(DIE_TO_ZONE, seed=seed)
    rows += [r for k in stages for r in l2_die_do_nothing_cpu(cfg, cal, DIE_TO_ZONE, nworld=nworld_cpu, seed=seed, stage=k)]
    use_gpu = torch.cuda.is_available() if gpu is None else gpu
    if use_gpu:
        rows += l_die_gpu(seed=seed, calibration=calibration)
    else:
        rows += [GateRow("L2", name, False, "not run: CUDA unavailable", "needs the batched env")
                 for name in ("do_nothing_batched", "random_policy")]  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sim-gates", description="L1/L2 gates (reach, die_to_zone) before any reward tuning")
    ap.add_argument("--task", default="reach")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nworld", type=int, default=16, help="die_to_zone: number of CPU do-nothing episodes")
    ap.add_argument("--out", type=Path, default=None, help="default runs/gates/<task>-<timestamp>.json")
    a = ap.parse_args(argv)
    if a.task not in GATE_TASKS:
        print(f"sim-gates: unknown task {a.task!r} (known: {', '.join(GATE_TASKS)})", file=sys.stderr)
        return 2
    if a.task == "reach":
        rows = run_reach_gates(calibration=a.calibration, seed=a.seed)
    else:
        rows = run_die_gates(calibration=a.calibration, seed=a.seed, nworld_cpu=a.nworld)
    w = max(len(r.name) for r in rows)
    for r in rows:
        print(f"{r.level}  {r.name:<{w}}  {'PASS' if r.passed else 'FAIL'}  {r.value}  [{r.threshold}]")
    ok = all(r.passed for r in rows)
    out = a.out or Path("runs/gates") / f"{a.task}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": a.task, "passed": ok, "rows": [asdict(r) for r in rows]}, indent=2))
    print(f"{'ALL PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

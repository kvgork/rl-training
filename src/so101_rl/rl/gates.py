"""Feasibility (L1) and predicate (L2) gates for reach, before any reward tuning (sim plan S4.2 reach part, MVP M3).

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

__all__ = ["GateRow", "independent_reach_success", "main", "run_reach_gates"]

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sim-gates", description="L1/L2 gates (reach) before any reward tuning")
    ap.add_argument("--task", default="reach")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default runs/gates/<task>-<timestamp>.json")
    a = ap.parse_args(argv)
    if a.task != "reach":
        print(f"sim-gates: task {a.task!r} is deferred in the MVP (plans/mvp-plan.md); only 'reach'", file=sys.stderr)
        return 2
    rows = run_reach_gates(calibration=a.calibration, seed=a.seed)
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

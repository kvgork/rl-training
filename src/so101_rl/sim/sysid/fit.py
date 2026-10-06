"""CMA-ES fit of the simulator parameters to an excitation run (FS7, sim-first track S2.3; pixi task ``sysid-fit``).

``python -m so101_rl.sim.sysid.fit --run runs/sysid/<ts> --backend {warp,cpu} --out calib/sim_params.yaml
[--heldout-kinds chirp,random] [--popsize N] [--generations G] [--fit kp,kv,...] [--fixed damping,...]``

Decision D-SYSID: fit ``kp, kv, frictionloss, armature, delay_ticks``; hold ``damping`` and ``forcerange`` at the
prior unless the held-out gate fails. Parameters in ``fixed`` win over ``fit``.

Search:

* ``delay_ticks`` is an outer integer grid (0..6). Every delay runs its own CMA-ES concurrently (one batch of worlds
  per delay); after a quarter of the generations only the two best delays continue.
* The continuous parameters are searched in **log space relative to the prior** (``x = ln(value / prior)``, mean 0 =
  the prior, ``sigma0`` 0.3, box bounds ``±bound``). The per-joint parameters form one CMA-ES block per joint
  (4-D for the default fit, not one 24-D problem), each driven by that joint's error; scalar parameters (die mass,
  frictions) form one global block driven by the total objective. All blocks share the same simulated worlds: world
  ``i`` runs candidate ``i`` of every block. Joint coupling (gravity, inertia) is second order on the solo-dominated
  error of each joint, and the separable blocks converge with the small populations a CPU can afford.
* Objective (per candidate) = mean over joints of ``e_j`` + 0.2 x max over joints, with ``e_j`` the mean over
  training segments of the unrounded-tick ``rms_pct`` (the quantised form has plateaus).
* ``spread`` = the larger of the final CMA-ES standard deviation per parameter in log space (~ relative spread) and
  the local-curvature :func:`profile_spread` (which exposes flat valleys CMA-ES does not), keyed
  ``"<field>.<joint>"``, plus ``"<field>"`` = the max over joints. A spread above ``UNIDENTIFIED_SPREAD`` (0.5) marks
  the parameter **unidentified** (S3 gives it a wide DR range instead of a point estimate).

The held-out RMS (the compare-real definition: quantised, replayed OPEN LOOP from each segment's first row, pooled
over the held-out segments: excitation kinds in ``--heldout-kinds`` plus a
seeded 20 % time window of every other segment) is the gate: ``sysid-fit`` writes ``--out`` only when every joint is
at most ``MAX_RMS_PCT`` (5 %); otherwise ``<run>/sim_params.unverified.yaml`` and exit 1. A
``<run>/fit_report.json`` (and ``history.csv``) is written either way.

Backends: ``warp`` (MuJoCo Warp, one candidate per world, ``WarpSim.set_world_params``) and ``cpu`` (a spawn pool of
``workers`` processes, default 16, each with its own ``MjModel``/``MjData``); the objective is identical (test-pinned).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from so101_rl.config import (
    JOINT_NAMES,
    N_JOINTS,
    MotorCalibration,
    So101Config,
    load_calibration,
    load_config,
)
from so101_rl.sim.sysid.dataset import ExcitationDataset, TickSegment
from so101_rl.sim.sysid.params import (
    MAX_DELAY_TICKS,
    PER_JOINT_FIELDS,
    SCALAR_FIELDS,
    SimParams,
    SimParamsMeta,
    mjcf_sha256,
)
from so101_rl.sim.sysid.replay import (
    SHOOTING_WINDOW_TICKS,
    TickMap,
    as_joint_dict,
    pooled_rms_pct,
    replay_segment_cpu,
)

DEFAULT_FIT: Final[tuple[str, ...]] = ("kp", "kv", "frictionloss", "armature", "delay_ticks")
DEFAULT_FIXED: Final[tuple[str, ...]] = ("damping", "forcerange")
UNIDENTIFIED_SPREAD: Final[float] = 0.5
QUANT_DEADZONE_TICKS: Final[float] = 0.5  # the log is rounded to whole ticks
PROFILE_STEP: Final[float] = 0.15  # finite-difference step in log space for profile_spread
PROFILE_CORR_TICKS: Final[int] = 6  # residual correlation length (~5 Hz excitation band at 30 Hz): N_eff = N / 6
PROFILE_FLOOR_PCT: Final[float] = 0.002  # residual floor (rounding) so a noise-free fit is not called "exact"
PROFILE_SIGMAS: Final[float] = 2.0  # spread = 2 standard errors
WORST_JOINT_WEIGHT: Final[float] = 0.2
DEFAULT_HELDOUT_KINDS: Final[tuple[str, ...]] = ("chirp", "random")


@dataclass
class FitResult:
    params: SimParams
    train_rms_pct: dict[str, float]
    heldout_rms_pct: dict[str, float]
    spread: dict[str, float]
    evals: int
    wall_s: float
    backend: str
    history: list[tuple[int, float]]
    unidentified: list[str] = field(default_factory=list)
    delay_scores: dict[int, float] = field(default_factory=dict)  # best training objective per delay tried
    objective: float = math.nan  # training objective of the returned params
    cma_spread: dict[str, float] = field(default_factory=dict)  # final CMA-ES std per parameter (log space)
    profile_spread: dict[str, float] = field(default_factory=dict)  # local-curvature spread (see profile_spread)

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = self.params.to_dict()
        d["delay_scores"] = {str(k): v for k, v in self.delay_scores.items()}
        return d


def objective_rms_pct(sim: np.ndarray, real: np.ndarray, span: np.ndarray, deadzone: float) -> np.ndarray:
    """Per-joint RMS (rows 1..T-1, % of range) of the error with a quantisation dead zone: a simulated position that
    rounds to the logged tick costs nothing (``|err| <= deadzone`` -> 0), larger errors count from the zone edge. Gives a
    continuous objective whose minimum is not biased by the log's rounding (plain unrounded RMS is floored at the
    rounding noise and pulled off the true parameters)."""
    err = np.asarray(sim, dtype=np.float64)[1:] - np.asarray(real, dtype=np.float64)[1:]
    err = np.sign(err) * np.maximum(np.abs(err) - deadzone, 0.0)
    return np.sqrt(np.mean(err**2, axis=0)) / span * 100.0


def objective(e: np.ndarray) -> np.ndarray:
    """``[..., 6]`` per-joint errors -> ``mean_j + WORST_JOINT_WEIGHT * max_j``."""
    return e.mean(axis=-1) + WORST_JOINT_WEIGHT * e.max(axis=-1)


# --------------------------------------------------------------------------------------------- evaluators
class _CpuEvaluator:
    """Spawn pool; every worker holds its own model/data and both segment sets."""

    def __init__(self, sets: dict[str, list[TickSegment]], calibration, cfg, *, workers: int | None, window: int) -> None:
        self.n = max(1, int(workers or min(16, os.cpu_count() or 1)))
        ctx = mp.get_context("spawn")
        self.pool = cf.ProcessPoolExecutor(self.n, mp_context=ctx, initializer=_cpu_init,
                                           initargs=(sets, calibration, cfg, window))
        self.sets = sets
        self.calibration, self.cfg, self.window = calibration, cfg, window
        self.evals = 0

    def evaluate(self, cands: Sequence[SimParams], which: str, *, deadzone: float = QUANT_DEADZONE_TICKS) -> np.ndarray:
        """``[n, n_segments, 6]`` objective rms_pct (unrounded sim, dead-zoned error) per candidate, segment, joint."""
        chunk = max(1, math.ceil(len(cands) / self.n))
        res = list(self.pool.map(_cpu_eval, [(p, which, deadzone) for p in cands], chunksize=chunk))
        self.evals += len(cands)
        return np.stack(res)

    def simulate(self, p: SimParams, which: str, *, quantize: bool = True, open_loop: bool = False) -> list[np.ndarray]:
        return self.pool.submit(_cpu_sim, (p, which, quantize, open_loop)).result()

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)


_W: dict[str, Any] = {}


def _cpu_init(sets, calibration, cfg, window) -> None:
    from so101_rl.sim.model import load_model

    model, data = load_model()
    span = np.array([calibration[n].range_max - calibration[n].range_min for n in JOINT_NAMES], dtype=np.float64)
    _W.update(model=model, data=data, sets=sets, cal=calibration, cfg=cfg, window=window, span=span,
              tmap=TickMap(calibration, model))


def _cpu_sim(args) -> list[np.ndarray]:
    p, which, quantize, open_loop = args
    w = _W
    p.apply_cpu(w["model"])
    return [
        replay_segment_cpu(s, w["model"], w["data"], w["cal"], w["cfg"], p,
                           shooting_window_ticks=len(s) if open_loop else w["window"],
                           quantize=quantize, tmap=w["tmap"], apply_params=False)
        for s in w["sets"][which]
    ]


def _cpu_eval(args) -> np.ndarray:
    p, which, deadzone = args
    sims = _cpu_sim((p, which, False, False))
    return np.stack([objective_rms_pct(sim, s.present_ticks, _W["span"], deadzone)
                     for sim, s in zip(sims, _W["sets"][which], strict=True)])


class _WarpEvaluator:
    """One ``WarpSim`` with ``capacity`` worlds; smaller candidate sets are padded with their last entry."""

    def __init__(self, sets: dict[str, list[TickSegment]], calibration, cfg, *, capacity: int, window: int,
                 device: str) -> None:
        from so101_rl.sim.model import load_model
        from so101_rl.sim.warp_model import WarpSim

        model, _ = load_model()
        self.ws = WarpSim(model, cfg.sim_warp, capacity, device=device)
        self.capacity = capacity
        self.sets, self.calibration, self.cfg, self.window = sets, calibration, cfg, window
        self.tmap = TickMap(calibration, model)
        self.span = np.array([calibration[n].range_max - calibration[n].range_min for n in JOINT_NAMES], dtype=np.float64)
        self.evals = 0

    def _run(self, cands: Sequence[SimParams], which: str, quantize: bool, open_loop: bool = False):
        import torch

        from so101_rl.sim.sysid.params import BatchedSimParams
        from so101_rl.sim.sysid.replay import replay_segments_warp

        out = []
        for i in range(0, len(cands), self.capacity):
            part = list(cands[i : i + self.capacity])
            n = len(part)
            part += [part[-1]] * (self.capacity - n)
            sims = replay_segments_warp(self.sets[which], self.ws, BatchedSimParams.stack(part, self.ws.device),
                                        self.calibration, self.cfg, shooting_window_ticks=(max(len(s) for s in self.sets[which]) if open_loop else self.window), quantize=quantize,
                                        tmap=self.tmap)
            out.append(sims[:n])
        self.evals += len(cands)
        return torch.cat(out)

    def evaluate(self, cands: Sequence[SimParams], which: str, *, deadzone: float = QUANT_DEADZONE_TICKS) -> np.ndarray:
        import torch

        sims = self._run(cands, which, False).double()
        span = torch.as_tensor(self.span, device=sims.device)
        res, t0 = [], 0
        for s in self.sets[which]:
            T = len(s)
            real = torch.as_tensor(s.present_ticks, dtype=torch.float64, device=sims.device)
            err = sims[:, t0 + 1 : t0 + T] - real[None, 1:]
            err = torch.sign(err) * (err.abs() - deadzone).clamp(min=0.0)
            res.append(torch.sqrt((err**2).mean(dim=1)) / span * 100.0)
            t0 += T
        return torch.stack(res, dim=1).cpu().numpy()

    def simulate(self, p: SimParams, which: str, *, quantize: bool = True, open_loop: bool = False) -> list[np.ndarray]:
        sims = self._run([p], which, quantize, open_loop)[0].cpu().numpy()
        out, t0 = [], 0
        for s in self.sets[which]:
            out.append(sims[t0 : t0 + len(s)])
            t0 += len(s)
        return out

    def close(self) -> None:
        self.ws.close()


# --------------------------------------------------------------------------------------------- the fit
def _import_cma():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # cma warns about a missing matplotlib on import
        import cma
    return cma


def _resolve_fields(fit: Sequence[str], fixed: Sequence[str]) -> tuple[list[str], list[str], bool]:
    known = set(PER_JOINT_FIELDS) | set(SCALAR_FIELDS) | {"delay_ticks", "deadband_ticks"}
    bad = [f for f in (*fit, *fixed) if f not in known]
    if bad:
        raise ValueError(f"unknown parameter(s) {bad}; known: {sorted(known)}")
    eff = [f for f in dict.fromkeys(fit) if f not in set(fixed)]
    if "deadband_ticks" in eff:
        raise ValueError("deadband_ticks is not fitted (D-BACKLASH: it is a DR range, not a point estimate)")
    return ([f for f in eff if f in PER_JOINT_FIELDS], [f for f in eff if f in SCALAR_FIELDS], "delay_ticks" in eff)


def fit_sim_params(
    ds_train: ExcitationDataset,
    ds_heldout: ExcitationDataset,
    *,
    prior: SimParams,
    fit: Sequence[str] = DEFAULT_FIT,
    fixed: Sequence[str] = DEFAULT_FIXED,
    backend: Literal["warp", "cpu"] = "warp",
    popsize: int = 256,
    generations: int = 60,
    seed: int = 0,
    log_dir: Path | None = None,
    calibration: dict[str, MotorCalibration] | None = None,
    cfg: So101Config | None = None,
    workers: int | None = None,
    sigma0: float = 0.3,
    bound: float = 1.6,
    delay_grid: Sequence[int] | None = None,
    shooting_window_ticks: int = SHOOTING_WINDOW_TICKS,
    device: str = "cuda:0",
    profile_step: float = PROFILE_STEP,
) -> FitResult:
    """Fit ``fit`` (minus ``fixed``) to ``ds_train`` starting from ``prior``; report on ``ds_heldout``."""
    t_start = time.perf_counter()
    if calibration is None or cfg is None:
        cfg = cfg if cfg is not None else load_config()
        calibration = calibration if calibration is not None else load_calibration(Path(cfg.calibration_path).expanduser())
    pj, sc, fit_delay = _resolve_fields(fit, fixed)
    if not ds_train.segments or not ds_heldout.segments:
        raise ValueError("both the training and the held-out set need at least one segment")
    if popsize < 2 and (pj or sc):
        raise ValueError("popsize must be >= 2")
    for f in pj:
        if (getattr(prior, f) <= 0).any():
            raise ValueError(f"prior {f} must be > 0 everywhere to fit it in log space")
    for f in sc:
        if getattr(prior, f) <= 0:
            raise ValueError(f"prior {f} must be > 0 to fit it in log space")
    delays = sorted(set(delay_grid if delay_grid is not None else range(MAX_DELAY_TICKS + 1))) if fit_delay else [prior.delay_ticks]
    if any(not 0 <= d <= MAX_DELAY_TICKS for d in delays):
        raise ValueError(f"delay grid {delays} outside 0..{MAX_DELAY_TICKS}")
    sets = {"train": list(ds_train.segments), "heldout": list(ds_heldout.segments)}
    has_cma = bool(pj or sc)
    pop = popsize if has_cma else 1
    if backend == "warp":
        ev: Any = _WarpEvaluator(sets, calibration, cfg, capacity=len(delays) * pop, window=shooting_window_ticks,
                                 device=device)
    elif backend == "cpu":
        ev = _CpuEvaluator(sets, calibration, cfg, workers=workers, window=shooting_window_ticks)
    else:
        raise ValueError(f"backend must be 'warp' or 'cpu', got {backend!r}")

    def make(d: int, xj: np.ndarray | None, xg: np.ndarray | None) -> SimParams:
        upd: dict[str, Any] = {"delay_ticks": int(d)}
        for i, f in enumerate(pj):
            upd[f] = getattr(prior, f) * np.exp(xj[:, i])  # xj: [6, len(pj)]
        for i, f in enumerate(sc):
            upd[f] = float(getattr(prior, f) * math.exp(xg[i]))
        return replace(prior, **upd)

    history: list[tuple[int, float]] = []
    best: dict[int, tuple[float, SimParams]] = {}
    es: dict[int, dict[Any, Any]] = {}
    try:
        if has_cma:
            cma = _import_cma()
            opts = {"popsize": pop, "bounds": [-bound, bound], "verbose": -9, "tolfun": 1e-9, "tolx": 1e-6,
                    "tolfunhist": 1e-12, "tolstagnation": 10**9}
            for d in delays:
                es[d] = {}
                if pj:
                    for j in range(N_JOINTS):
                        es[d][j] = cma.CMAEvolutionStrategy(np.zeros(len(pj)), sigma0,
                                                            {**opts, "seed": seed * 1000 + d * 10 + j + 1})
                if sc:
                    es[d]["g"] = cma.CMAEvolutionStrategy(np.zeros(len(sc)), sigma0, {**opts, "seed": seed * 1000 + d * 10 + 9})
            active = list(delays)
            elim_at = max(3, generations // 4)
            for g in range(generations):
                cands: list[SimParams] = []
                asked: dict[int, dict[Any, list[np.ndarray]]] = {}
                for d in active:
                    asked[d] = {b: e.ask() for b, e in es[d].items()}
                    for i in range(pop):
                        xj = np.stack([asked[d][j][i] for j in range(N_JOINTS)]) if pj else None
                        xg = asked[d]["g"][i] if sc else None
                        cands.append(make(d, xj, xg))
                E = ev.evaluate(cands, "train").mean(axis=1)  # [n, 6]
                E = np.nan_to_num(E, nan=1e9, posinf=1e9)  # a diverged candidate ranks last, never "best"
                J = objective(E)
                for k, d in enumerate(active):
                    sl = slice(k * pop, (k + 1) * pop)
                    for b, e in es[d].items():
                        vals = J[sl] if b == "g" else E[sl, b]
                        e.tell(asked[d][b], [float(v) for v in vals])
                    i = int(np.argmin(J[sl]))
                    if d not in best or J[sl][i] < best[d][0]:
                        best[d] = (float(J[sl][i]), cands[k * pop + i])
                history.append((ev.evals, min(v[0] for v in best.values())))
                if fit_delay and g + 1 == elim_at and len(active) > 2:
                    active = sorted(active, key=lambda d: best[d][0])[:2]
            finals = {d: make(d, np.stack([es[d][j].result.xfavorite for j in range(N_JOINTS)]) if pj else None,
                              es[d]["g"].result.xfavorite if sc else None) for d in active}
        else:
            active = list(delays)
            finals = {d: replace(prior, delay_ticks=int(d)) for d in active}
        order = list(finals)
        Jf = objective(np.nan_to_num(ev.evaluate([finals[d] for d in order], "train").mean(axis=1), nan=1e9))
        delay_scores = {d: best[d][0] for d in best}
        choice: tuple[float, SimParams, int] | None = None
        for d, jv in zip(order, Jf, strict=True):
            delay_scores[d] = min(delay_scores.get(d, math.inf), float(jv))
            for cand_j, cand in ((float(jv), finals[d]), best.get(d, (math.inf, None))):
                if cand is not None and (choice is None or cand_j < choice[0]):
                    choice = (cand_j, cand, d)
        assert choice is not None
        j_final, params, d_final = choice
        history.append((ev.evals, j_final))
        # reported / gated like compare-real: quantised, OPEN LOOP from each segment's first row (no re-init)
        train_q = pooled_rms_pct(ev.simulate(params, "train", open_loop=True), sets["train"], calibration)
        held_q = pooled_rms_pct(ev.simulate(params, "heldout", open_loop=True), sets["heldout"], calibration)
        spread: dict[str, float] = {}
        cma_spread: dict[str, float] = {}
        profile = profile_spread(ev, params, pj, h=profile_step) if pj and profile_step > 0 else {}
        if d_final in es:
            for i, f in enumerate(pj):
                for j, n in enumerate(JOINT_NAMES):
                    k = f"{f}.{n}"
                    cma_spread[k] = float(es[d_final][j].result.stds[i])
                    spread[k] = max(cma_spread[k], profile.get(k, 0.0))
                spread[f] = max(spread[f"{f}.{n}"] for n in JOINT_NAMES)
            for i, f in enumerate(sc):
                spread[f] = cma_spread[f] = float(es[d_final]["g"].result.stds[i])
        unidentified = sorted(k for k, v in spread.items() if v > UNIDENTIFIED_SPREAD and ("." in k or k in sc))
        evals = ev.evals
    finally:
        ev.close()
    res = FitResult(
        params=params,
        train_rms_pct=as_joint_dict(train_q),
        heldout_rms_pct=as_joint_dict(held_q),
        spread=spread,
        evals=evals,
        wall_s=time.perf_counter() - t_start,
        backend=backend,
        history=history,
        unidentified=unidentified,
        delay_scores=dict(sorted(delay_scores.items())),
        objective=j_final,
        cma_spread=cma_spread,
        profile_spread=profile,
    )
    if log_dir is not None:
        write_fit_log(Path(log_dir), res)
    return res


def profile_spread(ev: Any, params: SimParams, pj: Sequence[str], *, h: float = PROFILE_STEP) -> dict[str, float]:
    """Identifiability from the local curvature of each joint's mean squared error in log space (Gauss-Newton).

    Per joint ``j`` the Hessian ``H_j`` of ``g_j = e_j**2`` over that joint's fitted parameters is estimated by central
    differences (``1 + 2d + 2d(d-1)`` candidates; every joint is perturbed in the same pattern at once, its error
    depends on its own parameters to first order). For a least-squares fit the parameter covariance is
    ``2 * s2 / N_eff * H_j^-1`` with the residual variance ``s2 = g_j + PROFILE_FLOOR_PCT**2`` and ``N_eff`` the training
    rows over ``PROFILE_CORR_TICKS``; the spread of parameter ``i`` is ``PROFILE_SIGMAS`` marginal standard errors.
    Unlike the CMA-ES step size this exposes valleys: the MJCF position servo, for instance, behaves almost
    identically when ``kp``, ``kv`` and ``armature`` are scaled together.
    """
    d = len(pj)
    base = np.zeros((N_JOINTS, d))
    pats: list[np.ndarray] = [base]
    for i in range(d):
        for sgn in (1, -1):
            x = base.copy()
            x[:, i] = sgn * h
            pats.append(x)
    pairs = [(i, k) for i in range(d) for k in range(i + 1, d)]
    for i, k in pairs:
        for si, sk in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            x = base.copy()
            x[:, i] = si * h
            x[:, k] = sk * h
            pats.append(x)
    cands = [replace(params, **{f: getattr(params, f) * np.exp(x[:, i]) for i, f in enumerate(pj)}) for x in pats]
    g = ev.evaluate(cands, "train").mean(axis=1) ** 2  # [n, 6]
    n_eff = max(1.0, sum(len(sg) - 1 for sg in ev.sets["train"]) / PROFILE_CORR_TICKS)
    out: dict[str, float] = {}
    for j, name in enumerate(JOINT_NAMES):
        g0 = g[0, j]
        H = np.zeros((d, d))
        for i in range(d):
            H[i, i] = (g[1 + 2 * i, j] - 2 * g0 + g[2 + 2 * i, j]) / h**2
        o = 1 + 2 * d
        for n, (i, k) in enumerate(pairs):
            pp, pm, mp, mm = g[o + 4 * n : o + 4 * n + 4, j]
            H[i, k] = H[k, i] = (pp - pm - mp + mm) / (4 * h**2)
        w, v = np.linalg.eigh(H)
        w = np.maximum(w, 1e-12 * max(1.0, float(np.abs(w).max())))
        cov = (v / w) @ v.T
        var = 2.0 * (max(g0, 0.0) + PROFILE_FLOOR_PCT**2) / n_eff
        for i, f in enumerate(pj):
            out[f"{f}.{name}"] = float(min(10.0, PROFILE_SIGMAS * math.sqrt(var * max(cov[i, i], 0.0))))
    return out


def write_fit_log(log_dir: Path, res: FitResult) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "fit_report.json").write_text(json.dumps(res.summary(), indent=2))
    with open(log_dir / "history.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["evals", "best_objective_pct"])
        w.writerows(res.history)


def meta_for(res: FitResult, *, release: int, run: Path, git_sha: str | None = None) -> SimParamsMeta:
    if git_sha is None:
        try:
            git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            git_sha = "unknown"
    notes = [f"unidentified (spread > {UNIDENTIFIED_SPREAD}): {', '.join(res.unidentified)}"] if res.unidentified else []
    return SimParamsMeta(
        version=release,
        date=dt.datetime.now(dt.UTC).date().isoformat(),
        git_sha=git_sha,
        mjcf_sha256=mjcf_sha256(),
        excitation_run=str(run),
        fit_rms_pct=res.train_rms_pct,
        heldout_rms_pct=res.heldout_rms_pct,
        spread=res.spread,
        backend=res.backend,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------------- CLI
def _csv(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sysid-fit", description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="excitation run directory (plan.json + tick_log.bin)")
    ap.add_argument("--out", type=Path, required=True, help="sim_params yaml to write when the held-out gate passes")
    ap.add_argument("--backend", choices=("warp", "cpu"), default="warp")
    ap.add_argument("--heldout-kinds", type=_csv, default=DEFAULT_HELDOUT_KINDS)
    ap.add_argument("--heldout-fraction", type=float, default=0.2)
    ap.add_argument("--fit", type=_csv, default=DEFAULT_FIT)
    ap.add_argument("--fixed", type=_csv, default=DEFAULT_FIXED)
    ap.add_argument("--popsize", type=int, default=None, help="default: 256 (warp) / 32 (cpu)")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None, help="cpu backend processes (default: min(16, cpus))")
    ap.add_argument("--prior", type=Path, default=None, help="prior sim_params yaml (default: the MJCF values)")
    ap.add_argument("--release", type=int, default=1, help="meta.version of the written file (HS3: 1)")
    ap.add_argument("--calibration", type=Path, default=None, help="LeRobot calibration JSON (default: from config)")
    ap.add_argument("--max-pct", type=float, default=None, help="held-out gate (default: compare_real.MAX_RMS_PCT)")
    return ap


def main(argv: list[str] | None = None) -> int:
    from so101_rl.sim.compare_real import MAX_RMS_PCT
    from so101_rl.sim.model import load_model

    args = build_parser().parse_args(argv)
    base = load_config() if args.calibration is None else None
    calib_path = args.calibration if args.calibration is not None else base.calibration_path
    calibration = load_calibration(Path(calib_path).expanduser())
    cfg = load_config(calibration=calibration)
    ds = ExcitationDataset.from_run(args.run)
    train, held = ds.split(heldout_kinds=set(args.heldout_kinds), heldout_fraction_by_time=args.heldout_fraction,
                           seed=args.seed)
    if not held.segments:
        print("sysid-fit: the split left no held-out data (check --heldout-kinds)", file=sys.stderr)
        return 2
    prior = (SimParams.from_yaml(args.prior, allow_unverified=True) if args.prior is not None
             else SimParams.from_mjcf(load_model()[0]))
    popsize = args.popsize or (256 if args.backend == "warp" else 32)
    res = fit_sim_params(train, held, prior=prior, fit=args.fit, fixed=args.fixed, backend=args.backend,
                         popsize=popsize, generations=args.generations, seed=args.seed, log_dir=args.run,
                         calibration=calibration, cfg=cfg, workers=args.workers)
    gate = MAX_RMS_PCT if args.max_pct is None else args.max_pct
    meta = meta_for(res, release=args.release, run=args.run)
    print(f"{'joint':<15}{'train %':>10}{'held-out %':>12}")
    for n in JOINT_NAMES:
        print(f"{n:<15}{res.train_rms_pct[n]:>10.3f}{res.heldout_rms_pct[n]:>12.3f}")
    print(f"delay_ticks {res.params.delay_ticks}  (scores {res.delay_scores})  evals {res.evals}  {res.wall_s:.1f} s")
    if res.unidentified:
        print(f"unidentified (spread > {UNIDENTIFIED_SPREAD}): {', '.join(res.unidentified)}")
    bad = [n for n in JOINT_NAMES if not res.heldout_rms_pct[n] <= gate]
    if bad:
        path = args.run / "sim_params.unverified.yaml"
        res.params.to_yaml(path, meta=meta)
        print(f"FAIL: held-out RMS above {gate:g} % on {', '.join(bad)}; wrote {path}", file=sys.stderr)
        return 1
    res.params.to_yaml(args.out, meta=meta)
    print(f"PASS: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

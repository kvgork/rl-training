"""Massively parallel SAC on the batched env (sim plan S4.3 / S4.6, MVP M4). FROZEN API for the connector.

``pixi run train-sim-sac --task {reach,die_to_zone} --nworld 1024 --budget-min 60 [--seed S] [--dr-max-stage 4]
[--out DIR] [--resume CKPT] [--config YAML]``

Tasks: ``eval_sim.TASKS``. ``gamma`` defaults to the task spec's (``TASK_GAMMA``: reach 0.97, die_to_zone 0.98); the
policy input is the task's FS3 observation plus the goal (reach 24, die 29), taken from the env. ``die_to_zone``
(plans/die-to-zone-plan.md D1) refuses ``success_radius_mm`` (reach only) and, per sim plan S4.6, refuses to train
on unverified sim parameters (the v0 MJCF prior: gripper / die contact never identified) unless the config sets
``allow_unverified_sim_params: true`` -- the user's recorded waiver (die-to-zone plan decision 1), never a default.

Loop: warm-up random actions -> FastSAC acting on ``obs ++ goal`` (FS3 policy input) -> GPU replay (``next_obs`` =
the pre-reset ``final_obs`` so truncation bootstraps) -> ``grad_steps_per_env_step`` updates per vector step. Every
``log_every_steps`` env steps a row goes to ``metrics.csv`` (and ``hooks.on_metrics`` + TensorBoard through
``viz.TrainingMonitor``); every ``eval_every_steps`` a deterministic eval on ``eval_episodes`` episodes at the
CURRENT DR stage (``eval_sim.run_episodes``), a checkpoint ``ckpt_<step>.pt`` + ``ckpt_last.pt``, the review triggers
(``review.json``; printed and logged, never switching anything) and the widening rule
(``sim.domain_rand.next_stage``: one stage up at >= 70 % eval success, up to ``dr_max_stage``).

Die start curriculum (``die_start_curriculum: true``, die_to_zone only; D2a, plan §10 2026-10-16): the die starts
from ``DIE_TO_ZONE.die_start`` stage 0 (open table in front of the zone) instead of the well, and the same widening
rule (>= 70 % eval success at the current stages) first moves the START stage up, S0 -> S1 -> S2 -> S3 (the well),
with the DR stage held at ``dr_start_stage``; only at S3 do the DR stages widen as before. Evals run at the current
start and DR stages; the exit gate (``stop_at_gate``) needs start stage S3 AND ``dr_max_stage``, with the same numeric
thresholds. ``start_stage`` is a ``metrics.csv`` column (the well stage for die without the curriculum, empty for
reach) and a checkpoint field (resume continues from it); ``ckpt_best`` ranks ``(start_stage, dr_stage, success)``.
Off by default: without it the die starts in the well exactly as before.

Scripted demos (``demo_ratio`` > 0 with ``demo_episodes`` > 0, die_to_zone only; D2b, plan §10 2026-10-16): before
training, ``rl/scripted.py``'s privileged pick-and-place controller runs ``demo_episodes`` episodes per start stage
(S0..S3 with the curriculum, else the well) on the training env; the transitions of its SUCCESSFUL episodes go into a
separate demo replay in the online replay's exact format (no HER here: the loop never relabels), and every update
batch takes ``replay_gpu.demo_count(batch_size, demo_ratio)`` rows from it (RLPD-style mixing, ``sample_mixed``). The
demo replay is never written again and is not checkpointed (a resume collects it anew); ``run_meta.json`` ``demos``
records the per-stage scripted success and the transition count. Default 0: nothing is collected and every batch
is the same ``replay.sample`` draw as before.

Eval cost: an eval is also deferred until this segment's eval time is at most ``eval_max_fraction`` (10 %) of its
training time (the next eval, assumed as long as the last one, waits until it fits); ``eval_frac`` in ``metrics.csv`` is the measured
fraction (the first eval of a segment is never deferred). Only the last ``keep_ckpts`` (3) numbered checkpoints are kept, plus ``ckpt_last.pt`` and
``ckpt_best.pt`` (a hard link to the numbered checkpoint with the best ``(start_stage, dr_stage, eval success)``).

``train_success_rate`` is NOT a success rate of the policy: it is the fraction of the episodes that ENDED in the log
window that succeeded, with the stochastic (exploring) policy. Successful episodes end early, so it overstates
success (completion bias), and exploration noise passing through the 15 mm ball counts too. Judge the policy by
``eval_success_rate`` (deterministic, one episode per world).

Status: ``completed`` = the numeric exit gate passed at the final DR stage (``stop_at_gate``) or ``max_env_steps``
reached; ``budget_exhausted`` = ``budget_min`` minutes of training time used (thermal pauses excluded, evals
included). The budget spans resumes and has ONE owner: the checkpoint's ``train_s`` carries the history, so a
resume passes the run's FULL ``budget_min`` (the lab job does; never the remainder). ``metrics.csv`` rows after the
resumed checkpoint's ``env_step`` are dropped on resume. ``stopped`` = ``hooks.should_stop()`` or a thermal pause
longer than ``thermal_max_pause_min`` (checkpoint, then return; resumable); ``failed`` = a non-finite loss (the last
good checkpoint is kept). Exceptions (CUDA OOM included) are NOT caught (L12).

Thermal (plan §10 P8), standalone CLI only (``thermal_guard``; the lab job owns the one thermal guard and turns this
one off): the GPU core temperature is read through NVML (``pynvml``) every ``THERMAL_POLL_S``; >= 85 °C pauses
training, <= 75 °C resumes. A missing sample is fail-closed: it pauses like a hot one (the 6th consecutive missing
sample is flagged ``thermal_needs_human`` in the metrics). ``heartbeat`` is progress-coupled, so it is not called
while paused; ``on_metrics`` reports ``thermal_paused`` = 1 when a pause starts.

``metrics.csv`` is always written (header at start). The CLI exits 1 when ``metrics.csv`` has < 3 data rows or no
checkpoint exists (L12), 0 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import itertools
import json
import math
import os
import re
import shutil
import signal
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import torch

from so101_rl.env.tasks import DIE_TO_ZONE, REACH
from so101_rl.rl import eval_sim
from so101_rl.rl.fast_sac import FastSAC, SacConfig
from so101_rl.rl.replay_gpu import GpuReplay, sample_mixed

__all__ = [
    "METRICS_FIELDS",
    "SimSacConfig",
    "ThermalGuard",
    "TrainHooks",
    "TrainResult",
    "load_sim_sac_config",
    "main",
    "run_training",
]

CKPT_FORMAT: Final[str] = "so101_rl.sim_sac_ckpt"
CKPT_VERSION: Final[int] = 1
THERMAL_PAUSE_C: Final[float] = 85.0
THERMAL_RESUME_C: Final[float] = 75.0
THERMAL_POLL_S: Final[float] = 2.0
THERMAL_NEEDS_HUMAN_MISSING: Final[int] = 6
TASK_GAMMA: Final[dict[str, float]] = {s.name: s.gamma for s in (REACH, DIE_TO_ZONE)}  # FS4 task specs
# sim plan S4.5 / S4.6 review trigger "success < X after Y GPU-h" per task: (min eval success, after GPU-h)
TASK_LOW_SUCCESS_TRIGGER: Final[dict[str, tuple[float, float]]] = {"reach": (0.5, 1.0), "die_to_zone": (0.1, 4.0)}

METRICS_FIELDS: Final[tuple[str, ...]] = (
    "env_step", "wall_s", "train_s", "env_steps_per_s", "grads_per_s", "dr_stage", "train_success_rate",
    "train_episodes", "mean_return", "critic_loss", "actor_loss", "alpha", "entropy", "q_mean", "q_max",
    "action_saturation", "eval_success_rate", "eval_wilson_lo", "eval_final_err_p50", "eval_final_err_p90",
    "vram_mb", "gpu_temp_c", "thermal_paused_s", "review_triggers", "viz_disabled", "eval_frac", "start_stage",
)  # fmt: skip
_NUMBERED_CKPT: Final[re.Pattern[str]] = re.compile(r"^ckpt_(\d+)\.pt$")


@dataclass(frozen=True)
class SimSacConfig:
    """Strict-loaded with :func:`load_sim_sac_config`. Required: ``task``, ``nworld``, ``budget_min``, ``seed``."""

    task: str
    nworld: int
    budget_min: float
    seed: int
    dr_max_stage: int = 4
    eval_every_steps: int = 200_000
    eval_episodes: int = 1000
    # ---- additive fields (documented defaults)
    dr_start_stage: int = 0
    log_every_steps: int = 50_000  # metrics.csv row interval (eval rows come on top)
    max_env_steps: int | None = None  # None = unlimited (the time budget ends the run)
    stop_at_gate: bool = True  # end as "completed" when the numeric exit gate passes at the final stage
    replay_capacity: int = 1_000_000
    batch_size: int = 4096
    grad_steps_per_env_step: int = 8
    warmup_steps: int = 20  # vector steps of uniform random actions
    lr: float = 3e-4
    gamma: float | None = None  # None = the task's (reach 0.97)
    calibration: str | None = None  # None = so101.yaml calibration_path (the real arm's calibration)
    sim_params: str = "calib/sim_params.yaml"
    vram_fraction: float = 0.6
    thermal_max_pause_min: float = 60.0
    tensorboard: bool = True
    eval_max_fraction: float = 0.1  # defer an eval until evals are <= this fraction of the training time
    keep_ckpts: int = 3  # numbered checkpoints kept (plus ckpt_last.pt and ckpt_best.pt)
    thermal_guard: bool = True  # the trainer's own NVML pause; False under the lab (the job owns the guard)
    success_radius_mm: float | None = None  # None = the task's (reach 15 mm); plan §10 2026-10-14; reach only
    allow_unverified_sim_params: bool = False  # die_to_zone only: the user's S4.6 waiver (module docstring)
    die_start_curriculum: bool = False  # die_to_zone only: start stages S0..S3 before the DR stages (D2a)
    demo_ratio: float = 0.0  # die_to_zone only: share of each batch from the scripted demo replay (D2b); 0 = off
    demo_episodes: int = 0  # scripted episodes run per start stage to fill the demo replay (successes are kept)


_REQUIRED: Final[tuple[str, ...]] = ("task", "nworld", "budget_min", "seed")
_TYPES: Final[dict[str, tuple[type, bool]]] = {  # name -> (type, nullable)
    "task": (str, False), "nworld": (int, False), "budget_min": (float, False), "seed": (int, False),
    "dr_max_stage": (int, False), "eval_every_steps": (int, False), "eval_episodes": (int, False),
    "dr_start_stage": (int, False), "log_every_steps": (int, False), "max_env_steps": (int, True),
    "stop_at_gate": (bool, False), "replay_capacity": (int, False), "batch_size": (int, False),
    "grad_steps_per_env_step": (int, False), "warmup_steps": (int, False), "lr": (float, False),
    "gamma": (float, True), "calibration": (str, True), "sim_params": (str, False), "vram_fraction": (float, False),
    "thermal_max_pause_min": (float, False), "tensorboard": (bool, False), "eval_max_fraction": (float, False),
    "keep_ckpts": (int, False), "thermal_guard": (bool, False), "success_radius_mm": (float, True),
    "allow_unverified_sim_params": (bool, False), "die_start_curriculum": (bool, False),
    "demo_ratio": (float, False), "demo_episodes": (int, False),
}  # fmt: skip


@dataclass(frozen=True)
class TrainHooks:
    on_metrics: Callable[[int, Mapping[str, float]], None] | None = None  # (env_step, scalars) at each eval/log point
    on_checkpoint: Callable[[int, Path], None] | None = None
    should_stop: Callable[[], bool] | None = None  # polled every iteration; graceful stop = checkpoint then return
    heartbeat: Callable[[int], None] | None = None  # every training iteration with env_step (progress-coupled)


@dataclass(frozen=True)
class TrainResult:
    status: Literal["completed", "stopped", "budget_exhausted", "failed"]
    env_steps: int
    last_ckpt: Path | None
    metrics_csv: Path
    reason: str = ""  # additive: why the run ended (human-readable)


# ------------------------------------------------------------------------------------------------ config
def load_sim_sac_config(src: Path | Mapping) -> SimSacConfig:
    """Strict: a yaml/json file or a mapping with exactly known keys; required keys present; types and ranges
    checked (ints reject bools and floats; floats accept ints). Raises ``ValueError`` / ``TypeError``."""
    if isinstance(src, Mapping):
        raw = dict(src)
        where = "config"
    else:
        import yaml

        where = str(src)
        raw = yaml.safe_load(Path(src).read_text())
        if not isinstance(raw, dict):
            raise TypeError(f"{where}: expected a mapping")
    unknown = sorted(set(raw) - set(_TYPES))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{where}: missing required keys {missing}")
    vals: dict[str, Any] = {}
    for k, v in raw.items():
        typ, nullable = _TYPES[k]
        if v is None:
            if not nullable:
                raise TypeError(f"{where}: {k} must not be null")
            vals[k] = None
            continue
        if typ is bool:
            ok = isinstance(v, bool)
        elif typ is int:
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif typ is float:
            ok = isinstance(v, int | float) and not isinstance(v, bool)
            v = float(v) if ok else v
        else:
            ok = isinstance(v, str)
        if not ok:
            raise TypeError(f"{where}: {k} must be {typ.__name__}, got {type(v).__name__} {v!r}")
        vals[k] = v
    cfg = SimSacConfig(**vals)
    _check_ranges(cfg, where)
    return cfg


def _check_ranges(c: SimSacConfig, where: str) -> None:
    from so101_rl.sim.domain_rand import REACH_DR_STAGES

    def bad(msg: str) -> ValueError:
        return ValueError(f"{where}: {msg}")

    if c.task not in eval_sim.TASKS:
        raise bad(f"task {c.task!r} is not trainable (only {list(eval_sim.TASKS)})")
    if c.task != "reach" and c.success_radius_mm is not None:
        raise bad(f"success_radius_mm is a reach-only override (plan §10 2026-10-14); {c.task} success is the zone "
                  "square (env/tasks.py)")  # fmt: skip
    if c.task != "die_to_zone" and c.die_start_curriculum:
        raise bad(f"die_start_curriculum is a die_to_zone option; {c.task} has no die start stages")
    if not 0.0 <= c.demo_ratio < 1.0 or c.demo_episodes < 0:
        raise bad("demo_ratio must be in [0, 1) and demo_episodes >= 0")
    if (c.demo_ratio > 0) != (c.demo_episodes > 0):
        raise bad("demo mixing needs both demo_ratio > 0 and demo_episodes > 0 (or neither)")
    if c.demo_ratio > 0 and c.task != "die_to_zone":
        raise bad(f"demo mixing uses the scripted die controller (rl/scripted.py); {c.task} has none")
    if c.nworld < 1:
        raise bad("nworld must be >= 1")
    if not (math.isfinite(c.budget_min) and c.budget_min >= 0):
        raise bad("budget_min must be finite and >= 0")
    last = len(REACH_DR_STAGES) - 1
    if not 0 <= c.dr_start_stage <= c.dr_max_stage <= last:
        raise bad(f"need 0 <= dr_start_stage <= dr_max_stage <= {last}")
    for k in ("eval_every_steps", "eval_episodes", "log_every_steps", "replay_capacity", "batch_size",
              "grad_steps_per_env_step", "keep_ckpts"):  # fmt: skip
        if getattr(c, k) < 1:
            raise bad(f"{k} must be >= 1")
    if c.success_radius_mm is not None and not (math.isfinite(c.success_radius_mm) and c.success_radius_mm > 0):
        raise bad("success_radius_mm must be > 0 (or null = the task's)")
    if c.warmup_steps < 0 or (c.max_env_steps is not None and c.max_env_steps < 0):
        raise bad("warmup_steps / max_env_steps must be >= 0")
    if c.replay_capacity < c.nworld:
        raise bad("replay_capacity must be >= nworld")
    if not (0 < c.lr < 1) or (c.gamma is not None and not 0 < c.gamma < 1):
        raise bad("lr and gamma must be in (0, 1)")
    if not 0 < c.vram_fraction <= 1 or not c.thermal_max_pause_min > 0:
        raise bad("vram_fraction must be in (0, 1], thermal_max_pause_min > 0")
    if not 0 < c.eval_max_fraction <= 1:
        raise bad("eval_max_fraction must be in (0, 1]")


# ------------------------------------------------------------------------------------------------ thermal
_NVML_HANDLE: Any = None


def read_gpu_temp_c() -> float | None:
    """GPU 0 core temperature through NVML; None when it cannot be read (the caller fails closed)."""
    global _NVML_HANDLE
    try:
        import pynvml

        if _NVML_HANDLE is None:
            pynvml.nvmlInit()
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:  # noqa: BLE001 - any failure is a missing sample (fail-closed by the guard)
        _NVML_HANDLE = None  # re-initialise on the next sample
        return None


class ThermalGuard:
    """Hysteresis pause at ``pause_c`` / resume at ``resume_c``; a missing sample pauses (fail-closed)."""

    def __init__(self, reader: Callable[[], float | None] = read_gpu_temp_c, *, pause_c: float = THERMAL_PAUSE_C,
                 resume_c: float = THERMAL_RESUME_C) -> None:  # fmt: skip
        if not resume_c < pause_c:
            raise ValueError("resume_c must be below pause_c")
        self.reader, self.pause_c, self.resume_c = reader, pause_c, resume_c
        self.paused = False
        self.missing = 0
        self.last_c: float | None = None

    @property
    def needs_human(self) -> bool:
        return self.missing >= THERMAL_NEEDS_HUMAN_MISSING

    def sample(self) -> bool:
        """Take one sample; returns ``paused``."""
        t = self.reader()
        if t is None or not math.isfinite(t):
            self.missing += 1
            self.paused = True
            return True
        self.missing, self.last_c = 0, float(t)
        if t >= self.pause_c:
            self.paused = True
        elif self.paused and t <= self.resume_c:
            self.paused = False
        return self.paused


# ------------------------------------------------------------------------------------------------ review triggers
def review_triggers(history: list[dict[str, float]], *, train_h: float, n_updates_per_transition: float,
                    task: str = "reach") -> list[str]:  # fmt: skip
    """Plan S4.3 step 2 (reach subset; the low-success trigger per ``TASK_LOW_SUCCESS_TRIGGER``). Recorded only; the
    method is never switched automatically (plan §5, P7)."""
    out: list[str] = []
    if not history:
        return out
    h = history[-1]
    if n_updates_per_transition < 0.005:
        out.append(f"UTD {n_updates_per_transition:.4f} < 0.005 grads/transition")
    if h.get("alpha", 1.0) < 1e-4:
        out.append(f"alpha {h['alpha']:.2e} < 1e-4")
    rmax, gamma = h.get("reward_abs_max", 0.0), h.get("gamma", 0.97)
    if rmax > 0 and h.get("q_max", 0.0) > 2 * rmax / (1 - gamma):
        out.append(f"|Q| {h['q_max']:.1f} > 2 r_max/(1-gamma) = {2 * rmax / (1 - gamma):.1f}")
    if h.get("action_saturation", 0.0) > 0.5:
        out.append(f"action saturation {h['action_saturation']:.2f} > 0.5")
    evals = [x for x in history if "eval_success_rate" in x]
    if len(evals) >= 4:
        cl = [x.get("critic_loss") for x in evals[-4:]]
        if None not in cl and all(b > a for a, b in itertools.pairwise(cl)):
            out.append("critic loss rising over 3 evals")
        best_before = max(x["eval_success_rate"] for x in evals[:-3])
        if max(x["eval_success_rate"] for x in evals[-3:]) <= best_before < eval_sim.GATE_MIN_SUCCESS:
            out.append("no eval improvement over 3 evals")
    min_ok, after_h = TASK_LOW_SUCCESS_TRIGGER[task]
    if train_h >= after_h and evals and evals[-1]["eval_success_rate"] < min_ok:
        out.append(f"{task} success {evals[-1]['eval_success_rate']:.2f} < {min_ok:g} after {train_h:.2f} GPU-h")
    return out


def trigger_kind(trigger: str) -> str:
    """The trigger without its numbers (``"alpha 9.1e-05 < 1e-4"`` -> ``"alpha # < #"``): each kind is recorded once
    per run, not on every log line its condition still holds (die run 1 wrote dozens of copies to ``review.json``)."""
    return re.sub(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", "#", trigger)


# ------------------------------------------------------------------------------------------------ training
class _Csv:
    def __init__(self, path: Path, *, append: bool) -> None:
        self.path = path
        new = not (append and path.exists() and path.stat().st_size > 0)
        self._f = path.open("a" if not new else "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=list(METRICS_FIELDS), extrasaction="ignore")
        if new:
            self._w.writeheader()
            self._f.flush()

    def row(self, r: Mapping[str, Any]) -> None:
        self._w.writerow({k: ("" if r.get(k) is None else r[k]) for k in METRICS_FIELDS})
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _truncate_metrics(path: Path, env_step: int) -> None:
    """On resume: keep only the rows up to the checkpoint's ``env_step`` (later rows belong to the abandoned
    continuation) and rewrite the file with the current header."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    def step(r: Mapping[str, Any]) -> float:
        try:
            return float(r.get("env_step") or "nan")
        except ValueError:
            return math.nan

    keep = [r for r in rows if step(r) <= env_step]
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(METRICS_FIELDS), extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)
    tmp.replace(path)


def _prune_ckpts(out_dir: Path, keep: int) -> None:
    """Delete all but the ``keep`` highest-step ``ckpt_<step>.pt`` (``ckpt_last`` / ``ckpt_best`` are not numbered)."""
    numbered = sorted((int(m.group(1)), p) for p in out_dir.iterdir() if (m := _NUMBERED_CKPT.match(p.name)))
    for _, p in numbered[:-keep] if keep > 0 else numbered:
        p.unlink(missing_ok=True)


def _link_or_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    tmp.replace(dst)


def _save(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def unverified_sim_params(path: str | Path) -> str | None:
    """Why the ``sim_params`` file is not a fitted, verified release (None = release >= 1 and verified)."""
    from so101_rl.sim.compare_real import MAX_RMS_PCT
    from so101_rl.sim.sysid.params import read_sim_params_file

    meta = read_sim_params_file(Path(path), allow_unverified=True).meta
    if meta.version < 1:
        return f"release {meta.version} is the MJCF prior"
    problems = meta.verification_problems(MAX_RMS_PCT)
    return f"release {meta.version} unverified: {'; '.join(problems)}" if problems else None


def run_training(
    cfg: SimSacConfig, out_dir: Path, *, hooks: TrainHooks | None = None, resume: Path | None = None
) -> TrainResult:
    """Train until gate / budget / stop (module docstring). Always writes ``metrics.csv``; returns the outcome."""
    hooks = hooks or TrainHooks()
    if cfg.task == "die_to_zone" and not cfg.allow_unverified_sim_params:
        problem = unverified_sim_params(cfg.sim_params)
        if problem:
            raise ValueError(f"{cfg.sim_params}: {problem}. Sim plan S4.6: die_to_zone is not trained on unverified "
                             "sim parameters (gripper / die contact never identified; needs HS3 v1). A waiver is the "
                             "user's decision (plans/die-to-zone-plan.md decision 1): allow_unverified_sim_params")  # fmt: skip
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    ck: dict[str, Any] | None = None
    if resume is not None:
        ck = eval_sim.load_checkpoint(Path(resume))
        if ck["task"] != cfg.task:
            raise ValueError(f"resume: checkpoint task {ck['task']!r} != {cfg.task!r}")
    if ck is not None:
        _truncate_metrics(metrics_csv, int(ck["env_step"]))
    csvw = _Csv(metrics_csv, append=ck is not None)
    try:
        return _train(cfg, out_dir, hooks, ck, csvw)
    finally:
        csvw.close()


def _train(cfg: SimSacConfig, out_dir: Path, hooks: TrainHooks, ck: dict[str, Any] | None, csvw: _Csv) -> TrainResult:
    stage = int(ck["dr_stage"]) if ck is not None else cfg.dr_start_stage
    stage = min(stage, cfg.dr_max_stage)
    env = eval_sim.build_env(cfg.task, cfg.nworld, stage=stage, seed=cfg.seed, calibration=cfg.calibration,
                             sim_params=cfg.sim_params, success_radius_mm=cfg.success_radius_mm)  # fmt: skip
    try:
        return _train_with_env(cfg, out_dir, hooks, ck, csvw, env)
    finally:
        env.close()


def _train_with_env(cfg: SimSacConfig, out_dir: Path, hooks: TrainHooks, ck: dict[str, Any] | None, csvw: _Csv,
                    env: Any) -> TrainResult:  # fmt: skip
    eval_env: Any = None
    dev = torch.device(env.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.vram_fraction, dev)
        torch.cuda.reset_peak_memory_stats(dev)
    gamma = cfg.gamma if cfg.gamma is not None else TASK_GAMMA.get(cfg.task, 0.97)
    goal_dim = int(env.goal.shape[1])
    obs_dim, priv_dim, act_dim = int(env.obs_dim) + goal_dim, int(env.priv_dim), int(env.act_dim)
    sac_cfg = SacConfig(batch_size=cfg.batch_size, grad_steps_per_env_step=cfg.grad_steps_per_env_step, lr_actor=cfg.lr,
                        lr_critic=cfg.lr, lr_alpha=cfg.lr, gamma=gamma, warmup_env_steps=cfg.warmup_steps,
                        target_entropy=-float(act_dim) / 2.0, seed=cfg.seed)  # fmt: skip
    sac = FastSAC(obs_dim, priv_dim, act_dim, sac_cfg, device=dev)
    env_step, train_s0 = 0, 0.0
    if ck is not None:
        sac.load_state_dict(ck["sac"])
        env_step, train_s0 = int(ck["env_step"]), float(ck["train_s"])
    replay = GpuReplay(cfg.replay_capacity, obs_dim, priv_dim, act_dim, device=dev, gamma=gamma)
    # die start curriculum (module docstring): None for reach; the well stage unless the curriculum is on
    die_start = DIE_TO_ZONE.die_start if cfg.task == "die_to_zone" else None
    well_stage = die_start.well_stage if die_start is not None else None
    start_stage = well_stage
    if cfg.die_start_curriculum:
        start_stage = int(ck["start_stage"]) if ck is not None and ck.get("start_stage") is not None else 0
        env.set_start_stage(start_stage)
    # scripted demos (D2b, RLPD-style mixing): a separate replay, filled once (also after a resume), never with
    # online transitions; demo_ratio 0 builds nothing and samples exactly as before
    demo_replay, demo_stats = None, None
    if cfg.demo_ratio > 0:
        demo_replay, demo_stats = _collect_demo_replay(cfg, env, (obs_dim, priv_dim, act_dim), dev, gamma, well_stage,
                                                       start_stage)  # fmt: skip
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed + 12345)
    meta = {"config": dataclasses.asdict(cfg), "env": dict(getattr(env, "metadata", {})), "git_sha": eval_sim.git_sha(),
            "resumed_from_step": env_step if ck is not None else None, "sac": sac_cfg.to_dict(),
            "start_stages": [die_start.stage_name(k) for k in range(die_start.n_stages)] if die_start else None,
            "demos": demo_stats}  # fmt: skip
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    from so101_rl.viz.monitor import TrainingMonitor, VizConfig

    so101_cfg = getattr(env, "cfg", None)
    mon = TrainingMonitor(VizConfig(live_viewer=False, video_every_min=0.0, tensorboard=cfg.tensorboard),
                          getattr(env, "ws", None), out_dir, task=cfg.task, render_cfg=so101_cfg)  # fmt: skip
    guard = ThermalGuard(read_gpu_temp_c) if cfg.thermal_guard else None
    history: list[dict[str, float]] = []
    review: list[dict[str, Any]] = []
    if ck is not None and (out_dir / "review.json").exists():  # a resume keeps the run's triggers (each fires once)
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            review = [r for r in json.loads((out_dir / "review.json").read_text())["triggers"]
                      if int(r["env_step"]) <= env_step]  # fmt: skip
    last_ckpt: Path | None = (out_dir / "ckpt_last.pt") if (out_dir / "ckpt_last.pt").exists() and ck is not None else None
    status: str = "budget_exhausted"
    reason = ""
    budget_s = cfg.budget_min * 60.0
    t_wall0 = time.monotonic()
    train_s = train_s0
    paused_s = 0.0
    next_log = env_step + cfg.log_every_steps
    next_eval = env_step + cfg.eval_every_steps
    next_thermal = 0.0
    eval_s = 0.0  # this segment's eval time (eval_frac); the time cap below spans the segment
    eval_ok_at = train_s0  # train_s at which the next eval may run (eval_max_fraction)
    best_path, best_key = out_dir / "ckpt_best.json", None
    if ck is not None and best_path.exists():
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            b = json.loads(best_path.read_text())
            ss = b.get("start_stage", well_stage)
            best_key = (int(ss) if ss is not None else 0, int(b["dr_stage"]), float(b["success_rate"]))
    seg_train_s0 = train_s0
    n = int(env.nworld)
    # per-window accumulators (device)
    acc: dict[str, torch.Tensor] = {}
    n_upd = 0
    ep_done = torch.zeros((), device=dev)
    ep_succ = torch.zeros((), device=dev)
    ep_ret_sum = torch.zeros((), device=dev)
    rew_abs_max = torch.zeros((), device=dev)
    ret_run = torch.zeros(n, device=dev)
    win_t, win_step, win_upd = time.monotonic(), env_step, 0
    vec_steps = 0

    def checkpoint(last_eval: dict[str, Any] | None, *, numbered: bool) -> Path:
        nonlocal last_ckpt
        d = {
            "format": CKPT_FORMAT, "version": CKPT_VERSION, "task": cfg.task, "env_step": env_step, "train_s": train_s,
            "dr_stage": int(env.dr_stage), "start_stage": start_stage, "sac": sac.state_dict(),
            "config": dataclasses.asdict(cfg),
            "obs_layout_version": _obs_layout_version(), "policy_input_dim": obs_dim, "act_dim": act_dim,
            "sim_params_release": meta["env"].get("sim_params_release"), "git_sha": meta["git_sha"],
            "sim_params_verified": meta["env"].get("sim_params_verified"),
            "sim_params_sha256": meta["env"].get("sim_params_sha256"),
            "calibration_sha256": meta["env"].get("calibration_sha256"),
            "last_eval": last_eval or {}, "camera_pose_method": meta["env"].get("camera_pose_method", "none"),
        }  # fmt: skip
        nonlocal best_key
        if numbered:
            num = out_dir / f"ckpt_{env_step}.pt"
            _save(num, d)
            key = (start_stage if start_stage is not None else 0, int(env.dr_stage),
                   float((last_eval or {}).get("success_rate", -1.0)))  # fmt: skip
            if last_eval and (best_key is None or key >= best_key):
                best_key = key
                _link_or_copy(num, out_dir / "ckpt_best.pt")
                best_path.write_text(json.dumps({"env_step": env_step, "start_stage": start_stage, "dr_stage": key[1],
                                                 "success_rate": key[2]}))  # fmt: skip
            _prune_ckpts(out_dir, cfg.keep_ckpts)
        _save(out_dir / "ckpt_last.pt", d)
        last_ckpt = out_dir / "ckpt_last.pt"
        if hooks.on_checkpoint is not None:
            hooks.on_checkpoint(env_step, last_ckpt)
        return last_ckpt

    def emit_row(ev: dict[str, Any] | None) -> dict[str, float]:
        nonlocal acc, n_upd, ep_done, ep_succ, ep_ret_sum, win_t, win_step, win_upd
        now = time.monotonic()
        dt = max(now - win_t, 1e-9)
        m = {k: float(v) / max(n_upd, 1) for k, v in acc.items()} if n_upd else {}
        done = float(ep_done)
        row: dict[str, Any] = {
            "env_step": env_step, "wall_s": round(now - t_wall0, 3), "train_s": round(train_s, 3),
            "env_steps_per_s": round((env_step - win_step) / dt, 1), "grads_per_s": round((sac.updates - win_upd) / dt, 1),
            "dr_stage": int(env.dr_stage), "start_stage": start_stage,
            "train_success_rate": float(ep_succ) / done if done else None,
            "train_episodes": int(done), "mean_return": float(ep_ret_sum) / done if done else None,
            "vram_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if dev.type == "cuda" else 0.0,
            "gpu_temp_c": guard.last_c if guard is not None else None, "thermal_paused_s": round(paused_s, 1),
            "eval_frac": round(eval_s / max(train_s - seg_train_s0, 1e-9), 4), **{k: round(v, 6) for k, v in m.items()},
        }  # fmt: skip
        if ev is not None:
            row.update({"eval_success_rate": ev["success_rate"], "eval_wilson_lo": ev["wilson_lo"],
                        "eval_final_err_p50": ev["final_err_m"]["p50"], "eval_final_err_p90": ev["final_err_m"]["p90"]})  # fmt: skip
        hist = {k: float(v) for k, v in row.items() if isinstance(v, int | float) and not isinstance(v, bool)}
        hist.update({"reward_abs_max": float(rew_abs_max), "gamma": gamma})
        history.append(hist)
        trig = review_triggers(history, train_h=train_s / 3600.0, n_updates_per_transition=cfg.grad_steps_per_env_step / n,
                               task=cfg.task)  # fmt: skip
        seen = {trigger_kind(r["trigger"]) for r in review}
        new = [t for t in trig if trigger_kind(t) not in seen]
        for t in new:
            review.append({"env_step": env_step, "trigger": t})
            print(f"[train-sim-sac] REVIEW TRIGGER @ {env_step}: {t} (recorded; the method is not switched)", flush=True)
            mon.add_text(env_step, "review/trigger", t)
        if new:
            (out_dir / "review.json").write_text(json.dumps(
                {"triggers": review, "note": "review triggers never switch the method automatically (plan §5, P7)"},
                indent=2))  # fmt: skip
        row["review_triggers"] = len(review)
        row.update(mon.csv_fields())
        csvw.row(row)
        scalars = {k: float(v) for k, v in row.items() if isinstance(v, int | float) and not isinstance(v, bool)}
        mon.on_metrics(env_step, {**scalars, **({"eval_success_rate": ev["success_rate"]} if ev else {})})
        if hooks.on_metrics is not None:
            hooks.on_metrics(env_step, scalars)
        acc, n_upd = {}, 0
        ep_done.zero_()
        ep_succ.zero_()
        ep_ret_sum.zero_()
        win_t, win_step, win_upd = now, env_step, sac.updates
        return m

    def run_eval() -> dict[str, Any]:
        nonlocal eval_env
        if eval_env is None:
            eval_env = eval_sim.build_env(cfg.task, min(cfg.eval_episodes, cfg.nworld, 1024), stage=int(env.dr_stage),
                                          seed=cfg.seed + 1000, calibration=cfg.calibration, sim_params=cfg.sim_params,
                                          success_radius_mm=cfg.success_radius_mm)  # fmt: skip
        if getattr(eval_env, "randomizer", None) is not None:
            eval_env.randomizer.set_stage(int(env.dr_stage))
        if cfg.die_start_curriculum:
            eval_env.set_start_stage(start_stage)
        sac.actor.eval()
        st = eval_sim.run_episodes(eval_env, lambda x: sac.act(x, deterministic=True), cfg.eval_episodes)
        sac.actor.train()
        lo, hi = eval_sim.wilson_interval(st["successes"], st["episodes"])
        return {"success_rate": st["success_rate"], "wilson_lo": lo, "wilson_hi": hi, "episodes": st["episodes"],
                "successes": st["successes"], "final_err_m": st["final_err_m"], "dr_stage": int(env.dr_stage),
                "start_stage": start_stage, "env_step": env_step}  # fmt: skip

    obs, priv = (x.clone() for x in env.reset()[:2])
    last_ev: dict[str, Any] | None = dict(ck["last_eval"]) if ck is not None and ck.get("last_eval") else None
    stop_sig = hooks.should_stop or (lambda: False)
    try:
        while True:
            if stop_sig():
                status, reason = "stopped", "should_stop() returned True"
                break
            now = time.monotonic()
            if guard is not None and now >= next_thermal:
                next_thermal = now + THERMAL_POLL_S
                if guard.sample():
                    res = _thermal_pause(guard, stop_sig, hooks, env_step, cfg.thermal_max_pause_min)
                    paused_s += res[1]
                    win_t += res[1]  # pause time is not throughput time
                    if res[0] != "resumed":  # never "failed": a thermal timeout is a resumable stop (P8)
                        status, reason = ("stopped", "should_stop() during a thermal pause") if res[0] == "stopped" else (
                            "stopped", f"thermal_timeout: thermal pause longer than {cfg.thermal_max_pause_min} min")  # fmt: skip
                        break
                    next_thermal = time.monotonic() + THERMAL_POLL_S
            if train_s >= budget_s:
                status, reason = "budget_exhausted", f"{cfg.budget_min} min training budget used"
                break
            if cfg.max_env_steps is not None and env_step >= cfg.max_env_steps:
                status, reason = "completed", f"max_env_steps {cfg.max_env_steps} reached"
                break
            t_it = time.monotonic()
            obs_in = torch.cat([obs, env.goal], dim=1)
            if vec_steps < cfg.warmup_steps and ck is None:
                action = torch.rand((n, act_dim), generator=gen, device=dev) * 2.0 - 1.0
            else:
                action = sac.act(obs_in)
            st = env.step(action)
            next_in = torch.cat([st.final_obs, st.desired], dim=1)
            replay.add(obs_in, priv, action, st.reward, next_in, st.final_priv, st.terminated.float(), st.truncated.float())
            done = st.terminated | st.truncated
            ret_run += st.reward.float()
            ep_done += done.sum()
            ep_succ += st.info["success"].sum()  # terminated also covers a terminal trip
            ep_ret_sum += torch.where(done, ret_run, torch.zeros_like(ret_run)).sum()
            mon.on_step(env_step + n, success=st.info["success"], ep_return=ret_run)  # before the done worlds' reset
            ret_run.masked_fill_(done, 0.0)
            rew_abs_max = torch.maximum(rew_abs_max, st.reward.abs().max())
            obs, priv = st.obs, st.priv
            vec_steps += 1
            env_step += n
            if replay.size >= cfg.batch_size and (vec_steps > cfg.warmup_steps or ck is not None):
                for _ in range(cfg.grad_steps_per_env_step):
                    out = sac.update(sample_mixed(replay, demo_replay, cfg.batch_size, cfg.demo_ratio, generator=gen))
                    for k, v in out.items():
                        acc[k] = acc[k] + v if k in acc else v.clone()
                    n_upd += 1
            train_s += time.monotonic() - t_it
            if hooks.heartbeat is not None:
                hooks.heartbeat(env_step)
            ev = None
            if env_step >= next_eval and train_s >= eval_ok_at:
                next_eval += cfg.eval_every_steps * max(1, (env_step - next_eval) // cfg.eval_every_steps + 1)
                t_e = time.monotonic()
                ev = run_eval()
                dt_e = time.monotonic() - t_e
                train_s += dt_e
                eval_s += dt_e
                # cumulative, counting the next eval (assumed as long as this one): eval_s <= f train_s after it
                eval_ok_at = seg_train_s0 + (eval_s + dt_e) / cfg.eval_max_fraction - dt_e
            if ev is not None or env_step >= next_log:
                next_log = env_step + cfg.log_every_steps
                m = emit_row(ev)
                if m and not all(math.isfinite(v) for v in m.values()):
                    status, reason = "failed", f"non-finite training metrics at env_step {env_step}: {m}"
                    break
            if ev is not None:
                last_ev = ev
                checkpoint(ev, numbered=True)
                passed, _ = eval_sim.gate_verdict(ev["successes"], ev["episodes"], ev["dr_stage"])
                at_target_start = start_stage == well_stage  # reach: both None; die: the well (S3)
                if passed and cfg.stop_at_gate and ev["dr_stage"] >= cfg.dr_max_stage and at_target_start:
                    status, reason = "completed", f"numeric exit gate passed at stage {ev['dr_stage']}"
                    break
                if not at_target_start:  # the start curriculum first; the DR stage waits for the well
                    new_start = _next_stage(start_stage, ev["success_rate"], well_stage)
                    if new_start != start_stage:
                        start_stage = new_start
                        env.set_start_stage(start_stage)
                        print(f"[train-sim-sac] start stage {start_stage - 1} -> {start_stage} "
                              f"({die_start.stage_name(start_stage)}) at {env_step} (eval success "
                              f"{ev['success_rate']:.3f})", flush=True)  # fmt: skip
                    continue
                new_stage = _next_stage(int(env.dr_stage), ev["success_rate"], cfg.dr_max_stage)
                if new_stage != int(env.dr_stage):
                    env.randomizer.set_stage(new_stage)
                    print(f"[train-sim-sac] DR stage {new_stage - 1} -> {new_stage} at {env_step} "
                          f"(eval success {ev['success_rate']:.3f})", flush=True)  # fmt: skip
        nonfinite = status == "failed" and reason.startswith("non-finite")
        if not nonfinite and env_step > 0:
            m = emit_row(None) if (n_upd or float(ep_done) > 0) else {}
            finite = all(math.isfinite(v) for v in m.values()) and all(
                bool(torch.isfinite(p).all()) for p in sac.actor.parameters())
            if finite:  # the final weights (also after a thermal timeout); last_eval = the latest eval (may be older)
                checkpoint(last_ev, numbered=False)
            else:  # never overwrite the last good checkpoint with non-finite weights
                status, reason = "failed", f"non-finite training state at env_step {env_step}"
    finally:
        mon.close()
        if eval_env is not None:
            eval_env.close()
    print(f"[train-sim-sac] {status}: {reason} (env_steps {env_step}, train {train_s / 60:.1f} min, "
          f"thermal pause {paused_s:.0f} s)", flush=True)  # fmt: skip
    return TrainResult(status, env_step, last_ckpt, csvw.path, reason)  # type: ignore[arg-type]


def _collect_demo_replay(cfg: SimSacConfig, env: Any, dims: tuple[int, int, int], dev: torch.device, gamma: float,
                         well_stage: int | None, start_stage: int | None) -> tuple[GpuReplay, dict[str, Any]]:  # fmt: skip
    """Fill a demo replay with ``cfg.demo_episodes`` scripted episodes per start stage (every curriculum stage S0..S3
    with ``die_start_curriculum``, else the well) on the training env, keeping the successful ones
    (``scripted.collect_demos``); the env's start stage is restored. Raises when no demo succeeded (L12)."""
    from so101_rl.rl import scripted

    die_start = DIE_TO_ZONE.die_start
    stages = list(range(die_start.n_stages)) if cfg.die_start_curriculum else [well_stage]
    horizon = int(getattr(getattr(env, "task", None), "episode_steps", getattr(env, "episode_steps", 100)))
    demo = GpuReplay(max(cfg.demo_episodes * len(stages) * horizon, int(env.nworld)), *dims, device=dev, gamma=gamma)
    per_stage = {}
    t0 = time.monotonic()
    for k in stages:
        env.set_start_stage(k)
        st = scripted.collect_demos(env, demo, cfg.demo_episodes, policy=scripted.demo_policy(env))
        per_stage[die_start.stage_name(k)] = st
        print(f"[train-sim-sac] demos S{k} ({die_start.stage_name(k)}): {st['successes']}/{st['episodes']} scripted "
              f"successes, {st['transitions']} transitions", flush=True)  # fmt: skip
    env.set_start_stage(start_stage)
    if demo.size == 0:
        raise RuntimeError("demo_ratio > 0 but no scripted demo episode succeeded (rl/scripted.py); refusing to mix an "
                           "empty demo replay")  # fmt: skip
    return demo, {"ratio": cfg.demo_ratio, "episodes_per_stage": cfg.demo_episodes, "transitions": demo.size,
                  "collect_s": round(time.monotonic() - t0, 1), "per_stage": per_stage}  # fmt: skip


def _next_stage(stage: int, success: float, max_stage: int) -> int:
    from so101_rl.sim.domain_rand import next_stage

    return next_stage(stage, success, max_stage)


def _obs_layout_version() -> int:
    from so101_rl.env.obs import OBS_LAYOUT_VERSION

    return OBS_LAYOUT_VERSION


def _thermal_pause(guard: ThermalGuard, should_stop: Callable[[], bool], hooks: TrainHooks, env_step: int,
                   max_pause_min: float) -> tuple[str, float]:  # fmt: skip
    """Block while ``guard`` is paused. Returns ``("resumed" | "stopped" | "timeout", seconds paused)``."""
    t0 = time.monotonic()
    print(f"[train-sim-sac] THERMAL PAUSE at env_step {env_step}: gpu {guard.last_c} C, missing samples {guard.missing}",
          flush=True)  # fmt: skip
    if hooks.on_metrics is not None:
        hooks.on_metrics(env_step, {"thermal_paused": 1.0, "thermal_needs_human": float(guard.needs_human),
                                    **({"gpu_temp_c": guard.last_c} if guard.last_c is not None else {})})  # fmt: skip
    while guard.paused:
        if should_stop():
            return "stopped", time.monotonic() - t0
        if time.monotonic() - t0 > max_pause_min * 60.0:
            return "timeout", time.monotonic() - t0
        time.sleep(THERMAL_POLL_S)
        guard.sample()
        if guard.needs_human and hooks.on_metrics is not None and guard.missing == THERMAL_NEEDS_HUMAN_MISSING:
            hooks.on_metrics(env_step, {"thermal_paused": 1.0, "thermal_needs_human": 1.0})
    if hooks.on_metrics is not None:
        hooks.on_metrics(env_step, {"thermal_paused": 0.0, **({"gpu_temp_c": guard.last_c} if guard.last_c is not None else {})})
    return "resumed", time.monotonic() - t0


# ------------------------------------------------------------------------------------------------ CLI
def _artifacts_ok(res: TrainResult) -> tuple[bool, str]:
    rows = 0
    if res.metrics_csv.exists():
        with res.metrics_csv.open() as f:
            rows = max(0, sum(1 for _ in f) - 1)
    if rows < 3:
        return False, f"{res.metrics_csv} has {rows} data rows (< 3)"
    if res.last_ckpt is None or not res.last_ckpt.exists():
        return False, "no checkpoint written"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train-sim-sac", description="FastSAC on the batched sim env (reach, die_to_zone)")
    ap.add_argument("--config", type=Path, default=None, help="yaml with SimSacConfig keys (CLI flags override)")
    ap.add_argument("--task", default=None)
    ap.add_argument("--nworld", type=int, default=None)
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dr-max-stage", type=int, default=None)
    ap.add_argument("--dr-start-stage", type=int, default=None)
    ap.add_argument("--eval-every-steps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--success-radius-mm", type=float, default=None, help="reach only: success radius (default the task's 15 mm)")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--sim-params", default=None)
    ap.add_argument("--out", type=Path, default=None, help="default runs/sac-sim/<task>-<timestamp>/")
    ap.add_argument("--resume", type=Path, default=None)
    a = ap.parse_args(argv)
    raw: dict[str, Any] = {}
    if a.config is not None:
        import yaml

        raw.update(yaml.safe_load(a.config.read_text()) or {})
    for k in ("task", "nworld", "budget_min", "seed", "dr_max_stage", "dr_start_stage", "eval_every_steps",
              "eval_episodes", "calibration", "sim_params", "success_radius_mm"):  # fmt: skip
        v = getattr(a, k)
        if v is not None:
            raw[k] = v
    raw.setdefault("task", "reach")
    raw.setdefault("seed", 0)
    cfg = load_sim_sac_config(raw)
    out = a.out or Path("runs/sac-sim") / f"{cfg.task}-{time.strftime('%Y%m%d-%H%M%S')}"
    stop = {"flag": False}

    def on_signal(signum: int, _frame: Any) -> None:
        stop["flag"] = True
        print(f"[train-sim-sac] signal {signum}: stopping after this iteration (checkpoint first)", flush=True)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    res = run_training(cfg, out, hooks=TrainHooks(should_stop=lambda: stop["flag"]), resume=a.resume)
    ok, why = _artifacts_ok(res)
    print(f"[train-sim-sac] status={res.status} env_steps={res.env_steps} ckpt={res.last_ckpt} metrics={res.metrics_csv}")
    if not ok:
        print(f"[train-sim-sac] artifacts missing: {why}", file=sys.stderr)
        return 1
    return 0 if res.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())

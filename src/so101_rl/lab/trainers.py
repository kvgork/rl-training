"""Trainer plugins: a small protocol and a static table (``fake`` for tests, ``fastsac_sim`` for the MVP).

A plugin validates its config (strict: unknown keys are refused, so thermal limits, budgets etc. can never be set
through a config), says how the job process is launched, and runs train / evaluate inside that job process.
``fastsac_sim`` targets the frozen ``so101_rl.rl.{train_sim_sac,eval_sim,policy_export}`` API and imports it lazily,
only inside the job; nothing here imports torch.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from so101_rl.lab.errors import LabError
from so101_rl.lab.runs import REPO_ROOT

TASKS: dict[str, str] = {"reach": "Move the gripper tip to a goal in the workspace box (FS4 REACH, 15 mm, 100 steps)."}
EXIT_GATE = {"success_min": 0.95, "wilson_lo_min": 0.93, "episodes_min": 1000}  # D-EXIT-GATE (numeric part)


@dataclass(frozen=True)
class Capabilities:
    gpu: bool
    resumable: bool
    evaluates: bool
    exports: bool
    env_name: str  # the pixi environment the job runs in
    tasks: tuple[str, ...]


@dataclass
class Hooks:
    """Same shape as the frozen ``TrainHooks``; the job supplies the callables."""

    on_metrics: Callable[[int, dict[str, float]], None] = lambda step, scalars: None
    on_checkpoint: Callable[[int, str], None] = lambda step, path: None
    should_stop: Callable[[], bool] = lambda: False
    heartbeat: Callable[[int], None] = lambda step: None


@dataclass(frozen=True)
class TrainOutcome:
    status: str  # completed | stopped | budget_exhausted | failed
    env_steps: int
    last_ckpt: str | None
    metrics_csv: str | None
    reason: str = ""  # the trainer's own words for why it ended (``TrainResult.reason``); untrusted text


@dataclass(frozen=True)
class EvalOutcome:
    success_rate: float
    wilson_lo: float
    wilson_hi: float
    episodes: int
    numeric_gate_passed: bool
    eval_json: str | None
    policy: str | None = None


@dataclass(frozen=True)
class Param:
    type: type
    default: Any
    lo: float | None = None
    hi: float | None = None


class Trainer(Protocol):
    name: str
    version: str
    capabilities: Capabilities
    params: dict[str, Param]

    def available(self) -> bool: ...
    def validate(self, config: dict[str, Any], task: str) -> dict[str, Any]: ...
    def train(self, cfg: dict[str, Any], task: str, out_dir: Path, hooks: Hooks, resume: str | None) -> TrainOutcome: ...
    def evaluate(
        self, ckpt: str, task: str, *, episodes: int, dr_stage: int | None, out_dir: Path, render: int
    ) -> EvalOutcome: ...


def launch_argv(trainer: Trainer, run_dir: Path, *, python: Sequence[str] | None = None) -> list[str]:
    """The pinned argv of the job process. ``python`` replaces the pixi prefix (the subprocess fallback / tests)."""
    tail = ["-m", "so101_rl.lab.job", str(run_dir)]
    if python:
        return [*python, *tail]
    pixi = shutil.which("pixi") or "pixi"
    return [pixi, "run", "--frozen", "--manifest-path", str(REPO_ROOT / "pyproject.toml"),
            "-e", trainer.capabilities.env_name, "python", *tail]


def validate_params(params: dict[str, Param], config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise LabError("CONFIG_INVALID", "config must be an object", details={"path": ""})
    unknown = sorted(set(config) - set(params))
    if unknown:
        raise LabError(
            "CONFIG_INVALID", "unknown config key", details={"path": f"/{unknown[0]}", "allowed": sorted(params)},
            hint="thermal limits, budgets and gates are not configurable",
        )
    out: dict[str, Any] = {}
    for k, p in params.items():
        v = config.get(k, p.default)
        if p.type is float and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if v is None and p.default is None:
            out[k] = None
            continue
        if not isinstance(v, p.type) or isinstance(v, bool) != (p.type is bool):
            raise LabError("CONFIG_INVALID", f"{k} must be {p.type.__name__}", details={"path": f"/{k}"})
        if p.type in (int, float) and (
            (p.lo is not None and v < p.lo) or (p.hi is not None and v > p.hi) or not math.isfinite(v)
        ):
            raise LabError("CONFIG_INVALID", f"{k} out of range", details={"path": f"/{k}", "lo": p.lo, "hi": p.hi})
        out[k] = v
    return out


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def numeric_gate(success_rate: float, wilson_lo: float, episodes: int) -> bool:
    return (
        episodes >= EXIT_GATE["episodes_min"]
        and success_rate >= EXIT_GATE["success_min"]
        and wilson_lo >= EXIT_GATE["wilson_lo_min"]
    )


# ----------------------------------------------------------------------------------------------- fake
class FakeTrainer:
    """Deterministic fake job: writes metrics.csv, heartbeats and JSON checkpoints; no torch, no GPU."""

    name = "fake"
    version = "1"
    capabilities = Capabilities(gpu=False, resumable=True, evaluates=True, exports=True, env_name="cpu",
                                tasks=("reach",))
    params: ClassVar[dict[str, Param]] = {
        "steps": Param(int, 200, 1, 10_000_000),
        "step_s": Param(float, 0.01, 0.0, 5.0),
        "metric_every": Param(int, 10, 1, 1_000_000),
        "ckpt_every": Param(int, 50, 1, 1_000_000),
        "budget_min": Param(float, 10.0, 0.001, 360.0),
        "seed": Param(int, 0, 0, 2**31 - 1),
        "crash_at": Param(int, -1, -1, 10_000_000),
        "success_rate": Param(float, 0.97, 0.0, 1.0),
    }

    def available(self) -> bool:
        return True

    def validate(self, config: dict[str, Any], task: str) -> dict[str, Any]:
        return validate_params(self.params, config)

    def train(self, cfg: dict[str, Any], task: str, out_dir: Path, hooks: Hooks, resume: str | None) -> TrainOutcome:
        start = 0
        if resume:
            start = int(json.loads(Path(resume).read_text())["env_step"])
        metrics = out_dir / "metrics.csv"
        new = not metrics.exists()
        t0 = time.monotonic()
        last_ckpt = resume
        with metrics.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["env_step", "wall_s", "return", "success_rate"])
            step = start
            while step < cfg["steps"]:
                if hooks.should_stop():
                    last_ckpt = self._ckpt(out_dir, step, hooks, cfg)
                    return TrainOutcome("stopped", step, last_ckpt, str(metrics))
                if cfg["crash_at"] >= 0 and step >= cfg["crash_at"]:
                    raise RuntimeError(f"fake crash at step {step}")
                time.sleep(cfg["step_s"])
                step += 1
                hooks.heartbeat(step)
                if step % cfg["metric_every"] == 0 or step == cfg["steps"]:
                    frac = step / cfg["steps"]
                    scalars = {"return": round(-50.0 * (1.0 - frac), 4),
                               "success_rate": round(cfg["success_rate"] * frac, 4)}
                    w.writerow([step, round(time.monotonic() - t0, 3), scalars["return"], scalars["success_rate"]])
                    f.flush()
                    hooks.on_metrics(step, scalars)
                if step % cfg["ckpt_every"] == 0:
                    last_ckpt = self._ckpt(out_dir, step, hooks, cfg)
        last_ckpt = self._ckpt(out_dir, step, hooks, cfg)
        return TrainOutcome("completed", step, last_ckpt, str(metrics))

    @staticmethod
    def _ckpt(out_dir: Path, step: int, hooks: Hooks, cfg: dict[str, Any]) -> str:
        p = out_dir / f"ckpt_{step:09d}.json"
        p.write_text(json.dumps({"env_step": step, "success_rate": cfg["success_rate"]}))  # self-contained (S5)
        hooks.on_checkpoint(step, str(p))
        return str(p)

    def evaluate(
        self, ckpt: str, task: str, *, episodes: int, dr_stage: int | None, out_dir: Path, render: int
    ) -> EvalOutcome:
        src = json.loads(Path(ckpt).read_text())
        rate = float(src.get("success_rate", 0.97))
        succ = round(rate * episodes)
        lo, hi = wilson(succ, episodes)
        sr = succ / episodes
        out = {"task": task, "ckpt_env_step": src["env_step"], "episodes": episodes, "successes": succ,
               "success_rate": sr, "wilson_lo": lo, "wilson_hi": hi, "dr_stage": dr_stage,
               "numeric_gate_passed": numeric_gate(sr, lo, episodes), "gate": EXIT_GATE}
        (out_dir / "eval.json").write_text(json.dumps(out, indent=1))
        policy = None
        if out["numeric_gate_passed"]:
            policy = str(out_dir / "policy.json")
            Path(policy).write_text(json.dumps({"from_ckpt": ckpt}))
        return EvalOutcome(sr, lo, hi, episodes, out["numeric_gate_passed"], str(out_dir / "eval.json"), policy)


# ----------------------------------------------------------------------------------------------- fastsac_sim
class FastSacSimTrainer:
    """FastSAC in GPU sim (M2/M4) through the frozen ``rl.train_sim_sac`` / ``rl.eval_sim`` / ``rl.policy_export``."""

    name = "fastsac_sim"
    version = "1"
    capabilities = Capabilities(gpu=True, resumable=True, evaluates=True, exports=True, env_name="default",
                                tasks=("reach",))
    params: ClassVar[dict[str, Param]] = {
        "nworld": Param(int, 1024, 1, 65_536),  # 8 grads / 1024 worlds = UTD 0.0078 >= the 0.005 review trigger
        "budget_min": Param(float, 60.0, 1.0, 360.0),
        "seed": Param(int, 0, 0, 2**31 - 1),
        "dr_max_stage": Param(int, 4, 0, 4),
        "eval_every_steps": Param(int, 200_000, 1_000, 100_000_000),
        "eval_episodes": Param(int, 1000, 1, 10_000),
        "log_every_steps": Param(int, 50_000, 1_000, 100_000_000),
        "max_env_steps": Param(int, None, 1, 10_000_000_000),  # None: the time budget ends the run
    }

    def available(self) -> bool:
        return all(
            importlib.util.find_spec(m) is not None
            for m in ("so101_rl.rl.train_sim_sac", "so101_rl.rl.eval_sim", "so101_rl.rl.policy_export")
        )

    def validate(self, config: dict[str, Any], task: str) -> dict[str, Any]:
        return validate_params(self.params, config)

    def train(self, cfg: dict[str, Any], task: str, out_dir: Path, hooks: Hooks, resume: str | None) -> TrainOutcome:
        from so101_rl.rl import train_sim_sac as tss  # lazy: torch + warp live behind this import

        try:  # the job owns the one thermal guard (P8): the trainer's own pause is off under the lab
            sim_cfg = tss.load_sim_sac_config({"task": task, **cfg, "thermal_guard": False})
        except (TypeError, ValueError) as e:  # validate() already checked the lab's keys; this is a contract drift
            raise LabError("CONFIG_INVALID", f"train_sim_sac refused the config: {e}"[:300]) from None
        th = tss.TrainHooks(
            on_metrics=lambda step, scalars: hooks.on_metrics(int(step), dict(scalars)),
            on_checkpoint=lambda step, path: hooks.on_checkpoint(int(step), str(path)),
            should_stop=hooks.should_stop,
            heartbeat=lambda step: hooks.heartbeat(int(step)),
        )
        r = tss.run_training(sim_cfg, out_dir, hooks=th, resume=Path(resume) if resume else None)
        return TrainOutcome(str(r.status), int(r.env_steps), _s(r.last_ckpt), _s(r.metrics_csv),
                            str(getattr(r, "reason", "")))

    def evaluate(
        self, ckpt: str, task: str, *, episodes: int, dr_stage: int | None, out_dir: Path, render: int
    ) -> EvalOutcome:
        from so101_rl.rl import eval_sim, policy_export

        kw: dict[str, Any] = {"episodes": episodes, "out_dir": out_dir, "render": render}
        kw["dr_stage"] = 4 if dr_stage is None else dr_stage
        r = eval_sim.evaluate(Path(ckpt), task, **kw)
        policy = None
        if r.numeric_gate_passed:
            policy = _s(policy_export.export_policy(Path(ckpt), out_dir / "policy.pt"))
        return EvalOutcome(float(r.success_rate), float(r.wilson_lo), float(r.wilson_hi), int(r.episodes),
                           bool(r.numeric_gate_passed), _s(r.eval_json), policy)


def _s(p: Any) -> str | None:
    return None if p is None else str(p)


TRAINERS: dict[str, Trainer] = {"fake": FakeTrainer(), "fastsac_sim": FastSacSimTrainer()}


def get_trainer(name: str) -> Trainer:
    try:
        return TRAINERS[name]
    except (KeyError, TypeError):
        raise LabError("UNKNOWN_TRAINER", "no such trainer", details={"known": sorted(TRAINERS)}) from None


def describe(t: Trainer) -> dict[str, Any]:
    c = t.capabilities
    return {
        "name": t.name, "version": t.version, "available": t.available(), "gpu": c.gpu, "resumable": c.resumable,
        "tasks": list(c.tasks),
        "config": {k: {"type": p.type.__name__, "default": p.default, "min": p.lo, "max": p.hi}
                   for k, p in t.params.items()},
    }


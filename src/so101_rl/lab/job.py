"""The job entry point, run inside each transient unit: ``python -m so101_rl.lab.job <run_dir>``.

One invocation = one segment. It loads ``run.json``, runs the trainer (train, or evaluate for ``kind == "eval"``),
and owns the run's ``status.json`` / ``usage.json`` / ``heartbeat`` while it lives:

- **Heartbeat** is progress-coupled: it is written from the trainer's ``heartbeat(env_step)`` hook, never a thread.
- **GPU-h** is progress-holding time (P6): the sum of heartbeat intervals in which ``env_step`` advanced, each
  capped at 30 min; thermal holds and stalls do not count.
- **should_stop** is true on a pause/stop request in ``control.json``, SIGTERM, the run's own budget (progress
  time ≥ ``budget_min``), the wall deadline (budget + 60 min), or a thermal hold longer than 30 min.
- **Budget** has one owner per layer: the trainer gets the run's FULL ``budget_min`` (its checkpoint's ``train_s``
  carries the history across resumes); the job's own check (progress time of all segments vs ``budget_min``) is the
  lab's backstop. Passing the remainder would count the history twice (B1, 2026-10-08 review).
- **Thermal** (P8, GPU trainers only): the job's guard is the ONE guard (``fastsac_sim`` runs with the trainer's own
  pause off). The heartbeat blocks while the GPU is hot or NVML is silent (fail-closed); the status shows
  ``paused`` with ``pause_reason`` ``thermal_hot`` / ``thermal_unknown`` meanwhile; a hold longer than 30 min stops
  the run with a checkpoint (``stopped`` / ``thermal_timeout``, resumable). A trainer that still reports its own
  pause (``on_metrics`` with ``thermal_paused`` = 1) is treated as a known pause, not a stall.
- **Eval** runs evaluate the copy of the checkpoint in their own dir (``config.ckpt`` relative to it,
  ``ckpt_sha256`` checked), so a later resume of the training run cannot change what was evaluated.
- **Trainer outcome** -> state: ``completed`` -> completed; ``stopped`` / ``budget_exhausted`` -> stopped (or paused
  for an AI pause) with the job's stop reason; ``failed`` -> failed with the trainer's reason (untrusted text).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from so101_rl.lab import runs
from so101_rl.lab.launcher import boot_id, proc_start_time
from so101_rl.lab.thermal import NvmlSampler, ThermalGuard
from so101_rl.lab.trainers import Hooks, TrainOutcome, get_trainer

log = logging.getLogger("so101_rl.lab.job")

DEADLINE_GRACE_S = 60 * 60
EVAL_WALL_S = 4 * 3600
HB_WRITE_S = 2.0
USAGE_WRITE_S = 30.0


class Job:
    def __init__(
        self,
        run_dir: Path,
        *,
        thermal: ThermalGuard | None = None,
        use_nvml: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run = runs.read_run(self.run_dir)
        self.status = runs.read_status(self.run_dir)
        self.trainer = get_trainer(self.run["trainer"])
        self.clock = clock
        if thermal is None and (self.trainer.capabilities.gpu if use_nvml is None else use_nvml):
            thermal = ThermalGuard(NvmlSampler())
        self.thermal = thermal
        self.segment = int(self.status.get("segment", 0))
        usage = runs.read_usage(self.run_dir)
        self.prior_gpu_h = float(usage.get("gpu_h") or 0.0)
        self.progress_s = 0.0
        self.last_step = int(self.status.get("env_step") or 0)
        self.t_start = clock()
        self.last_hb_t = self.t_start
        self._hb_written = -1e18
        self._usage_written = self.t_start
        self.stop_reason: str | None = None
        self._signal_stop = False
        self._trainer_hold_t: float | None = None  # clock() when the trainer reported thermal_paused = 1
        cfg = self.run.get("config", {})
        self.budget_s = float(cfg.get("budget_min", 0.0)) * 60.0 if self.run.get("kind") != "eval" else 0.0
        remaining = self.budget_s - self.prior_gpu_h * 3600.0
        self.remaining_s = max(0.0, remaining)
        wall = EVAL_WALL_S if self.run.get("kind") == "eval" else self.remaining_s + DEADLINE_GRACE_S
        self.deadline = self.t_start + wall

    # ------------------------------------------------------------------ control
    def control_action(self) -> str | None:
        ctl = runs.read_json(self.run_dir / "control.json")
        if isinstance(ctl, dict) and ctl.get("action") in ("pause", "stop") and ctl.get("segment") == self.segment:
            return str(ctl["action"])
        return None

    def should_stop(self) -> bool:
        if self.stop_reason:
            return True
        action = self.control_action()
        if action == "pause":
            self.stop_reason = "ai_pause"
        elif action == "stop" or self._signal_stop:
            self.stop_reason = "user_stop"
        elif self.run.get("kind") != "eval" and self.progress_s >= self.remaining_s:
            self.stop_reason = "budget_exhausted"
        elif self.clock() >= self.deadline:
            self.stop_reason = "deadline"
        return self.stop_reason is not None

    # ------------------------------------------------------------------ hooks
    def heartbeat(self, env_step: int) -> None:
        now = self.clock()
        if env_step > self.last_step:
            self.progress_s += min(now - self.last_hb_t, runs.STALL_CAP_S)
            self.last_step = env_step
        self.last_hb_t = now
        if self.thermal is not None and not self._thermal_wait() and not self.should_stop():
            self.stop_reason = "thermal_timeout"
        if now - self._hb_written >= HB_WRITE_S:
            self._write_heartbeat(env_step)
        if now - self._usage_written >= USAGE_WRITE_S:
            self._write_usage(final=False)

    def _thermal_wait(self) -> bool:
        assert self.thermal is not None

        def on_hold(reason: str) -> None:
            core, mem = self.thermal.last_sample if self.thermal else (None, None)
            runs.update_status(self.run_dir, state="paused", pause_reason=f"thermal_{reason}", hold=reason)
            runs.append_event(self.run_dir, "thermal_hold", reason=reason, core_c=core, mem_c=mem)
            if reason == "unknown":
                runs.append_event(self.run_dir, "needs_human", reason="gpu_temperature_unobservable")

        def on_release() -> None:
            self.last_hb_t = self.clock()  # the hold is not progress-holding time
            runs.update_status(self.run_dir, state="running", pause_reason=None, hold=None)
            runs.append_event(self.run_dir, "thermal_release", core_c=self.thermal.last_sample[0])

        return self.thermal.wait(
            on_hold=on_hold, on_release=on_release,
            should_abort=lambda: self._signal_stop or self.control_action() is not None
        )

    def on_metrics(self, env_step: int, scalars: dict[str, float]) -> None:
        paused = scalars.get("thermal_paused")
        if paused is not None:
            self._trainer_thermal(env_step, paused >= 0.5, bool(scalars.get("thermal_needs_human", 0.0)))
            return  # a pause report is not progress (and must not block in the job's own thermal wait)
        self.heartbeat(env_step)

    def _trainer_thermal(self, env_step: int, paused: bool, needs_human: bool) -> None:
        """The trainer's own thermal pause (``train_sim_sac``): a known pause, not a stall."""
        now = self.clock()
        if paused:
            if self._trainer_hold_t is None:
                self._trainer_hold_t = now
                runs.update_status(self.run_dir, state="paused", pause_reason="thermal_trainer", hold="trainer")
                runs.append_event(self.run_dir, "thermal_hold", reason="trainer", env_step=int(env_step))
            if needs_human:
                runs.append_event(self.run_dir, "needs_human", reason="gpu_temperature_unobservable")
            return
        if self._trainer_hold_t is not None:
            self.deadline += now - self._trainer_hold_t  # the pause does not eat the wall deadline
            self._trainer_hold_t = None
            self.last_hb_t = now  # the pause is not progress-holding time
            runs.update_status(self.run_dir, state="running", pause_reason=None, hold=None)
            runs.append_event(self.run_dir, "thermal_release", reason="trainer", env_step=int(env_step))

    def on_checkpoint(self, env_step: int, path: str) -> None:
        runs.update_status(self.run_dir, last_ckpt=str(path), env_step=int(env_step))
        runs.append_event(self.run_dir, "checkpoint", env_step=int(env_step))

    def _write_heartbeat(self, env_step: int) -> None:
        self._hb_written = self.clock()
        t = runs.now_iso()
        runs.atomic_write_json(self.run_dir / "heartbeat", {"t": t, "env_step": env_step, "pid": os.getpid()})
        runs.update_status(self.run_dir, heartbeat_at=t, env_step=int(env_step))

    def _write_usage(self, *, final: bool) -> None:
        self._usage_written = self.clock()
        with runs.flocked(self.run_dir / ".lock"):
            u = runs.read_usage(self.run_dir)
            gpu_h = max(float(u.get("gpu_h") or 0.0), self.prior_gpu_h + self.progress_s / 3600.0)  # high-water
            segs = [s for s in u.get("segments", []) if s.get("seg") != self.segment]
            segs.append({"seg": self.segment, "progress_h": round(self.progress_s / 3600.0, 6),
                         "wall_h": round((self.clock() - self.t_start) / 3600.0, 6), "final": final})
            u.update(gpu_h=round(gpu_h, 6), segments=segs, updated_at=runs.now_iso(),
                     wall_h=round(sum(s["wall_h"] for s in segs), 6))
            runs.atomic_write_json(self.run_dir / "usage.json", u)

    # ------------------------------------------------------------------ main
    def run_segment(self) -> int:
        old_handler = signal.signal(signal.SIGTERM, self._on_sigterm)
        try:
            return self._run_segment()
        finally:
            signal.signal(signal.SIGTERM, old_handler)

    def _run_segment(self) -> int:
        runs.update_status(
            self.run_dir, state="running", active=True, pid=os.getpid(), pid_start=proc_start_time(os.getpid()),
            unit=os.environ.get("SO101_LAB_UNIT") or self.status.get("unit"), boot_id=boot_id(),
            pause_reason=None, hold=None, stop_reason=None, error=None,
        )
        runs.append_event(self.run_dir, "segment_started", segment=self.segment, pid=os.getpid())
        try:
            if self.run.get("kind") == "eval":
                return self._evaluate()
            return self._train()
        except Exception as e:  # noqa: BLE001 - every failure must land in status.json
            log.error("job failed: %s", traceback.format_exc())
            self._write_usage(final=True)
            runs.update_status(self.run_dir, state="failed", active=False, error=f"{type(e).__name__}: {e}"[:2000])
            runs.append_event(self.run_dir, "failed", error=f"{type(e).__name__}: {e}"[:512])
            return 1

    def _on_sigterm(self, signum: int, frame: Any) -> None:
        self._signal_stop = True

    def _train(self) -> int:
        cfg = dict(self.run["config"])  # the FULL budget_min: the trainer's train_s carries the history (B1)
        resume = self.status.get("last_ckpt") if self.segment > 0 else None
        if self.remaining_s <= 0.0:
            out = TrainOutcome("budget_exhausted", self.last_step, resume, None)
        else:
            hooks = Hooks(on_metrics=self.on_metrics, on_checkpoint=self.on_checkpoint,
                          should_stop=self.should_stop, heartbeat=self.heartbeat)
            out = self.trainer.train(cfg, self.run["task"], self.run_dir, hooks, resume)
        self._write_heartbeat(out.env_steps)
        self._write_usage(final=True)
        fields: dict[str, Any] = {"active": False, "env_step": int(out.env_steps), "hold": None}
        if out.last_ckpt:
            fields["last_ckpt"] = out.last_ckpt
        if out.status == "completed":
            fields.update(state="completed", stop_reason=None, pause_reason=None)
        elif out.status in ("stopped", "budget_exhausted"):
            reason = self.stop_reason or ("budget_exhausted" if out.status == "budget_exhausted" else "user_stop")
            if reason == "ai_pause":
                fields.update(state="paused", pause_reason="ai", stop_reason=None)
            else:
                fields.update(state="stopped", stop_reason=reason, pause_reason=None)
        else:
            fields.update(state="failed", error=f"trainer returned {out.status!r}: {out.reason}"[:500])
        st = runs.update_status(self.run_dir, **fields)
        runs.append_event(self.run_dir, st["state"], env_step=int(out.env_steps),
                          reason=st.get("stop_reason") or st.get("pause_reason"),
                          trainer_status=out.status, trainer_reason=out.reason[:300])
        return 1 if st["state"] == "failed" else 0

    def _evaluate(self) -> int:
        cfg = self.run["config"]
        if self.thermal is not None and not self._thermal_wait():
            runs.update_status(self.run_dir, state="stopped", active=False, stop_reason="thermal_timeout")
            runs.append_event(self.run_dir, "stopped", reason="thermal_timeout")
            return 0
        ckpt = Path(cfg["ckpt"])
        ckpt = ckpt if ckpt.is_absolute() else self.run_dir / ckpt
        if cfg.get("ckpt_sha256") and runs.file_sha256(ckpt) != cfg["ckpt_sha256"]:
            raise RuntimeError(f"the checkpoint copy {ckpt.name} does not match ckpt_sha256")
        t0 = self.clock()
        r = self.trainer.evaluate(str(ckpt), self.run["task"], episodes=int(cfg["episodes"]),
                                  dr_stage=cfg.get("dr_stage"), out_dir=self.run_dir, render=int(cfg.get("render", 0)))
        self.progress_s = self.clock() - t0  # an eval holds the GPU for its whole wall time
        self._write_usage(final=True)
        result = {"success_rate": r.success_rate, "wilson_lo": r.wilson_lo, "wilson_hi": r.wilson_hi,
                  "episodes": r.episodes, "numeric_gate_passed": r.numeric_gate_passed,
                  "eval_json": r.eval_json, "policy": r.policy, "ckpt": str(ckpt),
                  "source_ckpt": cfg.get("source_ckpt"), "ckpt_sha256": cfg.get("ckpt_sha256")}
        runs.atomic_write_json(self.run_dir / "result.json", result)
        runs.update_status(self.run_dir, state="completed", active=False)
        runs.append_event(self.run_dir, "eval_done", numeric_gate_passed=r.numeric_gate_passed,
                          success_rate=r.success_rate, wilson_lo=r.wilson_lo)
        return 0


def check_import_root(repo: str | None) -> str | None:
    """None when ``so101_rl`` is imported from ``<repo>/src`` (the launcher's ``SO101_LAB_REPO``), else the error.

    The pixi editable install is shared by every checkout; the launcher pins ``PYTHONPATH=<repo>/src``, and this
    check makes a job that would still run another checkout's code fail loudly instead."""
    import so101_rl

    pkg = Path(so101_rl.__file__).resolve().parent
    if not repo:
        return f"SO101_LAB_REPO is not set (jobs are started by so101_rl.lab.launcher); so101_rl is from {pkg}"
    src = (Path(repo) / "src").resolve()
    if not pkg.is_relative_to(src):
        return f"so101_rl is imported from {pkg}, not from the launching checkout's {src}"
    return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="so101_rl.lab.job", description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path)
    a = p.parse_args(argv)
    err = check_import_root(os.environ.get("SO101_LAB_REPO"))
    if err is not None:
        log.error("refusing to run: %s", err)
        try:
            runs.update_status(a.run_dir.resolve(), state="failed", active=False, error=f"import pinning: {err}"[:500])
            runs.append_event(a.run_dir.resolve(), "failed", error="import_pinning")
        except Exception:  # noqa: BLE001 - the refusal itself must not be masked
            log.error("could not record the refusal: %s", traceback.format_exc())
        return 3
    return Job(a.run_dir.resolve()).run_segment()


if __name__ == "__main__":
    sys.exit(main())

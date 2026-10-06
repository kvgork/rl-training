"""``TrainingMonitor``: the frozen interface FS11 of ``so101_rl.viz`` (plan Task V.3; V.1 and V.2 plug in here).

``rl/train_sim_sac.py`` and the env debugging tools call only this::

    mon = TrainingMonitor(VizConfig(live_viewer=args.viewer, ...), sim, run_dir, task="reach", render_cfg=cfg)
    mon.attach_rollout_env(video_env)              # additive: a small separate env for rollout videos (optional)
    ...
    mon.on_step(env_step, success=success, ep_return=ep_return)       # every env step: O(1), no host sync
    mon.on_metrics(env_step, {"success_rate": 0.4, "alpha": 0.1, ...})  # every log interval: TensorBoard scalars
    mon.maybe_record(env_step, policy)                                  # when due (and force=True at a checkpoint)
    mon.close()

**Never raises into the training loop.** Every public method catches ``Exception``; the first failure is logged once,
all visualisation is switched off for the rest of the run, ``viz_disabled`` becomes 1 and ``viz/status.json`` records
the error. The training loop writes ``mon.csv_fields()`` (``{"viz_disabled": 0|1}``) into ``metrics.csv``. This is the
one place swallowing an error is allowed: the error is visualisation-only. A missing display (headless) is not a
failure: the live viewer is simply left off (logged once) and videos / TensorBoard carry on. A missing
``tensorboard`` package likewise only turns the TensorBoard writer off.

TensorBoard logs go to ``<run_dir>/tensorboard`` (``pixi run dashboard --logdir runs/`` scans recursively).
``metrics.csv`` stays the source of truth. The logged scalar names (an unknown name is logged as given):
:data:`TB_TAGS`.

Imports: ``sim`` and ``env`` read-only; never ``safety``; ``rl`` imports ``viz``, not the reverse.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self

from so101_rl.viz.live_viewer import (
    ATTACH_REQUEST,
    LivePublisher,
    NoDisplayError,
    VizContext,
    require_display,
)

if TYPE_CHECKING:
    import torch

    from so101_rl.config.schema import So101Config
    from so101_rl.sim.warp_model import WarpSim
    from so101_rl.viz.video import RolloutEnv

__all__ = ["TB_TAGS", "TrainingMonitor", "VizConfig", "gpu_health"]

log = logging.getLogger("so101_rl.viz")

# name given by the training loop -> TensorBoard tag (dashboard sections)
TB_TAGS: Final[dict[str, str]] = {
    # training curves
    "success_rate": "train/success_rate",
    "eval_success_rate": "eval/success_rate",
    "mean_return": "train/mean_return",
    "episode_length": "train/episode_length",
    "dr_stage": "train/dr_stage",
    # SAC internals
    "alpha": "sac/alpha",
    "entropy": "sac/entropy",
    "q_mean": "sac/q_mean",
    "q_max": "sac/q_max",
    "critic_loss": "sac/critic_loss",
    "action_saturation": "sac/action_saturation",
    "clamp_active_fraction": "env/clamp_active_fraction",
    "tracker_visible_fraction": "env/tracker_visible_fraction",
    # system health (also filled in by the monitor itself)
    "env_steps_per_s": "system/env_steps_per_s",
    "grads_per_s": "system/grads_per_s",
}


@dataclass(frozen=True)
class VizConfig:
    live_viewer: bool = False  # --viewer
    viewer_worlds: tuple[str, ...] = ("0", "best", "worst")  # world ids or selectors
    viewer_hz: float = 30.0
    video_every_min: float = 10.0  # 0 disables
    video_worlds: int = 4
    video_seconds: float = 10.0
    tensorboard: bool = True


def gpu_health() -> dict[str, float]:
    """GPU temperature / memory / utilisation. NVML through ``pynvml`` when installed, else one ``nvidia-smi`` call;
    empty when neither works. (A call costs ~30 ms: the monitor throttles it.)"""
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "system/gpu_temp_c": float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
            "system/gpu_mem_used_mb": mem.used / 2**20,
            "system/gpu_util_pct": float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
        }
    except Exception as e:  # noqa: BLE001 - fall through to nvidia-smi
        log.debug("pynvml unavailable (%s); trying nvidia-smi", e)
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=temperature.gpu,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        t, m, u = (float(x) for x in out.strip().split(","))
        return {"system/gpu_temp_c": t, "system/gpu_mem_used_mb": m, "system/gpu_util_pct": u}
    except Exception:  # noqa: BLE001
        return {}


class TrainingMonitor:
    """One per training run; never raises into the training loop (see the module docstring)."""

    def __init__(self, cfg: VizConfig, sim: WarpSim, run_dir: Path, *, task: str, render_cfg: So101Config) -> None:
        self.cfg, self.sim, self.run_dir, self.task, self.render_cfg = cfg, sim, Path(run_dir), task, render_cfg
        self.viz_dir = self.run_dir / "viz"
        self.context = VizContext()  # the training loop may update its fields (by reference) for the overlay
        self.viz_disabled = 0
        self.error: str | None = None
        self.last_video: Path | None = None
        self._publisher: LivePublisher | None = None
        self._tb: Any = None
        self._rollout_env: RolloutEnv | None = None
        self._success: torch.Tensor | None = None
        self._ep_return: torch.Tensor | None = None
        self._last_env_step = 0
        self._housekeeping_at = 0.0
        self._t0 = time.monotonic()
        self._last_video_t = self._t0
        self._last_metrics_t: float | None = None
        self._last_metrics_step = 0
        self._last_gpu_t = -1e9
        self._gpu: dict[str, float] = {}
        self._warned: set[str] = set()
        self._closed = False
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.viz_dir / "status.json").unlink(missing_ok=True)  # a previous run's verdict is not this run's
            self._init_components()
        except Exception as e:  # noqa: BLE001 - visualisation-only
            self._disable(e)

    # ------------------------------------------------------------------ setup
    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning("viz: %s", msg)

    def _init_components(self) -> None:
        cfg = self.cfg
        if cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except ImportError as e:
                self._warn_once("tb", f"TensorBoard writer off ({e}); metrics.csv is unaffected")
        if cfg.live_viewer:
            try:
                require_display()
            except NoDisplayError as e:
                self._warn_once("display", f"live viewer off: {e}")
            else:
                self._make_publisher(spawn=True)

    def _make_publisher(self, spawn: bool) -> LivePublisher:
        if self._publisher is None:
            self._publisher = LivePublisher(
                self.sim, self.cfg.viewer_worlds, self.cfg.viewer_hz, self.viz_dir, task=self.task, spawn_viewer=spawn
            )
        return self._publisher

    def attach_rollout_env(self, env: RolloutEnv) -> None:
        """Provide the env :meth:`maybe_record` rolls the policy out on (a separate ``video_worlds``-world env)."""
        self._rollout_env = env

    # ------------------------------------------------------------------ failure handling
    def _disable(self, exc: BaseException) -> None:
        if self.viz_disabled:
            return
        self.viz_disabled = 1
        self.error = f"{type(exc).__name__}: {exc}"
        log.error("viz: visualisation disabled for the rest of the run (training continues): %s", self.error, exc_info=exc)
        self._teardown()
        try:
            self.viz_dir.mkdir(parents=True, exist_ok=True)
            (self.viz_dir / "status.json").write_text(json.dumps({"viz_disabled": 1, "error": self.error}))
        except OSError:
            pass

    def _teardown(self) -> None:
        pub, self._publisher = self._publisher, None
        tb, self._tb = self._tb, None
        for closer in (getattr(pub, "close", None), getattr(tb, "close", None)):
            if closer is not None:
                try:
                    closer()
                except Exception:  # noqa: BLE001, S110 - already failing; nothing more to report
                    pass

    def csv_fields(self) -> dict[str, int]:
        """Columns for ``metrics.csv``: ``viz_disabled`` is 1 once any visualisation component has crashed."""
        return {"viz_disabled": self.viz_disabled}

    # ------------------------------------------------------------------ FS11
    def on_step(self, env_step: int, *, success: torch.Tensor, ep_return: torch.Tensor) -> None:
        """Every env step. O(1): remembers references, and (when due) lets the publisher start one small copy."""
        if self.viz_disabled or self._closed:
            return
        try:
            self._success, self._ep_return, self._last_env_step = success, ep_return, env_step
            pub = self._publisher
            if pub is not None:
                pub.tick(env_step, success, ep_return, self.context)
            now = time.monotonic()
            if now >= self._housekeeping_at:
                self._housekeeping_at = now + 1.0
                self._housekeeping()
        except Exception as e:  # noqa: BLE001
            self._disable(e)

    def _housekeeping(self) -> None:
        req = self.viz_dir / ATTACH_REQUEST
        if req.exists():  # `viz-attach` asked for a feed
            try:
                req.unlink()
            except OSError:
                pass
            try:
                require_display()
                self._make_publisher(spawn=False).request_viewer()
            except NoDisplayError as e:
                self._warn_once("display", f"viewer attach ignored: {e}")
        pub = self._publisher
        if pub is not None:
            rc = pub.poll_child()
            if rc is not None and rc != 0:
                raise RuntimeError(f"viewer process crashed (exit code {rc})")

    def on_metrics(self, env_step: int, scalars: Mapping[str, float]) -> None:
        """TensorBoard scalars (name mapping in :data:`TB_TAGS`) plus the monitor's own throughput / GPU / return."""
        if self.viz_disabled or self._closed:
            return
        try:
            now = time.monotonic()
            out: dict[str, float] = {TB_TAGS.get(k, k): float(v) for k, v in scalars.items()}
            if self._last_metrics_t is not None and now > self._last_metrics_t:
                out.setdefault("system/env_steps_per_s", (env_step - self._last_metrics_step) / (now - self._last_metrics_t))
            self._last_metrics_t, self._last_metrics_step = now, env_step
            if self._success is not None and self._ep_return is not None:  # one host read per log interval
                out.setdefault("train/current_success_fraction", float(self._success.float().mean()))
                out.setdefault("train/current_mean_return", float(self._ep_return.float().mean()))
            if now - self._last_gpu_t >= 10.0:
                self._last_gpu_t, self._gpu = now, gpu_health()
            out.update(self._gpu)
            out["system/viz_disabled"] = float(self.viz_disabled)
            if self._tb is not None:
                for tag, val in out.items():
                    self._tb.add_scalar(tag, val, env_step)
                self._tb.flush()
        except Exception as e:  # noqa: BLE001
            self._disable(e)

    def add_text(self, env_step: int, tag: str, text: str) -> None:
        """Review triggers and other notes, as TensorBoard text."""
        if self.viz_disabled or self._closed:
            return
        try:
            if self._tb is not None:
                self._tb.add_text(tag, text, env_step)
        except Exception as e:  # noqa: BLE001
            self._disable(e)

    def maybe_record(
        self, env_step: int, policy: Callable[[torch.Tensor], torch.Tensor], *, force: bool = False
    ) -> Path | None:
        """Record a rollout video when ``video_every_min`` has elapsed since the last one, or when ``force`` (call it
        with ``force=True`` at every checkpoint). Returns the first video's path, or None when nothing was recorded.
        Blocks the caller for the duration of the recording (a few seconds, every ~10 minutes)."""
        if self.viz_disabled or self._closed:
            return None
        try:
            every = self.cfg.video_every_min
            now = time.monotonic()
            due = force or (every > 0 and now - self._last_video_t >= every * 60.0)
            if not due or (every <= 0 and not force):
                return None
            if self._rollout_env is None:
                self._warn_once("rollout", "maybe_record: no rollout env attached (attach_rollout_env); no video")
                return None
            from so101_rl.viz.video import record_rollout

            n = int(getattr(self._rollout_env, "nworld", self.cfg.video_worlds))
            worlds = list(range(min(self.cfg.video_worlds, n)))
            res = record_rollout(
                self._rollout_env, policy, env_step=env_step, out_dir=self.run_dir / "videos", worlds=worlds,
                seconds=self.cfg.video_seconds,
            )
            self._last_video_t = time.monotonic()
            self.last_video = res.paths[0]
            if self._tb is not None and res.tb_frames is not None:
                import torch

                vid = torch.from_numpy(res.tb_frames).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]
                self._tb.add_video("rollout/world0", vid, env_step, fps=10)
            return res.paths[0]
        except Exception as e:  # noqa: BLE001
            self._disable(e)
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._teardown()
        except Exception as e:  # noqa: BLE001
            log.error("viz: close failed: %s", e)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


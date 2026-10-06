"""Demo / smoke entry point for the training visualisation (``pixi run viz-demo`` / ``python -m so101_rl.viz.demo``).

A random-walk "policy" on a small reach-like env over :class:`~so101_rl.sim.warp_model.WarpSim`, driving the real
:class:`~so101_rl.viz.monitor.TrainingMonitor`. It stands in for ``rl/train_sim_sac.py`` (not written yet) and is
what the V.1 / V.2 acceptance commands run::

    python -m so101_rl.viz.demo --nworld 1024 --viewer --seconds 60        # window: worlds 0, best, worst
    python -m so101_rl.viz.demo --record 10 --out runs/viz-demo/           # >= 2 mp4s + TensorBoard scalars
    python -m so101_rl.viz.demo --nworld 1024 --overhead --seconds 20      # steps/s with / without the publisher

Never opens a serial device. ``MUJOCO_GL=egl`` is selected here (before mujoco is imported) so videos render headless.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from so101_rl.viz.video import RolloutStatus

SUBSTEPS_PER_TICK = 51  # 0.1 s of sim time at dt = 1/510 s: one 10 Hz policy step
EPISODE_STEPS = 50
SUCCESS_RADIUS_M = 0.03


class RandomWalkEnv:
    """Reach-like env: random goal, ctrl += 0.15 * action, return = -distance. Implements ``RolloutEnv``."""

    def __init__(self, nworld: int, *, seed: int = 0) -> None:
        from so101_rl.config import SimWarpConfig
        from so101_rl.sim.model import load_model
        from so101_rl.sim.warp_model import WarpSim

        self.model, _ = load_model()
        self.sim = WarpSim(self.model, SimWarpConfig(), nworld)
        self.nworld, self.device = nworld, self.sim.device
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        m = self.model
        self.lo = torch.as_tensor(m.actuator_ctrlrange[:, 0], dtype=torch.float32, device=self.device)
        self.hi = torch.as_tensor(m.actuator_ctrlrange[:, 1], dtype=torch.float32, device=self.device)
        self.rest_q = torch.as_tensor(m.key("rest").qpos, dtype=torch.float32, device=self.device)
        self.rest_ctrl = torch.as_tensor(m.key("rest").ctrl, dtype=torch.float32, device=self.device)
        self.goal = torch.zeros(nworld, 3, device=self.device)
        self.ep_step = torch.zeros(nworld, dtype=torch.long, device=self.device)
        self.ep_return = torch.zeros(nworld, device=self.device)
        self.success = torch.zeros(nworld, dtype=torch.bool, device=self.device)
        self.dr_stage = 0
        self._reset(torch.ones(nworld, dtype=torch.bool, device=self.device))
        # stagger the episodes so 'best' / 'worst' differ from the first step (rollout videos reset() fully)
        self.ep_step = torch.randint(0, EPISODE_STEPS, (nworld,), device=self.device, generator=self.gen)

    def _reset(self, mask: torch.Tensor) -> None:
        u = torch.rand(self.nworld, 3, device=self.device, generator=self.gen)
        goal = torch.tensor([0.10, -0.15, 0.03], device=self.device) + u * torch.tensor([0.15, 0.30, 0.12], device=self.device)
        self.goal = torch.where(mask[:, None], goal, self.goal)
        self.ep_step = torch.where(mask, torch.zeros_like(self.ep_step), self.ep_step)
        self.ep_return = torch.where(mask, torch.zeros_like(self.ep_return), self.ep_return)
        self.success = self.success & ~mask
        self.sim.reset_worlds(mask, self.rest_q, torch.zeros(self.model.nv, device=self.device), self.rest_ctrl)

    def reset(self) -> torch.Tensor:
        self._reset(torch.ones(self.nworld, dtype=torch.bool, device=self.device))
        return self._obs()

    def _obs(self) -> torch.Tensor:
        return self.sim.qpos[:, :6].clone()

    def step(self, action: torch.Tensor) -> torch.Tensor:
        ctrl = (self.sim.ctrl + 0.15 * action.clamp(-1, 1)).clamp(self.lo, self.hi)
        self.sim.ctrl.copy_(ctrl)
        self.sim.step(SUBSTEPS_PER_TICK)
        dist = (self.sim.site_xpos("gripper_tip") - self.goal).norm(dim=1)
        self.ep_return += -dist
        self.success = dist < SUCCESS_RADIUS_M
        self.ep_step += 1
        self._reset(self.ep_step >= EPISODE_STEPS)
        return self._obs()

    # --- RolloutEnv ---------------------------------------------------------------------------------------------
    def qpos(self, worlds: Sequence[int]) -> np.ndarray:
        return self.sim.qpos[list(worlds)].cpu().numpy().astype(np.float64)

    def status(self, worlds: Sequence[int]) -> RolloutStatus:
        w = list(worlds)
        return RolloutStatus(self.success[w].cpu().numpy(), self.ep_return[w].cpu().numpy(), self.dr_stage)

    def close(self) -> None:
        self.sim.close()


def random_policy(obs: torch.Tensor) -> torch.Tensor:
    return torch.rand(obs.shape[0], 6, device=obs.device) * 2 - 1


def _render_cfg() -> Any:
    from so101_rl.config import load_calibration, load_config

    try:
        return load_config()
    except Exception:  # noqa: BLE001 - no calibration in ~/.cache: use the repo's test calibration
        return load_config(calibration=load_calibration(Path(__file__).resolve().parents[3] / "tests" / "data" / "so101_follower.calib.json"))


def run_loop(env: RandomWalkEnv, mon: Any, seconds: float, *, log_every_s: float = 2.0) -> tuple[int, float]:
    """Step the env for ``seconds``; returns (steps, steps/s)."""
    obs = env._obs()  # keep the staggered episodes
    ctx = mon.context if mon is not None else None
    t0 = time.perf_counter()
    last_log = t0
    steps = 0
    while time.perf_counter() - t0 < seconds:
        obs = env.step(random_policy(obs))
        steps += 1
        if mon is not None:
            ctx.goal, ctx.episode_step, ctx.tracker_pos = env.goal, env.ep_step, env.sim.body_xpos("die")
            mon.on_step(steps, success=env.success, ep_return=env.ep_return)
            if time.perf_counter() - last_log > log_every_s:
                last_log = time.perf_counter()
                mon.on_metrics(steps, {"success_rate": float(env.success.float().mean()), "mean_return": float(env.ep_return.mean())})
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    return steps, steps / el


def measure_overhead(nworld: int, seconds: float, repeats: int) -> dict[str, float]:
    """Training steps/s without and with the publisher (no window), alternating, median of ``repeats``."""
    from so101_rl.viz.monitor import TrainingMonitor, VizConfig

    env = RandomWalkEnv(nworld)
    cfg = _render_cfg()
    import tempfile

    run_dir = Path(tempfile.mkdtemp(prefix="viz-overhead-"))
    run_loop(env, None, 2.0)  # warm-up (graph capture, allocator)
    base, with_pub = [], []
    for i in range(repeats):
        _, a = run_loop(env, None, seconds)
        mon = TrainingMonitor(VizConfig(live_viewer=False, video_every_min=0, tensorboard=False), env.sim, run_dir / f"r{i}", task="reach", render_cfg=cfg)
        mon._make_publisher(spawn=False)  # publisher only: no window, and always "active" (grace period)
        mon._publisher._grace_until = float("inf")
        _, b = run_loop(env, mon, seconds)
        pub = mon._publisher.published if mon._publisher else 0
        mon.close()
        base.append(a)
        with_pub.append(b)
        print(f"repeat {i}: without {a:7.1f} steps/s   with publisher {b:7.1f} steps/s   (published {pub})", flush=True)
    env.close()
    mb, mp = statistics.median(base), statistics.median(with_pub)
    return {"base_steps_per_s": mb, "pub_steps_per_s": mp, "overhead_pct": 100.0 * (mb - mp) / mb}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="so101_rl.viz.demo")
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=20.0, help="how long to run the training stand-in")
    ap.add_argument("--viewer", action="store_true", help="open the live 3D viewer (needs a display)")
    ap.add_argument("--record", type=float, default=0.0, metavar="SECONDS", help="also record rollout videos of this length")
    ap.add_argument("--out", type=Path, default=Path("runs/viz-demo"))
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--overhead", action="store_true", help="measure the publisher's training slowdown and exit")
    ap.add_argument("--repeats", type=int, default=5)
    ns = ap.parse_args(argv)
    if ns.nworld > 1024:
        ap.error("--nworld is capped at 1024 (shared GPU)")

    if ns.overhead:
        res = measure_overhead(ns.nworld, ns.seconds, ns.repeats)
        print(f"median without {res['base_steps_per_s']:.1f} steps/s, with publisher {res['pub_steps_per_s']:.1f} steps/s: overhead {res['overhead_pct']:+.2f} %")
        return 0

    from so101_rl.viz.monitor import TrainingMonitor, VizConfig

    env = RandomWalkEnv(ns.nworld)
    video_env = RandomWalkEnv(4, seed=1) if ns.record > 0 else None
    cfg = VizConfig(live_viewer=ns.viewer, video_every_min=0, video_seconds=ns.record or 10.0, tensorboard=not ns.no_tensorboard)
    mon = TrainingMonitor(cfg, env.sim, ns.out, task="reach", render_cfg=_render_cfg())
    try:
        if video_env is not None:
            mon.attach_rollout_env(video_env)
        steps, sps = run_loop(env, mon, ns.seconds)
        print(f"{steps} steps, {sps:.1f} steps/s on {ns.nworld} worlds")
        if video_env is not None:
            p = mon.maybe_record(steps, random_policy, force=True)
            vids = sorted((ns.out / "videos").glob("*.mp4"))
            print("videos:", *[str(v) for v in vids], sep="\n  ")
            if p is None or not vids:
                print(f"no video written (viz_disabled={mon.viz_disabled}: {mon.error})", file=sys.stderr)
                return 1
        print(f"viz_disabled={mon.viz_disabled}")
    finally:
        mon.close()
        env.close()
        if video_env is not None:
            video_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

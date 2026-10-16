"""Sim evaluation and the numeric exit gate for reach and die_to_zone (sim plan S4.3 step 3 / S4.6, MVP M4).

``pixi run eval-sim --ckpt PATH --task {reach,die_to_zone} --episodes 1000 --dr-stage 4 --render 16
--out runs/eval-sim/<ts>/ [--gate]``

Every world runs exactly one deterministic episode per round (worlds that finish early are not re-counted), so
the success rate is not biased toward short (successful) episodes. Writes ``eval.json`` (success rate, Wilson 95 %
interval, final error p50/p90, trips (``trip_episodes`` and per cause ``trip_causes``), the DR stage and its ranges, the gate verdict), ``review_checklist.md`` (the
human part of the gate) and, with ``--render K``, ``renders/*.mp4`` through ``viz/video.py``.

**Exit gate** (plan §5 S4, §10 D-EXIT-GATE): ``success >= 0.95`` over ``>= 1000`` episodes at the final DR stage
with the Wilson 95 % lower bound ``>= 0.93`` -> ``numeric_gate_passed``; the run passes the gate only when a human
also ticks the rendered-rollout checklist (recorded elsewhere; this module never marks it). ``--gate`` makes the CLI
exit 1 unless the numeric part passes. The CLI exits 2 when ``eval.json`` was not written.

Reach uses no camera (state observation): ``camera_pose_method`` is recorded as ``"none"``. The die task sees the
die through the simulated tracker on the ``calib/camera.yaml`` pose (``BatchedGoalEnv`` refuses a provisional pose),
so its ``camera_pose_method`` is that pose's method. Renders use the ``calib/camera.yaml`` pose for the D435 view (a
provisional pose is fine for a picture) and record its source.

Tasks (``TASKS``): ``reach`` (``env.tasks.REACH``) and ``die_to_zone`` (``env.tasks.DIE_TO_ZONE``, plans/
die-to-zone-plan.md D1; same gate numbers, DR stages and env factory). ``success_radius_mm`` (plan §10
2026-10-14) replaces REACH's 15 mm ball only; the die task refuses it (its predicate is the zone square).
"""

from __future__ import annotations

import os
import sys

if "mujoco" not in sys.modules:  # headless renders (viz/video.py); must precede the first mujoco import
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import hashlib
import dataclasses
import json
import math
import subprocess
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from so101_rl.safety.types import TripCause

__all__ = [
    "GATE_FINAL_STAGE",
    "GATE_MIN_EPISODES",
    "GATE_MIN_SUCCESS",
    "GATE_MIN_WILSON_LO",
    "EvalResult",
    "build_env",
    "evaluate",
    "gate_verdict",
    "run_episodes",
    "task_spec",
    "wilson_interval",
]

GATE_MIN_SUCCESS: Final[float] = 0.95
GATE_MIN_WILSON_LO: Final[float] = 0.93
GATE_MIN_EPISODES: Final[int] = 1000
GATE_FINAL_STAGE: Final[int] = 4  # the last entry of sim/domain_rand.REACH_DR_STAGES (test-enforced)
TASKS: Final[tuple[str, ...]] = ("reach", "die_to_zone")  # die_to_zone: plans/die-to-zone-plan.md D1
# per-cause counts of the episodes ended by an emulated trip (eval.json ``trip_causes``; env.tasks.TERMINAL_TRIPS)
TRIP_CAUSES: Final[tuple[tuple[str, TripCause], ...]] = (
    ("nan", TripCause.NAN), ("range", TripCause.RANGE), ("box_present", TripCause.BOX_PRESENT), ("stuck", TripCause.STUCK),
)
CAMERA_POSE_METHOD_REACH: Final[str] = "none"  # reach observation is state only; no camera in the policy loop
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class EvalResult:
    success_rate: float
    wilson_lo: float
    wilson_hi: float
    episodes: int
    numeric_gate_passed: bool
    eval_json: Path
    renders: tuple[Path, ...]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (``(0, 1)`` for ``n == 0``)."""
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside 0..{n}")
    p = successes / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def gate_verdict(successes: int, episodes: int, dr_stage: int) -> tuple[bool, list[str]]:
    """The numeric exit gate; returns ``(passed, reasons it failed)``."""
    lo, _ = wilson_interval(successes, episodes)
    rate = successes / episodes if episodes else 0.0
    reasons = []
    if episodes < GATE_MIN_EPISODES:
        reasons.append(f"episodes {episodes} < {GATE_MIN_EPISODES}")
    if dr_stage < GATE_FINAL_STAGE:
        reasons.append(f"dr_stage {dr_stage} < final stage {GATE_FINAL_STAGE}")
    if rate < GATE_MIN_SUCCESS:
        reasons.append(f"success {rate:.4f} < {GATE_MIN_SUCCESS}")
    if lo < GATE_MIN_WILSON_LO:
        reasons.append(f"wilson_lo {lo:.4f} < {GATE_MIN_WILSON_LO}")
    return not reasons, reasons


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ------------------------------------------------------------------------------------------------ env factory
def build_env(
    task: str,
    nworld: int,
    *,
    stage: int,
    seed: int,
    calibration: str | None = None,
    sim_params: str = "calib/sim_params.yaml",
    device: str | None = None,
    success_radius_mm: float | None = None,
) -> Any:
    """The batched env of ``task`` with the DR stages around ``sim_params`` (the v0 MJCF prior is accepted for
    reach, plan S4.5; its release is recorded). Needs CUDA, or ``device="cpu"`` (CPU MuJoCo through
    ``sim/cpu_sim.py``, for a few worlds). The training loop builds its envs through this function too (tests
    replace it with the toy env). ``success_radius_mm`` replaces REACH's 15 mm success radius
    (plan §10 2026-10-14; None = the task's); refused for the die task (:func:`task_spec`)."""
    base_spec = task_spec(task, success_radius_mm)
    dev = device or "cuda:0"
    if torch.device(dev).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the batched sim env needs CUDA (MuJoCo Warp)")
    from so101_rl.config import load_calibration, load_config
    from so101_rl.env.batched_env import BatchedGoalEnv
    from so101_rl.sim.compare_real import MAX_RMS_PCT
    from so101_rl.sim.domain_rand import DomainRandomizer
    from so101_rl.sim.model import load_model
    from so101_rl.sim.sysid.params import read_sim_params_file

    if calibration is None:
        cal_path = load_config().calibration_path  # so101.yaml calibration_path (the real arm's calibration)
    else:
        cal_path = Path(calibration)
    cal = load_calibration(cal_path)
    cfg = load_config(calibration=cal)
    spf = read_sim_params_file(Path(sim_params), allow_unverified=True)
    model = load_model()[0]
    rnd = DomainRandomizer(spf.params, spf.dr, nworld, stage=stage, seed=seed + 1, device=dev)
    env = BatchedGoalEnv(cfg, cal, base_spec, nworld, sim_params=spf.params, randomizer=rnd, model=model, seed=seed, device=dev)
    env.metadata.update({
        "calibration": str(cal_path), "calibration_sha256": _sha256(cal_path), "sim_params": str(sim_params), "sim_params_release": spf.meta.version,
        "sim_params_sha256": _sha256(Path(sim_params)),
        "sim_params_verified": not spf.meta.verification_problems(MAX_RMS_PCT), "success_radius_m": base_spec.success_radius_m,
    })  # fmt: skip
    if not base_spec.obs.has_die:  # die tasks keep the env's own camera_pose_method (the calib/camera.yaml pose)
        env.metadata["camera_pose_method"] = CAMERA_POSE_METHOD_REACH
    return env


def task_spec(task: str, success_radius_mm: float | None = None) -> Any:
    """The FS4 :class:`~so101_rl.env.tasks.TaskSpec` of ``task`` (``TASKS``). ``success_radius_mm`` replaces REACH's
    15 mm radius (plan §10 2026-10-14); the die task refuses it (its success is the zone square, not a radius).
    Raises ``ValueError``."""
    from so101_rl.env.tasks import DIE_TO_ZONE, REACH

    if task not in TASKS:
        raise ValueError(f"task {task!r} is not a sim task (only {TASKS})")
    if task == "die_to_zone":
        if success_radius_mm is not None:
            raise ValueError("success_radius_mm is a reach-only override (plan §10 2026-10-14); die_to_zone success "
                             "is the zone square (env/tasks.py), not a radius")  # fmt: skip
        return DIE_TO_ZONE
    if success_radius_mm is None:
        return REACH
    return dataclasses.replace(REACH, success_radius_m=success_radius_mm / 1000.0)


def policy_input(env: Any, obs: torch.Tensor) -> torch.Tensor:
    """FS3 policy input: the observation with the CURRENT desired goal appended."""
    return torch.cat([obs, env.goal], dim=1)


@torch.no_grad()
def run_episodes(env: Any, policy: Callable[[torch.Tensor], torch.Tensor], episodes: int) -> dict[str, Any]:
    """Run ``episodes`` deterministic episodes: rounds of one episode per world (full reset each round, no phase
    randomisation), counting each world's FIRST episode only. Returns host statistics."""
    n = int(env.nworld)
    horizon = int(getattr(getattr(env, "task", None), "episode_steps", getattr(env, "episode_steps", 50)))
    succ, err, ret, length, trips, causes = [], [], [], [], [], []
    left = int(episodes)
    while left > 0:
        k = min(n, left)
        obs = env.reset(phase_randomise=False)[0]
        done = torch.zeros(n, dtype=torch.bool, device=env.device)
        r_acc = torch.zeros(n, device=env.device)
        s_ep = torch.zeros(n, dtype=torch.bool, device=env.device)
        e_ep = torch.zeros(n, device=env.device)
        l_ep = torch.zeros(n, dtype=torch.int64, device=env.device)
        t_ep = torch.zeros(n, dtype=torch.bool, device=env.device)
        c_ep = torch.zeros(n, dtype=torch.int64, device=env.device)  # trip bits of the step that ended the episode
        for t in range(horizon):
            st = env.step(policy(policy_input(env, obs)))
            live = ~done
            r_acc += torch.where(live, st.reward.float(), torch.zeros_like(r_acc))
            end = live & (st.terminated | st.truncated)
            s_ep |= end & st.info["success"]  # terminated also covers a terminal trip (plan §10 2026-10-08)
            fe = (st.final_achieved - st.desired).norm(dim=1)
            e_ep = torch.where(end, fe, e_ep)
            l_ep = torch.where(end, torch.full_like(l_ep, t + 1), l_ep)
            tripped = end & st.info["tripped"]
            t_ep |= tripped
            step_trips = st.info.get("trips")
            if step_trips is not None:
                c_ep = torch.where(tripped, step_trips.to(torch.int64), c_ep)
            done |= end
            obs = st.obs
        succ.append(s_ep[:k].cpu())
        err.append(e_ep[:k].cpu())
        ret.append(r_acc[:k].cpu())
        length.append(l_ep[:k].cpu())
        trips.append(t_ep[:k].cpu())
        causes.append(c_ep[:k].cpu())
        left -= k
    s = torch.cat(succ).numpy()
    e = torch.cat(err).double().numpy()
    cause = torch.cat(causes)
    return {
        "episodes": int(s.size),
        "successes": int(s.sum()),
        "success_rate": float(s.mean()) if s.size else 0.0,
        "final_err_m": {"p50": float(np.percentile(e, 50)), "p90": float(np.percentile(e, 90))},
        "mean_return": float(torch.cat(ret).mean()),
        "mean_length": float(torch.cat(length).double().mean()),
        "trip_episodes": int(torch.cat(trips).sum()),
        "trip_causes": {name: int(((cause & int(c)) != 0).sum()) for name, c in TRIP_CAUSES},
        "returns": torch.cat(ret).numpy(),
    }


# ------------------------------------------------------------------------------------------------ checkpoints
def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        d = torch.load(Path(path), map_location="cpu", weights_only=True)
    except OSError:
        raise
    except Exception as exc:
        raise ValueError(f"{path}: not a loadable checkpoint ({type(exc).__name__}: {exc})") from exc
    if not isinstance(d, dict) or d.get("format") != "so101_rl.sim_sac_ckpt":
        raise ValueError(f"{path}: not a sim SAC checkpoint")
    return d


def actor_from_checkpoint(d: dict[str, Any], device: torch.device | str = "cpu") -> torch.nn.Module:
    from so101_rl.rl.fast_sac import Actor

    sac = d["sac"]
    obs_dim, _priv, act_dim = sac["dims"]
    actor = Actor(obs_dim, act_dim, tuple(sac["cfg"]["actor_hidden"]), tuple(sac["cfg"]["log_std_bounds"]))
    actor.load_state_dict(sac["actor"])
    return actor.to(device).eval()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CAL_CHANGED_MSG = ("the calibration file changed since this policy was trained (sha256 differs): the sim it was "
                   "trained in had other joint limits / zeros; re-evaluate before trusting it on the arm")


def calibration_changed(trained: str | None, now: str | None) -> bool | None:
    """True/False when both sha256 are known; None for a policy trained before the hash was recorded."""
    if not trained or not now:
        return None
    return trained != now


class _RenderEnv:
    """``viz.video.RolloutEnv`` over a small batched env: the policy input is FS3 obs + goal.

    Render semantics: one episode per world, no auto-reset. The wrapped env auto-resets a world the step its episode
    ends (new pose, new goal), which would lose the terminal frame; reset is suppressed during ``step`` and the
    return / flags stop updating once a world has ended, so the overlay return is the episode return.
    """

    def __init__(self, env: Any) -> None:
        self.env, self.nworld = env, env.nworld
        self._ret = torch.zeros(env.nworld, device=env.device)
        self._succ = torch.zeros(env.nworld, dtype=torch.bool, device=env.device)
        self._trunc = torch.zeros(env.nworld, dtype=torch.bool, device=env.device)

    def reset(self) -> torch.Tensor:
        obs = self.env.reset(phase_randomise=False)[0]
        self._ret.zero_()
        self._succ.zero_()
        self._trunc.zero_()
        return policy_input(self.env, obs)

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self.env.reset = lambda *a, **k: None  # type: ignore[method-assign]  # instance attr shadows the method
        try:
            st = self.env.step(action)
        finally:
            self.env.__dict__.pop("reset", None)
        live = ~(self._succ | self._trunc)
        self._ret += torch.where(live, st.reward.float(), torch.zeros_like(self._ret))
        self._succ |= live & st.info["success"]
        self._trunc |= live & (st.terminated | st.truncated) & ~st.info["success"]
        return policy_input(self.env, st.obs)

    def qpos(self, worlds: Sequence[int]) -> np.ndarray:
        return self.env.ws.qpos[list(worlds)].detach().cpu().double().numpy()

    def status(self, worlds: Sequence[int]) -> Any:
        from so101_rl.viz.video import RolloutStatus

        w = list(worlds)
        return RolloutStatus(
            self._succ[w].cpu().numpy(), self._ret[w].cpu().numpy(), self.env.dr_stage,
            terminated=self._succ[w].cpu().numpy(), truncated=self._trunc[w].cpu().numpy(),
            goal=self.env.goal[w].detach().cpu().double().numpy(), success_radius=float(self.env.task.success_radius_m),
        )  # fmt: skip


CHECKLIST: Final[str] = """# Reach exit gate: human review checklist

Run: `{ckpt}` - eval `{eval_json}` - DR stage {stage} - {episodes} episodes - success {rate:.4f}
(Wilson 95 % [{lo:.4f}, {hi:.4f}]) - numeric gate: **{verdict}**

Watch every video in `renders/` and tick each item. The exit gate passes only with the numeric part AND every
tick (plan §5 S4); the AI may review first, a human signs (`pixi run lab gate-sign <run>`).

- [ ] The tip reaches the goal and stops there (no oscillating in and out of the 15 mm ball).
- [ ] No hovering at the edge of the success radius to farm the per-step reward.
- [ ] No contact with the table or the tray; no gripper dragged along the table.
- [ ] Motions are smooth: no saturated bang-bang joint commands, no shaking at rest.
- [ ] No unusual poses (e.g. elbow flipped, wrist twisted) that would be unsafe on the real arm.
- [ ] The overlay success flag agrees with what the video shows.
"""

DIE_CHECKLIST: Final[str] = """# Die-to-zone exit gate: human review checklist

Run: `{ckpt}` - eval `{eval_json}` - DR stage {stage} - {episodes} episodes - success {rate:.4f}
(Wilson 95 % [{lo:.4f}, {hi:.4f}]) - numeric gate: **{verdict}**

Watch every video in `renders/` and tick each item. The exit gate passes only with the numeric part AND every
tick (plan §5 S4, sim plan S4.6); the AI may review first, a human signs (`pixi run lab gate-sign <run>`).

- [ ] The die ends at rest inside the 60 mm zone square (push and place both count; note which one, S4.6).
- [ ] No predicate gaming: the die is not held or hovered just under the 16 mm height bound, and success is not
      flagged while the die slides through the zone or the gripper still pushes it.
- [ ] No exploiting the tracker: success is never flagged while the die is occluded (by the gripper or the arm).
- [ ] No contact with the table or the tray beyond pushing the die; no gripper dragged along the table.
- [ ] The die is not flung, dropped from height or knocked off the table.
- [ ] Motions are smooth: no saturated bang-bang joint commands, no shaking at rest.
- [ ] No unusual poses (e.g. elbow flipped, wrist twisted) that would be unsafe on the real arm.
- [ ] The overlay success flag agrees with what the video shows.
"""
CHECKLISTS: Final[dict[str, str]] = {"reach": CHECKLIST, "die_to_zone": DIE_CHECKLIST}


def evaluate(ckpt: Path, task: str, *, episodes: int, dr_stage: int, out_dir: Path, render: int = 0) -> EvalResult:
    """Deterministic evaluation of ``ckpt`` on ``episodes`` episodes at ``dr_stage``; writes ``eval.json``,
    ``review_checklist.md`` and ``render`` videos under ``out_dir``. See the module docstring."""
    if task not in TASKS:
        raise ValueError(f"task {task!r} not supported (only {TASKS})")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_checkpoint(Path(ckpt))
    if d["task"] != task:
        raise ValueError(f"checkpoint task {d['task']!r} != {task!r}")
    run_cfg = d.get("config", {})
    t0 = time.monotonic()
    env = build_env(task, min(int(episodes), int(run_cfg.get("nworld", 1024)), 1024), stage=int(dr_stage),
                    seed=int(run_cfg.get("seed", 0)) + 7777, calibration=run_cfg.get("calibration"),
                    sim_params=run_cfg.get("sim_params", "calib/sim_params.yaml"),
                    success_radius_mm=run_cfg.get("success_radius_mm"))  # fmt: skip
    try:
        actor = actor_from_checkpoint(d, env.device)
        stats = run_episodes(env, actor, episodes)
        dr = env.randomizer.describe() if getattr(env, "randomizer", None) is not None else {"stage": int(dr_stage)}
        meta = dict(getattr(env, "metadata", {}))
    finally:
        env.close()
    cal_changed = calibration_changed(d.get("calibration_sha256"), meta.get("calibration_sha256"))
    if cal_changed:
        warnings.warn(f"{ckpt}: {CAL_CHANGED_MSG} ({meta.get('calibration')})", stacklevel=2)
    renders: list[Path] = []
    render_src = None
    if render > 0:
        renders, render_src = _render(d, actor, task, int(render), int(dr_stage), run_cfg, out_dir)
    lo, hi = wilson_interval(stats["successes"], stats["episodes"])
    passed, reasons = gate_verdict(stats["successes"], stats["episodes"], int(dr_stage))
    doc = {
        "task": task, "ckpt": str(ckpt), "ckpt_sha256": _sha256(Path(ckpt)), "env_step": int(d.get("env_step", 0)),
        "episodes": stats["episodes"], "successes": stats["successes"], "success_rate": stats["success_rate"],
        "wilson95": [lo, hi], "final_err_m": stats["final_err_m"], "mean_return": stats["mean_return"],
        "mean_length": stats["mean_length"], "trip_episodes": stats["trip_episodes"],
        "trip_causes": stats["trip_causes"], "dr_stage": int(dr_stage), "dr": dr,
        "gate": {"min_success": GATE_MIN_SUCCESS, "min_wilson_lo": GATE_MIN_WILSON_LO, "min_episodes": GATE_MIN_EPISODES,
                 "final_stage": GATE_FINAL_STAGE, "numeric_passed": passed, "reasons": reasons,
                 "human_checklist": "pending (review_checklist.md; signed outside this tool)"},
        "camera_pose_method": meta.get("camera_pose_method", CAMERA_POSE_METHOD_REACH), "render_camera_pose": render_src,
        "calibration_changed_since_training": cal_changed,
        "renders": [str(p) for p in renders], "env": meta, "git_sha": git_sha(), "wall_s": time.monotonic() - t0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }  # fmt: skip
    eval_json = out_dir / "eval.json"
    tmp = eval_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(eval_json)
    (out_dir / "review_checklist.md").write_text(CHECKLISTS[task].format(
        ckpt=ckpt, eval_json=eval_json, stage=dr_stage, episodes=stats["episodes"], rate=stats["success_rate"], lo=lo,
        hi=hi, verdict="PASS" if passed else "FAIL (" + "; ".join(reasons) + ")"))  # fmt: skip
    return EvalResult(stats["success_rate"], lo, hi, stats["episodes"], passed, eval_json, tuple(renders))


def _render(d: dict, actor: Any, task: str, k: int, stage: int, run_cfg: dict, out_dir: Path) -> tuple[list[Path], str]:
    from so101_rl.viz.video import d435_camera_pose, record_rollout

    env = build_env(task, k, stage=stage, seed=int(run_cfg.get("seed", 0)) + 8888, calibration=run_cfg.get("calibration"),
                    sim_params=run_cfg.get("sim_params", "calib/sim_params.yaml"),
                    success_radius_mm=run_cfg.get("success_radius_mm"))  # fmt: skip
    try:
        actor = actor.to(env.device)
        res = record_rollout(_RenderEnv(env), actor, env_step=int(d.get("env_step", 0)), out_dir=out_dir / "renders",
                             worlds=list(range(k)), seconds=env.task.episode_steps / 10.0, keep_tb_frames=False)  # fmt: skip
    finally:
        env.close()
    return res.paths, d435_camera_pose()[1]


def rerender(ckpt: Path, task: str, *, dr_stage: int, render: int, out_dir: Path) -> list[Path]:
    """Re-render the gate-review videos of an already evaluated ``ckpt`` (same seeds / worlds as ``evaluate``) into
    ``out_dir/renders``, WITHOUT a new evaluation: no ``eval.json`` is written or touched."""
    d = load_checkpoint(Path(ckpt))
    if d["task"] != task:
        raise ValueError(f"checkpoint task {d['task']!r} != {task!r}")
    actor = actor_from_checkpoint(d, "cpu")
    return _render(d, actor, task, int(render), int(dr_stage), d.get("config", {}), Path(out_dir))[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval-sim", description="Sim evaluation + numeric exit gate (reach, die_to_zone)")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--task", choices=TASKS, default="reach")
    ap.add_argument("--episodes", type=int, default=GATE_MIN_EPISODES)
    ap.add_argument("--dr-stage", type=int, default=GATE_FINAL_STAGE)
    ap.add_argument("--render", type=int, default=0, help="number of rollout videos (renders/*.mp4)")
    ap.add_argument("--out", type=Path, default=None, help="default runs/eval-sim/<task>-<timestamp>/")
    ap.add_argument("--render-only", action="store_true", help="only (re-)render --render videos of --ckpt into --out/renders")
    ap.add_argument("--gate", action="store_true", help="exit 1 unless the numeric exit gate passes")
    a = ap.parse_args(argv)
    out = a.out or Path("runs/eval-sim") / f"{a.task}-{time.strftime('%Y%m%d-%H%M%S')}"
    if a.render_only:
        paths = rerender(a.ckpt, a.task, dr_stage=a.dr_stage, render=a.render, out_dir=out)
        print(f"rendered {len(paths)} videos -> {out / 'renders'}")
        return 0
    res = evaluate(a.ckpt, a.task, episodes=a.episodes, dr_stage=a.dr_stage, out_dir=out, render=a.render)
    if not res.eval_json.exists():
        print(f"eval-sim: {res.eval_json} missing", file=sys.stderr)
        return 2
    print(f"success {res.success_rate:.4f} over {res.episodes} (Wilson 95 % [{res.wilson_lo:.4f}, {res.wilson_hi:.4f}]) "
          f"numeric gate {'PASS' if res.numeric_gate_passed else 'FAIL'} -> {res.eval_json}")  # fmt: skip
    return 1 if a.gate and not res.numeric_gate_passed else 0


if __name__ == "__main__":
    sys.exit(main())

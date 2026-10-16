"""Point the gripper tip at a base-frame position and hold it there (e.g. to mark the die zone on the table).

Usage (servo started after this, as for deploy-policy; the real arm needs --attended):
    pixi run -e cpu python scripts/point_tip.py --x 0.15 --y -0.13 --z 0.035 --hold-s 300 --attended --accept-no-watchdog

The reach policy (default: the 3 mm sim export, plan §10 2026-10-16) drives the tip to the point as one reach
episode with deploy-policy's checks and episode loop (servo classification, attended / no-watchdog refusal, HOLD
keep-alive, ``wait_for_run``). After the episode it sends a zero-delta RUN command every policy step for ``--hold-s``
(the arm stays still), then stops sending: the servo's stale-command stop takes the arm back to rest. The
point must lie inside the reach task's goal bounds (z >= 0.03 m in the base frame, about 3.5 cm above the table).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_POLICY = Path("runs/sac-sim/reach-3mm-20261015-201549/policy.pt")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--z", type=float, default=0.035)
    ap.add_argument("--hold-s", type=float, default=300.0)
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--attended", action="store_true")
    ap.add_argument("--accept-no-watchdog", action="store_true")
    ap.add_argument("--wait-servo-s", type=float, default=120.0)
    args = ap.parse_args(argv)

    import torch

    from so101_rl.config.loader import DEFAULT_CONFIG_PATH
    from so101_rl.env.backend import ShmBackend
    from so101_rl.env.goal_env import So101GoalEnv
    from so101_rl.env.tasks import REACH
    from so101_rl.rl import deploy_policy as dp
    from so101_rl.rl.policy_export import load_policy

    torch.set_num_threads(1)
    goal = np.array([args.x, args.y, args.z], np.float64)
    policy = load_policy(args.policy)
    dp.check_policy(policy, REACH)
    radius_mm = policy.meta.get("success_radius_mm")
    task = REACH if radius_mm is None else dataclasses.replace(REACH, success_radius_m=float(radius_mm) / 1000.0)
    if not bool(task.goal_bounds.contains_np(goal[None])[0]):
        print(f"point_tip: refused: {goal.tolist()} is outside the reach goal bounds", file=sys.stderr)
        return 2
    shm_name = str(dp._yaml_key(DEFAULT_CONFIG_PATH, "shm_name"))
    try:
        rs = dp.resolve_servo(shm_name, wait_s=args.wait_servo_s, attended=args.attended,
                              accept_no_watchdog=args.accept_no_watchdog, config=None, calibration=None)  # fmt: skip
    except dp.Refused as exc:
        print(f"point_tip: refused: {exc}", file=sys.stderr)
        return 2
    backend = ShmBackend(rs.cfg, calibration=rs.calibration)
    with contextlib.ExitStack() as keep:
        keep.enter_context(dp.hold_keepalive(backend))
        goals = dp.GoalList(np.array([[goal, goal]]))
        env = So101GoalEnv(backend, rs.cfg, task, goal_sampler=goals)
    dp.wait_for_run(backend, args.wait_servo_s, lambda m: print(f"point_tip: {m}", flush=True))
    ep = dp._run_episode(env, policy, 1.0, goals=goals, index=0, infer_ms=[])
    print(f"point_tip: reach episode: success {ep['success']}, error {ep['final_error_m'] * 1000:.1f} mm, "
          f"{ep['steps']} steps, tripped {ep['tripped']}", flush=True)
    if ep["tripped"] or ep["servo_stalled"]:
        print("point_tip: the servo left RUN; not holding", file=sys.stderr)
        return 1
    print(f"point_tip: holding at {goal.tolist()} for {args.hold_s:.0f} s (Ctrl-C to stop early)", flush=True)
    t_end = time.monotonic() + args.hold_s
    zero = np.zeros(6, np.float32)
    try:
        while time.monotonic() < t_end:
            # a zero-delta RUN command per 10 Hz policy step, as a policy holding still sends. Not HOLD: in RUN a HOLD
            # command keeps the servo alive but the stuck monitor keeps seeing the last policy delta with the arm
            # still, and trips STUCK after 3 s (seen on the arm 2026-10-16)
            _, _, _, _, info = env.step(zero)
            if "trip" in info or info.get("servo_stalled", False):
                print(f"point_tip: the servo left RUN while holding (trip {info.get('trip')})", file=sys.stderr)
                return 1
    except KeyboardInterrupt:
        pass
    print("point_tip: done; the servo returns the arm to rest after its stale-command stop", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

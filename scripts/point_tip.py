"""Point the gripper tip at a base-frame position and hold it there (e.g. to mark the die zone on the table).

Usage (servo started after this, as for deploy-policy; the real arm needs --attended):
    pixi run -e cpu python scripts/point_tip.py --x 0.15 --y -0.13 --z 0.035 --hold-s 300 --attended --accept-no-watchdog

The reach policy (default: the 3 mm sim export, plan §10 2026-10-16) drives the tip to the point as one reach
episode with deploy-policy's checks and episode loop (servo classification, attended / no-watchdog refusal, HOLD
keep-alive, ``wait_for_run``; shared with ``hand-eye --execute`` in ``so101_rl/rl/tip_control.py``). After the episode it sends a zero-delta RUN command every policy step for ``--hold-s``
(the arm stays still), then stops sending: the servo's stale-command stop takes the arm back to rest. The
point must lie inside the reach task's goal bounds (z >= 0.03 m in the base frame, about 3.5 cm above the table).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from so101_rl.rl.tip_control import DEFAULT_REACH_POLICY as DEFAULT_POLICY


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

    from so101_rl.rl import deploy_policy as dp
    from so101_rl.rl import tip_control as tc

    torch.set_num_threads(1)
    goal = np.array([args.x, args.y, args.z], np.float64)
    policy, task = tc.load_reach_policy(args.policy)
    if not bool(task.goal_bounds.contains_np(goal[None])[0]):
        print(f"point_tip: refused: {goal.tolist()} is outside the reach goal bounds", file=sys.stderr)
        return 2
    try:
        rs = tc.resolve(attended=args.attended, accept_no_watchdog=args.accept_no_watchdog, wait_servo_s=args.wait_servo_s)
    except dp.Refused as exc:
        print(f"point_tip: refused: {exc}", file=sys.stderr)
        return 2
    ctl = tc.attach(rs, policy, task, wait_servo_s=args.wait_servo_s, log=lambda m: print(f"point_tip: {m}", flush=True))
    try:
        ep = ctl.reach(goal)
    except tc.ServoLeftRun as exc:
        print(f"point_tip: {exc}; not holding", file=sys.stderr)
        return 1
    print(f"point_tip: reach episode: success {ep['success']}, error {ep['final_error_m'] * 1000:.1f} mm, "
          f"{ep['steps']} steps", flush=True)
    print(f"point_tip: holding at {goal.tolist()} for {args.hold_s:.0f} s (Ctrl-C to stop early)", flush=True)
    try:
        ctl.hold(args.hold_s)  # zero-delta RUN commands, not HOLD (rl/tip_control.py)
    except tc.ServoLeftRun as exc:
        print(f"point_tip: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    print("point_tip: done; the servo returns the arm to rest after its stale-command stop", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

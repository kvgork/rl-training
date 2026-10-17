"""Drive the gripper tip to base-frame points with the reach policy and hold it still (``scripts/point_tip.py``,
``hand-eye --execute``).

The steps are deploy-policy's (``rl/deploy_policy.py``), in this order:

1. :func:`resolve` - find and classify the running servo (read-only, nothing is sent): the real arm is refused
   without ``attended``, a real servo with ``--no-mcu`` without ``accept_no_watchdog``.
2. (the caller's slow set-up, e.g. opening the camera, BEFORE the arm can move: plan §10 2026-10-17 "Perception
   order")
3. :func:`attach` - ``ShmBackend`` attach, the env built under the HOLD keep-alive, ``wait_for_run``.
4. :meth:`TipController.reach` - one reach episode to the point (the env's reset ramp, then the policy).
5. :meth:`TipController.hold` - a zero-delta RUN command every 10 Hz policy step. Not HOLD: in RUN a HOLD command
   keeps the servo alive but the stuck monitor keeps seeing the last policy delta with the arm still, and trips STUCK
   after 3 s (seen on the arm 2026-10-16).

When the caller stops commanding, the servo's stale-command stop takes the arm back to rest (nothing to send).
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_REACH_POLICY = Path("runs/sac-sim/reach-3mm-20261015-201549/policy.pt")  # plan §10 2026-10-16


class ServoLeftRun(RuntimeError):
    """The servo tripped / left RUN / stalled during a reach or a hold: stop, command nothing more."""


def load_reach_policy(path: Path) -> tuple[Any, Any]:
    """(policy, task): the reach export, checked like deploy-policy; the task's success radius is the export's
    ``success_radius_mm`` when recorded (the 3 mm policy). Raises ``deploy_policy.Refused``."""
    from so101_rl.env.tasks import REACH
    from so101_rl.rl import deploy_policy as dp
    from so101_rl.rl.policy_export import load_policy

    policy = load_policy(path)
    dp.check_policy(policy, REACH)
    radius_mm = policy.meta.get("success_radius_mm")
    task = REACH if radius_mm is None else dataclasses.replace(REACH, success_radius_m=float(radius_mm) / 1000.0)
    return policy, task


def resolve(*, attended: bool, accept_no_watchdog: bool, wait_servo_s: float, shm_name: str | None = None,
            log: Callable[[str], None] | None = None) -> Any:  # fmt: skip
    """``deploy_policy.resolve_servo`` on ``shm_name`` (default ``so101.yaml`` ``shm_name``). Read-only."""
    from so101_rl.config.loader import DEFAULT_CONFIG_PATH
    from so101_rl.rl import deploy_policy as dp

    name = shm_name if shm_name is not None else str(dp._yaml_key(DEFAULT_CONFIG_PATH, "shm_name"))
    return dp.resolve_servo(name, wait_s=wait_servo_s, attended=attended, accept_no_watchdog=accept_no_watchdog,
                            config=None, calibration=None, log=log)  # fmt: skip


@dataclasses.dataclass
class TipController:
    env: Any
    policy: Any
    goals: Any
    backend: Any
    task: Any
    policy_hz: float = 10.0

    def reach(self, xyz: np.ndarray) -> dict[str, Any]:
        """One reach episode to ``xyz`` (base frame; must lie in the task's goal bounds). Raises :class:`ServoLeftRun`
        if the servo left RUN."""
        from so101_rl.rl import deploy_policy as dp

        goal = np.asarray(xyz, np.float64).reshape(3)
        if not bool(self.task.goal_bounds.contains_np(goal[None])[0]):
            raise ValueError(f"{goal.tolist()} is outside the reach goal bounds")
        self.goals.pairs[0] = (goal, goal)
        ep = dp._run_episode(self.env, self.policy, 1.0, goals=self.goals, index=0, infer_ms=[])
        if ep["tripped"] or ep["servo_stalled"]:
            raise ServoLeftRun(f"the servo left RUN during the reach to {goal.tolist()} (sm_state {ep['sm_state']}, "
                               f"trips {ep['trips']:#x})")  # fmt: skip
        return ep

    def hold(self, seconds: float, on_step: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any] | None:
        """Hold still for ``seconds`` (at least one step) with zero-delta RUN commands; ``on_step(info)`` after each
        step. Returns the last step's info. Raises :class:`ServoLeftRun`."""
        zero = np.zeros(6, np.float32)
        steps = max(1, round(seconds * self.policy_hz))
        info: dict[str, Any] | None = None
        for _ in range(steps):
            _, _, _, _, info = self.env.step(zero)
            if "trip" in info or info.get("servo_stalled", False):
                raise ServoLeftRun(f"the servo left RUN while holding (trip {info.get('trip')})")
            if on_step is not None:
                on_step(info)
        return info


def attach(rs: Any, policy: Any, task: Any, *, wait_servo_s: float, log: Callable[[str], None] | None = None) -> TipController:
    """Attach to the resolved servo ``rs`` (:func:`resolve`), build the reach env under the HOLD keep-alive and wait
    for RUN (HOLD never moves the arm). Raises ``ServoNotRunning`` / ``ServoStateStale``."""
    from so101_rl.env.backend import ShmBackend
    from so101_rl.env.goal_env import So101GoalEnv
    from so101_rl.rl import deploy_policy as dp

    backend = ShmBackend(rs.cfg, calibration=rs.calibration)
    with contextlib.ExitStack() as keep:
        keep.enter_context(dp.hold_keepalive(backend))
        goals = dp.GoalList(np.zeros((1, 2, 3)))
        env = So101GoalEnv(backend, rs.cfg, task, goal_sampler=goals)
    dp.wait_for_run(backend, wait_servo_s, log)
    return TipController(env, policy, goals, backend, task, float(rs.cfg.rates.policy_hz))

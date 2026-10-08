"""Goal-conditioned Gymnasium env for the real arm and the CPU reference rig (FS12; supersedes F7's
``So101ReachEnv``).

One class for both tasks, both backends, and one definition of everything that sim and real must agree on:

* **Observation:** FS3, built by :func:`~so101_rl.env.obs.build_policy_obs_np` from the servo's state record
  (present ticks, Present_Speed, the post-clamp ``applied_delta_ticks``) plus the FK tip; the die block comes from
  the perception object table (:class:`~so101_rl.perception.object_table.ObjectTableShm`) when one is given:
  ``visible`` = the tracker reported the die visible in a detection no older than one policy period plus two
  camera frames (and the camera-moved fiducial is OK), ``die_xyz`` = the last visible position (held while
  invisible, zeros until first seen this episode), ``die_age`` = camera frames since it was last seen. Without an
  object table the die is never visible.
* **Action:** ``delta_ticks = rint(clip(action, -1, 1) * max_delta_ticks)`` (the F7 rule, the only scaling in the
  code base: :func:`tasks.action_to_delta_np`, bit-identical to the batched env's ``action_to_delta_torch``). A
  NaN passes through unchanged: the servo's sanitizer trips on it (the env never pre-filters what the safety chain
  must see); ±inf clip to ±1.
* **Reward / success / termination:** ``env/tasks.py`` (FS4): ``compute_reward`` = :func:`tasks.step_reward_np`
  with the visibility and the trip bits of the stored transition (:func:`tasks.visible_from_info`,
  ``info["trips"]``), and ``step`` uses the very same function, so SB3 HER relabelling reproduces the step reward
  exactly and never turns a trip into a success. ``terminated`` = the debounced (and, for the die, settled) success
  predicate on a step with a good bus read and no trip (``info["is_success"]``), OR a terminal trip
  (``tasks.TERMINAL_TRIPS`` in ``fault_bits``: NAN, RANGE, BOX_PRESENT, STUCK; ``info["tripped"]``), whose reward is
  ``task.trip_penalty`` (plan §10 2026-10-08 "Trip penalty").
* **Truncation:** the step limit, any ``fault_bits`` in the state record or the servo leaving RUN (``info["trip"]``
  = the fault bits; the episode then needs ``reset``), or a stalled servo (``info["servo_stalled"]``).
* **Stale state:** a record with ``read_ok = 0`` repeats the last good reading; it is still observed (flagged in
  ``info["read_ok"]``) but such a step never terminates an episode as a success. The reach task reports
  ``info["tracker_visible"] = read_ok`` (the achieved goal was measured this step).

``info`` keys every step: ``applied_delta_ticks`` f32[6], ``target_ticks`` i32[6], ``clamped_mask`` bool[6],
``trips`` (int, the state record's ``fault_bits``), ``is_success``, ``tripped``, ``state_seq``, ``cmd_seq``,
``tracker_visible``, ``read_ok``, ``sm_state``, ``tick``, ``action_clamped`` (any joint clamped this step) and
``applied_action`` (``applied_delta_ticks / max_delta_ticks``, the logged L4 action) -- the last two feed the
optional ``clamp`` / ``action_l2`` reward terms; plus ``trip`` when truncated by a fault.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

import gymnasium
import numpy as np
from gymnasium import spaces

from so101_rl.config import N_JOINTS, So101Config
from so101_rl.safety.types import SmState

from .backend import EnvBackend, ServoStateStale
from .goals import BoxGoalSampler, FixedGoalSampler, GoalSampler, redraw_near_start_np
from .obs import DIE_AGE_MAX_FRAMES, GOAL_DIM, DieObs, build_policy_obs_np
from .tasks import (
    TaskSpec,
    action_to_delta_np,
    debounce_np,
    step_reward_np,
    success_np,
    tripped_from_info,
    visible_from_info,
)

__all__ = ["DIE_LABEL", "So101GoalEnv"]

_NS: Final[int] = 1_000_000_000
DIE_LABEL: Final[bytes] = b"die"
CAMERA_HZ: Final[float] = 30.0
START_JITTER_TICKS: Final[int] = 30  # start_pose "random_in_box": uniform +-30 ticks on the arm joints around home


class So101GoalEnv(gymnasium.Env):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        backend: EnvBackend,
        cfg: So101Config,
        task: TaskSpec,
        *,
        goal_sampler: GoalSampler | None = None,
        object_table: Any | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode {render_mode!r} not in {self.metadata['render_modes']}")
        self.backend = backend
        self.cfg = cfg
        self.task = task
        self.render_mode = render_mode
        self.object_table = object_table
        self._fk = backend.fk()
        if goal_sampler is None:
            if task.zone is not None:
                goal_sampler = FixedGoalSampler(task.zone.goal_xyz())
            else:
                goal_sampler = BoxGoalSampler(task.goal_bounds, fk=self._fk, limits=cfg.joints, seed=0)
        self.goal_sampler = goal_sampler
        dim = task.obs.dim
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, (dim,), np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, (GOAL_DIM,), np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, (GOAL_DIM,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)
        self._max_delta = cfg.action.max_delta_ticks.astype(np.float64)
        if not (self._max_delta > 0).all():
            raise ValueError("cfg.action.max_delta_ticks must be > 0 on every joint")
        self._policy_period_ns = round(_NS / cfg.rates.policy_hz)
        self._frame_ns = round(_NS / CAMERA_HZ)
        self._die_max_age_ns = self._policy_period_ns + 2 * self._frame_ns
        from so101_rl.servo.shm import new_state

        self._st = new_state()
        self._tab = None
        if object_table is not None:
            from so101_rl.perception.object_table import new_object_table

            self._tab = new_object_table()
        self._obs_buf = np.zeros(dim, np.float32)
        self._tip = np.zeros(3)
        self._goal = np.zeros(3)
        self._episode = 0
        self._steps = 0
        self._count = np.zeros((), np.int64)
        self._prev_achieved = np.zeros(3)
        self._needs_reset = True
        self._cmd_seq = 0
        self._die_xyz = np.zeros(3)
        self._die_seen_ns: int | None = None
        self._die_visible = False
        self._fiducial_ok = True

    # ------------------------------------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        """``options``: ``q0`` (start ticks, default per ``task.start_pose``), ``goal`` (xyz, default sampled; a
        sampled reach goal within 2 success radii of the start tip is redrawn once, as in ``BatchedGoalEnv``)."""
        super().reset(seed=seed)
        options = options or {}
        q0 = options.get("q0")
        if q0 is None:
            q0 = self._start_ticks()
        self.backend.reset(np.asarray(q0))
        self._episode += 1
        self._steps = 0
        self._count = np.zeros((), np.int64)
        self._die_xyz = np.zeros(3)
        self._die_seen_ns = None
        self._die_visible = False
        self._fiducial_ok = True
        if not self._read_state():
            raise ServoStateStale("no servo state record after reset")
        goal = options.get("goal")
        if goal is None:
            goal = self.goal_sampler.sample(self.np_random)
            if self.task.zone is None:  # reach: the batched env's near-start redraw (env/goals.py), sim == real
                tip0 = self._fk.gripper_xyz_from_ticks(self._st[0]["present_ticks"], np.empty(3))
                goal = redraw_near_start_np(goal, tip0, self.task.success_radius_m,
                                            lambda: self.goal_sampler.sample(self.np_random))
        self._goal = np.asarray(goal, np.float64)
        obs, achieved = self._observe()
        self._prev_achieved = achieved.astype(np.float64)
        self._needs_reset = False
        rec = self._st[0]
        return obs, {"state_seq": int(rec["seq"]), "read_ok": bool(rec["read_ok"]), "tracker_visible": self._visible()}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if self._needs_reset:
            raise RuntimeError("So101GoalEnv.step() needs reset() first (new env, or after a trip / stalled servo)")
        delta = action_to_delta_np(np.asarray(action).reshape(N_JOINTS), self._max_delta)  # NaN stays NaN
        state_seq = int(self._st[0]["seq"])
        self._cmd_seq = int(self.backend.send_delta(delta, state_seq, self._episode))
        stalled = False
        try:
            self.backend.wait_policy_tick()
        except ServoStateStale:
            stalled = True
        if not stalled and not self._read_state():
            stalled = True
        self._steps += 1
        rec = self._st[0]
        obs, achieved = self._observe()
        read_ok = bool(rec["read_ok"]) and not stalled
        fault = int(rec["fault_bits"])
        sm = int(rec["sm_state"])
        tripped = fault != 0 or sm != int(SmState.RUN)
        applied = rec["applied_delta_ticks"].astype(np.float32)
        clamped = rec["clamped_mask"].astype(bool)
        info: dict[str, Any] = {
            "applied_delta_ticks": applied.copy(),
            "target_ticks": rec["target_ticks"].astype(np.int32),
            "clamped_mask": clamped,
            "trips": fault,
            "state_seq": int(rec["seq"]),
            "cmd_seq": self._cmd_seq,
            "tracker_visible": self._visible() if self.task.zone is not None else read_ok,
            "read_ok": read_ok,
            "sm_state": sm,
            "tick": int(rec["tick"]),
            "action_clamped": bool(clamped.any()),
            "applied_action": np.clip(applied / self._max_delta, -1.0, 1.0).astype(np.float32),
        }
        desired = obs["desired_goal"]
        reward = float(self.compute_reward(obs["achieved_goal"], desired, info))
        visible = np.asarray(self._visible()) if self.task.zone is not None else None
        # the same float32 goals the reward (and HER) sees; a stale read or a trip never counts toward termination
        pred = success_np(obs["achieved_goal"], desired, self.task, visible=visible) & read_ok & (not tripped)
        settle = self.task.settle_m is not None
        success, self._count = debounce_np(
            pred,
            self._count,
            self.task,
            achieved=achieved if settle else None,
            prev_achieved=self._prev_achieved if settle else None,
        )
        self._prev_achieved = achieved.astype(np.float64)
        # a policy-caused trip (TERMINAL_TRIPS) is terminal: no bootstrap, compute_reward paid task.trip_penalty
        terminal_trip = bool(tripped_from_info(info))
        terminated = bool(success) or terminal_trip
        truncated = bool(tripped or stalled or self._steps >= self.task.episode_steps)
        info["is_success"] = bool(success)
        info["tripped"] = terminal_trip
        if self.task.zone is not None:
            info["fiducial_ok"] = self._fiducial_ok
        if tripped:
            info["trip"] = fault
        if stalled:
            info["servo_stalled"] = True
        if tripped or stalled:
            self._needs_reset = True
        elif terminated or truncated:
            hold = getattr(self.backend, "send_hold", None)
            if hold is not None:  # the episode is over: an explicit HOLD instead of a silently stale RUN command
                hold(int(rec["seq"]), self._episode)
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Any) -> np.ndarray:
        """= :func:`tasks.step_reward_np` (vectorised for SB3 HER): ``info`` is one dict or a sequence of the stored
        transitions' dicts; the die visibility comes from ``info["tracker_visible"]`` and the trip bits from
        ``info["trips"]`` (neither is ever assumed), so a relabelled terminal-trip step keeps ``task.trip_penalty``."""
        visible = visible_from_info(info, self.task)
        return step_reward_np(achieved_goal, desired_goal, self.task, visible=visible, info=info)

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        render = getattr(self.backend, "render", None)
        return None if render is None else render()

    def close(self) -> None:
        """Closes the backend; the object table belongs to the caller (perception owns the segment)."""
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()

    # ------------------------------------------------------------------------------------------------ helpers
    def _start_ticks(self) -> np.ndarray:
        home = np.asarray(self.cfg.safe_state.home_ticks, dtype=np.int64)
        if self.task.start_pose == "home":
            return home
        lo, hi = self.cfg.joints.min_ticks.astype(np.int64), self.cfg.joints.max_ticks.astype(np.int64)
        xyz = np.empty(3)
        for _ in range(100):
            q = home.copy()
            q[:5] += self.np_random.integers(-START_JITTER_TICKS, START_JITTER_TICKS + 1, size=5)
            q = np.clip(q, lo, hi)
            if self._fk.in_box(self._fk.gripper_xyz_from_ticks(q, xyz), self.cfg.box):
                return q
        return home

    def _read_state(self) -> bool:
        """The step's record: the one the backend's ``wait_policy_tick`` accepted when it offers it
        (``accepted_state``: age- and newer-tick-checked), else the latest record (3 seqlock attempts)."""
        accepted = getattr(self.backend, "accepted_state", None)
        if accepted is not None:
            return bool(accepted(self._st))
        return any(self.backend.read_state(self._st) for _ in range(3))

    def _visible(self) -> bool:
        return bool(self._die_visible)

    def _update_die(self, now_ns: int) -> None:
        visible = False
        tab = self._tab
        if self.object_table is not None and tab is not None and self.object_table.read(tab):
            rec = tab[0]
            self._fiducial_ok = bool(rec["fiducial_ok"])
            for o in rec["objects"][: min(int(rec["n"]), len(rec["objects"]))]:
                if bytes(o["label"]).rstrip(b"\0") != DIE_LABEL:
                    continue
                t = int(o["t_mono_ns"])
                xyz = np.asarray(o["xyz_base"], np.float64)
                # ``now_ns`` is the start of the servo tick whose record was read; perception may have published
                # a detection after it (same monotonic clock): up to one policy period "in the future" is fresh
                fresh = -self._policy_period_ns <= now_ns - t <= self._die_max_age_ns
                if bool(o["visible"]) and fresh and self._fiducial_ok and np.all(np.isfinite(xyz)):
                    visible = True
                    self._die_xyz = xyz.copy()
                    self._die_seen_ns = t
                break
        self._die_visible = visible

    def _die_age_frames(self, now_ns: int) -> int:
        if self._die_visible:
            return 0
        if self._die_seen_ns is None:
            return DIE_AGE_MAX_FRAMES
        return int(min(DIE_AGE_MAX_FRAMES, max(1, round((now_ns - self._die_seen_ns) / self._frame_ns))))

    def _observe(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        rec = self._st[0]
        present = rec["present_ticks"]
        self._fk.gripper_xyz_from_ticks(present, self._tip)
        die = None
        if self.task.obs.has_die:
            now = int(rec["t_mono_ns"])
            self._update_die(now)
            die = DieObs(xyz=self._die_xyz, visible=self._die_visible, age_frames=self._die_age_frames(now))
        build_policy_obs_np(
            self.task.obs,
            present_ticks=present,
            velocity=rec["velocity"],
            applied_delta_ticks=rec["applied_delta_ticks"],
            tip_xyz=self._tip,
            cfg=self.cfg,
            die=die,
            out=self._obs_buf,
        )
        achieved = self._die_xyz if self.task.zone is not None else self._tip
        obs = {
            "observation": self._obs_buf.copy(),
            "achieved_goal": achieved.astype(np.float32),
            "desired_goal": self._goal.astype(np.float32),
        }
        return obs, achieved.copy()

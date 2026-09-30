"""2-D point-mass goal env with the same Dict observation contract as ``env/goal_env.py`` (plan F7).

Only numpy + gymnasium. Used by the pixi solver spike, the TD-MPC2 smoke train and the HER unit tests.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WORLD = 1.0  # position bounds [-WORLD, WORLD] in both axes
GOAL_RANGE = 0.8
DT = 0.1
ACCEL = 1.0
DAMPING = 0.9


class ToyReachEnv(gym.Env):
    """Move a point mass onto a goal. ``terminated`` on success, ``truncated`` at ``episode_steps``."""

    metadata = {"render_modes": [], "render_fps": 10}  # noqa: RUF012

    def __init__(
        self, *, episode_steps: int = 50, success_radius: float = 0.05, success_bonus: float = 1.0
    ) -> None:
        super().__init__()
        self.episode_steps = episode_steps
        self.success_radius = success_radius
        self.success_bonus = success_bonus
        big = np.float32(np.inf)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-big, big, (4,), np.float32),  # pos(2), vel(2)
                "achieved_goal": spaces.Box(-big, big, (2,), np.float32),
                "desired_goal": spaces.Box(-big, big, (2,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._pos = np.zeros(2)
        self._vel = np.zeros(2)
        self._goal = np.zeros(2)
        self._t = 0

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "observation": np.concatenate([self._pos, self._vel]).astype(np.float32),
            "achieved_goal": self._pos.astype(np.float32),
            "desired_goal": self._goal.astype(np.float32),
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._pos = self.np_random.uniform(-GOAL_RANGE, GOAL_RANGE, 2)
        self._vel = np.zeros(2)
        self._goal = self.np_random.uniform(-GOAL_RANGE, GOAL_RANGE, 2)
        self._t = 0
        return self._obs(), {"is_success": False}

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Any = None) -> np.ndarray:
        """Vectorised over leading dims: ``-|a-g| + bonus * [|a-g| < r]``."""
        dist = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return (-dist + self.success_bonus * (dist < self.success_radius)).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._vel = DAMPING * self._vel + ACCEL * DT * a
        self._pos = np.clip(self._pos + DT * self._vel, -WORLD, WORLD)
        self._t += 1
        obs = self._obs()
        dist = float(np.linalg.norm(self._pos - self._goal))
        success = dist < self.success_radius
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"]))
        truncated = self._t >= self.episode_steps and not success
        return obs, reward, bool(success), bool(truncated), {"is_success": bool(success)}

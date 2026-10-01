"""Stuck-action monitor (port of robot_data_runner's SafetyMonitor, plan F5).

Trips when the policy keeps commanding the SAME non-zero delta while no joint is moving: the arm is blocked (or
the command path is frozen) and integrating further would only wind the target up against an obstacle.
"""

from __future__ import annotations

import numpy as np

from so101_rl.config import N_JOINTS

from .types import TripCause


class StuckActionMonitor:
    def __init__(self, repeat_ticks: int = 90, epsilon_ticks: float = 0.5) -> None:
        # 90 ticks at 30 Hz = 3 s of identical non-zero deltas while not moving
        self.repeat_ticks = repeat_ticks
        self.epsilon_ticks = epsilon_ticks
        self._ref = np.zeros(N_JOINTS, dtype=np.float32)
        self._scratch = np.zeros(N_JOINTS, dtype=np.float32)
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0

    def update(self, applied_delta: np.ndarray, moving: np.ndarray) -> TripCause:
        """One call per servo tick. ``STUCK`` once ``repeat_ticks`` consecutive ticks carry the same non-zero delta
        (every component within ``epsilon_ticks`` of the first one) with ``moving == 0`` on every joint; any change,
        a zero delta, or any motion resets the count."""
        eps = self.epsilon_ticks
        nonzero = float(np.abs(applied_delta).max()) > eps
        still = not bool(moving.any())
        if not (nonzero and still):
            self._count = 0
            return TripCause.NONE
        if self._count > 0:
            np.subtract(applied_delta, self._ref, out=self._scratch)
            if float(np.abs(self._scratch).max()) > eps:
                self._count = 0
        if self._count == 0:
            self._ref[...] = applied_delta
        self._count += 1
        return TripCause.STUCK if self._count >= self.repeat_ticks else TripCause.NONE

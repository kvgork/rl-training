"""Clocks and tick statistics (plan F6). ``FakeClock`` lets a 24 h soak run in seconds."""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

_SPIN_NS = 200_000  # sleep until 200 us before the deadline, then spin


class Clock(Protocol):
    def monotonic_ns(self) -> int: ...

    def sleep_until(self, t_ns: int) -> None: ...


class RealClock:
    """``time.monotonic_ns`` with a coarse sleep followed by a short spin for the last 200 us."""

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep_until(self, t_ns: int) -> None:
        remaining = t_ns - time.monotonic_ns()
        if remaining > _SPIN_NS:
            time.sleep((remaining - _SPIN_NS) / 1e9)
        while time.monotonic_ns() < t_ns:
            pass


class FakeClock:
    """Deterministic clock: time moves only through :meth:`advance` and :meth:`sleep_until`."""

    def __init__(self, start_ns: int = 0) -> None:
        self._now = start_ns

    def monotonic_ns(self) -> int:
        return self._now

    def advance(self, ns: int) -> None:
        if ns < 0:
            raise ValueError("cannot advance a clock backwards")
        self._now += ns

    def sleep_until(self, t_ns: int) -> None:
        """Advances exactly to ``t_ns`` (never backwards)."""
        self._now = max(self._now, t_ns)


class TickStats:
    """Preallocated ring of per-tick jitter/duration; percentiles are computed on demand.

    ``jitter_ns`` is how late the tick started relative to its absolute deadline (>= 0 in practice);
    the ``p*_ms`` / ``max_ms`` properties are over the jitter. ``duration_*`` are over the tick body.
    A tick is an overrun when its duration exceeds ``period_ns``. p99.9 needs >= 1e4 samples to mean anything.
    """

    def __init__(self, period_ns: int, capacity: int = 1 << 18) -> None:
        self.period_ns = period_ns
        self._jitter = np.zeros(capacity, dtype=np.int64)
        self._duration = np.zeros(capacity, dtype=np.int64)
        self._cap = capacity
        self.samples = 0
        self.overruns = 0
        self._max_jitter_ns = 0
        self._max_duration_ns = 0

    def record(self, jitter_ns: int, duration_ns: int) -> None:
        i = self.samples % self._cap
        self._jitter[i] = jitter_ns
        self._duration[i] = duration_ns
        self.samples += 1
        if duration_ns > self.period_ns:
            self.overruns += 1
        self._max_jitter_ns = max(self._max_jitter_ns, jitter_ns)
        self._max_duration_ns = max(self._max_duration_ns, duration_ns)

    def _valid(self, arr: np.ndarray) -> np.ndarray:
        return arr[: min(self.samples, self._cap)]

    def _pct_ms(self, arr: np.ndarray, q: float) -> float:
        v = self._valid(arr)
        return float(np.percentile(v, q)) / 1e6 if v.size else 0.0

    @property
    def p50_ms(self) -> float:
        return self._pct_ms(self._jitter, 50.0)

    @property
    def p99_ms(self) -> float:
        return self._pct_ms(self._jitter, 99.0)

    @property
    def p999_ms(self) -> float:
        return self._pct_ms(self._jitter, 99.9)

    @property
    def max_ms(self) -> float:
        return self._max_jitter_ns / 1e6

    @property
    def duration_p99_ms(self) -> float:
        return self._pct_ms(self._duration, 99.0)

    @property
    def duration_max_ms(self) -> float:
        return self._max_duration_ns / 1e6

    def summary(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "overruns": self.overruns,
            "p50_ms": self.p50_ms,
            "p99_ms": self.p99_ms,
            "p999_ms": self.p999_ms,
            "max_ms": self.max_ms,
            "duration_p99_ms": self.duration_p99_ms,
            "duration_max_ms": self.duration_max_ms,
        }

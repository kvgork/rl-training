"""Thermal and firmware-protection monitor (plan F5).

Semantics
---------
* ``update`` is called once per servo tick with the latest block read.
* **Trip:** any joint ``>= trip_c``, any reading of 0 or 255 (sensor dead / bus garbage), or no successful block
  read for more than ``stale_temp_s`` -> ``TripCause.THERMAL``.
* **Latch:** a trip latches. While latched, every ``update`` keeps returning ``THERMAL`` (even if the joint
  oscillates 59 <-> 61), and the trip is counted exactly ONCE. The latch clears only when ``may_resume()`` becomes
  true, i.e. every joint has been ``<= resume_c`` continuously for ``resume_dwell_s`` (hysteresis + dwell, so a
  joint hovering around the threshold cannot chatter the state machine).
* **Protection bits:** any non-zero ``status`` byte (Feetech firmware protection flags: voltage, sensor,
  temperature, current, angle, overload) -> ``TripCause.PROTECTION_BIT``. These are not latched here: the state
  machine treats them as a human-reset cause.
* **Stale temperature** (review S10): the timer starts at the first successful read, not at construction, and
  it restarts whenever ``update`` was not called for longer than ``stale_temp_s`` (the caller was blocked, e.g.
  in a multi-second ``reenergize``): a gap in *feeding* is not evidence of a dead sensor. Consecutive failed
  reads while being fed every tick are still a stale-temperature trip after ``stale_temp_s`` (and the state
  machine's ``max_bus_failures`` latches long before that).
* ``trips_last_hour`` counts latch events in the trailing 3600 s, for diagnostics and the soak report only. The
  automatic-resume limit (``max_trips_per_hour``) is counted in ONE place: ``SafetyStateMachine``.
"""

from __future__ import annotations

import numpy as np

from so101_rl.config import N_JOINTS, ThermalConfig
from so101_rl.servo.timing import Clock

from .types import TripCause

_HOUR_NS = 3600 * 1_000_000_000
_NS = 1_000_000_000
_SENSOR_UNREAD = 0  # short / never read
_SENSOR_GARBAGE = 255  # all-ones bus garbage
_RING = 256


class ThermalMonitor:
    def __init__(self, cfg: ThermalConfig, clock: Clock) -> None:
        self._cfg = cfg
        self._clock = clock
        self._stale_ns = int(cfg.stale_temp_s * _NS)
        self._dwell_ns = int(cfg.resume_dwell_s * _NS)
        self._latched = False
        self._last_good_ns: int | None = None  # no successful read yet: the stale timer has not started
        self._last_update_ns: int | None = None
        self._cool_since_ns: int | None = None
        self._trip_times = np.zeros(_RING, dtype=np.int64)
        self._n_trips = 0
        self.hottest_c: float = 0.0
        self.max_seen_c = np.zeros(N_JOINTS, dtype=np.int32)

    @property
    def latched(self) -> bool:
        return self._latched

    def update(self, temp_c: np.ndarray, status: np.ndarray, block_ok: bool) -> TripCause:
        now = self._clock.monotonic_ns()
        cause = TripCause.NONE
        if self._last_update_ns is not None and now - self._last_update_ns > self._stale_ns:
            self._last_good_ns = None  # not fed for longer than the stale window: restart the timer
        self._last_update_ns = now

        if block_ok:
            self._last_good_ns = now
            tmax = int(temp_c.max())
            self.hottest_c = float(tmax)
            np.maximum(self.max_seen_c, temp_c, out=self.max_seen_c)
            if int(status.max()) != 0:
                cause |= TripCause.PROTECTION_BIT
            dead = bool((temp_c == _SENSOR_UNREAD).any() or (temp_c == _SENSOR_GARBAGE).any())
            fault = dead or tmax >= self._cfg.trip_c
            if not dead and tmax <= self._cfg.resume_c:
                if self._cool_since_ns is None:
                    self._cool_since_ns = now
            else:
                self._cool_since_ns = None
        else:
            # Arrays hold the PREVIOUS values; never treat them as a fresh reading or as cooling time.
            self._cool_since_ns = None
            if self._last_good_ns is None:
                self._last_good_ns = now  # the stale window starts with the first unanswered read
            fault = (now - self._last_good_ns) > self._stale_ns

        if self._latched and not fault and self.may_resume():
            self._latched = False  # cooled down for the full dwell
        if fault and not self._latched:
            self._latched = True
            self._trip_times[self._n_trips % _RING] = now
            self._n_trips += 1
        if self._latched:
            cause |= TripCause.THERMAL
        return cause

    def may_resume(self) -> bool:
        """All temperatures ``<= resume_c`` for ``resume_dwell_s`` without interruption."""
        if self._cool_since_ns is None:
            return False
        return (self._clock.monotonic_ns() - self._cool_since_ns) >= self._dwell_ns

    def trips_last_hour(self) -> int:
        now = self._clock.monotonic_ns()
        n = min(self._n_trips, _RING)
        return int((self._trip_times[:n] > now - _HOUR_NS).sum())

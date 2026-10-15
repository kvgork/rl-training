"""Thermal and firmware-protection monitor (plan F5).

Semantics
---------
* ``update`` is called once per servo tick with the latest block read.
* **Trip:** any joint ``>= trip_c`` on ``CONFIRM_READS`` consecutive good reads, any reading of 0 or 255 (sensor
  dead / bus garbage, at once), or no successful block read for more than ``stale_temp_s`` -> ``TripCause.THERMAL``.
  Why consecutive: the real bus returns one-tick spikes (26 -> 69 -> 26 C on one joint, 19 in 4400 ticks on
  2026-10-11); real motor heat cannot rise that fast, and the firmware's own over-temperature cut stays the backstop.
  The same rule decides whether a joint is too warm for the cooling dwell (``> resume_c``).
* **Median filter** (:class:`MedianTemperature`, plan section 10, 2026-10-15): every trip, warm and hottest
  decision reads the per-joint median of the last ``MEDIAN_READS`` (11, 367 ms) good reads, not the raw byte. The
  real bus returns spiked reads on one joint (~1 % of reads, 40-150 C, in runs of 1-4 reads; 33 -> 94 -> 98 -> 65 ->
  33 C on 2026-10-12 tripped through ``CONFIRM_READS``), which a median of 11 removes: replayed over 740k real reads
  it stays within 2 C of the 1 s median and trips 0 times (the old rule tripped twice). A real temperature change
  reaches the median after 6 reads (200 ms), far faster than motor heat moves. Until 11 reads exist the median is
  over the reads so far. The dead-sensor values 0 / 255 bypass the filter (raw) and trip at once.
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
CONFIRM_READS = 3  # 100 ms at 30 Hz (plan section 10, 2026-10-11)
MEDIAN_READS = 11  # per-joint median window of the temperature filter (plan section 10, 2026-10-15)
OUTLIER_C = 3.0  # a raw read this far from the median is counted (diagnostics only)


class MedianTemperature:
    """Per joint: the median of the last ``n`` reads fed to ``update`` (module docstring, median filter). ``update(raw)``
    returns the filtered temperatures (a view owned by the filter)."""

    def __init__(self, n: int = MEDIAN_READS) -> None:
        self._buf = np.zeros((int(n), N_JOINTS), dtype=np.float64)
        self._i = 0
        self._count = 0
        self._out = np.zeros(N_JOINTS, dtype=np.float64)
        self.outliers = np.zeros(N_JOINTS, dtype=np.int64)  # raw reads > OUTLIER_C from the median (diagnostics)

    def update(self, raw: np.ndarray) -> np.ndarray:
        self._buf[self._i] = raw
        self._i = (self._i + 1) % len(self._buf)
        self._count = min(self._count + 1, len(self._buf))
        np.copyto(self._out, np.median(self._buf[: self._count], axis=0))
        self.outliers += np.abs(np.asarray(raw, dtype=np.float64) - self._out) > OUTLIER_C
        return self._out


class HotCheck:
    """``MedianTemperature`` then ``ConsecutiveAbove``: the trip test of one run of reads (re-energise)."""

    def __init__(self, threshold: float) -> None:
        self._median = MedianTemperature()
        self._run = ConsecutiveAbove(threshold)

    def update(self, temp_c: np.ndarray) -> bool:
        return self._run.update(self._median.update(temp_c))


class ConsecutiveAbove:
    """Per joint: true once a joint has read above a threshold on ``n`` consecutive reads fed to ``update``."""

    def __init__(self, threshold: float, *, strict: bool = False, n: int = CONFIRM_READS) -> None:
        self._threshold = threshold
        self._strict = strict  # ``>`` instead of ``>=``
        self._n = n
        self._count = np.zeros(N_JOINTS, dtype=np.int64)

    def update(self, temp_c: np.ndarray) -> bool:
        above = temp_c > self._threshold if self._strict else temp_c >= self._threshold
        self._count = np.where(above, self._count + 1, 0)
        return bool((self._count >= self._n).any())


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
        self._median = MedianTemperature()
        self._hot = ConsecutiveAbove(cfg.trip_c)
        self._warm = ConsecutiveAbove(cfg.resume_c, strict=True)
        self._trip_times = np.zeros(_RING, dtype=np.int64)
        self._n_trips = 0
        self.hottest_c: float = 0.0
        self.max_seen_c = np.zeros(N_JOINTS, dtype=np.int32)

    @property
    def outlier_reads(self) -> np.ndarray:
        """Per joint: raw temperature reads more than ``OUTLIER_C`` from the median (diagnostics, servo summary)."""
        return self._median.outliers

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
            dead = bool((temp_c == _SENSOR_UNREAD).any() or (temp_c == _SENSOR_GARBAGE).any())
            t = self._median.update(temp_c)
            self.hottest_c = float(t.max())
            np.maximum(self.max_seen_c, t.astype(np.int32), out=self.max_seen_c)
            if int(status.max()) != 0:
                cause |= TripCause.PROTECTION_BIT
            hot = self._hot.update(t)
            warm = self._warm.update(t)
            fault = dead or hot
            if not dead and not warm:
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

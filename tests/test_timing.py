import time

import pytest

from so101_rl.servo.timing import FakeClock, RealClock, TickStats


def test_fake_clock_sleep_until_advances_exactly():
    c = FakeClock(start_ns=1_000)
    c.sleep_until(5_000)
    assert c.monotonic_ns() == 5_000
    c.sleep_until(4_000)  # never backwards
    assert c.monotonic_ns() == 5_000
    c.advance(10)
    assert c.monotonic_ns() == 5_010
    with pytest.raises(ValueError):
        c.advance(-1)


def test_fake_clock_24h_soak_is_instant():
    c = FakeClock()
    period = 33_333_333
    t0 = time.perf_counter()
    deadline = 0
    for _ in range(30 * 3600 * 24 // 100):  # 1/100 of a day of 30 Hz ticks
        deadline += period
        c.sleep_until(deadline)
    assert c.monotonic_ns() == deadline
    assert time.perf_counter() - t0 < 5


def test_tick_stats_percentiles_known_distribution():
    stats = TickStats(period_ns=33_000_000, capacity=20_000)
    # jitter 1..10000 us, duration constant 1 ms
    for j in range(1, 10_001):
        stats.record(j * 1_000, 1_000_000)
    assert stats.samples == 10_000 and stats.overruns == 0
    assert stats.p50_ms == pytest.approx(5.0, abs=0.01)
    assert stats.p99_ms == pytest.approx(9.9, abs=0.01)
    assert stats.p999_ms == pytest.approx(9.99, abs=0.01)
    assert stats.max_ms == 10.0
    assert stats.duration_p99_ms == pytest.approx(1.0)


def test_tick_stats_overruns_and_ring_wrap():
    stats = TickStats(period_ns=1_000, capacity=8)
    stats.record(500_000_000, 900_000_000)  # spike: must be evicted from the percentile ring ...
    for i in range(1, 20):  # ... by 19 more samples; only the last 8 (i = 12..19) stay
        stats.record(i * 1_000_000, i * 1_000_000)  # distinct values, each duration exceeds the period
    assert stats.overruns == 20 and stats.samples == 20
    # ring holds 12..19 ms: a non-wrapping implementation would see the 500 ms spike and 1..11 ms
    assert stats.p50_ms == pytest.approx(15.5)
    assert stats.p99_ms == pytest.approx(12 + 0.99 * 7)
    assert stats.duration_p99_ms == pytest.approx(12 + 0.99 * 7)
    # max is a running max over every sample, so it still remembers the evicted spike
    assert stats.max_ms == 500.0 and stats.duration_max_ms == 900.0
    assert stats.summary()["overruns"] == 20


def test_tick_stats_empty():
    stats = TickStats(period_ns=1)
    assert stats.p999_ms == 0.0 and stats.max_ms == 0.0


def test_real_clock_sleeps_until_deadline():
    c = RealClock()
    t = c.monotonic_ns() + 3_000_000
    c.sleep_until(t)
    now = c.monotonic_ns()
    assert now >= t
    # Upper bound is only a sanity check against sleeping far too long; scheduler noise on a loaded CI host
    # can exceed a few ms, so it is deliberately generous (the real jitter budget is measured by H2).
    assert now - t < 250_000_000
    before = c.monotonic_ns()
    c.sleep_until(before - 1)  # past deadline returns immediately, never sleeps a period
    assert c.monotonic_ns() - before < 250_000_000
    assert isinstance(c.monotonic_ns(), int)
    assert c.monotonic_ns() >= now  # monotonic

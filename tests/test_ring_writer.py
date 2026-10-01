"""Task 6.2: RingWriter is allocation-free on push, drained by a thread, and flushes its tail on close."""

import threading
import time
import tracemalloc

import numpy as np

from so101_rl.logging.ring_writer import RingWriter

DT = np.dtype([("t", "<i8"), ("x", "<f4", (6,)), ("flag", "u1")])


def rec(i: int) -> np.ndarray:
    r = np.zeros((), DT)
    r["t"] = i
    r["x"] = float(i)
    r["flag"] = i % 2
    return r


def read_back(path):
    return np.fromfile(path, dtype=DT)


def test_close_flushes_tail(tmp_path):
    rw = RingWriter(tmp_path / "ticks.bin", DT, capacity=64, flush_interval_s=3600)  # thread never fires
    for i in range(10):
        rw.push(rec(i))
    rw.close()
    out = read_back(tmp_path / "ticks.bin")
    assert out["t"].tolist() == list(range(10))
    assert out["x"][7].tolist() == [7.0] * 6
    rw.close()  # idempotent


def test_flush_returns_count_and_is_incremental(tmp_path):
    rw = RingWriter(tmp_path / "t.bin", DT, capacity=64, flush_interval_s=3600)
    for i in range(5):
        rw.push(rec(i))
    assert rw.flush() == 5
    assert rw.flush() == 0
    for i in range(5, 8):
        rw.push(rec(i))
    assert rw.flush() == 3
    rw.close()
    assert read_back(tmp_path / "t.bin")["t"].tolist() == list(range(8))


def test_wraparound_preserves_order(tmp_path):
    rw = RingWriter(tmp_path / "t.bin", DT, capacity=8, flush_interval_s=3600)
    for i in range(6):
        rw.push(rec(i))
    rw.flush()
    for i in range(6, 12):  # wraps the physical buffer, never laps the reader
        rw.push(rec(i))
    rw.close()
    assert read_back(tmp_path / "t.bin")["t"].tolist() == list(range(12))
    assert rw.dropped == 0


def test_lapping_drops_oldest_and_counts(tmp_path):
    rw = RingWriter(tmp_path / "t.bin", DT, capacity=8, flush_interval_s=3600)
    for i in range(20):
        rw.push(rec(i))
    rw.close()
    assert read_back(tmp_path / "t.bin")["t"].tolist() == list(range(12, 20))
    assert rw.dropped == 12


def test_background_thread_drains(tmp_path):
    rw = RingWriter(tmp_path / "t.bin", DT, capacity=1024, flush_interval_s=0.01)
    try:
        for i in range(100):
            rw.push(rec(i))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (tmp_path / "t.bin").stat().st_size < 100 * DT.itemsize:
            time.sleep(0.01)
        assert (tmp_path / "t.bin").stat().st_size == 100 * DT.itemsize  # drained before close()
        assert any(t.name == "ring-flush" and t.is_alive() for t in threading.enumerate())
    finally:
        rw.close()
    assert not rw._thread.is_alive()


def test_push_is_allocation_free(tmp_path):
    rw = RingWriter(tmp_path / "t.bin", DT, capacity=100_000, flush_interval_s=3600)
    r = rec(1)
    for _ in range(1000):  # warm up
        rw.push(r)
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        for _ in range(20_000):
            rw.push(r)
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rw.close()
    # 20k pushes of a 33-byte record would retain ~660 kB if push allocated per call.
    assert after - before < 2048, f"push retained {after - before} bytes"


def test_push_does_not_alias_caller_buffer(tmp_path):
    rw = RingWriter(tmp_path / "t.bin", DT, capacity=8, flush_interval_s=3600)
    r = rec(1)
    rw.push(r)
    r["t"] = 999  # caller reuses its buffer
    rw.close()
    assert read_back(tmp_path / "t.bin")["t"].tolist() == [1]

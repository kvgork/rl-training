"""Preallocated binary ring for the servo tick log, drained by a low-priority thread.

``push`` copies one record into a preallocated numpy ring: no allocation, no I/O, no lock.
Single producer (servo loop) / single consumer (flusher): the producer only advances ``_head``, the
consumer only advances ``_tail``. If the producer laps the consumer the oldest unflushed records are
dropped and counted in ``dropped`` (the servo loop is never blocked).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


class RingWriter:
    def __init__(self, path: Path, dtype: np.dtype, capacity: int = 1_000_000, *, flush_interval_s: float = 0.2) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.capacity = capacity
        self._buf = np.zeros(capacity, dtype=self.dtype)
        self._head = 0  # total records pushed
        self._tail = 0  # total records written to disk
        self.dropped = 0
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._interval = flush_interval_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "ab")  # noqa: SIM115 - closed in close()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="ring-flush", daemon=True)
        self._thread.start()

    def push(self, rec: np.ndarray) -> None:
        self._buf[self._head % self.capacity] = rec
        self._head += 1

    def flush(self) -> int:
        """Write pending records to disk; returns the number written."""
        with self._flush_lock:
            if self._closed:
                return 0
            head = self._head
            tail = self._tail
            lag = head - tail
            if lag > self.capacity:  # overwritten before we got to them
                self.dropped += lag - self.capacity
                tail = head - self.capacity
            n = head - tail
            if n == 0:
                self._tail = tail
                return 0
            start = tail % self.capacity
            end = start + n
            if end <= self.capacity:
                self._fh.write(self._buf[start:end].tobytes())
            else:
                self._fh.write(self._buf[start:].tobytes())
                self._fh.write(self._buf[: end - self.capacity].tobytes())
            self._fh.flush()
            self._tail = head
            return n

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.flush()

    def close(self) -> None:
        """Stop the thread, flush the tail, close the file. Idempotent."""
        if self._closed:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self.flush()
        with self._flush_lock:
            self._closed = True
            self._fh.close()

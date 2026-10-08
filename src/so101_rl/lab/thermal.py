"""GPU thermal fail-closed for training jobs (plan §10 P8).

GPU core: hold at >= 85 °C, release at <= 75 °C. Memory junction: hold at >= 95 °C, release at <= 85 °C, only when
NVML reports it. Six consecutive empty samples (no core temperature) hold the job too (fail-closed) until a valid
cool sample arrives. A hold longer than ``max_hold_s`` makes the job stop with a checkpoint.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

log = logging.getLogger(__name__)

PAUSE_C, RESUME_C = 85.0, 75.0
MEM_PAUSE_C, MEM_RESUME_C = 95.0, 85.0
EMPTY_LIMIT = 6


class Sampler(Protocol):
    def sample(self) -> tuple[float | None, float | None]:
        """``(core_c, mem_junction_c)``; ``core_c is None`` is an empty sample."""
        ...


class NvmlSampler:
    """pynvml on GPU ``index``; any NVML failure is an empty sample, never an exception."""

    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._handle = None

    def sample(self) -> tuple[float | None, float | None]:
        try:
            import pynvml

            if self._handle is None:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._index)
            core = float(pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception as e:  # noqa: BLE001 - fail-closed: count it as empty
            log.warning("NVML sample failed: %s", e)
            self._handle = None
            return None, None
        mem = None
        try:
            v = pynvml.nvmlDeviceGetFieldValues(self._handle, [pynvml.NVML_FI_DEV_MEMORY_TEMP])[0]
            if v.nvmlReturn == 0 and v.value.uiVal > 0:
                mem = float(v.value.uiVal)
        except Exception:  # noqa: BLE001 - the memory junction is optional ("when observable")
            mem = None
        return core, mem


class ThermalGuard:
    def __init__(
        self,
        sampler: Sampler,
        *,
        interval_s: float = 5.0,
        max_hold_s: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.sampler = sampler
        self.interval_s = interval_s
        self.max_hold_s = max_hold_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self.empties = 0
        self.holding: str | None = None  # None | "hot" | "unknown"
        self.last_sample: tuple[float | None, float | None] = (None, None)

    def update(self, core: float | None, mem: float | None) -> str | None:
        self.last_sample = (core, mem)
        if core is None:
            self.empties += 1
            if self.empties >= EMPTY_LIMIT and self.holding is None:
                self.holding = "unknown"
            return self.holding
        self.empties = 0
        hot = core >= PAUSE_C or (mem is not None and mem >= MEM_PAUSE_C)
        cool = core <= RESUME_C and (mem is None or mem <= MEM_RESUME_C)
        if hot:
            self.holding = "hot"
        elif self.holding is not None and cool:
            self.holding = None
        return self.holding

    def poll(self, *, force: bool = False) -> str | None:
        now = self._clock()
        if force or self._last is None or now - self._last >= self.interval_s:
            self._last = now
            self.update(*self.sampler.sample())
        return self.holding

    def wait(
        self,
        *,
        on_hold: Callable[[str], None],
        on_release: Callable[[], None],
        should_abort: Callable[[], bool],
    ) -> bool:
        """Block while holding. True = continue training; False = abort (stop requested or hold too long)."""
        reason = self.poll()
        if reason is None:
            return True
        on_hold(reason)
        t0 = self._clock()
        while True:
            if should_abort() or self._clock() - t0 >= self.max_hold_s:
                return False
            self._sleep(self.interval_s)
            if self.poll(force=True) is None:
                on_release()
                return True

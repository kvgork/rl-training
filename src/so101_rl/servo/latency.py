"""Latency measurement hooks (plan F6 latency budget, Task 3.5).

* :func:`measure_bus` times ``n`` ``read_block`` round trips (standalone only: never while a servo process owns the
  bus) (and optionally ``write_goal`` of the value just read)
  on any ``Bus`` with an injected clock: FakeBus with a latency model in the tests, FeetechBus on the arm
  (``tests/test_latency.py`` hardware variant, budget p99 read < 6 ms).
* :func:`measure_obs_to_cmd` summarises the policy-side latency the servo records in its tick log
  (``TICK_LOG_DTYPE``): ``obs_to_cmd_ns`` = servo apply time - publish time of the state record the command was
  computed from (``CMD.state_seq``), and ``cmd_age_ns`` = servo receipt - policy write time (``CMD.t_mono_ns``).
* :func:`write_report` writes ``runs/hw/latency-<ts>.json``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.servo.timing import Clock, RealClock


def _stats_ms(ns: np.ndarray) -> dict[str, float]:
    if ns.size == 0:
        return {"n": 0, "p50": float("nan"), "p99": float("nan"), "max": float("nan")}
    ms = ns.astype(np.float64) / 1e6
    return {"n": int(ms.size), "p50": float(np.percentile(ms, 50)), "p99": float(np.percentile(ms, 99)), "max": float(ms.max())}


def measure_bus(bus: Bus, n: int = 300, *, clock: Clock | None = None, write: bool = False) -> dict[str, float]:
    """``read_block`` round-trip latency over ``n`` calls: ``{n, p50, p99, max, failures}`` in ms. The statistics
    cover the SUCCESSFUL reads only (``n`` = ``attempts - failures``): a failed read returns after a timeout and
    would distort the budget (U3 review item 16); ``failures`` counts the rest. With ``write`` each good read is
    followed by ``write_goal(present)`` (the arm holds still) and ``write_*`` keys are added. The caller must make
    sure torque is off on the real arm (the hardware test asserts it)."""
    clock = clock if clock is not None else RealClock()
    block = BlockRead()
    reads = np.zeros(n, dtype=np.int64)
    writes = np.zeros(n, dtype=np.int64)
    failures = 0
    write_failures = 0
    goal = np.zeros(block.position.shape, dtype=np.int32)
    nw = 0
    nr = 0
    for _ in range(n):
        t0 = clock.monotonic_ns()
        ok = bool(bus.read_block(block)) and block.ok
        t1 = clock.monotonic_ns()
        if not ok:
            failures += 1
            continue
        reads[nr] = t1 - t0
        nr += 1
        if write:
            goal[...] = block.position
            t2 = clock.monotonic_ns()
            if not bus.write_goal(goal):
                write_failures += 1
            writes[nw] = clock.monotonic_ns() - t2
            nw += 1
    out = _stats_ms(reads[:nr])
    out["attempts"] = n
    out["failures"] = failures
    if write:
        out.update({f"write_{k}": v for k, v in _stats_ms(writes[:nw]).items()})
        out["write_failures"] = write_failures
    return out


def measure_obs_to_cmd(tick_log: np.ndarray) -> dict[str, dict[str, float]]:
    """``{"obs_to_cmd": {...}, "cmd_age": {...}}`` in ms over the ticks of a tick log that consumed a command."""
    o2c = tick_log["obs_to_cmd_ns"]
    age = tick_log["cmd_age_ns"]
    return {"obs_to_cmd": _stats_ms(o2c[o2c > 0]), "cmd_age": _stats_ms(age[age > 0])}


def write_report(result: dict, root: Path = Path("runs/hw")) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"latency-{dt.datetime.now(dt.UTC).strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return path

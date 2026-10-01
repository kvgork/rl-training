"""Latency hooks (plan Task 3.5): FakeBus now, the real bus on the hardware track (never run here)."""

import json
import os

import numpy as np
import pytest

from so101_rl.servo.core import TICK_LOG_DTYPE, build_servo_core
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.latency import measure_bus, measure_obs_to_cmd, write_report
from so101_rl.servo.shm import CmdMode, LocalIpc, new_cmd, new_state
from so101_rl.servo.timing import FakeClock

MS = 1_000_000


def test_measure_bus_on_fakebus_with_a_latency_model(calibration):
    clock = FakeClock()
    bus = FakeBus(clock, calibration=calibration)
    bus.connect()
    bus.faults.latency_ms(lambda rng: 1.0 + 4.0 * rng.random())  # 1..5 ms per bus operation
    bus.faults.fail_next_read(3)
    r = measure_bus(bus, n=300, clock=clock, write=True)
    assert r["n"] == 300 and r["failures"] == 3
    assert 1.0 <= r["p50"] <= 5.0 and r["p50"] <= r["p99"] <= r["max"] <= 5.0 + 1e-6
    assert r["write_n"] == 297 and 1.0 <= r["write_p50"] <= 5.0


def test_measure_bus_constant_latency_is_exact(calibration):
    clock = FakeClock()
    bus = FakeBus(clock, calibration=calibration)
    bus.faults.latency_ms(2.0)
    r = measure_bus(bus, n=50, clock=clock)
    assert r["p50"] == r["p99"] == r["max"] == pytest.approx(2.0)


class _Ring:
    def __init__(self):
        self.recs = []

    def push(self, rec):
        self.recs.append(np.array(rec, dtype=TICK_LOG_DTYPE))


def test_measure_obs_to_cmd_from_the_servo_tick_log(cfg, calibration):
    clock = FakeClock(start_ns=10**12)
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.home_ticks)
    ipc = LocalIpc()
    ring = _Ring()
    core = build_servo_core(cfg, calibration, bus, ipc, None, clock, tick_log=ring, log=lambda m: None)
    core.start()
    st, cmd = new_state(), new_cmd()
    for step in range(20):
        ipc.read_state(st)  # the policy reads an observation ...
        clock.advance(7 * MS)  # ... computes for 7 ms ...
        cmd["state_seq"] = st["seq"][0]
        cmd["t_mono_ns"] = clock.monotonic_ns()
        cmd["mode"] = int(CmdMode.RUN)
        ipc.write_cmd(cmd)  # ... and writes the command; the servo picks it up 5 ms later
        clock.advance(5 * MS)
        for i in range(3):
            core.tick()
            clock.advance(core.period_ns - 12 * MS if i == 2 else core.period_ns)  # 12 ms of policy time next
    log = np.array(ring.recs, dtype=TICK_LOG_DTYPE)
    r = measure_obs_to_cmd(log)
    # policy write -> servo receipt: 5 ms; observation published -> command applied: one servo period
    assert r["cmd_age"]["n"] == 20 and r["cmd_age"]["p50"] == pytest.approx(5.0)
    assert r["obs_to_cmd"]["n"] == 20 and r["obs_to_cmd"]["p50"] == pytest.approx(core.period_ns / MS)


def test_write_report(tmp_path):
    p = write_report({"read": {"p99": 1.0}}, tmp_path / "hw")
    assert p.name.startswith("latency-") and json.loads(p.read_text())["read"]["p99"] == 1.0


@pytest.mark.hardware
def test_hw_feetech_read_latency_budget(cfg, calibration):  # pragma: no cover - hardware track only (SO101_HW=1)
    """Real STS3215 bus: p99 block read < 6 ms (plan latency table). Records runs/hw/latency-<ts>.json.
    Run attended, with the arm torque OFF (it only reads, and writes goal = present)."""
    from so101_rl.servo.feetech_bus import FeetechBus

    assert os.environ.get("SO101_HW") == "1"
    bus = FeetechBus(cfg, calibration)
    bus.connect()
    try:
        r = measure_bus(bus, n=300, write=True)
    finally:
        bus.disconnect()
    write_report({"bus": r, "port": cfg.port})
    assert r["failures"] == 0
    assert r["p99"] < 6.0

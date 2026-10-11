"""Env backends (FS12; Phase 0 Task 8.1; sim plan R1.1): ``SimBackend`` (in-process ServoCore + MujocoBus) and
``ShmBackend`` (policy side of the servo shm segment).

``ShmBackend`` is exercised against a ``servo --backend fake`` child process (and one ``--backend mujoco`` run),
never a serial device: the hardware rule of the plan.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from so101_rl.env.backend import (
    EnvBackend,
    ResetFailed,
    ServoNotRunning,
    ServoStateStale,
    ShmBackend,
    SimBackend,
)
from so101_rl.safety.fk import So101FK
from so101_rl.safety.types import SmState, TripCause
from so101_rl.servo.core import PHASE_LOOP, TICK_LOG_DTYPE
from so101_rl.servo.shm import ServoShm, new_state
from so101_rl.servo.timing import FakeClock

ROOT = Path(__file__).parent.parent
CALIB = ROOT / "tests" / "data" / "so101_follower.calib.json"
NS = 1_000_000_000


def _state(b) -> np.ndarray:
    st = new_state()
    assert b.read_state(st)
    return st[0]


# ================================================================================================ SimBackend


@pytest.fixture
def sim(cfg, calibration):
    b = SimBackend(cfg, calibration)
    b.reset(None)
    yield b
    b.close()


def test_backends_implement_the_f7_protocol(sim):
    assert isinstance(sim, EnvBackend)
    assert isinstance(sim.fk(), So101FK)


def test_sim_reset_populates_a_run_state_at_home(sim, cfg):
    rec = _state(sim)
    assert int(rec["sm_state"]) == SmState.RUN and int(rec["fault_bits"]) == 0 and bool(rec["read_ok"])
    home = np.asarray(cfg.safe_state.home_ticks)
    assert np.abs(rec["present_ticks"].astype(int) - home).max() <= 5
    assert np.abs(rec["target_ticks"].astype(int) - home).max() <= 5


def test_sim_10_policy_steps_advance_the_servo_tick_by_30(sim):
    t0 = int(_state(sim)["tick"])
    for _ in range(10):
        sim.send_delta(np.zeros(6), int(_state(sim)["seq"]), 1)
        sim.wait_policy_tick()
    assert int(_state(sim)["tick"]) - t0 == 30


def test_sim_send_delta_returns_monotone_cmd_seq_echoed_by_the_servo(sim):
    seqs = []
    for _ in range(5):
        seqs.append(sim.send_delta(np.zeros(6), 0, 1))
        sim.wait_policy_tick()
        assert int(_state(sim)["last_cmd_seq"]) == seqs[-1]
    assert seqs == sorted(seqs) and len(set(seqs)) == 5


@pytest.mark.parametrize("j", range(6))
def test_sim_unit_delta_is_applied_exactly_through_the_servo(sim, cfg, j):
    """Test 1 of the action units, through the backend: delta = max_delta_ticks[j] lands on the goal exactly."""
    before = _state(sim)["goal_ticks"].astype(int)
    d = np.zeros(6, np.float32)
    d[j] = cfg.action.max_delta_ticks[j]
    sim.send_delta(d, 0, 1)
    sim.wait_policy_tick()
    rec = _state(sim)
    assert float(rec["applied_delta_ticks"][j]) == float(cfg.action.max_delta_ticks[j])
    after = rec["goal_ticks"].astype(int)
    assert after[j] - before[j] == int(cfg.action.max_delta_ticks[j])
    assert np.array_equal(np.delete(after, j), np.delete(before, j))


def test_sim_nan_trips_the_servo_and_the_next_reset_boots_a_fresh_core(sim):
    goal = _state(sim)["goal_ticks"].copy()
    sim.send_delta(np.array([np.nan, 0, 0, 0, 0, 0]), 0, 1)
    sim.wait_policy_tick()
    rec = _state(sim)
    assert int(rec["fault_bits"]) & TripCause.NAN
    assert int(rec["sm_state"]) == SmState.SAFE_STOPPING
    assert np.array_equal(rec["goal_ticks"], goal)  # the NaN never moved the goal
    assert not rec["applied_delta_ticks"].any()
    sim.reset(None)
    rec = _state(sim)
    assert sim.boots == 2 and int(rec["sm_state"]) == SmState.RUN and int(rec["fault_bits"]) == 0


def test_sim_start_rest_boots_through_the_reenergise_ramp(cfg, calibration):
    b = SimBackend(cfg, calibration, start="rest")
    b.reset(None)
    rec = _state(b)
    assert int(rec["sm_state"]) == SmState.RUN
    assert np.abs(rec["present_ticks"].astype(int) - np.asarray(cfg.safe_state.home_ticks)).max() <= 5
    assert b.clock.ticks > 60  # the ramp from rest took real (sim) time


def test_sim_reset_q0_sets_the_limp_start_pose(cfg, calibration):
    b = SimBackend(cfg, calibration)
    q0 = np.asarray(cfg.safe_state.home_ticks).copy()
    q0[0] += 100  # inside the box: a valid re-energise start pose
    b.reset(q0)
    assert int(_state(b)["sm_state"]) == SmState.RUN  # re-energise ramped it back home
    with pytest.raises(ValueError, match="shape"):
        b.reset(np.zeros(5))
    with pytest.raises(ValueError, match="limits"):
        b.reset(np.full(6, 5000))


def test_sim_backend_rejects_a_bad_start(cfg, calibration):
    with pytest.raises(ValueError, match="start"):
        SimBackend(cfg, calibration, start="up")


def test_sim_backend_needs_reset_first(cfg, calibration):
    b = SimBackend(cfg, calibration)
    assert not b.read_state(new_state())
    with pytest.raises(RuntimeError, match="reset"):
        b.wait_policy_tick()
    with pytest.raises(RuntimeError, match="reset"):
        b.send_delta(np.zeros(6), 0, 0)


def test_sim_action_delay_is_forwarded_to_the_bus(cfg, calibration):
    b = SimBackend(cfg, calibration, action_delay_ticks=2)
    assert b.bus.action_delay_ticks == 2


class _Params:
    """Stand-in for FS7 ``SimParams`` (SU2): only ``apply_cpu(model)`` is used by SimBackend."""

    def __init__(self, kp: float) -> None:
        self.kp = kp
        self.calls = 0

    def apply_cpu(self, model) -> None:
        self.calls += 1
        model.actuator_gainprm[:, 0] = self.kp
        model.actuator_biasprm[:, 1] = -self.kp


def _first_step_progress(b: SimBackend, cfg) -> int:
    d = np.zeros(6, np.float32)
    d[0] = cfg.action.max_delta_ticks[0]
    p0 = int(_state(b)["present_ticks"][0])
    b.send_delta(d, 0, 1)
    b.wait_policy_tick()
    return abs(int(_state(b)["present_ticks"][0]) - p0)


def test_sim_backend_applies_sim_params(cfg, calibration):
    params = _Params(kp=20.0)
    b = SimBackend(cfg, calibration, sim_params=params)
    assert params.calls == 1
    b.reset(None)  # torque on restores the bus's SAVED gains: they must be the sim_params ones
    assert np.allclose(b.model.actuator_gainprm[b.bus.act_ids, 0], 20.0)
    assert np.allclose(b.model.actuator_biasprm[b.bus.act_ids, 1], -20.0)
    nominal = SimBackend(cfg, calibration)
    nominal.reset(None)
    assert nominal.model.actuator_gainprm[nominal.bus.act_ids[0], 0] > 900  # the MJCF sts3215 kp 998.22
    # a softer servo follows the same command less far in the same time
    assert _first_step_progress(b, cfg) < _first_step_progress(nominal, cfg)


def test_sim_tick_clock_is_monotone_across_resets(sim):
    t1 = sim.clock.monotonic_ns()
    sim.reset(None)
    assert sim.clock.monotonic_ns() > t1


# ================================================================================================ ShmBackend: unit


class _FakeServo:
    """An in-process segment owner publishing hand-made state records (no ServoCore)."""

    def __init__(self, cfg) -> None:
        self.name = f"so101envtest_{uuid.uuid4().hex[:10]}"
        self.shm = ServoShm.create(self.name)
        self.st = new_state()
        self.st["sm_state"] = int(SmState.RUN)
        self.st["read_ok"] = 1

    def publish(self, tick: int, t_ns: int, **fields) -> None:
        self.st["tick"] = tick
        self.st["t_mono_ns"] = t_ns
        for k, v in fields.items():
            self.st[k] = v
        self.shm.write_state(self.st)

    def close(self) -> None:
        self.shm.close()


@pytest.fixture
def fake_servo(cfg):
    s = _FakeServo(cfg)
    yield s
    s.close()


def _shm_backend(cfg, calibration, name, clock):
    return ShmBackend(dataclasses.replace(cfg, shm_name=name), clock, calibration=calibration, attach_timeout_s=1.0)


def test_shm_wait_raises_stale_when_no_newer_tick_arrives(cfg, calibration, fake_servo):
    clock = FakeClock(start_ns=10 * NS)
    b = _shm_backend(cfg, calibration, fake_servo.name, clock)
    try:
        fake_servo.publish(5, clock.monotonic_ns() + b.policy_period_ns)
        b.wait_policy_tick()  # tick 5 is new to it
        with pytest.raises(ServoStateStale, match="no servo state newer"):
            b.wait_policy_tick()  # the servo never published tick 6 (stalled): bounded, never hangs
    finally:
        b.close()


def test_shm_wait_raises_stale_on_an_old_record(cfg, calibration, fake_servo):
    clock = FakeClock(start_ns=10 * NS)
    fake_servo.publish(5, clock.monotonic_ns() - NS)  # a newer tick, but published 1 s ago
    b = _shm_backend(cfg, calibration, fake_servo.name, clock)
    try:
        with pytest.raises(ServoStateStale, match="old"):
            b.wait_policy_tick()
    finally:
        b.close()


def test_shm_wait_keeps_an_absolute_grid_and_counts_skipped_periods(cfg, calibration, fake_servo):
    clock = FakeClock(start_ns=10 * NS)
    b = _shm_backend(cfg, calibration, fake_servo.name, clock)
    tick = 0
    for _ in range(3):
        tick += 3
        fake_servo.publish(tick, clock.monotonic_ns() + b.policy_period_ns)
        b.wait_policy_tick()
    t3 = clock.monotonic_ns()
    clock.advance(int(2.5 * b.policy_period_ns))  # the caller was late by 2.5 periods
    tick += 3
    fake_servo.publish(tick, t3 + 3 * b.policy_period_ns)
    b.wait_policy_tick()
    assert b.overruns == 2
    assert b.tick_gaps == [3, 3, 3]
    b.close()


def test_shm_restarted_servo_is_stale_and_accepted_state_is_the_checked_record(cfg, calibration, fake_servo):
    clock = FakeClock(start_ns=10 * NS)
    b = _shm_backend(cfg, calibration, fake_servo.name, clock)
    try:
        st = new_state()
        assert not b.accepted_state(st)  # nothing accepted yet
        fake_servo.publish(5, clock.monotonic_ns() + b.policy_period_ns, last_cmd_seq=11)
        b.wait_policy_tick()
        fake_servo.publish(6, clock.monotonic_ns(), last_cmd_seq=12)  # a later record, not yet waited for
        assert b.accepted_state(st) and int(st["tick"][0]) == 5 and int(st["last_cmd_seq"][0]) == 11
        fake_servo.publish(2, clock.monotonic_ns() + b.policy_period_ns)  # tick went back: a new servo process
        with pytest.raises(ServoStateStale, match="restarted"):
            b.wait_policy_tick()
    finally:
        b.close()


def test_shm_send_delta_writes_a_run_command(cfg, calibration, fake_servo):
    from so101_rl.servo.shm import CmdMode, new_cmd

    b = _shm_backend(cfg, calibration, fake_servo.name, FakeClock())
    s1 = b.send_delta(np.arange(6, dtype=np.float32), 42, 7)
    s2 = b.send_hold(43, 7)
    assert s2 > s1
    cmd = new_cmd()
    assert fake_servo.shm.read_cmd(cmd)
    assert int(cmd["mode"][0]) == CmdMode.HOLD and int(cmd["state_seq"][0]) == 43 and int(cmd["episode"][0]) == 7
    b.close()


def test_shm_reset_refuses_a_servo_outside_run_and_hold(cfg, calibration, fake_servo):
    clock = FakeClock(start_ns=10 * NS)
    b = _shm_backend(cfg, calibration, fake_servo.name, clock)
    fake_servo.publish(3, clock.monotonic_ns() + b.policy_period_ns, sm_state=int(SmState.TORQUE_OFF),
                       fault_bits=int(TripCause.NAN))
    with pytest.raises(ServoNotRunning, match="TORQUE_OFF"):
        b.reset(np.asarray(cfg.safe_state.home_ticks))
    b.close()


def test_shm_attach_times_out_without_a_servo(cfg, calibration):
    with pytest.raises(FileNotFoundError):
        _shm_backend(cfg, calibration, f"so101none_{uuid.uuid4().hex[:8]}", FakeClock())


# ================================================================================================ ShmBackend vs servo process


def _servo_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "so101_rl.servo.main", "--calibration", str(CALIB), *args]


class ServoProc:
    """``servo --backend <kind>`` child process on a private shm name (FakeMcu in-process, never a serial port)."""

    def __init__(self, tmp_path: Path, backend: str = "fake", *extra: str) -> None:
        self.name = f"so101envtest_{uuid.uuid4().hex[:10]}"
        self.tick_log = tmp_path / "ticks.bin"
        self.summary = tmp_path / "summary.json"
        env = {k: v for k, v in os.environ.items() if k != "SO101_HW"}
        self.proc = subprocess.Popen(
            _servo_cmd("--backend", backend, "--shm-name", self.name, "--tick-log", str(self.tick_log),
                       "--summary", str(self.summary), *extra),
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def wait_run(self, timeout_s: float) -> None:
        t_end = time.monotonic() + timeout_s
        shm = None
        st = new_state()
        try:
            while time.monotonic() < t_end:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"servo exited {self.proc.returncode}: {self.proc.stderr.read()[-2000:]!r}")
                if shm is None:
                    try:
                        shm = ServoShm.attach(self.name)
                    except (FileNotFoundError, RuntimeError):
                        time.sleep(0.02)
                        continue
                if shm.read_state(st) and int(st["sm_state"][0]) == SmState.RUN:
                    return
                time.sleep(0.01)
            raise TimeoutError("servo never reached RUN")
        finally:
            if shm is not None:
                shm.close()

    def stop(self) -> int:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGCONT)
            self.proc.send_signal(signal.SIGINT)
            try:
                return self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                return self.proc.wait()
        return self.proc.returncode

    def ticks(self) -> np.ndarray:
        return np.fromfile(self.tick_log, dtype=TICK_LOG_DTYPE)


@pytest.fixture
def servo_fake(tmp_path):
    s = ServoProc(tmp_path, "fake")
    try:
        s.wait_run(30.0)
        yield s
    finally:
        s.stop()


def _backend_for(cfg, calibration, servo: ServoProc) -> ShmBackend:
    return ShmBackend(dataclasses.replace(cfg, shm_name=servo.name), calibration=calibration)


def test_shm_against_servo_fake_20_steps_cmd_seq_monotone_and_echoed(cfg, calibration, servo_fake):
    b = _backend_for(cfg, calibration, servo_fake)
    b.reset(np.asarray(cfg.safe_state.home_ticks))
    rng = np.random.default_rng(0)
    st = new_state()
    seqs, ticks = [], []
    for _ in range(20):
        assert b.read_state(st)
        d = np.rint(rng.uniform(-1, 1, 6) * cfg.action.max_delta_ticks * 0.3).astype(np.float32)
        seqs.append(b.send_delta(d, int(st["seq"][0]), b.episode))
        b.wait_policy_tick()
        assert b.read_state(st)
        assert int(st["last_cmd_seq"][0]) == seqs[-1]  # the servo received this very command
        assert int(st["sm_state"][0]) == SmState.RUN and int(st["fault_bits"][0]) == 0
        ticks.append(int(st["tick"][0]))
    assert all(b2 > a for a, b2 in itertools.pairwise(seqs))
    assert all(b2 > a for a, b2 in itertools.pairwise(ticks))
    assert 2 <= np.median(np.diff(ticks)) <= 4  # ~3 servo ticks per policy step
    b.close()


def test_shm_reset_ramps_without_a_per_tick_jump(cfg, calibration, servo_fake):
    b = _backend_for(cfg, calibration, servo_fake)
    home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int64)
    b.reset(home)
    away = home.copy()
    away[0] += 300  # shoulder_pan: stays inside the box
    away[3] -= 150
    b.reset(away)
    st = new_state()
    assert b.read_state(st)
    assert np.abs(st["target_ticks"][0].astype(int) - away).max() <= 2
    assert np.abs(st["present_ticks"][0].astype(int) - st["target_ticks"][0].astype(int)).max() < 5
    b.reset(home)
    b.close()
    servo_fake.stop()
    log = servo_fake.ticks()
    run = log[(log["phase"] == PHASE_LOOP) & np.isin(log["sm_state"], [int(SmState.RUN), int(SmState.HOLD)])]
    assert len(run) > 100
    step = np.abs(np.diff(run["target"].astype(np.int64), axis=0))
    assert step.max() <= cfg.action.max_ticks_per_tick.max()  # never a jump
    assert (step <= cfg.action.max_ticks_per_tick).all()


@pytest.mark.parametrize("deadband,ok", [(27, True), (45, False)], ids=["real-arm-27", "beyond-tolerance-45"])
def test_shm_reset_accepts_the_real_arm_steady_error_but_not_more(cfg, calibration, tmp_path, deadband, ok):
    """The real arm (P = 16) settles up to 27 ticks short of the reset target and stays there (2026-10-11): the
    reset converges once the arm is still within RESET_TRACK_TOL_TICKS, and fails beyond it."""
    from so101_rl.env.backend import RESET_TRACK_TOL_TICKS

    assert deadband < RESET_TRACK_TOL_TICKS if ok else deadband >= RESET_TRACK_TOL_TICKS
    servo = ServoProc(tmp_path, "fake", "--fake-deadband-ticks", str(deadband))
    try:
        servo.wait_run(30.0)
        b = _backend_for(cfg, calibration, servo)
        q0 = np.asarray(cfg.safe_state.home_ticks, dtype=np.int64) + np.array([150, -100, 120, 80, 60, 0])
        try:
            if ok:
                b.reset(q0)
                st = new_state()
                assert b.read_state(st)
                err = np.abs(st["present_ticks"][0].astype(int) - st["target_ticks"][0].astype(int))
                assert deadband - 1 <= err.max() <= deadband + 1  # really settled short of the goal
            else:
                with pytest.raises(ResetFailed, match="did not converge"):
                    b.reset(q0)
        finally:
            b.close()
    finally:
        servo.stop()


def test_shm_reset_after_a_trip_refuses_and_a_stalled_servo_is_stale(cfg, calibration, servo_fake):
    b = _backend_for(cfg, calibration, servo_fake)
    b.reset(np.asarray(cfg.safe_state.home_ticks))
    b.send_delta(np.array([np.nan, 0, 0, 0, 0, 0], np.float32), 0, b.episode)
    b.wait_policy_tick()
    st = new_state()
    assert b.read_state(st) and int(st["fault_bits"][0]) & TripCause.NAN
    with pytest.raises((ServoNotRunning, ResetFailed)):
        b.reset(np.asarray(cfg.safe_state.home_ticks))
    servo_fake.proc.send_signal(signal.SIGSTOP)  # the servo stops publishing
    try:
        time.sleep(0.05)
        with pytest.raises(ServoStateStale):
            for _ in range(3):
                b.wait_policy_tick()
    finally:
        servo_fake.proc.send_signal(signal.SIGCONT)
    b.close()


@pytest.mark.slow
def test_shm_against_servo_mujoco_reaches_home_and_steps(cfg, calibration, tmp_path):
    s = ServoProc(tmp_path, "mujoco")
    try:
        s.wait_run(90.0)
        b = _backend_for(cfg, calibration, s)
        home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int64)
        b.reset(home)
        st = new_state()
        assert b.read_state(st)
        assert np.abs(st["present_ticks"][0].astype(int) - home).max() < 5 + 2
        d = np.zeros(6, np.float32)
        d[0] = cfg.action.max_delta_ticks[0]
        b.send_delta(d, int(st["seq"][0]), b.episode)
        b.wait_policy_tick()
        assert b.read_state(st)
        assert float(st["applied_delta_ticks"][0][0]) == float(cfg.action.max_delta_ticks[0])
        b.close()
    finally:
        assert s.stop() == 0

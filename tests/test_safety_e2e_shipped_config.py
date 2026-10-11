"""End to end with the SHIPPED ``so101.yaml`` (review B1 e, B2, B6): FakeBus + FakeMcu + the safety modules, driven
by a minimal stand-in for the servo core loop (U3 is not written yet).

Before the wave-1 fixes this failed on the first RUN tick: the placeholder home (all 2047) put the tip outside the
box, so every start went RUN -> BOX_PRESENT -> SAFE_STOPPING -> TORQUE_OFF.

Core-loop rules modelled here, which ServoCore (U3) must follow too:
* one heartbeat per tick, only after that tick's read AND write succeeded;
* in TORQUE_OFF the loop keeps writing ``goal = present`` (harmless with torque off) so the heartbeat continues
  through the thermal dwell; otherwise the MCU latches and the automatic resume becomes a human reset.
"""

import numpy as np
import pytest

from so101_rl.safety.config_check import validate_geometry
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.pipeline import SafetyPipeline
from so101_rl.safety.reenergize import reenergize
from so101_rl.safety.safe_state import SafeStateSequencer
from so101_rl.safety.state_machine import SafetyStateMachine, SmInputs, SmOutputs
from so101_rl.safety.stuck import StuckActionMonitor
from so101_rl.safety.thermal import CONFIRM_READS, ThermalMonitor
from so101_rl.safety.types import SmState, TripCause, Verdict
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.heartbeat import RESET_HOLD_MS, FakeHeartbeatLink, FakeMcu
from so101_rl.servo.timing import FakeClock

S = 1_000_000_000


class MiniCore:
    def __init__(self, cfg, calibration):
        validate_geometry(cfg, calibration)  # config.load_config is pure; the safety stack validates the geometry
        self.cfg = cfg
        self.cal = calibration
        self.clock = FakeClock(start_ns=50 * S)
        self.period_ns = int(S / cfg.rates.servo_hz)
        self.mcu = FakeMcu(self.clock)
        self.mcu.reset_button(RESET_HOLD_MS)  # works whether the MCU boots DISARMED or LATCHED
        self.link = FakeHeartbeatLink(self.mcu)
        assert self.link.arm().armed
        self.seq_no = 0
        self.bus = FakeBus(self.clock, calibration=calibration, initial_ticks=cfg.safe_state.rest_ticks)
        self.fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)
        self.thermal = ThermalMonitor(cfg.thermal, self.clock)
        self.pipe = SafetyPipeline(cfg, self.fk, self.thermal, StuckActionMonitor())
        self.sm = SafetyStateMachine(cfg, self.clock)
        self.safe = SafeStateSequencer(
            cfg.safe_state,
            self.clock,
            tracking_error_ticks=cfg.action.tracking_error_ticks,
            motor_ids={n: c.id for n, c in calibration.items()},
            fk=self.fk,
        )
        self.block = BlockRead()
        self.v = Verdict()
        self.out = SmOutputs()
        self.states: list[SmState] = []
        self.trips = TripCause.NONE

    def heartbeat(self, ok: bool) -> None:
        if ok:
            assert self.link.send(self.seq_no), "MCU refused the heartbeat"
            self.seq_no = (self.seq_no + 1) & 0xFF

    def energize(self):
        reenergize(self.bus, self.cfg, self.cal, self.clock, self.block, tick_hook=self.heartbeat)
        self.sm.mark_energized()

    def tick(self, delta=None):
        got = bool(self.bus.read_block(self.block)) and self.block.ok
        if self.sm.state is SmState.RUN and got:
            if delta is not None:
                self.pipe.new_command(np.asarray(delta, dtype=np.float64), self.v)
            self.pipe.apply(self.block, 1, self.v)
        else:
            self.pipe.monitor_only(self.block, self.v)
        self.trips |= self.v.trips
        self.sm.step(
            SmInputs(block_ok=got, cmd_fresh=True, trips=self.v.trips, may_resume_thermal=self.thermal.may_resume()),
            self.out,
        )
        self.states.append(self.out.state)
        wrote = False
        if self.out.resync_target:
            self.pipe.resync(self.block.position)
        if self.out.run_safe_sequence:
            last = self.safe.run(self.bus, self.block, servo_hz=self.cfg.rates.servo_hz, tick_hook=self.heartbeat)
            assert not last.timed_out
            self.sm.notify_safe_done()
            return
        if self.out.state is SmState.RUN and self.out.use_policy:
            wrote = bool(self.bus.write_goal(self.v.target_ticks))
        elif self.out.torque_off_now and got:
            wrote = bool(self.bus.set_torque(False))  # acked: keeps the heartbeat alive (never a goal write: STS3215)
        self.heartbeat(got and wrote)
        if self.out.reenergize_requested:
            self.energize()
        self.clock.sleep_until(self.clock.monotonic_ns() + self.period_ns)

    def tip(self):
        return self.fk.gripper_xyz_from_ticks(self.block.position, np.zeros(3))


@pytest.fixture
def core(cfg, calibration):
    return MiniCore(cfg, calibration)


def test_start_from_rest_runs_at_home_without_box_present(core, cfg):
    core.energize()
    for i in range(90):  # 3 s of small policy commands around home
        core.tick(delta=[5, -5, 5, 0, 0, 0] if i % 6 < 3 else [-5, 5, -5, 0, 0, 0])
    assert not core.trips & TripCause.BOX_PRESENT
    assert set(core.states) == {SmState.RUN}
    assert core.fk.in_box(core.tip(), cfg.box)
    assert not core.link.status().latched


def test_thermal_trip_rests_torques_off_cools_and_resumes_by_full_reenergize(core, cfg):
    core.energize()
    for _ in range(10):
        core.tick(delta=[0] * 6)
    assert core.sm.state is SmState.RUN

    core.bus.set_temperature("elbow_flex", cfg.thermal.trip_c + 1)
    for _ in range(CONFIRM_READS):  # trip on the CONFIRM_READS-th hot read -> SAFE_STOPPING -> rest -> TORQUE_OFF
        core.tick()
    assert SmState.SAFE_STOPPING in core.states and core.sm.state is SmState.TORQUE_OFF
    assert not core.bus.torque_enabled().any()  # torque off at rest: never holding pose energised
    np.testing.assert_allclose(core.block.position[:5], cfg.safe_state.rest_ticks[:5], atol=30)

    core.bus.set_temperature("elbow_flex", cfg.thermal.resume_c - 5)
    n = 0
    while core.sm.state is SmState.TORQUE_OFF:
        core.tick()
        n += 1
        assert n < (cfg.thermal.resume_dwell_s + 5) * cfg.rates.servo_hz
        if core.sm.state is SmState.TORQUE_OFF:
            assert not core.bus.torque_enabled().any()
    assert n >= cfg.thermal.resume_dwell_s * cfg.rates.servo_hz - 2  # the dwell was enforced
    assert core.sm.state is SmState.RUN  # CHECK -> full reenergize -> mark_energized
    assert core.bus.torque_enabled().all()
    core.tick(delta=[0] * 6)
    assert core.sm.state is SmState.RUN and core.out.use_policy
    assert core.sm.thermal_trips_last_hour() == 1
    assert not core.link.status().latched  # the heartbeat survived trip, rest, dwell and re-energise

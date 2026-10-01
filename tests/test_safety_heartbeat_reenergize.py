"""Review B2: the deadman must not latch during a full re-energise or a full safe-state sequence.

The MCU (``FakeMcu``) must be ARMED before ``ping_all`` because the relay powers the servos. ``reenergize`` and
``SafeStateSequencer.run`` block for seconds, so they call ``tick_hook(ok)`` once per servo tick and the caller
sends a heartbeat when ``ok``. The MCU is brought to ARMED explicitly (physical reset button, then ARM), so the
test holds whether ``FakeMcu`` boots DISARMED or LATCHED (review S7, unit B).
"""

import numpy as np
import pytest

from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.reenergize import reenergize
from so101_rl.safety.safe_state import SafeStateSequencer
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.heartbeat import RESET_HOLD_MS, FakeHeartbeatLink, FakeMcu
from so101_rl.servo.timing import FakeClock

S = 1_000_000_000


class Heartbeat:
    """What ServoCore will pass as ``tick_hook``: one heartbeat per good tick, +1 sequence each."""

    def __init__(self, link: FakeHeartbeatLink) -> None:
        self.link = link
        self.seq = 0
        self.sent = 0
        self.ticks = 0

    def __call__(self, ok: bool) -> None:
        self.ticks += 1
        if ok:
            assert self.link.send(self.seq), "the MCU refused a heartbeat (latched or not armed)"
            self.seq = (self.seq + 1) & 0xFF
            self.sent += 1


@pytest.fixture
def clock():
    return FakeClock(start_ns=10 * S)


@pytest.fixture
def armed(clock):
    mcu = FakeMcu(clock)
    mcu.reset_button(RESET_HOLD_MS)  # physical reset: clears a boot latch (no-op if it booted DISARMED)
    link = FakeHeartbeatLink(mcu)
    status = link.arm()
    assert status.armed and status.relay_on and not status.latched
    return mcu, link


@pytest.fixture
def fk(cfg, calibration):
    return So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)


def test_full_reenergize_from_rest_with_heartbeat_hook_never_latches(cfg, calibration, clock, armed):
    mcu, link = armed
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.rest_ticks)
    hb = Heartbeat(link)
    t0 = clock.monotonic_ns()
    reenergize(bus, cfg, calibration, clock, BlockRead(), tick_hook=hb)
    elapsed_s = (clock.monotonic_ns() - t0) / S
    assert elapsed_s > 3 * mcu.timeout_ms / 1000  # long enough that a missing heartbeat WOULD have latched
    s = link.status()  # polls the MCU at "now"
    assert s.armed and s.relay_on and not s.latched
    assert hb.sent > 0.9 * hb.ticks


def test_control_without_the_hook_the_same_reenergize_latches(cfg, calibration, clock, armed):
    """Proves the test above can fail: no heartbeat for > 1000 ms latches the MCU."""
    _mcu, link = armed
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.rest_ticks)
    reenergize(bus, cfg, calibration, clock, BlockRead())
    s = link.status()
    assert s.latched and not s.relay_on


def test_full_safe_state_sequence_with_heartbeat_hook_never_latches(cfg, calibration, clock, armed, fk):
    mcu, link = armed
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.home_ticks)
    bus.set_torque(True)
    hb = Heartbeat(link)
    seq = SafeStateSequencer(
        cfg.safe_state,
        clock,
        tracking_error_ticks=cfg.action.tracking_error_ticks,
        motor_ids={n: c.id for n, c in calibration.items()},
        fk=fk,
    )
    t0 = clock.monotonic_ns()
    last = seq.run(bus, BlockRead(), servo_hz=cfg.rates.servo_hz, tick_hook=hb)
    assert last.done and not last.timed_out
    assert (clock.monotonic_ns() - t0) / S > 3 * mcu.timeout_ms / 1000
    s = link.status()
    assert s.armed and not s.latched
    assert not bus.torque_enabled().any()
    out = BlockRead()
    bus.read_block(out)
    assert np.abs(out.position - cfg.safe_state.rest_ticks).max() <= 25


def test_reenergize_then_safe_state_back_to_rest_one_armed_session(cfg, calibration, clock, armed, fk):
    """The whole thermal-resume shape on one MCU session: rest -> reenergize -> home -> safe state -> rest."""
    _mcu, link = armed
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.rest_ticks)
    hb = Heartbeat(link)
    reenergize(bus, cfg, calibration, clock, BlockRead(), tick_hook=hb)
    seq = SafeStateSequencer(
        cfg.safe_state,
        clock,
        tracking_error_ticks=cfg.action.tracking_error_ticks,
        motor_ids={n: c.id for n, c in calibration.items()},
        fk=fk,
    )
    assert not seq.run(bus, BlockRead(), servo_hz=cfg.rates.servo_hz, tick_hook=hb).timed_out
    assert not link.status().latched

import os
import threading
import time
import tty
from typing import ClassVar

import pytest
import serial

from so101_rl.servo import heartbeat as hb
from so101_rl.servo.heartbeat import (
    FakeHeartbeatLink,
    FakeMcu,
    HbSend,
    McuStatus,
    SerialHeartbeatLink,
)
from so101_rl.servo.timing import FakeClock, RealClock

MS = 1_000_000


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def raw_mcu(clock):
    """Straight out of an MCU reset: LATCHED, reason MCU_RESET."""
    return FakeMcu(clock)


@pytest.fixture
def mcu(raw_mcu):
    """An MCU whose boot latch was cleared with the physical reset button: DISARMED, ready for ARM."""
    raw_mcu.reset_button(500)
    return raw_mcu


@pytest.fixture
def link(mcu):
    return FakeHeartbeatLink(mcu)


def test_status_byte_bit_layout():
    s = McuStatus(relay_on=True, latched=False, armed=True, undervolt=False, reason=0)
    assert s.to_byte() == 0b0000_0101
    s = McuStatus(relay_on=False, latched=True, armed=False, undervolt=True, reason=hb.REASON_SEQ_ERROR)
    assert s.to_byte() == 0b0011_1010
    for b in range(256):
        assert McuStatus.from_byte(b).to_byte() == b
    assert McuStatus.from_byte(0x12) == McuStatus(False, True, False, False, 1)


def test_powers_up_latched_with_relay_open(raw_mcu):
    link = FakeHeartbeatLink(raw_mcu)
    assert (raw_mcu.relay_on, raw_mcu.armed, raw_mcu.latched) == (False, False, True)
    st = link.status()
    assert st == McuStatus(
        False, True, False, False, hb.REASON_MCU_RESET
    )  # firmware boot status 0x62 (PROTOCOL.md)
    assert st.to_byte() == 0x62


def test_boot_latch_refuses_serial_arm_and_only_the_button_clears_it(raw_mcu, clock):
    link = FakeHeartbeatLink(raw_mcu)
    for _ in range(3):
        assert link.arm().armed is False
        assert link.send(0) is HbSend.DELIVERED_NO_ECHO
        assert link.status().latched
        clock.advance(2000 * MS)
    assert raw_mcu.state == "LATCHED" and not raw_mcu.relay_on
    raw_mcu.reset_button(499)
    assert raw_mcu.state == "LATCHED"
    raw_mcu.reset_button(500)
    assert raw_mcu.state == "DISARMED" and raw_mcu.reason == hb.REASON_MCU_RESET  # reason kept until ARM
    assert link.arm().armed and raw_mcu.reason == hb.REASON_NONE


def test_boot_latch_not_cleared_while_estop_asserted(raw_mcu):
    raw_mcu.set_estop(True)
    raw_mcu.reset_button(2000)
    assert raw_mcu.state == "LATCHED"
    raw_mcu.set_estop(False)
    raw_mcu.reset_button(500)
    assert raw_mcu.state == "DISARMED"


def test_set_button_level_hold_is_timed_on_the_clock(raw_mcu, clock):
    raw_mcu.set_button(True)
    clock.advance(499 * MS)
    raw_mcu.poll()
    assert raw_mcu.latched
    clock.advance(1 * MS)
    raw_mcu.poll()
    assert raw_mcu.state == "DISARMED"
    raw_mcu.set_button(False)


def test_timeout_is_ge_exactly_1000_ms_latches(mcu, link, clock):
    link.arm()
    link.send(0)
    clock.advance(999 * MS)
    mcu.poll()
    assert mcu.state == "ARMED"
    clock.advance(1 * MS)  # exactly 1000 ms: the firmware uses >=
    mcu.poll()
    assert mcu.latched and mcu.reason == hb.REASON_TIMEOUT


def test_heartbeat_at_exactly_1000_ms_cannot_rescue(mcu, link, clock):
    link.arm()
    link.send(0)
    clock.advance(1000 * MS)
    assert link.send(1) is HbSend.DELIVERED_NO_ECHO
    assert mcu.reason == hb.REASON_TIMEOUT


def test_arm_checks_psu_side_not_servo_side(mcu, link):
    """B4: with the relay open the servo side reads 0 V; ARM must not look at it, and a sagging servo side while
    ARMED is not an undervolt."""
    mcu.set_servo_voltage(0.0)
    assert link.arm().armed
    mcu.set_servo_voltage(0.5)
    assert link.send(0) is HbSend.ACKED and mcu.state == "ARMED" and not mcu.undervolt


def test_psu_side_undervolt_refuses_arm_even_if_servo_side_is_fine(mcu, link):
    mcu.set_bus_voltage(8.0)  # servo side follows the relay (0 V): only the PSU-side reading can refuse here
    assert link.arm().armed is False and link.status().undervolt


def test_arm_refused_while_servo_side_still_live(mcu, link):
    mcu.set_servo_voltage(3.0)
    assert link.arm().armed is False
    mcu.set_servo_voltage(0.0)
    assert link.arm().armed


def test_relay_weld_detected_and_cannot_be_cleared_while_live(mcu, link, clock):
    link.arm()
    link.send(0)
    mcu.set_servo_voltage(12.0)  # relay will open on the timeout, but the servo bus stays up
    clock.advance(1001 * MS)
    mcu.poll()
    assert mcu.reason == hb.REASON_TIMEOUT
    clock.advance(499 * MS)
    mcu.poll()
    assert mcu.reason == hb.REASON_TIMEOUT
    clock.advance(1 * MS)
    mcu.poll()
    assert mcu.latched and mcu.reason == hb.REASON_WELD and not mcu.relay_on
    mcu.reset_button(600)
    assert mcu.latched and mcu.reason == hb.REASON_WELD  # the weld timer keeps running: re-latched at once
    assert link.arm().armed is False
    mcu.set_servo_voltage(None)  # bus finally dead
    mcu.reset_button(600)
    assert mcu.state == "DISARMED" and mcu.reason == hb.REASON_WELD  # reason kept until ARM
    assert link.arm().armed


def test_servo_side_decay_after_trip_is_not_a_weld(mcu, link, clock):
    mcu.drop_time_ms = 200
    link.arm()
    link.send(0)
    clock.advance(1001 * MS)
    mcu.poll()
    clock.advance(1000 * MS)
    mcu.poll()
    assert mcu.reason == hb.REASON_TIMEOUT


def test_partial_frame_dropped_after_100ms(mcu):
    mcu.feed(bytes([0x5A]))
    mcu.feed(bytes([0xA5, 0]))
    mcu.clock.advance(100 * MS)
    assert mcu.feed(bytes([0xFF])) == b""  # the stale head was discarded; 0xFF is line noise
    assert mcu.feed(bytes([0xA5, 0, 0xFF])) == bytes([mcu.status_byte()])


def test_partial_frame_gap_of_99ms_still_completes(mcu):
    mcu.feed(bytes([0x5A]))
    mcu.feed(bytes([0xA5, 0]))
    mcu.clock.advance(99 * MS)
    assert mcu.feed(bytes([0xFF])) == bytes([mcu.status_byte()])


def test_arm_only_from_disarmed(mcu, link):
    s = link.arm()
    assert s.armed and s.relay_on and not s.latched
    assert mcu.state == "ARMED"
    # a second ARM while ARMED changes nothing (in particular does not re-baseline the sequence)
    assert link.send(10) is HbSend.ACKED
    assert mcu.feed(bytes([0x5A])) == bytes([mcu.status_byte()])
    assert mcu.state == "ARMED" and mcu._last_seq == 10


def test_heartbeat_ignored_while_disarmed(mcu, link):
    assert link.send(0) is HbSend.DELIVERED_NO_ECHO  # not armed: echo says so
    assert mcu.state == "DISARMED"


def test_heartbeat_sequence_ok_and_wraps(mcu, link, clock):
    link.arm()
    for seq in list(range(250, 256)) + list(range(5)):
        clock.advance(33 * MS)
        assert link.send(seq) is HbSend.ACKED
    assert mcu.state == "ARMED"


def test_wrong_seq_latches_reason_3(mcu, link):
    link.arm()
    assert link.send(0) is HbSend.ACKED
    assert link.send(1) is HbSend.ACKED
    assert link.send(5) is HbSend.DELIVERED_NO_ECHO
    assert mcu.latched and mcu.reason == hb.REASON_SEQ_ERROR and not mcu.relay_on
    s = link.status()
    assert s.latched and s.reason == 3 and not s.relay_on


def test_repeated_seq_latches(mcu, link):
    link.arm()
    assert link.send(7) is HbSend.ACKED
    assert link.send(7) is HbSend.DELIVERED_NO_ECHO
    assert mcu.reason == hb.REASON_SEQ_ERROR


def test_bad_complement_latches_reason_3(mcu, link):
    link.arm()
    mcu.feed(bytes([0xA5, 3, 0x00]))
    assert mcu.latched and mcu.reason == hb.REASON_SEQ_ERROR


def test_timeout_999_ms_still_armed_1001_latched(mcu, link, clock):
    link.arm()
    link.send(0)
    clock.advance(999 * MS)
    mcu.poll()
    assert mcu.state == "ARMED" and mcu.relay_on
    clock.advance(2 * MS)  # 1001 ms since the last heartbeat
    mcu.poll()
    assert mcu.latched and mcu.reason == hb.REASON_TIMEOUT and not mcu.relay_on


def test_timeout_is_measured_from_last_valid_heartbeat(mcu, link, clock):
    link.arm()
    for seq in range(40):
        clock.advance(500 * MS)
        assert link.send(seq) is HbSend.ACKED
    assert mcu.state == "ARMED"  # 20 s total, never a 1 s gap


def test_timeout_without_any_heartbeat_after_arm(mcu, link, clock):
    link.arm()
    clock.advance(1001 * MS)
    assert link.status().latched  # feed() polls first


def test_late_heartbeat_after_timeout_does_not_unlatch(mcu, link, clock):
    link.arm()
    link.send(0)
    clock.advance(1500 * MS)
    assert link.send(1) is HbSend.DELIVERED_NO_ECHO
    assert mcu.latched and mcu.reason == hb.REASON_TIMEOUT


def test_latched_ignores_arm(mcu, link, clock):
    link.arm()
    clock.advance(2000 * MS)
    mcu.poll()
    assert mcu.latched
    s = link.arm()
    assert s.latched and not s.armed and not s.relay_on
    assert mcu.state == "LATCHED"


def test_reset_button_500ms_returns_to_disarmed(mcu, link, clock):
    link.arm()
    clock.advance(2000 * MS)
    mcu.poll()
    mcu.reset_button(499)
    assert mcu.latched
    mcu.reset_button(500)
    assert mcu.state == "DISARMED" and not mcu.latched and not mcu.relay_on
    assert mcu.reason == hb.REASON_TIMEOUT  # the firmware keeps the trip reason until the next ARM
    assert link.arm().armed  # and it can be armed again
    assert mcu.reason == hb.REASON_NONE


def test_reset_button_does_not_arm_and_not_when_disarmed(mcu):
    mcu.reset_button(600)
    assert mcu.state == "DISARMED" and not mcu.relay_on


def test_undervoltage_latches_after_50ms_dwell(mcu, link, clock):
    link.arm()
    link.send(0)
    mcu.set_bus_voltage(5.0)
    assert link.status().undervolt
    clock.advance(40 * MS)
    mcu.poll()
    assert mcu.state == "ARMED"
    clock.advance(20 * MS)
    mcu.poll()
    assert mcu.latched and mcu.reason == hb.REASON_UNDERVOLT


def test_undervoltage_glitch_shorter_than_50ms_is_tolerated(mcu, link, clock):
    link.arm()
    link.send(0)
    mcu.set_bus_voltage(5.0)
    clock.advance(30 * MS)
    mcu.set_bus_voltage(12.0)
    clock.advance(100 * MS)
    assert link.send(1) is HbSend.ACKED


def test_arm_refused_when_undervolt(mcu, link):
    mcu.set_bus_voltage(5.0)
    assert link.arm().armed is False


def test_estop_pin_latches_reason_4(mcu, link):
    link.arm()
    link.send(0)
    mcu.set_estop(True)
    assert mcu.latched and mcu.reason == hb.REASON_ESTOP_PIN
    mcu.reset_button(600)
    assert mcu.latched  # pin still pressed


def test_test_drop_returns_ms_with_jumper(mcu, link):
    mcu.test_jumper = True
    mcu.drop_time_ms = 37
    link.arm()
    assert link.test_drop() == 37
    assert mcu.latched and mcu.reason == hb.REASON_TEST and not mcu.relay_on


def test_test_drop_reports_2000_when_bus_never_falls(mcu, link):
    mcu.test_jumper = True
    link.arm()
    mcu.set_servo_voltage(12.0)
    assert link.test_drop() == 2000


def test_test_drop_rejected_without_jumper_returns_promptly(mcu, link, clock):
    link.arm()
    t0 = clock.monotonic_ns()
    with pytest.raises(hb.TestDropRefused, match="TEST") as exc:
        link.test_drop()
    assert clock.monotonic_ns() - t0 < 10 * MS  # no 2 s wait
    assert exc.value.status.armed and isinstance(exc.value, RuntimeError)
    assert mcu.state == "ARMED"


def test_test_drop_when_not_armed_is_refused_without_sending_e1(mcu, link):
    mcu.test_jumper = True
    with pytest.raises(hb.TestDropRefused):
        link.test_drop()  # DISARMED
    assert mcu.state == "DISARMED" and mcu.reason == hb.REASON_MCU_RESET


def test_test_drop_without_jumper_matches_firmware_status_only(mcu):
    """Firmware: 0xE1 without the jumper is answered with the status byte, nothing else, state unchanged."""
    mcu.feed(bytes([0x5A]))
    assert mcu.feed(bytes([0xE1])) == bytes([mcu.status_byte()])
    assert mcu.state == "ARMED" and mcu.poll() == b""


def test_test_drop_while_latched_does_not_overwrite_reason(mcu):
    mcu.test_jumper = True
    mcu.feed(bytes([0x5A]))
    mcu.clock.advance(1000 * MS)
    mcu.poll()
    assert mcu.reason == hb.REASON_TIMEOUT
    mcu.feed(bytes([0xE1]))
    assert mcu.reason == hb.REASON_TIMEOUT


def test_power_cycle_reports_mcu_reset_latched_and_opens_relay(mcu, link):
    link.arm()
    mcu.power_cycle()
    s = link.status()
    assert not s.relay_on and not s.armed and s.latched and s.reason == hb.REASON_MCU_RESET
    assert link.arm().armed is False  # an MCU reset needs the button again
    mcu.reset_button(500)
    assert link.arm().armed


def test_fake_link_unplug_returns_false_and_reports_dead(mcu, link):
    link.arm()
    assert link.send(0) is HbSend.ACKED and link.dead is False
    link.unplug()
    assert link.send(1) is HbSend.NOT_DELIVERED and link.dead is True


def test_fake_link_write_timeout_returns_false_and_frame_is_lost(mcu, link, clock):
    link.arm()
    assert link.send(0) is HbSend.ACKED
    link.fail_writes(1)
    assert link.send(1) is HbSend.NOT_DELIVERED
    assert mcu._last_seq == 0  # never arrived
    assert link.send(1) is HbSend.ACKED  # sequence continues unharmed


def test_split_frame_is_buffered(mcu):
    mcu.feed(bytes([0x5A]))
    assert mcu.feed(bytes([0xA5, 0])) == b""
    assert mcu.feed(bytes([0xFF])) == bytes([mcu.status_byte()])
    assert mcu._last_seq == 0


def test_line_noise_gets_no_reply(mcu):
    assert mcu.feed(bytes([0x00, 0x11, 0x22])) == b""


# -- SerialHeartbeatLink over a fake pyserial wired to FakeMcu ---------------------------------
class FakeSerial:
    instances: ClassVar[list["FakeSerial"]] = []
    mcu: ClassVar[FakeMcu]

    def __init__(self, port, baudrate, timeout=None, write_timeout=None):
        self.port, self.baudrate, self.timeout, self.write_timeout = port, baudrate, timeout, write_timeout
        self._rx = bytearray()
        self.writes = []
        self.raise_timeout = False
        self.unplugged: Exception | None = None  # set to an exception: every port access raises it
        self.touched_after_unplug = 0
        self.swallow_reply = 0  # next n replies never arrive
        self.closed = False
        FakeSerial.instances.append(self)

    def _check(self):
        if self.unplugged is not None:
            self.touched_after_unplug += 1
            raise self.unplugged

    @property
    def in_waiting(self):
        self._check()
        self._rx += FakeSerial.mcu.poll()  # the MCU loop runs concurrently: late drop_ms bytes arrive here
        return len(self._rx)

    def read(self, n=1):
        self._check()
        data, self._rx = bytes(self._rx[:n]), self._rx[n:]
        return data

    def write(self, data):
        self._check()
        if self.raise_timeout:
            raise serial.SerialTimeoutException("write timeout")
        self.writes.append(bytes(data))
        reply = FakeSerial.mcu.feed(bytes(data))
        if self.swallow_reply > 0 and reply:
            self.swallow_reply -= 1
            return len(data)
        self._rx += reply
        return len(data)

    def close(self):
        self.closed = True


@pytest.fixture
def ser_link(monkeypatch, mcu, clock):
    FakeSerial.instances.clear()
    FakeSerial.mcu = mcu
    monkeypatch.setattr(serial, "Serial", FakeSerial)
    link = SerialHeartbeatLink("/dev/ttyACM1", clock)
    return link, FakeSerial.instances[0]


def test_serial_link_opens_port_with_spec_settings(ser_link):
    _, ser = ser_link
    assert (ser.port, ser.baudrate, ser.timeout, ser.write_timeout) == ("/dev/ttyACM1", 115200, 0, 0.002)


def test_serial_link_happy_path(ser_link, mcu, clock):
    link, ser = ser_link
    assert link.arm().armed
    for seq in range(20):
        clock.advance(33 * MS)
        assert link.send(seq) is HbSend.ACKED
    assert ser.writes[0] == b"\x5a"
    assert ser.writes[1] == bytes([0xA5, 0, 0xFF])
    assert mcu.state == "ARMED"
    assert link.status().armed


def test_serial_link_write_timeout_returns_false(ser_link):
    link, ser = ser_link
    link.arm()
    assert link.send(0) is HbSend.ACKED
    ser.raise_timeout = True
    assert link.send(1) is HbSend.NOT_DELIVERED


def test_serial_link_missing_echo_returns_false_then_recovers(ser_link):
    link, ser = ser_link
    link.arm()
    assert link.send(0) is HbSend.ACKED
    ser.swallow_reply = 1
    assert link.send(1) is HbSend.ACKED  # echo for 1 not yet due when send(1) returns... it was swallowed
    assert link.send(2) is HbSend.DELIVERED_NO_ECHO  # ...so the next send sees an unacknowledged previous heartbeat
    assert link.send(3) is HbSend.ACKED  # the lost echo was forgotten; link recovers


def test_serial_link_latched_echo_returns_false(ser_link, mcu, clock):
    link, _ = ser_link
    link.arm()
    assert link.send(0) is HbSend.ACKED
    clock.advance(1500 * MS)
    assert link.send(1) is HbSend.ACKED  # this send's own echo has not been read yet
    assert link.send(2) is HbSend.DELIVERED_NO_ECHO  # the echo of the previous one says LATCHED
    assert mcu.latched


@pytest.mark.parametrize(
    "exc",
    [
        serial.SerialException("device disconnected"),
        OSError(5, "Input/output error"),
        ValueError("closed port"),
    ],
    ids=["SerialException", "OSError", "ValueError"],
)
@pytest.mark.parametrize("where", ["write", "in_waiting"])
def test_serial_link_unplug_returns_false_and_reports_dead(ser_link, exc, where, monkeypatch):
    link, ser = ser_link
    link.arm()
    assert link.send(0) is HbSend.ACKED and link.dead is False
    if where == "write":
        monkeypatch.setattr(ser, "write", lambda data: (_ for _ in ()).throw(exc))
    else:
        monkeypatch.setattr(FakeSerial, "in_waiting", property(lambda self: (_ for _ in ()).throw(exc)))
    assert link.send(1) is HbSend.NOT_DELIVERED  # must not raise in the tick path
    assert link.dead is True and link.last_error is exc


def test_serial_link_dead_link_no_longer_touches_the_port(ser_link):
    link, ser = ser_link
    link.arm()
    ser.unplugged = serial.SerialException("gone")
    assert link.send(0) is HbSend.NOT_DELIVERED and link.dead
    before = ser.touched_after_unplug
    for seq in range(1, 6):
        assert link.send(seq) is HbSend.NOT_DELIVERED
    assert ser.touched_after_unplug == before


def test_serial_link_write_timeout_is_transient_not_dead(ser_link):
    link, ser = ser_link
    link.arm()
    assert link.send(0) is HbSend.ACKED
    ser.raise_timeout = True
    assert link.send(1) is HbSend.NOT_DELIVERED and link.dead is False
    ser.raise_timeout = False
    assert link.send(2) is HbSend.ACKED  # a timed-out write is just a missed heartbeat: the link recovers at once


def test_serial_link_arm_times_out_without_reply(ser_link):
    link, ser = ser_link
    ser.swallow_reply = 1
    with pytest.raises(TimeoutError):
        link.arm()


def test_serial_link_test_drop(ser_link, mcu):
    link, _ = ser_link
    mcu.test_jumper = True
    mcu.drop_time_ms = 258  # needs both bytes of the uint16
    link.arm()
    assert link.test_drop() == 258


def test_serial_link_test_drop_refused_without_jumper_is_prompt(ser_link, mcu, clock):
    link, ser = ser_link
    link.arm()
    t0 = clock.monotonic_ns()
    with pytest.raises(hb.TestDropRefused, match="TEST") as exc:
        link.test_drop()
    assert (
        clock.monotonic_ns() - t0 < 10 * MS
    )  # the refused 0xE1 gets a status byte: no 2 s wait, no TimeoutError
    assert exc.value.status.armed
    assert mcu.state == "ARMED"
    assert ser.writes[-1] == b"\xe1"


def test_serial_link_test_drop_when_disarmed_does_not_send_e1(ser_link, mcu):
    link, ser = ser_link
    mcu.test_jumper = True
    with pytest.raises(hb.TestDropRefused):
        link.test_drop()
    assert b"\xe1" not in ser.writes


def test_serial_link_test_drop_reports_2000_if_bus_never_falls(ser_link, mcu):
    link, _ = ser_link
    mcu.test_jumper = True
    link.arm()
    mcu.set_servo_voltage(12.0)
    assert link.test_drop() == 2000


def test_serial_link_close(ser_link):
    link, ser = ser_link
    link.close()
    assert ser.closed


# -- tri-state send (U3 review item 2) ---------------------------------------------------------
def test_hbsend_values_are_bool_compatible():
    assert HbSend.ACKED == True  # legacy bool test doubles keep their meaning
    assert HbSend.NOT_DELIVERED == False
    assert HbSend.DELIVERED_NO_ECHO not in (True, False)


def test_fake_link_lost_echo_is_delivered_and_mcu_counted_it(mcu, link):
    link.arm()
    assert link.send(0) is HbSend.ACKED
    link.drop_echoes(1)
    assert link.send(1) is HbSend.DELIVERED_NO_ECHO
    assert mcu._last_seq == 1  # it arrived: the next one must be 2, repeating 1 would latch
    assert link.send(2) is HbSend.ACKED and not mcu.latched


def test_fake_link_last_status_tracks_every_reply(mcu, link, clock):
    assert link.last_status is None
    link.arm()
    assert link.last_status.armed
    link.send(0)
    clock.advance(1100 * MS)
    assert link.send(1) is HbSend.DELIVERED_NO_ECHO
    assert link.last_status.latched and link.last_status.reason == hb.REASON_TIMEOUT


def test_serial_link_exposes_last_status(ser_link, mcu, clock):
    link, _ = ser_link
    assert link.last_status is None
    link.arm()
    assert link.last_status.armed and not link.last_status.latched


def test_relay_closed_follows_the_mcu_state(mcu, link, clock):
    assert not mcu.relay_closed()
    link.arm()
    assert mcu.relay_closed()
    clock.advance(1000 * MS)
    assert not mcu.relay_closed()  # the timeout is applied by relay_closed() itself


# -- SerialHeartbeatLink on a real pyserial over a pty (no serial device): delivery from bytes written --------
class _PtyMcu:
    """FakeMcu on the master side of a pty, pumped synchronously by the test."""

    def __init__(self, clock):
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)
        os.set_blocking(self.master, False)
        self.port = os.ttyname(self.slave)
        self.mcu = FakeMcu(clock)
        self.mcu.reset_button(500)
        self.rx = bytearray()

    def pump(self):
        try:
            data = os.read(self.master, 256)
        except BlockingIOError:
            data = b""
        self.rx += data
        reply = self.mcu.feed(data) if data else b""
        if reply:
            os.write(self.master, reply)

    def close(self):
        os.close(self.master)
        os.close(self.slave)


@pytest.fixture
def pty_link(clock):
    p = _PtyMcu(clock)
    link = SerialHeartbeatLink(p.port, RealClock())
    yield link, p
    link.close()
    p.close()


def _arm_over_pty(link, p):
    def pump_for_200_ms():
        for _ in range(200):
            p.pump()
            time.sleep(0.001)

    t = threading.Thread(target=pump_for_200_ms)
    t.start()
    st = link.arm()
    t.join()
    assert st.armed


def test_serial_link_pyserial_timeout_after_a_full_write_is_still_delivered(pty_link, monkeypatch):
    """pyserial raises SerialTimeoutException also AFTER os.write took the whole frame (preemption). The link
    must not believe that: reusing the number of a frame the MCU received latches SEQ_ERROR."""
    link, p = pty_link
    _arm_over_pty(link, p)

    def ambiguous_write(self, data):
        os.write(self.fd, data)  # the bytes DID go out ...
        raise serial.SerialTimeoutException("Write timeout")  # ... and pyserial still raises

    monkeypatch.setattr(serial.Serial, "write", ambiguous_write)
    for seq in range(10):
        res = link.send(seq)
        assert res is not HbSend.NOT_DELIVERED
        p.pump()
    assert not p.mcu.latched and p.mcu._last_seq == 9


def test_serial_link_no_byte_accepted_is_not_delivered(pty_link, monkeypatch):
    link, p = pty_link
    _arm_over_pty(link, p)

    def full(fd, data):
        raise BlockingIOError

    monkeypatch.setattr(hb.os, "write", full)
    assert link.send(0) is HbSend.NOT_DELIVERED and not link.dead
    monkeypatch.undo()
    assert link.send(0) is not HbSend.NOT_DELIVERED  # the same number again: the MCU never saw it
    p.pump()
    assert p.mcu._last_seq == 0 and not p.mcu.latched


def _one_byte_then_full(monkeypatch):
    """The next ``os.write`` accepts only the first byte; every later one would block (until ``monkeypatch.undo``)."""
    real_write = hb.os.write
    calls = {"n": 0}

    def write(fd, data):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, data[:1])
        raise BlockingIOError

    monkeypatch.setattr(hb.os, "write", write)


class _SkewClock(RealClock):
    """Real time plus a test-controlled offset: ages the link's kept tail without waiting."""

    def __init__(self) -> None:
        self.skew_ns = 0

    def monotonic_ns(self) -> int:
        return time.monotonic_ns() + self.skew_ns

    def sleep_until(self, t_ns: int) -> None:
        super().sleep_until(t_ns - self.skew_ns)


@pytest.fixture
def skew_link(clock):
    p = _PtyMcu(clock)
    sk = _SkewClock()
    link = SerialHeartbeatLink(p.port, sk)
    yield link, p, sk
    link.close()
    p.close()


def test_serial_link_partial_write_is_not_delivered_until_the_frame_is_complete(pty_link, monkeypatch):
    """U3 verification N2: the MCU counts a heartbeat only at its third byte, so a partial write is NOT_DELIVERED
    (the caller reuses the number); the next send of that number only completes the kept tail (no duplicate)."""
    link, p = pty_link
    _arm_over_pty(link, p)
    p.rx.clear()  # the ARM byte
    _one_byte_then_full(monkeypatch)
    assert link.send(0) is HbSend.NOT_DELIVERED and not link.dead  # 1 of 3 bytes: the MCU has counted nothing
    monkeypatch.undo()
    p.pump()
    assert p.mcu._last_seq is None
    assert link.send(0) is not HbSend.NOT_DELIVERED  # the same number again: the tail completes frame 0
    p.pump()
    assert bytes(p.rx) == bytes([0xA5, 0, 0xFF])  # exactly one frame 0, no duplicate (a repeat latches)
    assert p.mcu._last_seq == 0 and not p.mcu.latched
    assert link.send(1) is not HbSend.NOT_DELIVERED
    p.pump()
    assert bytes(p.rx[-3:]) == bytes([0xA5, 1, 0xFE]) and p.mcu._last_seq == 1 and not p.mcu.latched


def test_serial_link_partial_write_then_a_new_number_sends_the_tail_first(pty_link, monkeypatch):
    link, p = pty_link
    _arm_over_pty(link, p)
    p.rx.clear()  # the ARM byte
    _one_byte_then_full(monkeypatch)
    assert link.send(0) is HbSend.NOT_DELIVERED
    monkeypatch.undo()
    p.pump()
    assert link.send(1) is not HbSend.NOT_DELIVERED  # tail of frame 0 first, then frame 1
    p.pump()
    assert bytes(p.rx) == bytes([0xA5, 0, 0xFF, 0xA5, 1, 0xFE])
    assert p.mcu._last_seq == 1 and not p.mcu.latched


def test_serial_link_stale_partial_tail_is_dropped_not_sent(skew_link, clock, monkeypatch):
    """U3 verification N2: a tail older than 80 ms is never sent (past the MCU's 100 ms window it would be line
    noise: here 0x5A ~0x5A = ARM, then a new 0xA5 head). Nothing is written until the MCU has dropped the head,
    then the same number goes out as a whole frame and the MCU counts it."""
    link, p, sk = skew_link
    _arm_over_pty(link, p)
    p.rx.clear()  # the ARM byte
    _one_byte_then_full(monkeypatch)
    assert link.send(0x5A) is HbSend.NOT_DELIVERED
    monkeypatch.undo()
    p.pump()
    assert bytes(p.rx) == bytes([0xA5])  # the MCU holds a frame head
    for ms in (85, 15):  # 85 ms, 100 ms after the head: dropped, and still within the quiet window
        sk.skew_ns += ms * MS
        clock.advance(ms * MS)
        assert link.send(0x5A) is HbSend.NOT_DELIVERED and not link.dead
        p.pump()
        assert bytes(p.rx) == bytes([0xA5])  # nothing written: neither the stale tail nor a new frame
    sk.skew_ns += 25 * MS
    clock.advance(25 * MS)
    assert link.send(0x5A) is not HbSend.NOT_DELIVERED  # 125 ms: the MCU dropped the head; a whole frame
    p.pump()
    assert bytes(p.rx) == bytes([0xA5, 0xA5, 0x5A, 0xA5])
    assert p.mcu._last_seq == 0x5A and not p.mcu.latched and p.mcu.armed


def test_serial_link_young_tail_is_still_completed(skew_link, clock, monkeypatch):
    link, p, sk = skew_link
    _arm_over_pty(link, p)
    p.rx.clear()  # the ARM byte
    _one_byte_then_full(monkeypatch)
    assert link.send(7) is HbSend.NOT_DELIVERED
    monkeypatch.undo()
    p.pump()
    sk.skew_ns += 70 * MS
    clock.advance(70 * MS)  # inside the MCU's 100 ms window
    assert link.send(7) is not HbSend.NOT_DELIVERED
    p.pump()
    assert bytes(p.rx) == bytes([0xA5, 7, 0xF8]) and p.mcu._last_seq == 7 and not p.mcu.latched

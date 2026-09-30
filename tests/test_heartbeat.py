from typing import ClassVar

import pytest
import serial

from so101_rl.servo import heartbeat as hb
from so101_rl.servo.heartbeat import (
    FakeHeartbeatLink,
    FakeMcu,
    McuStatus,
    SerialHeartbeatLink,
)
from so101_rl.servo.timing import FakeClock

MS = 1_000_000


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def mcu(clock):
    return FakeMcu(clock)


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


def test_powers_up_disarmed_with_relay_open(mcu, link):
    assert (mcu.relay_on, mcu.armed, mcu.latched) == (False, False, False)
    assert link.status() == McuStatus(False, False, False, False, hb.REASON_MCU_RESET)  # matches firmware boot (PROTOCOL.md)


def test_arm_only_from_disarmed(mcu, link):
    s = link.arm()
    assert s.armed and s.relay_on and not s.latched
    assert mcu.state == "ARMED"
    # a second ARM while ARMED changes nothing (in particular does not re-baseline the sequence)
    assert link.send(10)
    assert mcu.feed(bytes([0x5A])) == bytes([mcu.status_byte()])
    assert mcu.state == "ARMED" and mcu._last_seq == 10


def test_heartbeat_ignored_while_disarmed(mcu, link):
    assert link.send(0) is False  # not armed: echo says so
    assert mcu.state == "DISARMED"


def test_heartbeat_sequence_ok_and_wraps(mcu, link, clock):
    link.arm()
    for seq in list(range(250, 256)) + list(range(5)):
        clock.advance(33 * MS)
        assert link.send(seq) is True
    assert mcu.state == "ARMED"


def test_wrong_seq_latches_reason_3(mcu, link):
    link.arm()
    assert link.send(0)
    assert link.send(1)
    assert link.send(5) is False
    assert mcu.latched and mcu.reason == hb.REASON_SEQ_ERROR and not mcu.relay_on
    s = link.status()
    assert s.latched and s.reason == 3 and not s.relay_on


def test_repeated_seq_latches(mcu, link):
    link.arm()
    assert link.send(7)
    assert link.send(7) is False
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
        assert link.send(seq)
    assert mcu.state == "ARMED"  # 20 s total, never a 1 s gap


def test_timeout_without_any_heartbeat_after_arm(mcu, link, clock):
    link.arm()
    clock.advance(1001 * MS)
    assert link.status().latched  # feed() polls first


def test_late_heartbeat_after_timeout_does_not_unlatch(mcu, link, clock):
    link.arm()
    link.send(0)
    clock.advance(1500 * MS)
    assert link.send(1) is False
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
    assert link.arm().armed  # and it can be armed again


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
    assert link.send(1)


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


def test_test_drop_rejected_without_jumper(mcu, link):
    link.arm()
    with pytest.raises(RuntimeError, match="TEST"):
        link.test_drop()
    assert mcu.state == "ARMED"


def test_power_cycle_reports_mcu_reset_and_opens_relay(mcu, link):
    link.arm()
    mcu.power_cycle()
    s = link.status()
    assert not s.relay_on and not s.armed and s.reason == hb.REASON_MCU_RESET


def test_fake_link_write_timeout_returns_false_and_frame_is_lost(mcu, link, clock):
    link.arm()
    assert link.send(0)
    link.fail_writes(1)
    assert link.send(1) is False
    assert mcu._last_seq == 0  # never arrived
    assert link.send(1) is True  # sequence continues unharmed


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
        self.swallow_reply = 0  # next n replies never arrive
        self.closed = False
        FakeSerial.instances.append(self)

    @property
    def in_waiting(self):
        return len(self._rx)

    def read(self, n=1):
        data, self._rx = bytes(self._rx[:n]), self._rx[n:]
        return data

    def write(self, data):
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
        assert link.send(seq) is True
    assert ser.writes[0] == b"\x5a"
    assert ser.writes[1] == bytes([0xA5, 0, 0xFF])
    assert mcu.state == "ARMED"
    assert link.status().armed


def test_serial_link_write_timeout_returns_false(ser_link):
    link, ser = ser_link
    link.arm()
    assert link.send(0)
    ser.raise_timeout = True
    assert link.send(1) is False


def test_serial_link_missing_echo_returns_false_then_recovers(ser_link):
    link, ser = ser_link
    link.arm()
    assert link.send(0)
    ser.swallow_reply = 1
    assert link.send(1) is True  # echo for 1 not yet due when send(1) returns... it was swallowed
    assert link.send(2) is False  # ...so the next send sees an unacknowledged previous heartbeat
    assert link.send(3) is True  # the lost echo was forgotten; link recovers


def test_serial_link_latched_echo_returns_false(ser_link, mcu, clock):
    link, _ = ser_link
    link.arm()
    assert link.send(0)
    clock.advance(1500 * MS)
    assert link.send(1) is True  # this send's own echo has not been read yet
    assert link.send(2) is False  # the echo of the previous one says LATCHED
    assert mcu.latched


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


def test_serial_link_close(ser_link):
    link, ser = ser_link
    link.close()
    assert ser.closed

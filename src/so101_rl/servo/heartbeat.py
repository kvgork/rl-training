"""Heartbeat link to the watchdog MCU (plan F4, Task 1.3).

Wire protocol (host -> MCU, 115200 8N1 over USB-CDC)::

    0x5A                 ARM        accepted only from DISARMED, bus voltage sane, not latched
    0xA5 seq ~seq        HEARTBEAT  seq == last + 1 (mod 256); the first one after ARM sets the baseline
    0xE0                 STATUS_REQ
    0xE1                 TEST_DROP  only with the physical TEST jumper; reply is status + uint16 LE drop_ms

MCU -> host: one status byte after every accepted frame::

    bit0 relay_on, bit1 latched, bit2 armed, bit3 undervolt, bits4-7 trip_reason

Firmware invariants modelled by :class:`FakeMcu` (same as ``firmware/watchdog_mcu/PROTOCOL.md``): the relay is open
at reset; ARMED -> LATCHED after more than ``timeout_ms`` without a valid heartbeat, on a bad sequence or
complement, on bus voltage below threshold for more than 50 ms, or on the e-stop input; LATCHED clears only by the
physical reset button held 500 ms and lands in DISARMED.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

from so101_rl.servo.timing import Clock

CMD_ARM = 0x5A
CMD_HEARTBEAT = 0xA5
CMD_STATUS_REQ = 0xE0
CMD_TEST_DROP = 0xE1

REASON_NONE = 0
REASON_TIMEOUT = 1
REASON_UNDERVOLT = 2
REASON_SEQ_ERROR = 3
REASON_ESTOP_PIN = 4
REASON_TEST = 5
REASON_MCU_RESET = 6

DISARMED, ARMED, LATCHED = "DISARMED", "ARMED", "LATCHED"

RESET_HOLD_MS = 500
UNDERVOLT_DWELL_MS = 50
UNDERVOLT_THRESHOLD_V = 9.0

_MS = 1_000_000


class McuStatus(NamedTuple):
    relay_on: bool
    latched: bool
    armed: bool
    undervolt: bool
    reason: int

    @classmethod
    def from_byte(cls, b: int) -> McuStatus:
        return cls(bool(b & 1), bool(b & 2), bool(b & 4), bool(b & 8), (b >> 4) & 0xF)

    def to_byte(self) -> int:
        return (
            int(self.relay_on)
            | int(self.latched) << 1
            | int(self.armed) << 2
            | int(self.undervolt) << 3
            | (self.reason & 0xF) << 4
        )


class HeartbeatLink(Protocol):
    def arm(self) -> McuStatus: ...

    def send(self, seq: int) -> bool:
        """Non-blocking; False on write timeout or bad/missing echo -> counts as a missed heartbeat."""

    def status(self) -> McuStatus: ...

    def test_drop(self) -> int:
        """drop_ms."""


class FakeMcu:
    """Host-side model of the firmware state machine, driven by an injected clock."""

    def __init__(self, clock: Clock, timeout_ms: int = 1000) -> None:
        self.clock = clock
        self.timeout_ms = timeout_ms
        self.state = DISARMED
        self.relay_on = False
        self.latched = False
        self.armed = False
        self.reason = REASON_MCU_RESET  # firmware boots with reason 6 (PROTOCOL.md)
        self.undervolt = False
        self.test_jumper = False
        self.drop_time_ms = 12  # what TEST_DROP reports
        self.bus_voltage_v = 12.0
        self._estop = False
        self._last_hb_ns = 0
        self._last_seq: int | None = None
        self._undervolt_since_ns: int | None = None
        self._rx = bytearray()

    # -- state helpers -------------------------------------------------------------------------
    def _set_state(self, state: str, reason: int | None = None) -> None:
        self.state = state
        self.armed = state == ARMED
        self.latched = state == LATCHED
        self.relay_on = state == ARMED
        if reason is not None:
            self.reason = reason

    def _latch(self, reason: int) -> None:
        self._set_state(LATCHED, reason)

    def status_byte(self) -> int:
        return McuStatus(self.relay_on, self.latched, self.armed, self.undervolt, self.reason).to_byte()

    # -- environment inputs for tests ----------------------------------------------------------
    def set_bus_voltage(self, volts: float) -> None:
        self.poll()
        self.bus_voltage_v = volts
        if volts < UNDERVOLT_THRESHOLD_V:
            if self._undervolt_since_ns is None:
                self._undervolt_since_ns = self.clock.monotonic_ns()
        else:
            self._undervolt_since_ns = None
        self.undervolt = volts < UNDERVOLT_THRESHOLD_V

    def set_estop(self, pressed: bool) -> None:
        self._estop = pressed
        self.poll()

    def reset_button(self, hold_ms: float) -> None:
        """Physical reset button held for ``hold_ms``. Only a full 500 ms hold clears LATCHED."""
        self.poll()
        if self.state == LATCHED and hold_ms >= RESET_HOLD_MS and not self._estop:
            self._set_state(DISARMED, REASON_NONE)
            self._last_seq = None

    def power_cycle(self) -> None:
        self._set_state(DISARMED, REASON_MCU_RESET)
        self._last_seq = None
        self._rx.clear()

    # -- firmware loop -------------------------------------------------------------------------
    def poll(self) -> None:
        """Apply timeouts at ``clock.monotonic_ns()``."""
        now = self.clock.monotonic_ns()
        if self.state != ARMED:
            return
        if self._estop:
            self._latch(REASON_ESTOP_PIN)
        elif self._undervolt_since_ns is not None and now - self._undervolt_since_ns > UNDERVOLT_DWELL_MS * _MS:
            self._latch(REASON_UNDERVOLT)
        elif now - self._last_hb_ns > self.timeout_ms * _MS:
            self._latch(REASON_TIMEOUT)

    def feed(self, frame: bytes) -> bytes:
        """Consume host bytes; returns the status byte(s) for every accepted frame. Partial frames are buffered."""
        self._rx += frame
        out = bytearray()
        while self._rx:
            self.poll()
            cmd = self._rx[0]
            if cmd == CMD_HEARTBEAT:
                if len(self._rx) < 3:
                    break  # wait for the rest of the frame
                seq, inv = self._rx[1], self._rx[2]
                del self._rx[:3]
                self._on_heartbeat(seq, inv)
                out.append(self.status_byte())
            elif cmd == CMD_ARM:
                del self._rx[:1]
                self._on_arm()
                out.append(self.status_byte())
            elif cmd == CMD_STATUS_REQ:
                del self._rx[:1]
                out.append(self.status_byte())
            elif cmd == CMD_TEST_DROP:
                del self._rx[:1]
                if self.test_jumper:
                    self._latch(REASON_TEST)
                    out.append(self.status_byte())
                    out += int(self.drop_time_ms).to_bytes(2, "little")
            else:
                del self._rx[:1]  # line noise: not a frame, no reply
        return bytes(out)

    def _on_arm(self) -> None:
        if self.state == DISARMED and not self.undervolt and not self._estop:
            self._set_state(ARMED, REASON_NONE)
            self._last_hb_ns = self.clock.monotonic_ns()
            self._last_seq = None

    def _on_heartbeat(self, seq: int, inv: int) -> None:
        if self.state != ARMED:
            return  # DISARMED/LATCHED ignore heartbeats
        if inv != (~seq & 0xFF) or (self._last_seq is not None and seq != (self._last_seq + 1) & 0xFF):
            self._latch(REASON_SEQ_ERROR)
            return
        self._last_seq = seq
        self._last_hb_ns = self.clock.monotonic_ns()


class FakeHeartbeatLink:
    """Routes bytes straight into a :class:`FakeMcu`."""

    def __init__(self, mcu: FakeMcu) -> None:
        self.mcu = mcu
        self._drop_writes = 0

    def fail_writes(self, n: int = 1) -> None:
        """The next ``n`` sends behave like a write timeout: nothing reaches the MCU, ``send`` returns False."""
        self._drop_writes += n

    def arm(self) -> McuStatus:
        reply = self.mcu.feed(bytes([CMD_ARM]))
        return McuStatus.from_byte(reply[-1])

    def send(self, seq: int) -> bool:
        if self._drop_writes > 0:
            self._drop_writes -= 1
            return False
        reply = self.mcu.feed(bytes([CMD_HEARTBEAT, seq & 0xFF, ~seq & 0xFF]))
        if not reply:
            return False
        s = McuStatus.from_byte(reply[-1])
        return s.armed and not s.latched

    def status(self) -> McuStatus:
        return McuStatus.from_byte(self.mcu.feed(bytes([CMD_STATUS_REQ]))[-1])

    def test_drop(self) -> int:
        reply = self.mcu.feed(bytes([CMD_TEST_DROP]))
        if len(reply) != 3:
            raise RuntimeError("TEST_DROP not accepted (TEST jumper missing?)")
        return int.from_bytes(reply[1:3], "little")


class SerialHeartbeatLink:
    """pyserial link. ``send`` never blocks: it writes (<= ``write_timeout_s``) and drains the echo via ``in_waiting``.

    The echo of heartbeat *k* is normally there before heartbeat *k+1*; if the previous one is still unacknowledged
    when the next is sent (or the write times out, or the echo says LATCHED / not armed), ``send`` returns False.
    """

    _REPLY_WAIT_S = 0.1
    _DROP_WAIT_S = 2.0

    def __init__(self, port: str, clock: Clock, write_timeout_s: float = 0.002) -> None:
        import serial

        self._serial_mod = serial
        self._ser = serial.Serial(port, 115200, timeout=0, write_timeout=write_timeout_s)
        self._clock = clock
        self._pending = 0
        self._last_status: McuStatus | None = None

    def close(self) -> None:
        self._ser.close()

    def _drain(self) -> None:
        n = self._ser.in_waiting
        if n:
            for b in self._ser.read(n):
                self._last_status = McuStatus.from_byte(b)
                self._pending = max(0, self._pending - 1)

    def _read_exact(self, n: int, wait_s: float) -> bytes:
        deadline = self._clock.monotonic_ns() + int(wait_s * 1e9)
        buf = bytearray()
        while len(buf) < n:
            avail = self._ser.in_waiting
            if avail:
                buf += self._ser.read(min(avail, n - len(buf)))
                continue
            if self._clock.monotonic_ns() >= deadline:
                raise TimeoutError(f"watchdog MCU: expected {n} reply bytes, got {len(buf)}")
            self._clock.sleep_until(self._clock.monotonic_ns() + 1_000_000)
        return bytes(buf)

    def _flush_input(self) -> None:
        n = self._ser.in_waiting
        if n:
            self._ser.read(n)
        self._pending = 0

    def arm(self) -> McuStatus:
        self._flush_input()
        self._ser.write(bytes([CMD_ARM]))
        self._last_status = McuStatus.from_byte(self._read_exact(1, self._REPLY_WAIT_S)[0])
        return self._last_status

    def send(self, seq: int) -> bool:
        self._drain()
        acked_previous = self._pending == 0
        try:
            self._ser.write(bytes([CMD_HEARTBEAT, seq & 0xFF, ~seq & 0xFF]))
        except self._serial_mod.SerialTimeoutException:
            return False
        # A lost echo is forgotten (pending=1) so one missed heartbeat does not poison every later one.
        self._pending = 1 if not acked_previous else self._pending + 1
        if not acked_previous:
            return False
        st = self._last_status
        return st is None or (st.armed and not st.latched)

    def status(self) -> McuStatus:
        self._flush_input()
        self._ser.write(bytes([CMD_STATUS_REQ]))
        self._last_status = McuStatus.from_byte(self._read_exact(1, self._REPLY_WAIT_S)[0])
        return self._last_status

    def test_drop(self) -> int:
        self._flush_input()
        self._ser.write(bytes([CMD_TEST_DROP]))
        reply = self._read_exact(3, self._DROP_WAIT_S)
        self._last_status = McuStatus.from_byte(reply[0])
        return int.from_bytes(reply[1:3], "little")

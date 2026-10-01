"""Heartbeat link to the watchdog MCU (plan F4, Task 1.3).

Wire protocol (host -> MCU, 115200 8N1 over USB-CDC)::

    0x5A                 ARM        accepted only from DISARMED, PSU-side bus voltage sane, servo side dead, not latched
    0xA5 seq ~seq        HEARTBEAT  seq == last + 1 (mod 256); the first one after ARM sets the baseline
    0xE0                 STATUS_REQ
    0xE1                 TEST_DROP  only with the TEST jumper fitted and state ARMED; reply is the status byte and
                                    then, later, uint16 LE drop_ms. Otherwise the reply is the status byte alone

MCU -> host: one status byte after every accepted frame::

    bit0 relay_on, bit1 latched, bit2 armed, bit3 undervolt, bits4-7 trip_reason

Firmware invariants modelled by :class:`FakeMcu` (same as ``firmware/watchdog_mcu/PROTOCOL.md``): the relay is open
at reset and the MCU boots LATCHED (reason MCU_RESET); ARMED -> LATCHED after ``timeout_ms`` (>=) without a valid
heartbeat, on a bad sequence or complement, on PSU-side bus voltage below threshold for more than 50 ms, or on the
e-stop input; LATCHED clears only by the physical reset button held 500 ms and lands in DISARMED (reason kept). A
servo-side bus that stays live 500 ms after the relay is commanded open latches reason WELD.
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
REASON_WELD = 7

DISARMED, ARMED, LATCHED = "DISARMED", "ARMED", "LATCHED"

RESET_HOLD_MS = 500
UNDERVOLT_DWELL_MS = 50
UNDERVOLT_THRESHOLD_V = 9.0
DROP_THRESHOLD_V = 1.0
DROP_TIMEOUT_MS = 2000
WELD_MS = 500
PARTIAL_FRAME_MS = 100

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


class TestDropRefused(RuntimeError):
    """The MCU answered TEST_DROP with a plain status byte: jumper missing or state not ARMED. Nothing was tripped."""

    __test__ = False  # not a pytest class despite the name

    def __init__(self, status: McuStatus) -> None:
        super().__init__(
            f"TEST_DROP refused (TEST jumper missing or not ARMED): status {status.to_byte():#04x}"
        )
        self.status = status


class HeartbeatLink(Protocol):
    def arm(self) -> McuStatus: ...

    def send(self, seq: int) -> bool:
        """Non-blocking and never raises; False on write timeout, bad/missing echo or a dead link (USB unplugged)."""

    def status(self) -> McuStatus: ...

    def test_drop(self) -> int:
        """drop_ms. Raises :class:`TestDropRefused` at once (no 2 s wait) when the MCU refuses (no jumper / not ARMED)."""


class FakeMcu:
    """Host-side model of the firmware ``Core`` (``firmware/watchdog_mcu/main.py``), driven by an injected clock.

    ``_poll`` mirrors ``Core.poll`` and ``feed`` mirrors ``Core.feed``; ``tests/test_firmware_vectors.py`` replays
    every shared vector through both, so any drift fails there.
    """

    def __init__(self, clock: Clock, timeout_ms: int = 1000) -> None:
        self.clock = clock
        self.timeout_ms = timeout_ms
        self.test_jumper = False
        self.drop_time_ms = (
            12  # servo-side decay time after the relay opens (only when no servo voltage is forced)
        )
        self.bus_voltage_v = 12.0  # PSU side, before the relay: ARM check and undervolt trip
        self.servo_voltage_v: float | None = None  # servo side, after the relay; None = follows the relay
        self._estop = False
        self._btn = False
        self._boot()

    def _boot(self) -> None:
        now = self.clock.monotonic_ns()
        self.state = LATCHED  # an MCU reset boots LATCHED: only the physical button re-enables power
        self.reason = REASON_MCU_RESET
        self.relay_on = self.armed = False
        self.latched = True
        self.undervolt = False
        self._live = False  # servo side >= 1 V
        self._open_since_ns = (
            now - 10**15
        )  # when the relay last opened (servo-side decay model): long ago at boot
        self._last_hb_ns = now
        self._last_seq: int | None = None
        self._uv_since_ns: int | None = None
        self._rst_since_ns: int | None = None
        self._weld_since_ns: int | None = None
        self._drop_since_ns: int | None = None
        self._rx = bytearray()
        self._rx_ns = now
        self._tx = bytearray()  # bytes the MCU sent outside a feed() reply (drop_ms)

    # -- state helpers -------------------------------------------------------------------------
    def _set_state(self, state: str, reason: int | None = None) -> None:
        if self.relay_on and state != ARMED:
            self._open_since_ns = self.clock.monotonic_ns()
        self.state = state
        self.armed = state == ARMED
        self.latched = state == LATCHED
        self.relay_on = state == ARMED
        if reason is not None:
            self.reason = reason

    def _latch(self, reason: int) -> None:
        self._set_state(LATCHED, reason)
        self._last_seq = None

    def status_byte(self) -> int:
        return McuStatus(self.relay_on, self.latched, self.armed, self.undervolt, self.reason).to_byte()

    def _servo_v(self, now: int) -> float:
        if self.servo_voltage_v is not None:
            return self.servo_voltage_v
        if self.relay_on or now - self._open_since_ns < self.drop_time_ms * _MS:
            return self.bus_voltage_v
        return 0.0

    # -- environment inputs for tests ----------------------------------------------------------
    def set_bus_voltage(self, volts: float) -> None:
        """PSU-side (before the relay) bus voltage: used for the ARM check and the undervolt trip."""
        self.bus_voltage_v = volts
        self._poll()

    def set_servo_voltage(self, volts: float | None) -> None:
        """Servo-side (after the relay) bus voltage, used for the drop time and the weld check.

        ``None`` (default) follows the relay: the PSU voltage while the relay is closed and for ``drop_time_ms``
        after it opens, then 0 V. A fixed value models a welded relay (or a slow decay).
        """
        self.servo_voltage_v = volts
        self._poll()

    def set_estop(self, pressed: bool) -> None:
        self._estop = pressed
        self._poll()

    def set_button(self, pressed: bool) -> None:
        """Level of the physical reset button; the 500 ms hold is measured on the injected clock."""
        self._btn = pressed
        self._poll()

    def reset_button(self, hold_ms: float) -> None:
        """Convenience: press the physical button for ``hold_ms`` (the injected clock advances by that much).

        Only a full 500 ms hold clears LATCHED (never while the e-stop is asserted); the trip reason is kept.
        The elapsed time matters: after a trip the servo side needs ``drop_time_ms`` to decay, and ARM is refused
        until it has (just like the firmware), which a real 500 ms press always outlasts.
        """
        self._poll()
        self.clock.sleep_until(self.clock.monotonic_ns() + int(hold_ms * _MS))
        self._poll()
        if self.state == LATCHED and hold_ms >= RESET_HOLD_MS and not self._estop:
            self._set_state(DISARMED)
            self._last_seq = None
            self._poll()  # a still-live servo side re-latches WELD once its 500 ms window has run

    def power_cycle(self) -> None:
        """MCU reset (power, hardware WDT, USB re-enumeration): boots LATCHED, reason MCU_RESET, relay open."""
        self._boot()

    # -- firmware loop -------------------------------------------------------------------------
    def poll(self) -> bytes:
        """Run the firmware loop body at ``clock.monotonic_ns()``; returns the drop_ms bytes if a measurement ended."""
        self._poll()
        out, self._tx = bytes(self._tx), bytearray()
        return out

    def _poll(self) -> None:
        now = self.clock.monotonic_ns()
        sv = self._servo_v(now)
        self.undervolt = self.bus_voltage_v < UNDERVOLT_THRESHOLD_V
        self._live = sv >= DROP_THRESHOLD_V
        if not self.undervolt:
            self._uv_since_ns = None
        elif self._uv_since_ns is None:
            self._uv_since_ns = now
        uv_trip = self.undervolt and now - self._uv_since_ns > UNDERVOLT_DWELL_MS * _MS
        if self.state == ARMED and (
            self._estop or uv_trip or now - self._last_hb_ns >= self.timeout_ms * _MS
        ):
            self._latch(REASON_ESTOP_PIN if self._estop else REASON_UNDERVOLT if uv_trip else REASON_TIMEOUT)
        held = self.state == LATCHED and self._btn and not self._estop
        if not held:
            self._rst_since_ns = None
        elif self._rst_since_ns is None:
            self._rst_since_ns = now
        if held and now - self._rst_since_ns >= RESET_HOLD_MS * _MS:
            self._set_state(DISARMED)  # reason kept until the next ARM
            self._last_seq = None
            self._rst_since_ns = None
        if self._live and self.state != ARMED:
            if self._weld_since_ns is None:
                self._weld_since_ns = now
        else:
            self._weld_since_ns = None
        if self._weld_since_ns is not None and now - self._weld_since_ns >= WELD_MS * _MS:
            self._latch(REASON_WELD)
        if self._rx and now - self._rx_ns >= PARTIAL_FRAME_MS * _MS:
            self._rx.clear()  # a partial frame never waits longer than PARTIAL_FRAME_MS
        if self._drop_since_ns is not None:
            elapsed = now - self._drop_since_ns
            if sv < DROP_THRESHOLD_V or elapsed >= DROP_TIMEOUT_MS * _MS:
                self._tx += min(elapsed // _MS, 0xFFFF).to_bytes(2, "little")
                self._drop_since_ns = None

    def feed(self, frame: bytes) -> bytes:
        """Consume host bytes; returns the status byte for every accepted frame. Partial frames are buffered.

        The ``drop_ms`` bytes of an accepted TEST_DROP are not part of this reply: like the firmware they appear from
        :meth:`poll` once the servo-side bus has fallen below 1 V (or after 2000 ms).
        """
        self._poll()
        out = bytearray()
        now = self.clock.monotonic_ns()
        for byte in frame:
            self._rx.append(byte)
            self._rx_ns = now
            cmd = self._rx[0]
            if cmd == CMD_HEARTBEAT and len(self._rx) < 3:
                continue
            if cmd == CMD_ARM:
                self._on_arm(now)
            elif cmd == CMD_HEARTBEAT:
                self._on_heartbeat(self._rx[1], self._rx[2], now)
            elif cmd == CMD_TEST_DROP and self.test_jumper and self.state == ARMED:
                self._latch(REASON_TEST)
                self._drop_since_ns = now
            if cmd in (CMD_ARM, CMD_HEARTBEAT, CMD_STATUS_REQ, CMD_TEST_DROP):
                out.append(self.status_byte())
            self._rx.clear()
        return bytes(out)

    def _on_arm(self, now: int) -> None:
        if self.state == DISARMED and not (self.undervolt or self._estop or self._live):
            self._set_state(ARMED, REASON_NONE)
            self._last_hb_ns = now
            self._last_seq = None

    def _on_heartbeat(self, seq: int, inv: int, now: int) -> None:
        if self.state != ARMED:
            return  # DISARMED/LATCHED ignore heartbeats
        if inv != (~seq & 0xFF) or (self._last_seq is not None and seq != (self._last_seq + 1) & 0xFF):
            self._latch(REASON_SEQ_ERROR)
            return
        self._last_seq = seq
        self._last_hb_ns = now


class FakeHeartbeatLink:
    """Routes bytes straight into a :class:`FakeMcu`."""

    dead = False  # True after :meth:`unplug`, like :attr:`SerialHeartbeatLink.dead`

    def __init__(self, mcu: FakeMcu) -> None:
        self.mcu = mcu
        self._drop_writes = 0

    def fail_writes(self, n: int = 1) -> None:
        """The next ``n`` sends behave like a write timeout: nothing reaches the MCU, ``send`` returns False."""
        self._drop_writes += n

    def unplug(self) -> None:
        """USB cable pulled: every later ``send`` returns False and ``dead`` is True."""
        self.dead = True

    def arm(self) -> McuStatus:
        reply = self.mcu.feed(bytes([CMD_ARM]))
        return McuStatus.from_byte(reply[-1])

    def send(self, seq: int) -> bool:
        if self.dead:
            return False
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
        """Waits (on the fake clock) for the servo-side bus to fall, like the real link; raises at once on refusal."""
        before = self.status()
        if not before.armed:
            raise TestDropRefused(before)
        st = McuStatus.from_byte(self.mcu.feed(bytes([CMD_TEST_DROP]))[-1])
        if not (st.latched and st.reason == REASON_TEST):
            raise TestDropRefused(st)
        clock = self.mcu.clock
        for _ in range(DROP_TIMEOUT_MS + 200):
            reply = self.mcu.poll()
            if reply:
                return int.from_bytes(reply[:2], "little")
            clock.sleep_until(clock.monotonic_ns() + _MS)
        raise TimeoutError("watchdog MCU: no drop_ms reply")


class SerialHeartbeatLink:
    """pyserial link. ``send`` never blocks or raises: it writes (<= ``write_timeout_s``) and drains the echo.

    The echo of heartbeat *k* is normally there before heartbeat *k+1*; if the previous one is still unacknowledged
    when the next is sent (or the write times out, or the echo says LATCHED / not armed), ``send`` returns False.
    Any other serial failure (USB unplugged: ``SerialException`` / ``OSError``, a closed port, ...) marks the link
    :attr:`dead`: ``send`` returns False from then on without touching the port. There is no auto-reconnect: the
    MCU would time out and latch long before, so recovery is a new process plus the physical button.
    """

    _REPLY_WAIT_S = 0.1
    _DROP_WAIT_S = 2.5  # the MCU reports at most 2000 ms

    def __init__(self, port: str, clock: Clock, write_timeout_s: float = 0.002) -> None:
        import serial

        self._serial_mod = serial
        self._ser = serial.Serial(port, 115200, timeout=0, write_timeout=write_timeout_s)
        self._clock = clock
        self._pending = 0
        self._last_status: McuStatus | None = None
        self.dead = False
        self.last_error: BaseException | None = None

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
        if self.dead:
            return False
        try:
            self._drain()
            acked_previous = self._pending == 0
            try:
                self._ser.write(bytes([CMD_HEARTBEAT, seq & 0xFF, ~seq & 0xFF]))
            except self._serial_mod.SerialTimeoutException:
                return False  # transient: a missed heartbeat, the link may recover
        except Exception as exc:  # noqa: BLE001 - the tick path must never raise: unplug, closed port, termios error
            self.dead, self.last_error = True, exc
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
        """Measure the relay drop time. Raises :class:`TestDropRefused` straight away if the MCU refuses.

        A refused 0xE1 is answered with a status byte that is not "latched, reason TEST" (jumper missing); a
        status request first rejects a state that is not ARMED without sending 0xE1 at all.
        """
        before = self.status()
        if not before.armed:
            raise TestDropRefused(before)
        self._flush_input()
        self._ser.write(bytes([CMD_TEST_DROP]))
        st = McuStatus.from_byte(self._read_exact(1, self._REPLY_WAIT_S)[0])
        self._last_status = st
        if not (st.latched and st.reason == REASON_TEST):
            raise TestDropRefused(st)
        return int.from_bytes(self._read_exact(2, self._DROP_WAIT_S), "little")

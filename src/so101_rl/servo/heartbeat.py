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

import os
import select
import time
from enum import IntEnum
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


class HbSend(IntEnum):
    """Result of :meth:`HeartbeatLink.send` (U3 review item 2).

    * ``NOT_DELIVERED``: the frame never left the host (write timeout, dead link). The MCU saw nothing, so the
      caller must NOT advance its sequence number: the next heartbeat reuses it (a gap would latch SEQ_ERROR).
    * ``DELIVERED_NO_ECHO``: the frame was written, but no acknowledging echo came back (echo lost or late, or
      the echo reports the MCU not armed / latched: see ``last_status``). The MCU may have counted it, so the
      sequence DOES advance (repeating a sequence number also latches SEQ_ERROR).
    * ``ACKED``: written and acknowledged by an armed, unlatched MCU.

    The values are chosen so that a legacy ``bool`` compares sensibly: ``True == ACKED`` and
    ``False == NOT_DELIVERED``.
    """

    NOT_DELIVERED = 0
    ACKED = 1
    DELIVERED_NO_ECHO = 2


class HeartbeatLink(Protocol):
    last_status: McuStatus | None  # the most recent status byte the MCU sent (any echo / reply); None = none yet

    def arm(self) -> McuStatus: ...

    def send(self, seq: int) -> HbSend:
        """Non-blocking and never raises; see :class:`HbSend` for the three outcomes."""

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

    def relay_closed(self) -> bool:
        """Relay state at ``clock.monotonic_ns()`` (runs the firmware loop first). Power source for
        ``FakeBus(power=...)``: the servo side is live only while the relay is closed."""
        self._poll()
        return self.relay_on

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
        self._drop_echoes = 0
        self.last_status: McuStatus | None = None

    def fail_writes(self, n: int = 1) -> None:
        """The next ``n`` sends behave like a write timeout: nothing reaches the MCU (``NOT_DELIVERED``)."""
        self._drop_writes += n

    def drop_echoes(self, n: int = 1) -> None:
        """The next ``n`` sends reach the MCU but their echo is lost (``DELIVERED_NO_ECHO``)."""
        self._drop_echoes += n

    def unplug(self) -> None:
        """USB cable pulled: every later ``send`` returns False and ``dead`` is True."""
        self.dead = True

    def arm(self) -> McuStatus:
        reply = self.mcu.feed(bytes([CMD_ARM]))
        self.last_status = McuStatus.from_byte(reply[-1])
        return self.last_status

    def send(self, seq: int) -> HbSend:
        if self.dead:
            return HbSend.NOT_DELIVERED
        if self._drop_writes > 0:
            self._drop_writes -= 1
            return HbSend.NOT_DELIVERED
        reply = self.mcu.feed(bytes([CMD_HEARTBEAT, seq & 0xFF, ~seq & 0xFF]))
        if self._drop_echoes > 0:
            self._drop_echoes -= 1
            return HbSend.DELIVERED_NO_ECHO
        if not reply:
            return HbSend.DELIVERED_NO_ECHO
        s = self.last_status = McuStatus.from_byte(reply[-1])
        return HbSend.ACKED if s.armed and not s.latched else HbSend.DELIVERED_NO_ECHO

    def status(self) -> McuStatus:
        self.last_status = McuStatus.from_byte(self.mcu.feed(bytes([CMD_STATUS_REQ]))[-1])
        return self.last_status

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

    The echo of heartbeat *k* is normally there before heartbeat *k+1*, so ``ACKED`` here means "written, and the
    previous heartbeat was acknowledged by an armed, unlatched MCU" (one heartbeat behind). A previous heartbeat
    still unacknowledged, or an echo that says LATCHED / not armed, gives ``DELIVERED_NO_ECHO``; a write timeout
    gives ``NOT_DELIVERED``. Any other serial failure (USB unplugged: ``SerialException`` / ``OSError``, a closed
    port, ...) marks the link :attr:`dead`: ``send`` returns ``NOT_DELIVERED`` from then on without touching the
    port. There is no auto-reconnect: the MCU would time out and latch long before, so recovery is a new process
    plus the physical button.

    Delivery is decided from the bytes the kernel accepted, NOT from pyserial's ``SerialTimeoutException``:
    pyserial raises that also AFTER ``os.write`` took the whole frame (when ``write_timeout`` expired while the
    thread was preempted, or the fd was not writable again in time). Treating that as "not delivered" would reuse
    the sequence number of a frame the MCU did receive and latch it with SEQ_ERROR. So with a real pyserial port
    the frame goes through non-blocking ``os.write`` on its fd (at most ``write_timeout_s`` of waiting for
    writability): 0 bytes accepted = ``NOT_DELIVERED``.

    **Partial writes** (practically impossible for 3 bytes on an empty USB-CDC buffer; U3 verification N2). The MCU
    counts a heartbeat only when its third byte arrives, and it discards a frame head that gets no further byte
    within ``PARTIAL_FRAME_MS`` (100 ms, ``firmware/watchdog_mcu/PROTOCOL.md`` "Partial frames"); bytes after that
    are line noise, and a heartbeat tail ``seq ~seq`` read as noise can be a command (``0x5A`` ARM, ``0xA5`` a new
    head, ``0xE1`` TEST_DROP). So a frame counts as delivered only once all 3 bytes were accepted:

    * a partial write returns ``NOT_DELIVERED`` (the MCU has not counted it, the caller reuses the number) and keeps
      the unsent tail with its age;
    * on the next ``send`` a tail whose last written byte is younger than ``_TAIL_MAX_AGE_S`` (80 ms, 20 ms under
      the MCU window, which restarts on every byte) is written first. Once it is complete the kept frame is delivered: if the caller asks for the same number again (the
      normal case after ``NOT_DELIVERED``) nothing more is written and the result is that of a delivered frame;
      a different number is written after it as a new frame;
    * an older tail is DROPPED, never sent. The MCU never counted that frame, so ``NOT_DELIVERED`` stays true for
      it. Until ``_HEAD_EXPIRY_S`` (120 ms) after its last written byte nothing is written at all and ``send``
      returns ``NOT_DELIVERED``: a new frame would otherwise complete the stale head the MCU may still hold and
      latch it with SEQ_ERROR. After that the requested frame is written normally.
    """

    _TAIL_MAX_AGE_S = 0.08  # a tail older than this is dropped, not sent (the MCU window is PARTIAL_FRAME_MS)
    _HEAD_EXPIRY_S = (PARTIAL_FRAME_MS + 20) / 1000  # after a dropped tail: write nothing until the MCU dropped the head

    _REPLY_WAIT_S = 0.1
    _DROP_WAIT_S = 2.5  # the MCU reports at most 2000 ms

    def __init__(self, port: str, clock: Clock, write_timeout_s: float = 0.002) -> None:
        import serial

        self._serial_mod = serial
        self._ser = serial.Serial(port, 115200, timeout=0, write_timeout=write_timeout_s)
        self._write_timeout_s = write_timeout_s
        self._tail = b""  # unsent rest of a partially written heartbeat frame
        self._tail_seq = 0  # sequence number of that frame
        self._tail_ns = 0  # when its last byte was written (``clock``)
        self._quiet_until_ns = 0  # after a dropped tail: no byte before this (the MCU still holds the stale head)
        self._clock = clock
        self._pending = 0
        self._last_status: McuStatus | None = None
        self.dead = False
        self.last_error: BaseException | None = None

    @property
    def last_status(self) -> McuStatus | None:
        return self._last_status

    def close(self) -> None:
        self._ser.close()

    def _write_frame(self, frame: bytes) -> int:
        """Bytes of ``frame`` the OS accepted (0..3); waits at most ``write_timeout_s`` for writability.

        Uses the pyserial fd directly (non-blocking ``os.write``) when there is one. Test doubles without an int
        ``fd`` go through ``write()``, where a ``SerialTimeoutException`` means "nothing written"."""
        fd = getattr(self._ser, "fd", None)
        if not isinstance(fd, int):
            try:
                self._ser.write(frame)
            except self._serial_mod.SerialTimeoutException:
                return 0
            return len(frame)
        written = 0
        deadline = time.monotonic() + self._write_timeout_s
        while written < len(frame):
            try:
                written += os.write(fd, frame[written:])
                continue
            except (BlockingIOError, InterruptedError):
                pass
            left = deadline - time.monotonic()
            if left <= 0:
                break
            select.select([], [fd], [], left)
        return written

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

    def send(self, seq: int) -> HbSend:
        if self.dead:
            return HbSend.NOT_DELIVERED
        try:
            self._drain()
            acked_previous = self._pending == 0
            now = self._clock.monotonic_ns()
            completed = None  # sequence number of a kept frame completed by this call
            if self._tail and now - self._tail_ns >= int(self._TAIL_MAX_AGE_S * 1e9):
                # too old: the MCU drops (or already dropped) the head; sent now, the tail would be read as commands
                self._tail = b""
                self._quiet_until_ns = self._tail_ns + int(self._HEAD_EXPIRY_S * 1e9)
            if now < self._quiet_until_ns:
                return HbSend.NOT_DELIVERED  # the stale head may still be buffered in the MCU: write nothing
            if self._tail:  # finish the previous, partially written frame first
                n = self._write_frame(self._tail)
                self._tail = self._tail[n:]
                if n:
                    self._tail_ns = now  # the MCU's partial-frame timer restarts on every byte
                if self._tail:
                    return HbSend.NOT_DELIVERED  # still incomplete: the MCU has counted nothing
                completed = self._tail_seq
            if completed is None or completed != seq & 0xFF:
                frame = bytes([CMD_HEARTBEAT, seq & 0xFF, ~seq & 0xFF])
                n = self._write_frame(frame)
                if n < len(frame):
                    if n > 0:  # head written: keep the tail; the frame is delivered only once it is complete
                        self._tail, self._tail_seq, self._tail_ns = frame[n:], seq & 0xFF, now
                    return HbSend.NOT_DELIVERED  # not complete, so not counted (0 bytes: transient, may recover)
        except Exception as exc:  # noqa: BLE001 - the tick path must never raise: unplug, closed port, termios error
            self.dead, self.last_error = True, exc
            return HbSend.NOT_DELIVERED
        # A lost echo is forgotten (pending=1) so one missed heartbeat does not poison every later one.
        self._pending = 1 if not acked_previous else self._pending + 1
        if not acked_previous:
            return HbSend.DELIVERED_NO_ECHO
        st = self._last_status
        return HbSend.ACKED if st is None or (st.armed and not st.latched) else HbSend.DELIVERED_NO_ECHO

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

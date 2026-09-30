"""SO-101 deadman watchdog (MicroPython, RP2040). Protocol: PROTOCOL.md. Relay is ON only while ARMED."""

T_MS, UV_MV, UV_HOLD, RST_HOLD, DROP_MV, DROP_TMO = 1000, 9000, 50, 500, 1000, 2000
NONE, TIMEOUT, UNDERVOLT, SEQ, ESTOP, TEST, RESET = range(7)
DISARMED, ARMED, LATCHED = range(3)


class Core:  # pure logic, no hardware: the host tests drive exactly this class
    def __init__(self, now, diff):  # MCU: time.ticks_diff (wrap-safe); host tests inject an equivalent
        self.d, self.state, self.reason, self.uv, self.es = diff, DISARMED, RESET, False, False
        self.last, self.seq, self.buf, self.lo, self.rst, self.drop = now, None, b"", None, None, None

    def status(self):
        a = self.state == ARMED
        return a | (self.state == LATCHED) << 1 | a << 2 | self.uv << 3 | self.reason << 4

    def trip(self, reason):
        self.state, self.reason, self.seq = LATCHED, reason, None

    def poll(self, now, mv, estop, btn):  # call every loop; returns extra bytes (drop_ms)
        self.uv, self.es = mv < UV_MV, estop
        self.lo = (self.lo if self.lo is not None else now) if self.uv else None
        uv_trip = self.uv and self.d(now, self.lo) > UV_HOLD
        if self.state == ARMED and (estop or uv_trip or self.d(now, self.last) >= T_MS):
            self.trip(ESTOP if estop else UNDERVOLT if uv_trip else TIMEOUT)
        held = self.state == LATCHED and btn and not estop  # only the physical button clears a latch
        self.rst = (self.rst if self.rst is not None else now) if held else None
        if held and self.d(now, self.rst) >= RST_HOLD:
            self.state, self.rst = DISARMED, None
        if self.drop is None or not (mv < DROP_MV or self.d(now, self.drop) >= DROP_TMO):
            return b""
        ms, self.drop = min(self.d(now, self.drop), 0xFFFF), None
        return ms.to_bytes(2, "little")

    def feed(self, now, data, jumper):  # returns one status byte per accepted frame
        out = b""
        for b in data:
            self.buf += bytes((b,))
            c = self.buf[0]
            if c == 0xA5 and len(self.buf) < 3:
                continue
            if c == 0x5A and self.state == DISARMED and not self.uv and not self.es:
                self.state, self.reason, self.last, self.seq = ARMED, NONE, now, None
            elif c == 0xA5 and self.state == ARMED:
                s, n = self.buf[1], self.buf[2]
                good = n == (~s & 255) and (self.seq is None or s == (self.seq + 1) & 255)
                self.seq, self.last = (s, now) if good else (None, self.last)
                if not good:
                    self.trip(SEQ)
            elif c == 0xE1 and jumper and self.state == ARMED:
                self.trip(TEST)
                self.drop = now
            out += bytes((self.status(),)) if c in (0x5A, 0xA5, 0xE0, 0xE1) else b""
            self.buf = b""
        return out


def main():
    import micropython, select, sys  # noqa: I001
    from machine import ADC, WDT, Pin
    from time import sleep_ms, ticks_diff, ticks_ms

    micropython.kbd_intr(-1)  # 0x03 is a legal seq byte: never let it raise KeyboardInterrupt
    relay = Pin(15, Pin.OUT, value=0)  # GP15 -> transistor base; LOW at reset = fail-open
    estop, btn, jmp = [Pin(n, Pin.IN, Pin.PULL_UP) for n in (14, 13, 12)]
    adc, wdt, core, p = ADC(26), WDT(timeout=250), Core(ticks_ms(), ticks_diff), select.poll()
    p.register(sys.stdin, select.POLLIN)
    while True:
        wdt.feed()
        now = ticks_ms()
        out = core.poll(now, adc.read_u16() * 3300 * 57 // 655350, estop.value() == 1, btn.value() == 0)
        while p.poll(0):  # a flood starves wdt.feed -> MCU reset -> relay opens (fail-safe)
            out += core.feed(now, sys.stdin.buffer.read(1), jmp.value() == 0)
        relay.value(core.state == ARMED)
        sys.stdout.buffer.write(out)
        sleep_ms(1)


if __name__ == "__main__":
    main()

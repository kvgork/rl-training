# Watchdog MCU wire protocol (F4)

Normative spec for the host <-> watchdog MCU link. The host side is `src/so101_rl/servo/heartbeat.py`
(`SerialHeartbeatLink`, `FakeMcu`); the reference firmware is `main.py` in this directory. All three must
satisfy `test_vectors.json`.

Transport: dedicated USB-CDC serial, 115200 8N1 (the baud rate is nominal on CDC). The host must write each
frame with **one `write()` call** (a heartbeat is 3 bytes): a frame split across a stall would swallow the next
command byte as its own payload.

## Host -> MCU

| Byte | Meaning | Payload |
|---|---|---|
| `0x5A` | ARM (request relay close) | none. Accepted only in DISARMED, V_bus sane (>= 9.0 V), e-stop not asserted |
| `0xA5` | HEARTBEAT | `seq8`, `~seq8` (3 bytes total). `seq8 == last + 1 mod 256` required |
| `0xE0` | STATUS_REQ | none |
| `0xE1` | TEST_DROP | none. Accepted only with the physical TEST jumper fitted and state ARMED: opens the relay and measures the time until V_bus < 1 V |

Any other byte outside a frame is ignored silently (no response, no state change).

## MCU -> host

One **status byte** after every complete frame (`0x5A`, `0xA5`, `0xE0`, `0xE1`), whether or not the frame
changed anything (a refused ARM or a heartbeat in DISARMED/LATCHED is answered with the unchanged status).

```
bit0 relay_on   bit1 latched   bit2 armed   bit3 undervolt   bits4-7 trip_reason
```

`relay_on` and `armed` are always equal (the relay is energised only in ARMED). `undervolt` follows the live
V_bus (below 9.0 V), it is not latched. `trip_reason`: `0` NONE, `1` TIMEOUT, `2` UNDERVOLT, `3` SEQ_ERROR,
`4` ESTOP_PIN, `5` TEST, `6` MCU_RESET.

After `0xE1` the MCU later sends `drop_ms` as one uint16 little-endian, when V_bus falls below 1 V. If it has not
fallen after 2000 ms the value is 2000 (the test failed: treat `>= 2000` as "relay did not drop the bus").

Example status bytes: `0x60` boot (DISARMED, reason MCU_RESET); `0x05` ARMED; `0x12` LATCHED after timeout;
`0x42` LATCHED by e-stop; `0x2A` LATCHED by undervolt while V_bus is still low.

## State machine

`state in {DISARMED, ARMED, LATCHED}`. Boot state is DISARMED with reason `6` (MCU_RESET), relay LOW.

| From | Event | To |
|---|---|---|
| DISARMED | `0x5A`, V_bus sane, e-stop clear | ARMED (reason cleared to NONE, heartbeat window starts, seq base unset) |
| ARMED | valid heartbeat | ARMED (window restarts) |
| ARMED | no valid heartbeat for >= 1000 ms (wrap-safe `ticks_diff`) | LATCHED, reason 1 |
| ARMED | bad complement, or seq != last + 1 (including a repeated seq) | LATCHED, reason 3 |
| ARMED | V_bus < 9.0 V continuously for > 50 ms | LATCHED, reason 2 |
| ARMED | e-stop input asserted | LATCHED, reason 4 |
| ARMED | `0xE1` with TEST jumper | LATCHED, reason 5, `drop_ms` measurement starts |
| LATCHED | reset button held 500 ms continuously, e-stop not asserted | DISARMED (reason kept until the next ARM) |
| any | MCU reset (power, hardware WDT, USB re-enumeration that resets the board) | DISARMED, reason 6 |

Details that the vectors pin down:

- The **first** heartbeat after ARM may carry any `seq8`; it sets the base. Later ones must increment by 1 mod 256.
- A timeout is tested with `>=`; the vectors avoid the exact 1000 ms boundary (999 ms ARMED, 1001 ms LATCHED).
- Timeouts are evaluated before each incoming frame, so a late heartbeat cannot rescue an expired window.
- **Serial can never clear a latch.** Only the physical button (GPIO, held 500 ms, released restarts the hold).
- Releasing the button, or pressing the e-stop, restarts/blocks the 500 ms hold.

## Firmware invariants

1. The relay pin is LOW at reset (fail-open): transistor off, coil de-energised, contacts open (normally open).
2. The relay is driven high only while `state == ARMED`; it is recomputed every loop iteration.
3. Hardware WDT 250 ms, fed once per loop iteration. A hung or flooded loop resets the MCU, which opens the relay.
4. `micropython.kbd_intr(-1)`: `0x03` is a legal `seq8`/`~seq8` byte and must not raise KeyboardInterrupt.
5. V_bus is sampled through a 47k/10k divider (`V_bus = V_adc * 5.7`) on every loop iteration (about 1 kHz).
6. The e-stop input is fail-safe: wired to the e-stop's NC auxiliary contact to GND, a pressed e-stop *or a
   broken wire* reads HIGH = asserted.
7. Lost USB link, host crash, `kill -9` of the servo process and host reboot all stop the heartbeats, so they all
   end in TIMEOUT within about 1 s.

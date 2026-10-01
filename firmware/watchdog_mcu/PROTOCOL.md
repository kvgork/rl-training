# Watchdog MCU wire protocol (F4)

Normative spec for the host <-> watchdog MCU link. The host side is `src/so101_rl/servo/heartbeat.py`
(`SerialHeartbeatLink`, `FakeMcu`); the reference firmware is `main.py` in this directory. All three must
satisfy `test_vectors.json` (`tests/test_firmware_vectors.py` replays **every** vector through the firmware `Core`
and through `FakeMcu`).

Transport: dedicated USB-CDC serial, 115200 8N1 (the baud rate is nominal on CDC). The host must write each
frame with **one `write()` call** (a heartbeat is 3 bytes): the MCU discards a partial frame that stalls for
>= 100 ms, and a frame split across a longer stall would otherwise swallow the next command byte as its payload.

## Wire-format compatibility

The command bytes, the heartbeat frame and the status byte layout are unchanged. What changed in the wave-1 fix
(all semantic, none breaks a correct host): the MCU **boots LATCHED** (status `0x62`, was `0x60`), trip reason `7`
(`WELD`) exists, ARM additionally requires a dead servo-side bus, `0xE1` is answered with the status byte alone
unless it is accepted, and `drop_ms` is measured on the servo-side divider.

## Host -> MCU

| Byte | Meaning | Payload |
|---|---|---|
| `0x5A` | ARM (request relay close) | none. Accepted only in DISARMED, **PSU-side** V_bus sane (>= 9.0 V), servo-side bus dead (< 1 V), e-stop not asserted |
| `0xA5` | HEARTBEAT | `seq8`, `~seq8` (3 bytes total). `seq8 == last + 1 mod 256` required |
| `0xE0` | STATUS_REQ | none |
| `0xE1` | TEST_DROP | none. Accepted only with the physical TEST jumper fitted **and** state ARMED: opens the relay and measures the time until the servo-side bus < 1 V. Otherwise: status byte only, no state change, no `drop_ms` |

Any other byte outside a frame is ignored silently (no response, no state change).

## MCU -> host

One **status byte** after every complete frame (`0x5A`, `0xA5`, `0xE0`, `0xE1`), whether or not the frame
changed anything (a refused ARM, a refused TEST_DROP or a heartbeat in DISARMED/LATCHED is answered with the
unchanged status).

```
bit0 relay_on   bit1 latched   bit2 armed   bit3 undervolt   bits4-7 trip_reason
```

`relay_on` and `armed` are always equal (the relay is energised only in ARMED). `undervolt` follows the live
**PSU-side** V_bus (below 9.0 V), it is not latched. `trip_reason`: `0` NONE, `1` TIMEOUT, `2` UNDERVOLT,
`3` SEQ_ERROR, `4` ESTOP_PIN, `5` TEST, `6` MCU_RESET, `7` WELD (new).

After an **accepted** `0xE1` (the status byte shows LATCHED, reason 5) the MCU later sends `drop_ms` as one uint16
little-endian, when the servo-side V_bus falls below 1 V. If it has not fallen after 2000 ms the value is 2000 (the
test failed: treat `>= 2000` as "relay did not drop the bus"). A refused `0xE1` gets only the status byte, so the
host can tell at once: accepted = `latched` and reason `5`; anything else = refused (`TestDropRefused` in the host
link, no 2 s wait). A host should also check ARMED with `0xE0` first, so a refused attempt does not even send `0xE1`.

Example status bytes: `0x62` boot (LATCHED, reason MCU_RESET); `0x60` after the button cleared the boot latch
(DISARMED, reason kept); `0x05` ARMED; `0x12` LATCHED after timeout; `0x42` LATCHED by e-stop; `0x2A` LATCHED by
undervolt while the PSU side is still low; `0x72` LATCHED by relay weld.

## Voltage sensing: two dividers (B4)

| Divider | Position | ADC | Used for |
|---|---|---|---|
| PSU side | after the fuse, **before** the relay (and before the e-stop) | GP26 (ADC0) | ARM undervolt check, `undervolt` bit, undervolt trip |
| Servo side | **after** the relay, on the servo bus | GP27 (ADC1) | `drop_ms` measurement, ARM "bus dead" check, weld detection |

Both are 47k/10k (`V = V_adc * 5.7`). With the relay open the servo side reads about 0 V, so the ARM check can
only use the PSU side (a single servo-side divider would refuse ARM forever).

## State machine

`state in {DISARMED, ARMED, LATCHED}`. Boot state is **LATCHED** with reason `6` (MCU_RESET), relay LOW.

| From | Event | To |
|---|---|---|
| LATCHED | reset button held 500 ms continuously (`>=`), e-stop not asserted | DISARMED (reason kept until the next ARM) |
| DISARMED | `0x5A`, PSU-side V_bus sane, servo side dead, e-stop clear | ARMED (reason cleared to NONE, heartbeat window starts, seq base unset) |
| ARMED | valid heartbeat | ARMED (window restarts) |
| ARMED | no valid heartbeat for >= 1000 ms (wrap-safe `ticks_diff`) | LATCHED, reason 1 |
| ARMED | bad complement, or seq != last + 1 (including a repeated seq) | LATCHED, reason 3 |
| ARMED | PSU-side V_bus < 9.0 V continuously for > 50 ms | LATCHED, reason 2 |
| ARMED | e-stop input asserted | LATCHED, reason 4 |
| ARMED | `0xE1` with TEST jumper | LATCHED, reason 5, `drop_ms` measurement starts |
| DISARMED / LATCHED | servo-side V_bus >= 1 V continuously for >= 500 ms while the relay is commanded open | LATCHED, reason 7 (WELD) |
| any | MCU reset (power, hardware WDT, USB re-enumeration that resets the board) | **LATCHED**, reason 6 |

Details that the vectors pin down:

- **An MCU reset never re-enables power by itself.** A crash, a WDT reset or a USB re-enumeration leaves the relay
  open and the MCU LATCHED; only the physical button (500 ms) returns it to DISARMED, then ARM. Serial ARM in the
  boot latch is refused like in any latch.
- The **first** heartbeat after ARM may carry any `seq8`; it sets the base. Later ones must increment by 1 mod 256.
- The timeout is tested with `>=` (exactly 1000 ms latches; 999 ms is ARMED). The undervolt dwell is `>` (exactly
  50 ms is still ARMED) and the button hold is `>=` (exactly 500 ms clears; 499 ms does not).
- Timeouts are evaluated before each incoming frame, so a late heartbeat cannot rescue an expired window.
- **Serial can never clear a latch.** Only the physical button (GPIO, held 500 ms, released restarts the hold).
- Releasing the button, or pressing the e-stop, restarts/blocks the 500 ms hold.
- **Weld:** the weld timer runs while the relay is commanded open (any state but ARMED) and the servo side reads
  >= 1 V. A weld latch is re-asserted as soon as the button clears it while the bus is still live, and ARM is
  refused while the servo side is live. If a TEST_DROP never sees the bus fall, the latch becomes reason 7 at
  500 ms and `drop_ms` is still reported as 2000.
- **Partial frames:** a frame head (`0xA5` without its 2 payload bytes) is dropped if no further byte arrives within
  100 ms; the remaining bytes are then ordinary line noise.
- Tick counters wrap at 2**30 on the RP2040 (MicroPython `ticks_ms`); `ticks_diff` handles it (`wrap_bits: 30`
  vector).

## Firmware invariants

1. The relay pin is LOW at reset (fail-open): transistor off, coil de-energised, contacts open (normally open).
2. The relay is driven high only while `state == ARMED`; it is recomputed every loop iteration.
3. Hardware WDT 250 ms, fed once per loop iteration. A hung or flooded loop resets the MCU, which opens the relay
   and boots LATCHED.
4. `micropython.kbd_intr(-1)`: `0x03` is a legal `seq8`/`~seq8` byte and must not raise KeyboardInterrupt.
5. Both bus voltages are sampled through 47k/10k dividers (`V = V_adc * 5.7`) on every loop iteration (about 1 kHz).
6. The e-stop input is fail-safe: wired to the e-stop's NC auxiliary contact to GND, a pressed e-stop *or a
   broken wire* reads HIGH = asserted.
7. Lost USB link, host crash, `kill -9` of the servo process and host reboot all stop the heartbeats, so they all
   end in TIMEOUT within about 1 s.

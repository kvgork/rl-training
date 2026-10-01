# Watchdog MCU (RP2040, MicroPython)

Off-host deadman: a 12 V automotive relay (normally open, 30/40 A) sits in the servo supply feed and is
energised only while valid heartbeats arrive. See `PROTOCOL.md` for the wire protocol and state machine, and
`docs/HARDWARE_CHECKLIST.md` (H2, H3) for the power path and the bring-up procedure.

**Read the safety section of `docs/HARDWARE_CHECKLIST.md` before wiring anything.** Nothing in this repository
flashes the board for you.

## Files

| File | Purpose |
|---|---|
| `main.py` | Firmware. `Core` is pure logic (host-testable); `main()` does the pin and USB glue. Limit: 95 lines |
| `PROTOCOL.md` | Normative protocol and invariants |
| `test_vectors.json` | Shared vectors: `tests/test_firmware_vectors.py` (host), `FakeMcu`, and the real MCU (H3) |

## Pin map (Raspberry Pi Pico / RP2040)

| Pin | Direction | Function |
|---|---|---|
| GP15 | out, LOW at reset | Relay driver: 1 k ohm to the base/gate of the coil transistor (see checklist H2) |
| GP14 | in, pull-up | E-stop NC auxiliary contact to GND. HIGH = pressed **or** wire broken = asserted |
| GP13 | in, pull-up | Physical reset button to GND (NO). Held 500 ms clears a latch, **including the boot latch** (the MCU boots LATCHED) |
| GP12 | in, pull-up | TEST jumper to GND. Fitted only during the `0xE1` test, removed afterwards |
| GP26 (ADC0) | analog in | **PSU-side** V_bus (after the fuse, before the e-stop and relay) through 47k (top) / 10k (bottom); 100 nF across the 10k. About 2.1 V at 12 V. ARM check and undervolt trip |
| GP27 (ADC1) | analog in | **Servo-side** V_bus (after the relay), same 47k/10k + 100 nF. Drop-time measurement, "bus dead" check for ARM, relay-weld detection |
| USB | | CDC serial to the host (heartbeats). Use a dedicated port/cable |

## Flash (manual, human-operated; not done by any script here)

1. Hold BOOTSEL while plugging the Pico into the host **with the relay coil supply and the servo supply
   disconnected**, and copy the MicroPython UF2 for the RP2040 from micropython.org onto the `RPI-RP2` drive.
2. Copy `main.py` to the board filesystem as `main.py` (for example `mpremote cp firmware/watchdog_mcu/main.py :main.py`).
   It runs automatically at every boot.
3. Reconnect and check the boot status: send `0xE0`, expect `0x62` (LATCHED, reason MCU_RESET). Hold the reset
   button 500 ms: `0xE0` now answers `0x60` (DISARMED, reason kept). Serial ARM before the button must be refused.
4. Continue with H3 in `docs/HARDWARE_CHECKLIST.md`; do not connect the servo supply until H3's bench steps pass
   with a **dummy load (lamp or resistor)** on the relay.

Because `main.py` disables Ctrl-C (`kbd_intr(-1)`), to re-flash use `mpremote` with a soft reset, or BOOTSEL again.

## Host-side test

```
pixi run -e cpu pytest tests/test_firmware_vectors.py -q
```

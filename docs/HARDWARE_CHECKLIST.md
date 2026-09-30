# Hardware checklist (Phase 0, H1-H13)

Step-by-step bench and arm procedures for Phase 0. Plan reference: `plans/phase0-implementation-plan.md`
(H-series table) and `so101-self-learning-rl-plan.md` (Tier 1 power path, safe-state sequence, post-trip behaviour).
Each section gives the parts, the procedure, the artifact it must produce (under `runs/hw/`, read by
`check_milestone_phase0.py`), and the **shown-to-fail** variant: a safety function is not accepted until you
have seen it fail on purpose and the system caught it.

## Safety warnings (read first)

- **Every step is Tier 0: a human is present, with a hand within reach of the e-stop, until H13 signs off Tier 1.**
  Never leave the arm energised and unattended.
- **Mains voltage:** the PSU's AC input is lethal. Do not open the PSU, and do not wire anything on the mains
  side. Use a fused, enclosed, pre-certified 12 V DC supply and a mains plug you can pull.
- **DC current hazard:** a 12 V supply capable of 8-10 A can heat wire to glowing, weld relay contacts and start
  fires. Fuse the positive feed (next to the PSU, 15 A for 14-16 AWG wire minimum), use wire rated for the
  current, crimp or screw terminals (no twisted-and-taped joints), and keep polarity unambiguous (red/black).
- **DC arcs do not self-extinguish.** Only parts rated for DC at the supply current break the load: the relay
  contacts (30/40 A DC rated) and the e-stop (utilisation category **DC-13** at or above the supply's continuous
  current). An AC-only e-stop or an AC SSR can latch on or weld on DC. Do not substitute either.
- **Do not use a USB or GPIO relay board as the deadman.** It does not fail open when the host dies, the MCU
  resets or the USB link drops. The only accepted topology is the one below.
- **Power-up order:** bench-test the MCU and relay with a **dummy load** (12 V lamp or 10 ohm 50 W resistor)
  before connecting any servo. Connect the arm only after H2 and H3 pass on the dummy load.
- **Arm mechanics:** torque-off lets the arm fall. Foam under the arm and the camera mount clear of limp falls;
  clamp the base; keep hands, cables and the D435 out of the swing volume. Pinch points at every joint.
- **Stalled servos heat and draw current.** Keep the thermal trip (60 C) enabled; do not hold a stall longer
  than needed for H6; stop if any servo is too hot to touch.
- **Lithium/PSU hygiene:** no loose conductors on the bench; insulate the relay terminals; TVS and flyback
  diodes are polarised (diode) or bidirectional (TVS) - verify with a meter before power.
- **This repository never flashes or energises anything by itself.** Flashing the RP2040 and every
  power-on is a manual, human-confirmed act. Results tables below are filled by hand after each test.
- If anything smells, sparks, buzzes or gets hot: hit the e-stop, pull the mains plug, then investigate.

## Power-path topology (one topology, Tier 1)

```
 mains -> [PSU 12 V >= 8-10 A, fused, enclosed]
            |+12 V                                       0 V (common, star point at the PSU)
            +--[Fuse 15 A]--[E-STOP, DC-13, latching NC]--[RELAY contacts NO 30/40 A]--+--[TVS 15-16 V bidirectional]-+
                                                                                        |   (across the servo-side bus)
                                                                                        +--> servo bus / 6 servos
                                                                                        +--> divider 47k/10k -> RP2040 GP26

 Relay coil (12 V, from the PSU +12 V through the fuse, NOT from the contact side):

   +12 V (coil supply) --+-------------- coil ---------+
                         |                              |
                      [1N4007 / UF4007 flyback,         +--- collector/drain
                       cathode to +12 V]                     NPN (e.g. BC337 / 2N2222) or logic-level N-MOSFET
                         +-------------- coil ---------+     emitter/source -> 0 V
                                                        base: RP2040 GP15 --[1 k]-- base   (+ [100 k] base -> 0 V pulldown)

 RP2040:  GP14 <- e-stop NC aux contact -> GND (HIGH = asserted/broken)   GP13 <- reset button (NO) -> GND
          GP12 <- TEST jumper -> GND                                        USB-CDC <-> host (dedicated port)
```

Rules: relay normally open (NO) so any loss of coil power opens the load path; the e-stop is DC-13 rated and in
the **load path** (alternative: in the coil circuit *with* a mechanical break in the load path - the rated NC
e-stop in the load path is the default here); trip latching lives in the MCU (physical reset only); no USB/GPIO
relay; no AC SSR; TVS across the servo-side bus (after the relay, so it also clamps the back-EMF when the arm is
pushed while power is cut).

## Bill of materials

| Item | Spec | Qty | Notes |
|---|---|---|---|
| PSU | 12 V DC, >= 8-10 A continuous, enclosed, fused mains input | 1 | One stalled servo must not brown out the bus. Verify servo voltage range of the STS3215 before choosing 12 V |
| Relay | 12 V coil automotive relay, SPST normally open, 30/40 A DC contacts (Bosch-style 4-pin) with socket | 1 | Coil current about 150-200 mA. Socket with screw or 6.3 mm spade terminals |
| Coil driver | NPN BC337 / 2N2222 (Ic >= 500 mA) or logic-level N-MOSFET (e.g. 2N7000 / IRLZ44N), 1 k base resistor, 100 k pulldown | 1 set | GP15 must drive via the transistor; GPIO never sources the coil |
| Flyback diode | 1N4007 or UF4007 across the coil, cathode to +12 V | 1 | Polarity check with a meter |
| TVS | Bidirectional, 15-16 V standoff (e.g. SMCJ15CA / P6KE16CA), >= 600 W | 1 | Across the servo-side bus, short leads |
| E-stop | Latching (twist-release) mushroom, **DC-13 rated >= supply current**, NC load contact(s) + NC auxiliary contact | 1 | Aux contact feeds GP14. AC-13-only parts are not acceptable |
| Fuse | 15 A blade fuse + holder, in the +12 V feed at the PSU | 1 | Coil supply taps after the fuse |
| Wire | 14-16 AWG, red/black, crimped ferrules/ring terminals | - | Short, star-grounded |
| MCU | Raspberry Pi Pico (RP2040) + MicroPython UF2 | 1 | Dedicated USB cable, separate from the servo controller |
| Divider | 47 k (top), 10 k (bottom), 1 % + 100 nF across the 10 k | 1 set | Vbus = Vadc x 5.7 |
| Reset button | Momentary NO panel button | 1 | Physically reachable, not remotely actuatable |
| TEST jumper | 2-pin header + shunt | 1 | Fitted only during `0xE1` tests |
| Dummy load | 12 V 50 W halogen/lamp or 10 ohm 50 W resistor | 1 | Bench tests before any servo is connected |
| Instruments | Multimeter, optional scope (preferred for drop timing), camera/phone for photos | - | |
| Servo fan | Small 12 V or 5 V fan + mount | 1 | H4 |
| Foam, clamps | Foam pad under the arm, base clamp | - | H5, H6 |

## Results table (fill by hand after each artifact)

Header (append one row per artifact; the H-track, not the software waves, writes these rows):

| Date | H-id | Artifact path | Key values | Pass (Y/N) | Operator | Notes |
|---|---|---|---|---|---|---|

---

## H1 - NVIDIA driver, chassis, GPU thermals

- **Parts:** the workstation; no robot hardware.
- **Needs:** `cuda-version` re-pinned (plan 0.1) and `smoke-tdmpc2 --device cuda` (0.4).
- **Procedure:**
  1. `dkms status`; if the module is missing for kernel 7.0.0-34: `sudo dkms autoinstall`. Otherwise boot the
     previous kernel or switch driver branch.
  2. `hostnamectl chassis` and `cat /sys/class/dmi/id/chassis_type` (laptop vs desktop affects sustained power).
  3. `nvidia-smi` must show the RTX 3080 (10 GB).
  4. Run a 20-minute `torch.matmul` loop while logging `nvidia-smi -q -d TEMPERATURE,POWER` every 10 s.
  5. `pixi install -e default`, then `pixi run smoke-tdmpc2 --device cuda`.
- **Artifact:** `runs/hw/gpu-<ts>.json {driver, cuda, chassis, max_temp_c, throttled: false}`.
- **Pass:** RTX 3080 10 GB visible; `pixi install -e default` solves; smoke exit 0; `throttled` false.
- **Shown-to-fail:** not applicable; record the `nvidia-smi` failure output if the driver is missing before the fix.

## H2 - Power path and e-stop

- **Parts:** full BOM above, dummy load first.
- **Needs:** this document's topology diagram.
- **Procedure:**
  1. Build per the diagram with the **PSU unplugged**. Fuse in place, e-stop in the load path, TVS across the
     servo-side bus, flyback diode across the coil (meter: diode conducts coil-side to +12 V only).
  2. Continuity with the relay coil unpowered: load path **open** (infinite ohms between PSU+ and bus+).
  3. Connect the dummy load. Plug in the PSU. Drive the coil by hand (a 12 V jumper to the transistor base via
     1 k for this test only, not GP15 yet): contacts close, dummy load powers, e-stop twist-release behaviour
     known. Press the e-stop: load drops.
  4. Measure: bus voltage at the dummy load under load (sag < 0.5 V), TVS not warm, relay contacts not warm
     after 10 minutes at the dummy load's current.
  5. E-stop timing: use the MCU ADC via `0xE1` (H3 firmware, TEST jumper) or a scope on the bus; press the
     e-stop with the relay closed; V_bus must be < 1 V within 50 ms.
  6. Photograph the assembled power path (labels visible).
- **Artifacts:** photo; `runs/hw/power-path-<ts>.json {psu_a, relay_part, tvs_part, estop_rating}`;
  `runs/hw/estop-<ts>.json {drop_ms, v_after}`.
- **Pass:** continuity open when de-energised; e-stop opens the load path; `drop_ms` <= 50 and `v_after` < 1 V;
  estop_rating is DC-13 >= psu_a; no USB/GPIO relay and no AC SSR in the path.
- **Shown-to-fail:** remove the coil supply (pull the relay socket or the coil fuse) with the dummy load running:
  the load must drop. Also confirm that unplugging the USB cable to the MCU (H3) opens the relay.

## H3 - Watchdog MCU bring-up (deadman)

- **Parts:** RP2040 Pico, MicroPython, transistor driver board from H2, reset button, TEST jumper, e-stop aux
  contact, divider. **Dummy load on the relay, no servos.**
- **Needs:** `firmware/watchdog_mcu/main.py` (7.1), `SerialHeartbeatLink` (1.3), servo process (3.2).
- **Procedure:**
  1. Flash per `firmware/watchdog_mcu/README.md` (manual). Check the boot status byte is `0x60`.
  2. Wire GP15 (via 1 k to the transistor), GP14 (e-stop aux), GP13 (reset), GP12 (TEST), GP26 (divider).
  3. Replay `firmware/watchdog_mcu/test_vectors.json` against the real MCU over serial (all steps; vectors that
     set `vbus_mv`/`estop`/`btn`/`jumper` need the matching physical action - at least run every `portable` vector
     and the e-stop, reset-button and undervolt ones by hand).
  4. No-heartbeat timing: ARM, stop sending; scope or ADC log: relay opens at 1.0 +/- 0.1 s.
  5. With the servo process heartbeating: `kill -9` it -> relay opens <= 1.5 s. Unplug the MCU USB -> open
     <= 1.5 s. Reboot the host -> relay stays/opens open.
  6. Latch persistence: after any trip the relay stays open; a serial `0x5A` ARM does not close it.
     Only the physical button held 500 ms returns the MCU to DISARMED (status `0x1_`/`0x4_`, relay open until ARM).
- **Artifact:** `runs/hw/deadman-<ts>.json {timeout_s, kill9_s, usb_unplug_s, host_reboot_open, latch_persists,
  serial_reset_rejected, vectors_passed}`.
- **Pass:** timeout 1.0 +/- 0.1 s; kill -9 <= 1.5 s; USB unplug <= 1.5 s; host reboot opens; latch persists;
  serial reset rejected; all vectors pass on the real MCU.
- **Shown-to-fail:** deliberately try to clear a latch over serial (`0x5A`, garbage, repeated STATUS) and
  record that the latch stays (`serial_reset_rejected: true`). Also ARM with the e-stop pressed and with V_bus
  below 9 V (bench supply turned down): ARM must be refused.

## H4 - Servo fan and thermal duty-cycle study

- **Parts:** small fan mounted on the servos; temperature logging from the servo process.
- **Needs:** soak (3.3, `max_temp_c`) and 2.3.
- **Procedure:** run the 20 s on / 10 s off duty cycle at random motion; add a **10-minute hold-still segment**
  (holding is worse than moving). Log per-joint temperature at 1 Hz. Stop on any joint >= 60 C.
- **Artifact:** `runs/hw/thermal-<ts>.json {duty, max_temp_c per joint, time_to_60c_s or null}`; update
  `so101.yaml` `duty_on_s` / `duty_off_s` if needed.
- **Pass:** no joint reaches 60 C in the planned duty, or the duty is reduced until none does.
- **Shown-to-fail:** run the hold-still segment without the fan for a short period and note the temperature slope
  (stop well below 60 C).

## H5 - Rest pose, torque-off, sag map, stale-goal snap

- **Parts:** foam under the arm, camera recording the arm and the D435 mount.
- **Needs:** safe-state (2.4), `servo --backend feetech` (3.2), keyframe (4.1).
- **Procedure:** from `rest_ticks`, disable torque: the arm must not move (video). Sample limp falls from the box
  corners and the highest poses (camera). None may touch the D435 mount. Deliberately leave a **stale goal** and
  re-energise with foam present to see the snap, then repeat with the real re-energise sequence (goal = present
  before `Torque_Enable`).
- **Artifact:** `runs/hw/rest-pose-<ts>.json {rest_ticks, moved_ticks_after_torque_off}` (<= 5), videos in
  `runs/hw/sag-map-<ts>/`; set `verified_on_hardware: true` in `so101.yaml`.
- **Pass:** moved ticks <= 5; no fall touches the D435 mount; no snap with the real sequence.
- **Shown-to-fail:** the stale-goal re-energise (with foam) must visibly snap; the real sequence must not.

## H6 - Register configuration proof and thermal-trip race

- **Parts:** a foam-padded hard stop for one joint.
- **Needs:** `write_registers` (1.2) and 2.3.
- **Procedure:** read back all F1 registers after `configure`. Stall one joint against the stop: firmware
  fold-back must be visible (Status bit or load drop), and the 60 C software trip must fire before any 70 C
  firmware event during H4. Keep each stall short.
- **Artifact:** `runs/hw/registers-<ts>.json {readback, foldback_observed: true, sw_trip_first: true}`; commit the
  chosen `protection_current` / `over_current_protection_time` to `so101.yaml`.
- **Pass:** readback matches; fold-back observed; software trip first.
- **Shown-to-fail:** the stall itself; a deliberately wrong register value in a scratch config must be caught by
  the readback.

## H7 - Calibration read-back on the real arm

- **Needs:** `calib-check` (2.4).
- **Procedure:** `pixi run calib-check` (all joints match). Then run against a **deliberately corrupted copy** of
  the calibration JSON (`homing_offset` +1).
- **Artifact:** `runs/hw/calib-check-<ts>.json {ok: true, broken_copy_rejected: true}`.
- **Pass:** real calibration OK; corrupted copy refused, no torque enable.
- **Shown-to-fail:** the corrupted copy is the failure variant.

## H8 - Hand-eye calibration and table fiducial

- **Parts:** red marker on the gripper tip; ArUco marker on the table in view of the camera.
- **Needs:** D435 driver (5.4), loader/fiducial (5.3), servo (3.2).
- **Procedure:** `pixi run hand-eye --execute` (>= 10 poses, attended). Place the ArUco marker, record
  `expected_corners_px`. Nudge the camera by 2 mm: the gate must fail. Put it back.
- **Artifacts:** `calib/hand_eye.yaml` (residual <= 3 mm, n >= 10) and `calib/fiducial.yaml` committed;
  `runs/hw/camera-moved-<ts>.json {nudged_err_px > 2, restored_err_px <= 2}`.
- **Pass:** as listed. **Shown-to-fail:** the 2 mm nudge.

## H9 - Gravity-return tray for the 16 mm die

- **Parts:** a tilted tray with a start well inside the safety box and in camera view.
- **Procedure:** 20 drops from random points; count returns to the start well.
- **Artifact:** `runs/hw/tray-<ts>.json {returns}` and photos.
- **Pass:** returns >= 19/20. **Shown-to-fail:** a flatter tray angle, recorded as a failing configuration.

## H10 - 30-minute attended random-motion soak (Tier 0)

- **Needs:** soak (3.3), H2-H7 passed.
- **Procedure:** `REQUIRE_RT=1 pixi run soak --backend feetech --minutes 30` with the camera and a dummy MPPI
  load process running (`stress-ng` on the other cores). **A human stands at the e-stop for the whole run.**
- **Artifact:** `runs/soak-<ts>/report.json`.
- **Pass:** `check_soak_report(strict=True)`: p99.9 overrun < 5 ms, zero heartbeat gaps > 200 ms, zero thermal
  trips, zero calibration mismatches, max temperature per servo logged.
- **Shown-to-fail:** `kill -STOP` the servo process mid-soak in a separate short run: the relay must open
  (repeat of the H3 deadman with the real arm's load).

## H11 - Software safe-stop timing

- **Needs:** `CmdMode.STOP_REQ` (3.1), the tick log.
- **Procedure:** while moving, send `STOP_REQ`. From the tick log: `goal_ticks` frozen on the next tick and joint
  velocity below threshold within 100 ms. Repeat with an injected fake temperature (`--inject-temp 61`, hardware
  bus otherwise real).
- **Artifact:** `runs/hw/safe-stop-<ts>.json {goal_frozen: true, vel_below_thresh_ms}` (<= 100).
- **Pass:** as listed. **Shown-to-fail:** the injected temperature variant must trip the thermal path.

## H12 - Sysid check against >= 10 real trajectories

- **Needs:** `compare_real` (4.3) and logs (10.1).
- **Procedure:** use the H10 soak or 10 random-policy episodes; run `python -m so101_rl.sim.compare_real runs/<id>`.
- **Artifact:** `runs/hw/sysid-<ts>.json {rms_pct per joint}`.
- **Pass:** <= 5 %. Above 5 % is **recorded, not fixed** here (the parallel sysid track fits it).
- **Shown-to-fail:** not applicable.

## H13 - Tier promotion review (0 to 1)

- **Needs:** `check-milestone` (10.2) and every artifact above.
- **Procedure:** a human reviews every results-table row and artifact, then signs
  `runs/hw/tier.json {tier: 1, by, date, milestone_sha}`. The 2-hour goal-directed soak stays a Phase 2
  precondition.
- **Pass:** `pixi run check-milestone` exits 0.
- **Shown-to-fail:** run `check-milestone` with one artifact removed: it must fail.

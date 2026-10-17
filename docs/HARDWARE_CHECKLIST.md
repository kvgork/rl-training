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
            +--[Fuse 15 A]--+--[E-STOP, DC-13, latching NC]--[RELAY contacts NO 30/40 A]--+--[TVS 15-16 V bidirectional]-+
                            |                                                             |   (across the servo-side bus)
                            |                                                             +--> servo bus / 6 servos
                            |                                                             +--> divider B 47k/10k -> RP2040 GP27
                            +--> divider A 47k/10k -> RP2040 GP26  (PSU side: before the e-stop and the relay)

 Divider A (PSU side, GP26): ARM undervolt check, undervolt bit and trip. It stays live with the relay open, so ARM
 can be judged. Divider B (servo side, GP27): relay drop time (0xE1), "servo bus dead" check before ARM, and relay
 weld detection (relay commanded open but divider B still reads >= 1 V for 500 ms -> reason 7). Each divider is
 47k (top) / 10k (bottom) with 100 nF across the 10k; Vbus = Vadc x 5.7. A single divider on the servo side would
 read ~0 V with the relay open and ARM would be refused forever.

 Relay coil (12 V, from the PSU +12 V through the fuse, NOT from the contact side):

   +12 V (coil supply) --+-------------- coil ---------+
                         |                              |
                      [1N4007 / UF4007 flyback,         +--- collector/drain
                       cathode to +12 V]                     NPN (e.g. BC337 / 2N2222) or a 3.3 V-gate MOSFET
                         +-------------- coil ---------+     emitter/source -> 0 V
                                                        base: RP2040 GP15 --[1 k]-- base   (+ [100 k] base -> 0 V pulldown)

 RP2040:  GP14 <- e-stop NC aux contact -> GND (HIGH = asserted/broken)   GP13 <- reset button (NO) -> GND (boots LATCHED)
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
| Coil driver | NPN BC337 / 2N2222 (Ic >= 500 mA), 1 k base resistor, 100 k pulldown (default). A MOSFET only if Rds(on) is **guaranteed at Vgs = 2.5-3.3 V** (e.g. IRLML6344, AO3400A) | 1 set | GP15 must drive via the transistor; GPIO never sources the coil. **Do not use a 2N7000**: its Rds(on) is specified at 4.5-10 V and Vgs(th) can reach 3 V, so at the Pico's 3.3 V gate drive it is marginal and may not pull the coil in reliably. Measure coil current and the transistor drop in H2 whichever part is used |
| Flyback diode | 1N4007 or UF4007 across the coil, cathode to +12 V | 1 | Polarity check with a meter |
| TVS | Bidirectional, 15-16 V standoff (e.g. SMCJ15CA / P6KE16CA), >= 600 W | 1 | Across the servo-side bus, short leads |
| E-stop | Latching (twist-release) mushroom, **DC-13 rated >= supply current**, NC load contact(s) + NC auxiliary contact | 1 | Aux contact feeds GP14. AC-13-only parts are not acceptable |
| Fuse | 15 A blade fuse + holder, in the +12 V feed at the PSU | 1 | Coil supply taps after the fuse |
| Wire | 14-16 AWG, red/black, crimped ferrules/ring terminals | - | Short, star-grounded |
| MCU | Raspberry Pi Pico (RP2040) + MicroPython UF2 | 1 | Dedicated USB cable, separate from the servo controller |
| Dividers | 2 sets of 47 k (top), 10 k (bottom), 1 % + 100 nF across the 10 k | 2 sets | A: PSU side (after the fuse, before e-stop/relay) to GP26; B: servo side (after the relay) to GP27. Vbus = Vadc x 5.7 |
| Reset button | Momentary NO panel button | 1 | Physically reachable, not remotely actuatable. Also the only way out of the boot latch after any MCU reset |
| TEST jumper | 2-pin header + shunt | 1 | Fitted only during `0xE1` tests |
| Dummy load | 12 V 50 W halogen/lamp or 10 ohm 50 W resistor | 1 | Bench tests before any servo is connected |
| Instruments | Multimeter, **2-channel scope or logic analyser with an analog channel (>= 10 kS/s) - required for the H2 e-stop timing**, camera/phone for photos | - | The e-stop timing must not rely on the MCU relay path (see H2) |
| Servo fan | Small 12 V or 5 V fan + mount | 1 | H4 |
| Foam, clamps | Foam pad under the arm, base clamp | - | H5, H6 |

## Results table (fill by hand after each artifact)

Header (append one row per artifact; the H-track, not the software waves, writes these rows):

| Date | H-id | Artifact path | Key values | Pass (Y/N) | Operator | Notes |
|---|---|---|---|---|---|---|
| 2026-10-11 | HS4 (no-watchdog variant) | `runs/deploy/20261011-092435-7438/` (+ servo tick logs `runs/deploy/ticks-092435.bin`, `ticks-095342.bin`) | 70/70 successes (0.5: 10/10, 0.75: 10/10, 1.0: 50/50), Wilson 95 % LB 0.929 at 1.0, 0 trips, final error p50 12.6 mm at 1.0 (sim eval 12.3 mm), `finetune_eligible` true, exit 0 | Y (no-watchdog: not valid for Tier 1) | Koen van Gorkom | M6 attempt-3 policy, seed 7, `servo.mcu` false, `no_watchdog_accepted` true. One resume (episode 43): the reset settled 27 ticks off on elbow_flex (tolerance then 25, now 40, ARCHITECTURE 0.41). Real reset steady error up to ±21 ticks pan, 27 elbow_flex (P = 16 deadband). |
| 2026-10-14 | HS5 (15 mm, stopped) | `runs/sac-her/20261012-191349-203e/` (+ `runs/sac-her/ticks-*.bin`, `runner.log`) | 13469 / 60000 steps; baseline 1.00; frozen-actor training success 1.00 (p50 13.0 mm); anchored learning 0.92 (p50 12.7 mm); evals 0.50 @ 12010, 0.90 @ 13025; 3 train-window rollbacks → `rollback_limit` | N (stopped: no headroom at 15 mm) | Claude (unattended, plan §10 2026-10-13) | Earlier sessions: false THERMAL (corrupted temperature reads), STALE_CMD (4.3 s trainer gap), collapse under the old recipe (fixed by the anchor, 0.48), one real overload 0x40 (wrist_flex blocked at ~2990, calib max 3068). Next: a 5 mm sim policy, then a new 5 mm run (plan §10 2026-10-14). |
| 2026-10-14 | HS4-style zero-shot, 5 mm policy (no-watchdog) | `runs/deploy/20261014-170905-dbdc/` (+ `runs/deploy/ticks-20261014-170856.bin`) | success radius 5 mm: 62/70 (0.5: 8/10, 0.75: 8/10, 1.0: 46/50), Wilson 95 % LB 0.812 at 1.0, 0 trips, final error p50 4.3 mm / p90 4.9 mm at 1.0 (sim eval 0.971, 4.1 / 4.9 mm); the 8 misses ended at 5.3–8.4 mm (timeout); `finetune_eligible` true, exit 0 | Y (5 mm; no-watchdog: not valid for Tier 1) | Claude (unattended, user: "run the zero-shot 5 mm check yourself") | Policy `runs/sac-sim/reach-5mm-20261014-065212/policy.pt` (plan §10 2026-10-14), seed 7, servo `--no-mcu`. Servo ended TORQUE_OFF with STALE_CMD + USER_STOP (the normal end: deploy-policy exits, then SIGINT), 0 overruns. No hardware issue: shoulder_pan's deadband did not block 5 mm. |
| 2026-10-14 | HS5 (5 mm, stopped) | `runs/sac-her5/20261014-174624-bbcf/` (+ `runs/sac-her5/ticks-*.bin`, `runner.log`, `diag-critic/`, `awr-offline/`) | 9098 / 60000 steps; baseline 0.80 (5 mm); frozen-actor evals 0.9–1.0, p50 ~4.5 mm; actor collapsed after unfreezing (~7000): rollback 7598, eval 0.1 / 17.7 mm @ 8098, STUCK (0x80) in eval @ 9098 → `eval_trip`; first start failed (corrupt `Goal_Position` read, id 5) | N (stopped: the critic is uninformative, plan §10 2026-10-14) | Claude (unattended, plan §10 2026-10-13) | Offline diagnosis and AWR prototype: plan §10 2026-10-14 rows |
| 2026-10-15 | HS4-style zero-shot, offline-AWR 5 mm policy (no-watchdog) | `runs/deploy/20261015-164020-085f/` (+ `runs/deploy/ticks-20261015-164011.bin`; policy `runs/sac-her5/awr-offline/policy_awr_b05_s500.pt`) | success radius 5 mm: 69/70 (0.5: 10/10, 0.75: 9/10, 1.0: 50/50), Wilson 95 % LB 0.929 at 1.0, 0 trips, final error p50 4.1 mm at 1.0, mean 18.0 steps; paired vs the 2026-10-14 zero-shot (same goals): 8 vs 1 discordant successes, McNemar p 0.039; servo end TORQUE_OFF, fault bits 0x1100 (normal), 0 overruns; `finetune_eligible` true, exit 0 | Y (no-watchdog: not valid for Tier 1) | Claude (unattended, plan §10 2026-10-13) | Replaces the zero-shot 5 mm policy (plan §10 2026-10-15) |
| 2026-10-16 | HS4-style zero-shot, 3 mm sim policy (no-watchdog) | `runs/deploy/20261016-133104-3f10/` (+ `runs/deploy/ticks-20261016-133056.bin`; policy `runs/sac-sim/reach-3mm-20261015-201549/policy.pt`) | success radius 3 mm: 70/70 (0.5: 10/10, 0.75: 10/10, 1.0: 50/50), Wilson 95 % LB 0.929 at 1.0, 0 trips, final error p50 2.6 mm at 1.0 (sim eval 0.957, p50 2.57 mm); servo end TORQUE_OFF, fault bits 0x1100 (normal), 0 overruns; `finetune_eligible` true, exit 0 | Y (no-watchdog: not valid for Tier 1) | Claude (unattended, plan §10 2026-10-13) | Becomes the reach policy (plan §10 2026-10-16); the attempt before failed to start (corrupted Goal_Position read id 4) |

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
  5. **E-stop contact timing, measured independently of the relay command path.** The `0xE1` TEST_DROP measures
     the *relay* (the MCU opens it and times the bus), so it says nothing about the e-stop contact and must not be
     used for this step. Instead:
     1. Close the relay with the **hand-driven coil jumper from step 3** (MCU GP15 disconnected, or the MCU not
        powered), so that nothing but the e-stop contact can open the load path. Dummy load running at its
        rated current.
     2. Scope ch1 on the e-stop's own contact: the voltage across the NC load contact (or the aux contact line to
        GND), triggered on its edge = the instant the contact opens. Scope ch2 on the servo-side bus (divider B
        input or the bus itself, probe ground at the star point). Same-timebase capture, >= 10 kS/s, 100 ms/div or
        faster. (Equivalent: a second, independent MCU/logic analyser sampling those two nodes with its own clock;
        never the watchdog MCU's ADC.)
     3. Press the e-stop firmly 5 times, each time starting from the relay closed and the bus at full voltage;
        keep the 5 captures.
     4. Per capture, read `t_open` (contact edge) and `t_below_1v` (servo bus < 1 V) and compute
        `drop_ms = t_below_1v - t_open`. Report the worst of the five.
     5. Then repeat once with the MCU live and ARMED (normal operation): additionally the MCU must latch
        reason 4 (`0x42`) and the relay must open; the contact timing above must not change.
  6. Photograph the assembled power path (labels visible) and the scope screenshots.
- **Artifacts:** photo + scope captures; `runs/hw/power-path-<ts>.json {psu_a, relay_part, tvs_part, estop_rating}`;
  `runs/hw/estop-<ts>.json {drop_ms_worst, drop_ms_each[5], method: "scope" | "independent_logger", v_after,
  relay_driven_by: "hand_jumper"}`.
- **Pass:** continuity open when de-energised; e-stop opens the load path; worst `drop_ms` <= 50 and `v_after`
  < 1 V, measured with the relay driven by the hand jumper (a number obtained through `0xE1` or through the MCU
  ADC does not count for H2); estop_rating is DC-13 >= psu_a; no USB/GPIO relay and no AC SSR in the path.
- **Shown-to-fail:** remove the coil supply (pull the relay socket or the coil fuse) with the dummy load running:
  the load must drop. Also confirm that unplugging the USB cable to the MCU (H3) opens the relay. Add a second
  shown-to-fail for the timing method itself: with the coil jumper still driving the relay, release the e-stop
  (twist) and confirm the scope shows the bus coming back (the test setup can see both edges).

## H3 - Watchdog MCU bring-up (deadman)

- **Parts:** RP2040 Pico, MicroPython, transistor driver board from H2, reset button, TEST jumper, e-stop aux
  contact, both dividers (A: PSU side GP26, B: servo side GP27). **Dummy load on the relay, no servos.**
- **Needs:** `firmware/watchdog_mcu/main.py` (7.1), `SerialHeartbeatLink` (1.3), servo process (3.2).
- **Procedure:**
  1. Flash per `firmware/watchdog_mcu/README.md` (manual). Check the boot status byte is `0x62` (**LATCHED**,
     reason MCU_RESET; the MCU never boots armed or disarmed).
  2. Wire GP15 (via 1 k to the transistor), GP14 (e-stop aux), GP13 (reset), GP12 (TEST), GP26 (divider A, PSU
     side), GP27 (divider B, servo side).
  3. Boot-latch proof: serial `0x5A` right after boot must answer `0x62` and leave the relay open; hold the reset
     button 500 ms (not with the e-stop pressed): `0xE0` answers `0x60`; only now `0x5A` arms (`0x05`). Also
     power-cycle the MCU (unplug/replug USB) while ARMED: relay open and `0x62` again.
     Then ARM with the relay open and the servo bus at 0 V (divider B reads about 0): must be accepted, i.e. the
     ARM check uses divider A.
  4. Replay `firmware/watchdog_mcu/test_vectors.json` against the real MCU over serial (all steps; vectors that
     set `vbus_mv`/`servo_mv`/`estop`/`btn`/`jumper` need the matching physical action - at least run every
     `portable` vector and the e-stop, reset-button, undervolt, relay-weld and TEST_DROP ones by hand. Every
     vector begins with the 500 ms button press that clears the boot latch, except the `boot_*` ones).
  5. No-heartbeat timing: ARM, stop sending; scope or ADC log: relay opens at 1.0 +/- 0.1 s.
  6. With the servo process heartbeating: `kill -9` it -> relay opens <= 1.5 s. Unplug the MCU USB -> open
     <= 1.5 s. Reboot the host -> relay stays/opens open.
  7. Latch persistence: after any trip the relay stays open; a serial `0x5A` ARM does not close it.
     Only the physical button held 500 ms returns the MCU to DISARMED (status `0x1_`/`0x4_`, relay open until
     ARM; the trip reason is kept until the next ARM).
  8. Relay-weld detection (dummy load only, servo disconnected): with the relay open (DISARMED or tripped), feed
     12 V from a **current-limited bench supply (<= 100 mA)** directly into divider B only (never into the servo
     bus or the PSU rail). After 0.5 s the status must become `0x72` (reason 7); the button must not hold DISARMED
     while the injection stays; ARM must be refused. Remove the injection, press the button, ARM works.
  9. TEST_DROP: with the TEST jumper fitted and ARMED, `0xE1` -> status `0x52` then `drop_ms` (informational
     here; the e-stop timing is measured in H2 step 5). Without the jumper, or when not ARMED, `0xE1` must be
     answered with a plain status byte and no `drop_ms`.
  10. Arm timeout (FW1-HW; flash the FW1 `main.py` and replay every vector incl. the FW1 ones in step 4): press the
      button 500 ms and send no ARM. `0xE0` answers DISARMED with the previous reason kept (`0x80` after an
      arm-timeout latch, `0x60` after the boot latch) for 120 s, then `0x82` (LATCHED, reason 8 ARM_TIMEOUT); the
      relay stays open throughout (scope or ADC log; time the latch, 120 +/- 1 s). Press again: `0x80` (DISARMED,
      reason 8 kept). ARM within 2 min of that press: `0x05`. Pass: latch at 120 +/- 1 s with `0x82` and the
      relay open throughout; the re-press gives `0x80`; the ARM gives `0x05`.
- **Artifact:** `runs/hw/deadman-<ts>.json {timeout_s, kill9_s, usb_unplug_s, host_reboot_open, latch_persists,
  serial_reset_rejected, boot_latched, button_required_after_reset, arm_ok_with_relay_open, weld_detected,
  vectors_passed, arm_timeout_s}` (`arm_timeout_s` from step 10, about 120).
- **Pass:** timeout 1.0 +/- 0.1 s; kill -9 <= 1.5 s; USB unplug <= 1.5 s; host reboot opens; latch persists;
  serial reset rejected; boots `0x62` and needs the button after every MCU reset; ARM accepted with the relay
  open (divider A); weld injection gives `0x72`; all vectors pass on the real MCU.
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

- **Status (2026-10-10):** the rest pose is measured (`rest_ticks` = [2151, 828, 3071, 2854, 2203, 2010], joint-probe
  with torque off, after `zero-adjust`; plan §10 2026-10-10) and passes the geometry check at the measured table
  (-0.0045 m). Still to do with the servo running (needs H2/H3): the torque-off "does not move" test, the sag map and
  the stale-goal snap below. `verified_on_hardware` stays false until then. If the ramp into rest ever stalls on the
  shoulder_lift / elbow stops (the pose sits 3 and 5 ticks inside them), back it off by ~20 ticks.

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
- **Procedure:** stop the servo process first (calib-check refuses while it holds the port, exit 2). `pixi run
  calib-check` (read-only: pings and reads ID, Homing_Offset, Min/Max_Position_Limit; never writes, never enables
  torque; `--port` / `--calibration` override `so101.yaml`). It prints a per-joint file/servo table; exit 0 = all
  joints match. Then run against a **deliberately corrupted copy** of the calibration JSON (`homing_offset` +1):
  `cp <calibration_path> /tmp/broken.json`, edit one `homing_offset`, `pixi run calib-check --calibration
  /tmp/broken.json` → exit 1 naming that joint and `homing_offset`. Exit 2 = could not read (port missing, servo
  holds the port, a motor silent).
- **Artifact:** `runs/hw/calib-check-<ts>.json` (`ok: true`, the per-joint file/servo values) for the real
  calibration; the corrupted-copy run writes `runs/hw/failed/calib-check-<ts>.json` (`ok: false`, `mismatches`).
  Failed runs go to `failed/` so they never satisfy the `sysid-excite --backend real` artifact gate.
- **Pass:** real calibration exit 0; corrupted copy exit 1 with the field named; no register write, no torque enable.
- **Shown-to-fail:** the corrupted copy is the failure variant.

## H7a - Joint zeros from the 2-pose camera fit (`zero-adjust`, plan §10 2026-10-10)

- **Needs:** H7 passing; the camera intrinsics (`calib/camera.yaml`); a `camera-calibrate --method arm-shape --poses 2`
  run whose `report.json` `diagnostics.tilt_vs_offsets` says verdict `joint_offsets` (e.g. `runs/camera-calib/20261010-084545`:
  shoulder_lift −4.94°, wrist_flex +1.00°, elbow_flex −0.16°). Offsets are `true = read + offset`.
- **Camera poses (by hand, torque off):** pose 1 is whatever posture the arm holds by itself with torque off and the
  gripper resting on the table (e.g. upper arm leaning forward, wrist bent down; it need NOT be folded as at rest).
  Between poses turn ONLY the base, about 60° (at least 45°); leave the other joints. The whole arm including the
  gripper must be in the image, and you out of view, when you press Enter.
- **Procedure:**
  1. Stop the servo process (`zero-adjust` refuses while it holds the port). Torque must be off on every motor.
  2. Dry-run: `pixi run zero-adjust --from-report runs/camera-calib/<ts>`. It prints, per joint, the offset, Δticks,
     Homing_Offset / Min / Max old → new and the present reading → predicted; joints under 0.5° are skipped. Nothing
     is written. Check the signs: a negative shoulder_lift offset lowers its reading (Δ < 0), the homing offset rises.
  3. Apply: `pixi run zero-adjust --from-report runs/camera-calib/<ts> --apply --attended`. Keep the arm still. It
     backs up the calibration JSON (`<json>.bak-<ts>`), writes Lock / Homing_Offset / Min / Max_Position_Limit of the
     chosen joints only, reads them back, checks the present reading moved by Δ (±2 ticks), writes the JSON and
     re-compares. On a failure after a write (exit 3) it prints every write and the restore command
     `pixi run zero-adjust --restore <backup> --apply --attended`; run it, then `pixi run calib-check`.
  4. `pixi run calib-check` → exit 0 (the servo registers match the new JSON).
  5. `pixi run camera-calibrate --method arm-shape --poses 2` again (same posture rules): every
     |offset| in `tilt_vs_offsets` must now be < 1°.
- **Artifact:** `runs/hw/zero-adjust-<ts>.json` (inputs, shifts, registers and present readings before / after, every
  write, the backup path, the calib-check comparison); the new `runs/hw/calib-check-<ts>.json`; the new camera run.
- **Pass:** apply exit 0, calib-check exit 0, the re-run camera fit shows all |offsets| < 1°.
- **Note:** `so101.yaml` `rest_ticks` / `home_ticks` are tick values and are not changed: the physical pose they name
  moves by Δ (shoulder_lift −56 ticks for the 2026-10-10 run, inside `rest_tolerance_ticks` 100). Re-check H5 if
  the rest pose no longer settles. A second apply of the same report is refused (the JSON is newer than the report).
- **Status 2026-10-10:** applied (`runs/hw/zero-adjust-20261010-074742.json`), calib-check exit 0. Re-run
  `20261010-095104` (shoulder_lift 2012, elbow_flex 2450): shoulder_lift / elbow_flex / wrist_flex +0.11 / −0.15 /
  +0.62°. Re-run `20261010-163550` in another posture (upper arm leaning forward: shoulder_lift 2253, elbow_flex
  2168, wrist_flex 2860), calibration JSON unchanged: +1.47 / −2.14 / +2.20° (σ ≈ 0.06–0.09°, verdict
  `joint_offsets`). The two runs disagree by far more than σ, so the estimate depends on the posture (unloaded joint
  backlash with the gripper resting on the table, kinematic / mesh error); the bootstrap σ does not cover that.
  Two more runs: `20261010-164720` (tip gate failed: the gripper at the image edge in pose 1) +1.61 / −1.15 /
  +1.16°; `20261010-165407` (pose 1 close to the morning posture) −0.69 / −0.00 / +1.34°. Over the 4 runs:
  shoulder_lift −0.69…+1.61 (mean +0.63), elbow_flex −2.14…0.00 (mean −0.86), wrist_flex +0.62…+2.20 (mean +1.33);
  table z −4.2…−5.1 mm, base tilt 0.2–0.6°. The per-run scatter (±1–1.5°) is the method's repeatability for joint
  offsets, not a zero error: no further zero-adjust. wrist_flex is positive in every run (+1.3° ≈ 2 mm at the tip,
  well inside the 15 mm success radius). The pass bar "< 1°" is below this repeatability.

## H8 - Hand-eye calibration and table fiducial

- **Parts:** the table fiducial: ArUco `DICT_4X4_50` id 0, **50 mm**, printed on A4 (plan §10 2026-10-17 "H8
  software"); a red sticker or tape square of 12-16 mm on the gripper tip (the hand-eye marker, found by the red die
  tracker); a ruler.
- **Needs:** D435 driver (5.4), loader/fiducial (5.3), servo (3.2), the 3 mm reach policy
  (`runs/sac-sim/reach-3mm-20261015-201549/policy.pt`), `calib/camera.yaml` with its arm-shape pose.
- **Procedure** (in this order; start every command BEFORE the servo, plan §10 2026-10-17 "Perception order"):
  1. **Print** `pixi run python scripts/make_fiducial_print.py` → `runs/hw/fiducial/aruco_4x4_50_id0_50mm_A4.pdf`
     (the old `aruco_4x4_50_id0_30mm_A4.pdf` is obsolete: 30 mm is ~25 px in the image). Print at 100 %, check the
     black square is 50.0 mm. Glue it flat with its centre at base **(0.09, 0.05)**, sides parallel to the base x / y
     axes, the printed text side (TOP edge) towards +x, away from the base (measure from the base origin, the shoulder_pan
     axis; x 0.09 lies outside the reach goal bounds, so `scripts/point_tip.py` cannot point at it). It must stay clear of the zone, the tray and the start
     regions (it does at that spot).
  2. **Fiducial** (arm at rest, nothing over the marker, no servo needed): `pixi run fiducial-capture` → 30 frames,
     refused unless found in >= 90 % with <= 0.5 px rms corner jitter → `calib/fiducial.yaml`. A "rotated" note or a
     prior error above 10 px means the marker is turned or misplaced: fix and repeat.
  3. **Hand-eye** (attended, hand at the e-stop; remove the die and other red objects from the table; stick the red
     marker on the gripper tip, facing up / towards the camera): start
     `pixi run -e cpu hand-eye --execute --attended [--accept-no-watchdog]` first (it loads the policy and opens the
     camera), then the servo, then press reset. It visits up to 24 targets in view (2 cm grid, z 0.04-0.15 m),
     wants 12 (min 10) usable poses, holds each still 1.5 s with zero-delta RUN commands, then stops commanding (the
     servo's stale-command stop returns the arm to rest). If the sticker does not sit at the FK tip point, pass its
     centre with `--marker-offset-m X Y Z` (gripper body frame, metres). Outputs `runs/hw/hand-eye-<ts>/`
     (`poses.json`, `hand_eye.yaml`, `report.json`) and, only at residual <= 3 mm and n >= 10, `calib/hand_eye.yaml`.
     Read the printed RECOMMENDATION (camera-parity plan C.6): AGREE = keep the lower residual (folding a hand-eye
     pose into `calib/camera.yaml` is a separate, reviewed step: the tool never writes it); DISAGREE = investigate
     before training any die task.
  4. **Camera moved** (no servo): `pixi run camera-moved-check --procedure`: baseline (must pass) → at the prompt
     nudge the camera by about 2 mm (a light push on the mount), Enter → nudged (the gate must FAIL) → put it back
     exactly, Enter → restored (must pass) → `runs/hw/camera-moved-<ts>.json`. If the restored check fails, the
     camera did not go back: redo steps 2-4 (the fiducial records the camera where it is; the hand-eye too).
- **Artifacts:** `calib/fiducial.yaml` and `calib/hand_eye.yaml` (residual <= 3 mm, n >= 10) committed with
  `runs/hw/hand-eye-<ts>/report.json`; `runs/hw/camera-moved-<ts>.json {nudged_err_px > 2, restored_err_px <= 2,
  pass}`.
- **Pass:** as listed. **Shown-to-fail:** the 2 mm nudge.
- **Sim twin** (no hardware): `servo --backend mujoco` + `pixi run -e cpu hand-eye --execute --camera-backend sim
  --marker-offset-m 0 0.012 -0.01 --shm-name <servo shm>` (`tests/test_hand_eye_execute.py`: 10/10 poses, residual
  0.13 mm, 0.011 deg / 0.20 mm from the calibrated pose).

## H9 - Gravity-return tray for the 16 mm die

- **Parts:** a tilted tray with a start well inside the safety box and in camera view.
- **Procedure:** 20 drops from random points; count returns to the start well.
- **Artifact:** `runs/hw/tray-<ts>.json {returns}` and photos.
- **Pass:** returns >= 19/20. **Shown-to-fail:** a flatter tray angle, recorded as a failing configuration.

## H10 - 30-minute attended random-motion soak (Tier 0)

- **Needs:** soak (3.3), H2-H7 passed.
- **Procedure:** `REQUIRE_RT=1 pixi run -e robot soak --backend feetech --minutes 30` with the camera and a dummy MPPI
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
- **Superseded by HS1-HS3 (sim track, 2026-10-06):** the sysid track fits the parameters from dedicated excitation
  runs and the gate is now **<= 1 %** held-out RMS per joint (plan §10 2026-10-06). See the HS sections below.

## H13 - Tier promotion review (0 to 1)

- **Needs:** `check-milestone` (10.2) and every artifact above.
- **Procedure:** a human reviews every results-table row and artifact, then signs
  `runs/hw/tier.json {tier: 1, by, date, milestone_sha}`. The 2-hour goal-directed soak stays a Phase 2
  precondition.
- **Pass:** `pixi run check-milestone` exits 0.
- **Shown-to-fail:** run `check-milestone` with one artifact removed: it must fail.

---

# Sim-track hardware tasks (HS1-HS3, sysid; `plans/sim-track-implementation-plan.md`)

Appended 2026-10-06 (SU2 sync, integration). All attended (Tier 0): a human at the e-stop for the whole run. The
software never opens a serial device by itself: the operator starts the servo on the real bus, and the excitation
runner only attaches to its shared memory. Prerequisites for every HS run: **H2** (power path + e-stop, its artifact
`runs/hw/estop-*.json`), **H3** (deadman, `runs/hw/deadman-*.json`), **H5** (rest pose, `runs/hw/rest-pose-*.json`)
and **H7** (calibration read-back, `runs/hw/calib-check-*.json`); `sysid-excite --backend real` refuses (exit 2)
without them and without `--attended`. H6 (register proof) is recommended.

## HS1 - Reach-space excitation run on the real arm

- **Parts:** the arm on its clamp, foam under the reach space, the e-stop in hand; the camera is not needed.
- **Needs:** H2, H3, H5, H7 passed; `sysid-excite` (S2.2), dry-run first on sim:
  `pixi run -e cpu sysid-excite --backend sim --plan short --out runs/sysid/dry` (exit 0).
- **Procedure:**
  1. Terminal 1 (the servo owns the bus): `REQUIRE_RT=1 pixi run -e robot servo --backend feetech --heartbeat-port <mcu>
     --tick-log runs/sysid/<ts>/tick_log.bin`. Press the reset button when asked; wait for RUN at home.
  2. Terminal 2: `pixi run sysid-excite --backend real --attended --plan default --out runs/sysid/<ts>` (~10 min of
     arm motion; every segment starts and ends at the start pose and was previewed inside the box).
  3. Watch the temperatures in the servo output: a 60 °C trip ends the segment (safe state, rest, torque off);
     resume after the cool-down with a fresh run directory.
  4. Stop the servo with Ctrl-C (normal shutdown to rest).
- **Artifact:** `runs/sysid/<ts>/{episodes/*.h5, tick_log.bin, plan.json}`; every `check_episode` OK (the runner
  exits 1 otherwise).
- **Pass:** exit 0; `plan.json` clamp-active fraction in [0.05, 0.4] for the multisine and chirp segments (else lower
  the amplitudes and repeat: fully clamped data is blind to the servo dynamics).
- **Shown-to-fail:** run the same command without `--attended`, and once with one H-artifact moved away: both must
  refuse with exit 2 and no motion.

## HS2 - Gripper and die excitation

- **Needs:** HS1 done; H9 (tray) optional.
- **Procedure:** as HS1 with `--with-gripper-die` and the die in the start well: gripper open/close cycles at three
  speeds, closing on the die and on air.
- **Artifact:** `runs/sysid/<ts>-grip/` as in HS1. **Pass:** exit 0, every episode OK.
- **Shown-to-fail:** not applicable (HS1 covers the gates).

## HS3 - Sysid fit to `calib/sim_params.yaml` release 1

- **Needs:** HS1 (and HS2 for the die task); the GPU host.
- **Procedure:** `pixi run sysid-fit --run runs/sysid/<ts> --backend warp --heldout-kinds chirp,random --out
  calib/sim_params.yaml`. The default fit is the servo ratio group (one scale per joint on kp, kv, armature) plus
  the integer delay; frictionloss stays at the prior and is listed unidentified (plan §10 2026-10-06). Inspect the
  printed group scales, `delay_ticks`, the spread (`fit_report.json`) and the held-out table.
  **Deadband prior (plan §10 2026-10-11):** fit from a copy of `calib/sim_params.yaml` with `deadband_ticks: [32, 14,
  16, 12, 9, 9]` passed as `--prior` (the deadband measured from the HS4 deploy logs; it is held fixed, not fitted).
- **Artifact:** `calib/sim_params.yaml` release 1 (committed; `meta.heldout_rms_pct` <= 1 % on every joint,
  `servo_group`, `dr` ranges) and `runs/hw/sysid-<ts>.json` (H12's artifact, now from the fitted model:
  `pixi run compare-real --tick-log runs/sysid/<ts>/tick_log.bin --sim-params calib/sim_params.yaml`).
- **Pass:** `sysid-fit` exit 0 (it writes `--out` only when every joint's held-out RMS is <= 1 %).
- **On failure (exit 1):** the candidate is in `runs/sysid/<ts>/sim_params.unverified.yaml`. Widen the fit once
  (`--fit servo_group,delay_ticks,damping`), repeat; if a joint still fails, record it in the results table as a
  negative result and treat that joint's parameters as unidentified (wide DR) — do not loosen the gate (it is a
  plan §10 decision).
- **Shown-to-fail:** `pixi run sysid-fit ... --fixed damping,forcerange,frictionloss,delay_ticks` (delay held at 0)
  must exit 1 when the arm has a command delay (the synthetic twin fails at 1.5 %).

## HS4 - Attended zero-shot reach eval on the real arm (MVP M7)

Added 2026-10-08 (`plans/mvp-plan.md` M7, plan §10 D-ZEROSHOT). Attended Tier 0: a hand on the e-stop for the whole
session. It does not go through the lab connector.

- **Parts:** the arm on its clamp, foam under the reach space, the e-stop in hand. Optional: tape over the orange tip
  marker.
- **Needs:** H2, H3 (with FW1), H5 and H7 passed (H2/H3 not for the no-watchdog variant below); the servo bus powered; the camera calibrated
  (`pixi run camera-calibrate --method arm-shape` and `pixi run camera-overlay` passing); the sim exit gate signed
  (`pixi run lab gate-sign <training run>`; the eval run is refused). Status 2026-10-10: camera calibrated (`runs/camera-calib/20261010-165407`, overlay IoU 0.82) and the gate signed for the attempt-3 run `fastsac_sim-reach-20261008-125912-fefa` (eval `…130617-58ef`, `gate.json`). Dry run first against the sim servo: `pixi run -e cpu servo --backend
  mujoco` in terminal 1, then the terminal-2 command below without `--attended` (exit 0).
- **Procedure:**
  1. Terminal 1 (the servo owns the bus): `pixi run -e robot servo --backend feetech --heartbeat-port <mcu port> --tick-log
     runs/deploy/ticks-<ts>.bin --summary runs/deploy/servo-<ts>.json`. Do not press the reset button yet.
  2. Terminal 2: `pixi run -e cpu deploy-policy --policy runs/lab/<eval run>/policy.pt --protocol zero-shot --seed 7
     --attended`. Wait (~7 s) for "waiting for the servo to reach RUN".
  3. Press the MCU reset button. The ladder runs 10 episodes at speed 0.5, 10 at 0.75, then 50 at 1.0. It climbs a
     rung only if the finished rung had no trip and at least 8 of 10 successes; otherwise it stops (exit 1,
     `halt_reason: rung_below_threshold` in `summary.json`): that is a result, not a fault.
  4. On a trip or stall the ladder stops (exit 1): restart the servo (step 1, new tick-log name) and rerun step 2 as
     `pixi run -e cpu deploy-policy --resume runs/deploy/<ts> --attended` (the run dir printed by the first run). It
     keeps the finished episodes and continues the same goal list at the next episode; do not change `--policy`,
     `--seed`, `--protocol` or `--rung-episodes` (refused). A trip in the 0.5 or 0.75 rung still counts in that
     rung's advance check, so after the resume that rung is finished and the ladder stops there; a trip in the
     full-speed rung is resumed to the end. A real servo started with `--no-mcu` is refused (but see the no-watchdog
     variant below).
  5. When done, Ctrl-C the servo (normal shutdown to rest).
- **No-watchdog variant (plan §10 2026-10-10; until the H2/H3 parts exist).** No e-stop and no MCU: the e-stop is
  your hand on a switch that cuts the servo supply (a switched power strip, or the PSU's DC plug within reach).
  Nothing cuts power if the PC or the servo process hangs (the servos then hold their last goal), so you watch the
  arm for the whole session and never leave it. H5 still applies (rest pose, foam under the reach space).
  1. Clear the reach space, foam down, hand on the power switch. Starting the servo energises the arm AT ONCE
     (there is no reset button to wait for): it ramps from rest to home within a few seconds, then is in RUN and
     safe-stops (STALE_CMD) 0.5 s later unless `deploy-policy` is already attached. So start deploy-policy FIRST
     (it waits up to 120 s for the servo), then the servo (found 2026-10-11: the other order ends in TORQUE_OFF).
  2. Terminal 1: `pixi run -e cpu deploy-policy --policy runs/lab/fastsac_sim-reach-20261008-130617-58ef/policy.pt
     --protocol zero-shot --seed 7 --attended --accept-no-watchdog`. It prints `looking for the servo`.
  3. Terminal 2, within 120 s: `pixi run -e robot servo --backend feetech --no-mcu --tick-log
     runs/deploy/ticks-<ts>.bin --summary runs/deploy/servo-<ts>.json`. Deploy prints `NO WATCHDOG`, then the ladder
     runs as in step 3 above (no button press). To resume: deploy-policy with `--resume <run dir> --attended
     --accept-no-watchdog` first, then the servo again.
  4. Cut power at once on any contact with the table, the camera mount or a cable, on shaking, or on any motion you
     did not expect; the arm then falls limp onto the foam. Afterwards restart from step 1 and `--resume`.
  5. `summary.json` records `servo.mcu: false` and `servo.no_watchdog_accepted: true`; a pass in this variant is
     marked as such in the results table and does not count for the Tier 1 (unattended) entry, which still needs
     H2/H3.
- **Artifact:** `runs/deploy/<ts>/{episodes.jsonl, summary.json}` and the servo tick log.
- **Pass:** exit 0, and `summary.json` `verdict.finetune_eligible` true (Wilson 95 % lower bound >= 0.60 with n >= 50
  on the full-speed rung). Fill the results table.
- **Shown-to-fail:** step 2 without `--attended` against the feetech servo must refuse (exit 2) with no motion.

## HS5 - Real-arm fine-tuning of reach (SAC + HER)

Added 2026-10-12 (plan §10 2026-10-12 "Real fine-tuning software started" and "Real-arm fine-tuning without the
watchdog MCU"; ARCHITECTURE 0.45). Attended Tier 0: you watch the arm for the whole session, hand on the servo power
switch. **No watchdog MCU is required**: the servo's own safety layer ends the session on overheating (60 °C on 3
reads in a row), a firmware overload / overcurrent bit, a blocked joint (tracking error), STUCK or a box violation.

**Since 2026-10-13 (plan §10 "HS5 run by Claude, unattended"):** Claude runs the sessions itself, unattended. It
resumes only after `session_cap`, stale-command or THERMAL stops (600 s cool-down first); any other stop leaves the
arm torque-off for the user.

- **Parts:** as HS4 (arm on its clamp, foam under the reach space, the reach space clear, hand on the power switch).
- **Needs:** HS4 passed (`finetune_eligible`). Dry run first against the sim servo: terminal 1
  `pixi run train-reach-sac --init-from runs/lab/fastsac_sim-reach-20261008-130617-58ef/policy.pt --backend shm
  --total-steps 300 --learning-starts 100 --baseline-eval-every 200 --eval-episodes 2`, terminal 2
  `pixi run -e cpu servo --backend mujoco` (exit 0).
- **Procedure:**
  1. Start the trainer FIRST (as in HS4's no-watchdog variant, the servo is in RUN seconds after it starts and
     safe-stops 0.5 s later without a client): `pixi run train-reach-sac --init-from
     runs/lab/fastsac_sim-reach-20261008-130617-58ef/policy.pt --backend shm --total-steps 60000 --attended`. It
     prints `looking for the servo` and waits up to 120 s.
  2. Terminal 2, within 120 s: `pixi run -e robot servo --backend feetech --no-mcu --tick-log
     runs/sac-her/ticks-<ts>.bin --summary runs/sac-her/servo-<ts>.json` (or with `--heartbeat-port` once H3 exists).
     The trainer prints `NO WATCHDOG` (no-MCU servo), runs the baseline eval (10 fixed goals, ~2 min), then trains:
     100-step episodes with the updates in between (the arm holds still at the start pose for ~1 s while they run).
  3. A session ends by itself after 20 min (`session_cap`, the thermal duty cycle), or on any trip / stall / failed
     reset. Everything is saved; the servo safe-stops the arm to rest. Wait 10 min (cool-down) after a
     `session_cap`; after a trip, look at the cause first (servo summary `fault_bits`, the trainer's `halt`). A STALE_CMD
     (`0x100`) halt means the trainer stopped sending: send `timing.jsonl` and the trainer's `keep-alive starved` /
     `policy step gap` lines (plan §10 2026-10-13).
  4. Resume: the trainer first, `pixi run train-reach-sac --resume runs/sac-her/<ts> --attended`, then the servo
     again (step 2, new tick-log name). Repeat until `--total-steps` (exit 0).
     Run `20261012-191349-203e` (collapsed at step ~4060 under the old recipe): resume it ONCE with `--reset-actor`
     (plan §10 2026-10-13 "Anchored fine-tuning"), later resumes without it.
  5. Cut power at once on any contact with the table, the camera mount or a cable, on shaking, or on any motion you
     did not expect; then resume from step 4.
  6. Every eval is in `evals.jsonl`; a rollback (two evals below baseline − 0.15, a tripped eval, or the last 10
     training episodes below baseline − 0.15) restores the best actor (`rollbacks.jsonl`); the 3rd in a session ends it. Review
     trigger (plan §5): success below 10 % after 40k steps, or no drop in the mean final distance over 3 evals.
- **Artifact:** `runs/sac-her/<ts>/{model.zip, replay_buffer.pkl, baseline.json, evals.jsonl, best_policy.pt,
  policy_export.pt, sessions.jsonl, timing.jsonl, rollbacks.jsonl, summary.json}` and the servo tick logs.
- **Pass:** `--total-steps` reached (exit 0); then the fine-tuned `policy_export.pt` evaluated with `deploy-policy`
  (HS4 procedure, the same seed) against the HS4 result. Fill the results table.
- **5 mm (plan §10 2026-10-14):** the 15 mm run is stopped (the zero-shot actor is at the ceiling). The next run starts
  from a sim policy trained at 5 mm (`train-sim-sac ... --success-radius-mm 5`) and adds `--success-radius-mm 5` to
  step 1 (new run only; resumes keep it). Episodes that miss 5 mm run the full 100 steps (10 s). Compare final
  errors (`final_error_m_p50` in `evals.jsonl`), not only the success rate. `deploy-policy` scores at the export's radius
  (the 5 mm sim export: 5 mm; M6: 15 mm), so its zero-shot ladder (8/10 per rung) is a 5 mm check for a 5 mm policy.
- **Temperature median filter (plan §10 2026-10-15):** thermal decisions read the median of 11 reads; the servo
  summary's `temp_outlier_reads` counts raw reads > 3 °C off it (~0.4 % of reads per joint is normal on this arm).
- **wrist_flex limit (plan §10 2026-10-15):** `joints.max_ticks` wrist_flex is 2960 (real overloads at 2989 / 3013
  with the elbow flexed). Goal pairs for a seed changed with it: compare runs before and after unpaired.
- **Before any critic-based fine-tune session (plan §10 2026-10-14):** replay the actor update against the run's
  critic offline and score the moved actor in sim (`scripts/finetune_critic_diag.py`); if it degrades in sim, do not
  unfreeze the actor on the arm. Resume a rolled-back run from `best_policy.pt`, not `model.zip`'s actor.
- **Shown-to-fail:** step 1 without `--attended` against the feetech servo must refuse (exit 2) with no motion.

## HS6 - Zero-shot die-to-zone eval on the real arm (software ready, not runnable yet)

Added 2026-10-16 (plan §10 "Die support in deploy-policy (D5)"; ARCHITECTURE 0.62; sim plan HS6). Attended.

- **Needs (none met yet):** the die policy retrained on a verified `sim_params` release >= 1 (HS1-HS3; the v0 attempt-2
  policy is refused on the real arm by `deploy-policy`, plan §10 2026-10-16), H8 (hand-eye with the fiducial), the
  perception process writing the object table (D3), H9 (the tray; until it is measured only `--manual-die-reset`
  runs), the Phase 4 feasibility gate (D4), the verifier's false-accept check <= 2 %.
- **Dry run first** against the sim servo with the perception process (or a fake writer) on the object table:
  `deploy-policy --policy <die export> --protocol zero-shot` then `servo --backend mujoco` (a v0 export is allowed
  there, with a note).
- **Procedure:** as HS4 (start `deploy-policy` first, then the servo), with the perception process already running:
  `deploy-policy --policy <die export> --protocol zero-shot --seed 7 --attended [--accept-no-watchdog]
  [--object-table <name>] [--manual-die-reset]`. The task comes from the export; the ladder is 10 / 10 / 30 episodes
  at 0.5 / 0.75 / 1.0 with the HS4 advance rule; the goal is the zone. Refused before anything moves: a v0 /
  unverified die export, no or stale object table (> 1 s) or the fiducial not OK, and (no tray) a run without
  `--manual-die-reset`.
- **Manual die reset (until H9):** before every episode the arm goes home and holds; place the die at its start, take
  your hand out, then press Enter in the deploy terminal (or `touch <run dir>/die_ready`; `<run dir>/die_reset_waiting`
  shows the episode it waits for). With the tray, the attendant resets nothing.
- **Stops:** a trip / stall as in HS4; the tracker UNKNOWN on > 50 % of an episode's steps (`halt_reason`
  `tracker_unknown`: look at the camera, the die and perception, then `--resume`); the object table going stale or
  the fiducial failing before an episode (`perception_stale`).
- **Artifact:** `runs/deploy/<ts>/{episodes.jsonl, summary.json, episodes/*.h5}`; per episode `visible_fraction`,
  `final_die_xyz`, success and trips. Count push vs place from the tracker trace (the die block of the observations
  in the episode files) or the camera.
- **Pass:** n = 30 at full speed with the Wilson 95 % lower bound >= 0.60 (`verdict.finetune_eligible`).
- **Shown-to-fail:** the v0 attempt-2 export (`runs/sac-sim/die-a2-20261016-185514`) against the feetech servo must
  refuse (exit 2) with no motion.

# Wave 1 review: `integration/phase0-wave1` vs `main`

Reviewed on 2026-09-30, using an adversarial grill review: 7 attacker reviewers, then 3 defenders to challenge their findings. The verdict was **REQUEST_CHANGES**, with 6 blocking findings, 11 should-fix and 9 nits.

The user decided on 2026-10-01 (logged in plan §10):

- **B6:** a thermal trip goes to the rest pose with torque off. The arm never holds pose while energised.
- **MCU boot:** the MCU boots LATCHED.
- **Scope:** fix every blocking and should-fix item now. The nits are optional.

Each item below names the fix unit that owns it:

| Unit | Scope |
|---|---|
| A | `safety/` and `config/` |
| B | `firmware/`, `servo/heartbeat.py`, `docs/HARDWARE_CHECKLIST.md` |
| C | `sim/`, plus backend re-energise tests |

## BLOCKING

**B1. Rest pose vs BOX_PRESENT: the transit path is never validated** (A).
- **Code:** `safety/pipeline.py:137-140`, `reenergize.py:_ramp_home (~333-362)`, `safe_state.py:292-298`, and `config/loader.py`, which checks home and rest only against the joint limits.
- **Evidence:**
  - The real gravity-stable rest pose `[2050,2200,2301,2745,2040,2065]` puts the tip at (0.208, -0.001, -0.012) m. That is below the box floor of z = 0.01.
  - The placeholder home of 2047 puts the tip at (0.391, 0, 0.226), outside the box. Every start therefore goes RUN, then BOX_PRESENT on the first tick, then SAFE_STOPPING, then TORQUE_OFF. It fails safe, but the arm can never run.
  - `ramp_toward` moves each joint independently at 8 ticks per tick, and a sampled path dips to z = -0.023, below the table.
- **Fix:**
  - (a) Loader: FK(home) is inside the box with at least 10 mm margin. FK(rest) is at least 5 mm above the table and clear of the D435.
  - (b) Loader: sample FK along the exact ramp, tick by tick, from rest to home and from home to rest. Require z ≥ table + 5 mm.
  - (c) Run time: a transit zone that applies only to `_ramp_home` and `SafeStateSequencer`, and is already time-bounded.
    - Check the table floor on the ramp target every tick.
    - Order the ramp so the shoulder and elbow lift first.
    - Hold any joint whose motion would lower the tip.
    - If the ramp still cannot finish, time out to torque-off.
  - (d) `reenergize` refuses to start unless the present pose is near rest or inside the box.
  - (e) Add an end-to-end test that loads the shipped `so101.yaml`.
- **Example in-box home:** [2047, 960, 2700, 2748, 2047, 2047] puts the tip at (0.20, 0, 0.15). It is a placeholder until H5 measures real values.

**B2. `reenergize` sends no heartbeat, so the deadman latches during re-energise** (A).
- **Code:** `reenergize.py:268-362`. The MCU must be ARMED before `ping_all`, because the relay powers the servos. `reenergize` then blocks for up to about 8.9 s with no heartbeat, past the 1000 ms MCU timeout.
- **Fix:** add a per-tick hook to `reenergize` and to `SafeStateSequencer`, either a `tick_hook` callable or wrapped Bus and Clock objects.
  - The hook sends a heartbeat only when that tick's `read_block` succeeded and `write_goal` returned True.
  - Keep the pre-torque phase under 0.8 s.
  - Pin the hook in F5/F6, and test it against `FakeMcu`: no latch during a full re-energise.

**B3. `reenergize` writes the protection registers with raw `write_reg`** (A).
- **Code:** `reenergize.py:308-311`. There is no Lock=0, no check that torque is off, and no read-back. This bypasses the guards in `FeetechBus.write_registers` (`feetech_bus.py:170-194`).
- **Fix:**
  - Write Lock=0 first.
  - Confirm Torque_Enable == 0 with `read_reg`.
  - Read back every field on every motor, and raise before `set_torque(True)` if any value differs.

**B4. The voltage divider is on the servo side of the relay, so ARM is refused forever** (B).
- **Code:** `docs/HARDWARE_CHECKLIST.md:40-43`, and `firmware/watchdog_mcu/main.py:42`, where ARM requires `not self.uv`. With the relay open the servo side reads about 0 V.
- **Fix:**
  - Add a second divider on the PSU side, before the relay, and use it for the ARM undervolt check.
  - Keep the servo-side divider for drop timing and weld detection.
  - Update `PROTOCOL.md`, the test vectors and `FakeMcu` to match.

**B5. The safe-state 5 s bound is too short for the ramp speed** (A).
- **Code:** `safe_state.py:283` and `so101.yaml:47-48`. The ramp moves 240 ticks/s. From the shoulder_lift limit to home takes about 4.6 s, and home to rest about 2.9 s, so the sequence times out and torques off mid-air. From the wrist_roll limit alone it takes about 8.5 s.
- **Fix:** the loader must require `bound_s ≥ (max|limit − home| + max|home − rest|) / (step × hz) + margin`. Alternatively, raise the step or give each phase its own bound.

**B6. COOLDOWN keeps the arm energised, which the main plan forbids** (A). **Decided: go to rest and turn torque off.**
- **Code:** `state_machine.py:175-189`.
- **Fix:**
  - A thermal trip runs the safe-state sequence to rest and turns torque off.
  - After the dwell, it resumes through a full `reenergize`, using a new edge from thermal-only TORQUE_OFF back to CHECK. The automatic resume stays limited to 3 per hour.
  - Update ARCHITECTURE §6.5 to match.

## SHOULD-FIX

- **S1 (C).** Sim re-energise crashes: `sim/mujoco_bus.py:166-170` raises KeyError from `read_reg('Goal_Position')`, so `reenergize` fails on MujocoBus.
  - Serve live registers, as `fake_bus.py:278-284` does.
  - Add re-energise tests on FakeBus and MujocoBus. `tests/test_reenergize.py` uses only a private LoggingBus.
  - Also find the test that writes `MUJOCO_LOG.TXT` ("Nan, Inf or huge value in QACC at DOF 0"). Either make it intentional and silence the log, or fix the instability.
- **S2 (A).** `SafeStateSequencer(cfg, clock)`, built per the frozen F5 signature, silently loses the tracking clamp and uses motor ids 1-6 instead of the calibration ids (`safe_state.py:243-261`). Make both arguments required.
- **S3 (A).** There is no margin between the box clamp and the box trip (`pipeline.py:137-147`), so 1 mm of sag at the z floor causes a stop that needs a fresh process. Add about 5 mm of tolerance to the present-position check.
- **S4 (A).** LATCHED sends torque-off only on entry (`state_machine.py:137-138`). Emit it every tick; it is idempotent.
- **S5 (A).** A failure after `set_torque(True)` leaves torque on (`reenergize.py:325-330`).
  - Wrap that phase in try/except and turn torque off on failure.
  - Check the temperature and status bits before energising.
- **S6 (A).** A resume can go ahead on a failed bus read: the COOLDOWN-to-RUN transition omits `block_ok` (`state_machine.py:188`). This is superseded by B6, but no resume path may ignore `block_ok`.
- **S7 (B).** The deadman latch can be cleared without the button (`firmware/main.py:10/42/72`), because an MCU reset boots DISARMED. **Decided: boot LATCHED with reason MCU_RESET.** Update the vectors, `FakeMcu`, `PROTOCOL.md` and the checklist to match.
- **S8 (B).** `FakeMcu` drifts from the firmware (`servo/heartbeat.py:136,151-154,178-187`), and the vectors don't catch it:
  - The timeout test uses `>` where the firmware uses `>=`.
  - TEST_DROP without the jumper sends no reply and latches from any state.
  - The reset button clears the trip reason.
  - The vectors skip the exact 1000 ms boundary, and `FakeMcu` replays only the portable cases (`tests/test_firmware_vectors.py:101`). Broaden the cross-check.
- **S9 (B).** Serial link robustness (`heartbeat.py`):
  - `send` catches only `SerialTimeoutException`, so unplugging the USB raises. Return False and report the link as dead instead.
  - A refused TEST_DROP blocks for 2 s and then raises TimeoutError.
- **S10 (A).** Thermal edge cases:
  - `thermal.py:138`: the stale-temperature timer starts at construction, so the first failed read after a slow re-energise causes a spurious trip.
  - `state_machine.py:56`: the thermal ring holds 32 entries, but the loader allows `max_trips_per_hour` up to 10^6. Bound it in the loader.
  - Thermal trips are counted in two places (the monitor and the state machine). Pick one source.
- **S11 (A).** The calibration check is partly tautological: `drive_mode` is copied from config, so comparing it proves nothing. Document that this needs a motion-sign test, which belongs to H-track/U3, and drop the false claim.
- **S12 (A).** Tests that pass vacuously:
  - `test_thermal.py`'s "failed block never counts as cooling" runs for 0.5 s against a 30 s dwell.
  - `test_reenergize.py` never tests the ramp timeout. Replacing the deadline check with `if False:` survives, which would spin forever with torque on.
  - `test_reenergize.py` never tests `write_goal` or `set_torque` returning False, and uses an instant-follow plant.
  - `test_calib_check.py:67` has a near-vacuous `or` assert.
- **S13 (B).** HARDWARE_CHECKLIST H2: the e-stop timing test via 0xE1 measures the relay, not the e-stop contact. Require a scope or an equivalent independent measurement.

## NIT (optional)

- Backend drift: `write_goal` error handling and `torque_enabled` (a property in one backend, a method in another) differ, and FakeBus `write_registers` has no torque guard.
- The position sign-bit-15 decode is untested.
- MujocoBus ignores the calibrated position limits, and ignores the `drive_mode` sign on velocity.
- `test_mujoco_bus.py:28` has a stale "not runtime_checkable" comment and no `isinstance` check.
- `test_vectors.json` tests a 2^32 wrap, but RP2040 MicroPython wraps at 2^30.
- The 2N7000 is a marginal relay-coil driver at 3.3 V gate drive.
- The firmware's partial-frame buffer never times out.
- The stuck monitor is fed the commanded delta. This is documented, but the change to F5 is not declared.
- ARCHITECTURE §6 doesn't mention the safe-state timeout behaviour or the sequencer arguments.

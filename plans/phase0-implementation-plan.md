# Phase 0: Foundation + Stage A Critical-Path Scaffolding — Implementation Plan

> **Version 1 (2026-09-30).** Source plan: `so101-self-learning-rl-plan.md` v2 (§1, §2, §2A, §3, §5 Stage A critical path / Budgets / Latency budget / Phase 0, §6, §10). Architecture: `docs/ARCHITECTURE.md` v0.1. §10 decisions are settled and are **not** re-opened here.
>
> Scope discipline (anti-hypertrophy): Phase 0 deliverables plus *scaffolding* (frozen interfaces, skeletons, smoke tests) for the five Stage A critical-path items. No Stage B/C work, no C++/Rust, no ROS2, no SAM2/GDINO, no pixel world model, no sysid parameter fitting, no `pip install` support.

## Overview

| Item | Value |
|---|---|
| Objective | A pixi package `so101_rl` in which the same random-policy script runs safely in MuJoCo and on the real arm with logging; a policy-independent safety layer in its own 30 Hz process with a hardware deadman; frozen interfaces for env, servo↔policy IPC, safety, logging, perception and RL so Phases 1–3 can start in parallel. |
| Deliverables | `pyproject.toml` + `pixi.lock`; `src/so101_rl/{config,servo,safety,env,perception,sim,wm,rl,logging,_vendor/tdmpc2}`; `assets/so101/`; `firmware/watchdog_mcu/`; `tests/`; `docs/HARDWARE_CHECKLIST.md`; hardware artifacts (`calib/hand_eye.yaml`, `calib/fiducial.yaml`, soak / e-stop / deadman reports under `runs/`). |
| Domain | Real-time robot control (Python), RL infrastructure, packaging |
| Technologies | python 3.12, numpy 2.2, pixi 0.69, lerobot 0.6.1 (PyPI, driver only), feetech-servo-sdk, conda-forge pytorch (cpu/gpu metapackages), torchrl + tensordict, stable-baselines3, gymnasium 1.x, MuJoCo 3.x, pyrealsense2 2.58 (PyPI), opencv, h5py, hypothesis, MicroPython on RP2040 |
| Unit of this plan | Master passed no `Unit Id` / `Owns Files`; this phase is treated as its own sole unit owning the whole (currently empty) source tree. The plan partitions that tree into sub-units below (`execution_shape`). |
| Preconditions (human) | H1 NVIDIA driver restore (GPU smoke only; CPU path is accepted for the spike). H2/H3 power path + MCU before any *unattended* motion. |

### Environment facts checked on 2026-09-30

- Workspace is a git repo (branch `main`, one commit: plan + architecture doc). `.gitignore` already covers `.pixi/`, `.agent-state/`, `runs/`, `data/`, `checkpoints/`, `*.hdf5`.
- `pixi 0.69.0` installed. `nvidia-smi` fails (driver not loaded) — GPU tasks are CPU-first.
- Calibration file exists at `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`, keyed by motor name with `id, drive_mode, homing_offset, range_min, range_max` (ids 1–6; `wrist_roll` range 0–4095, i.e. a full turn).
- lerobot 0.6.1 (PyPI): `requires_python >=3.12`; `torch>=2.7,<2.12`, `torchvision>=0.22,<0.27`, `numpy>=2.0,<2.3`, `gymnasium>=1.1.1,<2`, `draccus`; extras `feetech` (= `feetech-servo-sdk<2`, `pyserial`, `deepdiff`) and `grpcio-dep` (= `grpcio`, `protobuf`) confirmed. `intelrealsense` extra pins `pyrealsense2<2.57`, so it is **not** used; `pyrealsense2 2.58.4` has a cp312 manylinux wheel.
- conda-forge: `pytorch-gpu` 2.7–2.13 (cuda12.8 / cuda13.0 builds), `pytorch-cpu`, `torchvision`, `torchcodec` ≤0.16, `cuda-version` ≤13.4, `mujoco-python` 3.13. PyPI: `torchrl` 0.9.0…0.14.0, `tensordict` 0.14.2, `stable-baselines3` 2.9.0, `mujoco` 3.14.0.
- TD-MPC2 (`nicklashansen/tdmpc2`, MIT, no packaging files) third-party imports: `torch`, `numpy`, `hydra`, `omegaconf`, `termcolor`, `tensordict`, `torchrl`, `gymnasium`, `pandas`. `common/buffer.py` hard-codes `torch.device('cuda:0')` and `torch.cuda.mem_get_info()` — a CPU patch is required for the spike.
- SO-ARM100 (`TheRobotStudio/SO-ARM100`, **Apache-2.0**): `Simulation/SO101/{so101_new_calib.xml, scene.xml, joints_properties.xml, assets/*.stl (17 files)}`.
- LeRobot 0.6.0 Feetech STS3215 control table (verified in the installed `LIT` env): `Return_Delay_Time (7,1)`, `Min_Position_Limit (9,2)`, `Max_Position_Limit (11,2)`, `Protection_Current (28,2)`, `Homing_Offset (31,2)`, `Operating_Mode (33,1)`, `Protection_Time (35,1)`, `Overload_Torque (36,1)`, `Over_Current_Protection_Time (38,1)`, `Torque_Enable (40,1)`, `Goal_Position (42,2)`, `Lock (55,1)`, `Present_Position (56,2)`, `Present_Velocity (58,2)`, `Present_Load (60,2)`, `Present_Voltage (62,1)`, `Present_Temperature (63,1)`, `Status (65,1)`, `Moving (66,1)`, `Present_Current (69,2)`, `Maximum_Acceleration (85,1)`. Sign-magnitude bits: `Present_Load` 10, `Homing_Offset` 11, `Goal_Position`/`Present_Position`/`Present_Velocity` 15. Resolution 4096 ticks/rev (0.0879°/tick).
- LeRobot API: `FeetechMotorsBus(port, motors: dict[str, Motor], calibration=None, protocol_version=0)` exposes `port_handler`, `packet_handler`, `sync_read(name, motors=None, *, normalize=True, num_retry=0)`, `sync_write(...)`, `read(name, motor, *, normalize=True)`, `_sync_read(addr, length, ids)`, `configure_motors(return_delay_time=0, maximum_acceleration=254, acceleration=254)` (also clears Phase bit 4 on sts3215), `read_calibration()`, `enable_torque()/disable_torque()` (both also write `Lock`). `SOFollower.configure()` writes `Operating_Mode=POSITION, P_Coefficient=16, I=0, D=32`; `disable_torque_on_disconnect=True`, `use_degrees=True`, `max_relative_target=None` are the defaults. No mock bus ships in the wheel — we write our own.
- Reusable code: `LIT/src/lerobot-isaac-deploy/src/lerobot_isaac_deploy/arm_motor_writer.py` (`compute_targets`, `write_targets`, `home_targets`, `ramped_home`, `read_joint_limits`), `LIT/scripts/_arm_kinematics.py` (`WorkspaceBox`, `DEFAULT_BOX = x(0.08,0.34) y(-0.22,0.22) z(0.010,0.28)`, `TABLE_Z_M=-0.02`, `clamp_joint_step`; its FK uses placo via `lerobot.model.kinematics` — **not reused**, see L7), `LIT/scripts/calibrate_hand_eye.py`, `LIT/scripts/_object_pose_depth.py`, `robot_data_runner/src/robot_data_runner/safety.py` (`SafetyMonitor` stuck-action watchdog, SIGINT handler). `safety_limits.py` exists at `LIT/src/robot-data-runner/src/robot_data_runner/safety_limits.py`, not in the standalone `robot_data_runner`. It was corrected on 2026-09-30. `clamp_action` and the NaN handling are still re-implemented in `safety/clamp.py`, with that file as a reference, because the units convention differs (§F1).

## Specialist Guidance Summary

**real-time-control-specialist** (servo loop, IPC, watchdog):
- One `SharedMemory` segment, two 64-byte-aligned blocks (command: policy→servo; state: servo→policy), header with magic/version/size asserted at attach. Single-writer **seqlock** (seq odd → payload → seq even) **plus a CRC32** over the payload so torn-read detection does not depend on x86 store ordering. Reader bounds retries (3) and treats failure as "no new data" — the servo never blocks on the policy.
- Command carries the `state_seq` it was computed from; the servo applies a delta **once per command seq** (dedupe), else a 10 Hz delta would be integrated three times at 30 Hz.
- Stale detection on the servo's own `monotonic_ns`: age > 150 ms → HOLD; N=5 consecutive 100 ms windows without a new seq → SAFE_STOPPING.
- Interpolate the 10 Hz goal over 3 ticks (no step input). Clamp order per tick: interpolate → joint limits → per-tick velocity → tracking clamp |target−present| ≤ E → write. **Anti-windup: the stored target is the clamped value.** E must exceed `vmax × (2–3 ticks + read latency)` or the loop limit-cycles; log the clamp-active fraction. Gripper gets its own E.
- `SCHED_FIFO` via `os.sched_setscheduler` in try/except, `rt_active` flag, `REQUIRE_RT=1` makes failure fatal on the robot, unset in CI. Deadline loop resyncs on a large overrun instead of bursting. `gc.collect(); gc.freeze(); gc.disable()` after preallocation. Preallocated ring for tick jitter; p99.9 needs ≥10⁴ samples.
- Heartbeat: one write per tick from the loop thread **after** read+write succeeded, `write_timeout=0.002`, a timeout counts as a missed heartbeat. Never from a timer thread (liveness must be causally after the work). Protocol with a rolling counter so a stuck repeating sender is rejected.
- State machine: RUN, HOLD, COOLDOWN, SAFE_STOPPING, TORQUE_OFF, LATCHED; fault code is a bitmask; resume rule per cause; re-sync target=present on every resume.
- Tests: inject `Clock`, `Bus`, heartbeat sink; `FakeClock` makes a 24 h soak run in seconds; `FakeBus` with first-order plant, latency distribution and fault injection; `FakeMcu` running the same protocol; hypothesis property tests (no NaN reaches the bus, |Δ| ≤ vmax, |target−present| ≤ E, 0 ≤ goal ≤ 4095). At least one HIL test with the real bus stays on the release checklist.

**lerobot-specialist** (driver; report based on 0.5.1, cross-checked here against the installed 0.6.0 and PyPI 0.6.1):
- Drive the bus directly for the 30 Hz loop; `send_action` does a second bus read per call, clamps relative to *present* not the last target, and normalises. Use `SOFollower` only for connect/calibrate/configure, or replicate `configure()` on the bus.
- Block read of `56..66` needs the SDK's `GroupSyncRead` on `bus.port_handler/packet_handler`; decode sign-magnitude manually on the raw path.
- Read-back check: compare `Homing_Offset` (decoded), `Min/Max_Position_Limit`, `drive_mode`, id against the JSON; the raw `_read` returns sign-magnitude (e.g. −1434 stored as 2048+1434) — a naive compare gives a false mismatch.
- Goal=present before torque enable, verify the read-back, then enable; set `Acceleration` before enabling. Failed reads must never be treated as present=0. Torque-off drops shoulder/elbow — ramp to a rest pose first.
- Pixi: conda `pytorch` maps to PyPI `torch`, so lerobot's torch requirement should be satisfied by the conda package — **verify with `pixi list`** that exactly one torch exists and it is conda; fall back to listing lerobot's runtime deps by hand with `--no-deps` if not.

## Frozen Interfaces

Everything below is the contract between units. A unit may add to these files' behaviour but may not change a listed signature, dtype, enum value or ordering without updating this section **and** `docs/ARCHITECTURE.md` §5/§6 in the same change.

### F1. Units and configuration — `src/so101_rl/config/`

One convention everywhere (§10 "Action space", plan §5 "Action units"): **raw servo ticks** (0.0879°/tick, 0–4095) inside the servo process; radians only in FK/env; **no [-1,1]-to-degrees hidden scaling anywhere except `env/goal_env.py` where `delta_ticks = round(action × max_delta_ticks)`**.

```python
# src/so101_rl/config/schema.py
TICKS_PER_REV: Final[int] = 4096
JOINT_NAMES: Final[tuple[str, ...]] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
N_JOINTS: Final[int] = 6

def ticks_to_rad(ticks: np.ndarray, calib: dict[str, MotorCalibration]) -> np.ndarray: ...   # uses homing_offset; result shape [6]
def rad_to_ticks(q: np.ndarray, calib: dict[str, MotorCalibration]) -> np.ndarray: ...      # int32, NOT clipped

@dataclass(frozen=True)
class JointLimits:            # ticks, int32[6]; intersection of config and calibration range_min/max
    min_ticks: np.ndarray
    max_ticks: np.ndarray

@dataclass(frozen=True)
class ActionConfig:
    max_delta_ticks: np.ndarray          # int32[6]; |Δ| per 10 Hz command; default arm 45, gripper 60
    tracking_error_ticks: np.ndarray     # E, int32[6]; default arm 120, gripper 400
    max_ticks_per_tick: np.ndarray       # per 30 Hz tick velocity cap; default 20
    interp_ticks: int = 3

@dataclass(frozen=True)
class ThermalConfig:
    trip_c: float = 60.0
    resume_c: float = 50.0
    resume_dwell_s: float = 30.0
    max_trips_per_hour: int = 3
    duty_on_s: float = 1200.0
    duty_off_s: float = 600.0
    stale_temp_s: float = 1.0            # temperature older than this is a fault

@dataclass(frozen=True)
class RateConfig:
    servo_hz: float = 30.0
    policy_hz: float = 10.0
    stale_cmd_ms: float = 150.0
    max_consecutive_misses: int = 5
    heartbeat_timeout_ms: int = 1000
    max_bus_failures: int = 3            # consecutive read/write failures → LATCHED

@dataclass(frozen=True)
class SafeStateConfig:
    bound_s: float = 5.0
    ramp_ticks_per_tick: int = 8
    home_ticks: np.ndarray               # int32[6]
    rest_ticks: np.ndarray               # int32[6]; gravity-stable, verified by H5
    torque_off_order: tuple[str, ...] = ("gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan")

@dataclass(frozen=True)
class WorkspaceBox:                      # metres, base frame; defaults from LIT DEFAULT_BOX
    x: tuple[float, float] = (0.08, 0.34)
    y: tuple[float, float] = (-0.22, 0.22)
    z: tuple[float, float] = (0.010, 0.28)

@dataclass(frozen=True)
class RegisterConfig:                    # written in configure(); ALL of them, every connect
    return_delay_time: int = 0
    operating_mode: int = 0              # POSITION
    p_coefficient: int = 16
    i_coefficient: int = 0
    d_coefficient: int = 32
    maximum_acceleration: int = 254
    acceleration: int = 254
    protection_current: int = 250        # gripper default from lerobot; arm value measured in H6
    overload_torque: int = 25
    over_current_protection_time: int = 20   # UNVERIFIED default; H6 proves fold-back

@dataclass(frozen=True)
class So101Config:
    port: str
    heartbeat_port: str | None
    calibration_path: Path
    shm_name: str
    joints: JointLimits
    action: ActionConfig
    thermal: ThermalConfig
    rates: RateConfig
    safe_state: SafeStateConfig
    box: WorkspaceBox
    registers: RegisterConfig
    kinematics_path: Path                # config/so101_kinematics.yaml

def load_config(path: Path | None = None, *, calibration: dict[str, MotorCalibration] | None = None) -> So101Config: ...
def load_calibration(path: Path) -> dict[str, MotorCalibration]: ...
```

Defaults live in **one file**, `src/so101_rl/config/so101.yaml`. A test (`tests/test_config_single_source.py`) greps `src/` for numeric literals matching known limit values outside `config/` and fails if any appear (L4: one source of truth; launch environments are diffed by `so101_rl.config.dump_env()`).

### F2. Bus protocol — `src/so101_rl/servo/bus.py`

```python
class BlockRead:                        # preallocated by the owner; all int32[6] unless noted
    position: np.ndarray                 # ticks 0..4095 (sign bit 15 decoded)
    velocity: np.ndarray                 # raw units, sign bit 15 decoded
    load: np.ndarray                     # sign bit 10 decoded; 1000 = 100 %
    voltage_dv: np.ndarray               # 0.1 V units
    temperature_c: np.ndarray
    status: np.ndarray                   # Status (65) protection bitfield, uint8 semantics
    moving: np.ndarray
    t_mono_ns: int
    ok: bool                             # False on ANY comm failure; arrays then hold the PREVIOUS values

class Bus(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...                      # never changes torque implicitly
    def ping_all(self) -> dict[int, bool]: ...             # id → answered
    def read_block(self, out: BlockRead) -> bool: ...      # ONE sync_read of addr 56, length 11 (56..66); returns out.ok
    def write_goal(self, goal_ticks: np.ndarray) -> bool: ...   # sync_write Goal_Position, raw; caller guarantees int32 in 0..4095
    def set_torque(self, enabled: bool, ids: Sequence[int] | None = None) -> bool: ...
    def read_reg(self, name: str, motor_id: int) -> int: ...      # sign-decoded
    def write_reg(self, name: str, motor_id: int, value: int) -> None: ...
    def read_calibration_regs(self) -> dict[str, MotorCalibration]: ...   # Homing_Offset (decoded), Min/Max_Position_Limit, id; drive_mode from config
```

The plan's latency table names an 8-byte read (56..63). We read **11 bytes** so `Status (65)` is in the same packet and the protection bits are checked every tick (plan Phase 0: "any firmware Protection status bit counts as a trip"). Expected cost ≈ +0.2 ms. If T3.6 measures > 0.5 ms extra, fall back to length 8 plus `Status` every 10th tick and record that in `ARCHITECTURE.md` §6.

Implementations: `servo/feetech_bus.py: FeetechBus(cfg, calibration)` (wraps `FeetechMotorsBus`; builds one `scservo_sdk.GroupSyncRead(port_handler, packet_handler, 56, 11)` at connect; decodes sign-magnitude itself), `servo/fake_bus.py: FakeBus(clock, plant=FirstOrderPlant(), faults=FaultInjector())`, `sim/mujoco_bus.py: MujocoBus(model, data, calibration, cfg, action_delay_ticks=0)`.

### F3. Servo ↔ policy shared memory — `src/so101_rl/servo/shm.py`

```python
SHM_MAGIC: Final[int] = 0x534F31303152_4C31          # "SO101RL1"
SHM_LAYOUT_VERSION: Final[int] = 1

class CmdMode(IntEnum): IDLE = 0; RUN = 1; HOLD = 2; STOP_REQ = 3

HEADER_DTYPE = np.dtype([("magic", "<u8"), ("version", "<u4"), ("cmd_size", "<u4"), ("state_size", "<u4"), ("_pad", "V44")])   # 64 B

CMD_DTYPE = np.dtype([                     # written by policy, read by servo
    ("seq", "<u8"), ("t_mono_ns", "<i8"), ("state_seq", "<u8"),
    ("mode", "<u4"), ("episode", "<u4"),
    ("delta_ticks", "<f4", (6,)),
    ("crc32", "<u4"), ("_pad", "V4"),
], align=True)                              # itemsize asserted == 64

STATE_DTYPE = np.dtype([                   # written by servo, read by policy
    ("seq", "<u8"), ("t_mono_ns", "<i8"), ("tick", "<u8"), ("overruns", "<u8"),
    ("present_ticks", "<i4", (6,)), ("velocity", "<i4", (6,)), ("load", "<i4", (6,)),
    ("voltage_dv", "<u1", (6,)), ("temp_c", "<u1", (6,)), ("status", "<u1", (6,)), ("moving", "<u1", (6,)),
    ("target_ticks", "<i4", (6,)), ("goal_ticks", "<i4", (6,)),
    ("applied_delta_ticks", "<f4", (6,)),   # the post-clamp delta actually applied this command (L4)
    ("clamped_mask", "<u1", (6,)),
    ("fault_bits", "<u4"), ("sm_state", "<u4"), ("last_cmd_seq", "<u8"), ("hb_ok", "<u1"),
    ("crc32", "<u4"),
], align=True)                              # itemsize asserted == 232 (corrected 2026-09-30: natural alignment; block stride 256 B)

class SeqlockBlock:                         # single writer, many readers, no locks
    def __init__(self, buf: memoryview, dtype: np.dtype) -> None: ...
    def write(self, rec: np.ndarray) -> None: ...         # seq |= 1 → payload → crc → seq = (seq|1)+1
    def read(self, out: np.ndarray, retries: int = 3) -> bool: ...   # False = torn/no data; `out` untouched on False

class ServoShm:
    @classmethod
    def create(cls, name: str) -> "ServoShm": ...          # servo process only; unlinks on close
    @classmethod
    def attach(cls, name: str) -> "ServoShm": ...          # policy side; unregisters from resource_tracker (py3.12)
    def write_state(self, rec: np.ndarray) -> None: ...
    def read_state(self, out: np.ndarray) -> bool: ...
    def write_cmd(self, rec: np.ndarray) -> None: ...
    def read_cmd(self, out: np.ndarray) -> bool: ...
    def close(self) -> None: ...

class ServoIpc(Protocol):                  # what ServoCore needs; ServoShm and LocalIpc (in-process, sim) implement it
    def read_cmd(self, out: np.ndarray) -> bool: ...
    def write_state(self, rec: np.ndarray) -> None: ...
```

Rules: timestamps are `time.monotonic_ns()` (valid across processes on one host); the servo owns the segment; the policy never unlinks; a repeated `seq` is "no new command".

### F4. Heartbeat / watchdog MCU wire protocol — `src/so101_rl/servo/heartbeat.py` + `firmware/watchdog_mcu/PROTOCOL.md`

Host → MCU (over a dedicated USB-CDC serial, 115200 8N1):

| Byte | Meaning | Payload |
|---|---|---|
| `0x5A` | ARM (request relay close) | — ; accepted only in DISARMED, bus voltage sane, not latched |
| `0xA5` | HEARTBEAT | `seq8`, `~seq8` (3 bytes total); MCU requires `seq8 == last+1 mod 256` |
| `0xE0` | STATUS_REQ | — |
| `0xE1` | TEST_DROP | — ; opens the relay deliberately and measures time until V_bus < 1 V; only accepted when a physical TEST jumper is present |

MCU → host: one status byte after every accepted frame: `bit0 relay_on, bit1 latched, bit2 armed, bit3 undervolt, bits4–7 trip_reason` (0 NONE, 1 TIMEOUT, 2 UNDERVOLT, 3 SEQ_ERROR, 4 ESTOP_PIN, 5 TEST, 6 MCU_RESET). After `0xE1` the MCU also sends `drop_ms` as one uint16 LE.

Firmware invariants (any language; reference implementation MicroPython on RP2040, ≈50 lines): relay pin defaults LOW at reset (fail-open); `state ∈ {DISARMED, ARMED, LATCHED}`; ARMED→LATCHED after 1000 ms without a valid heartbeat (`ticks_diff`, wrap-safe), on bad seq, on V_bus below threshold for >50 ms, or on the e-stop input; LATCHED clears **only** by the physical reset button held 500 ms → DISARMED; hardware WDT 250 ms fed in the loop; ADC on a 47k/10k divider sampled ≥1 kHz.

```python
class McuStatus(NamedTuple): relay_on: bool; latched: bool; armed: bool; undervolt: bool; reason: int

class HeartbeatLink(Protocol):
    def arm(self) -> McuStatus: ...
    def send(self, seq: int) -> bool: ...              # non-blocking; False on write timeout or bad echo → counts as a missed heartbeat
    def status(self) -> McuStatus: ...
    def test_drop(self) -> int: ...                    # drop_ms

class SerialHeartbeatLink(HeartbeatLink): def __init__(self, port: str, clock: Clock, write_timeout_s: float = 0.002) -> None
class FakeMcu:                                          # host-side model of the firmware for tests
    def __init__(self, clock: Clock, timeout_ms: int = 1000) -> None
    relay_on: bool; latched: bool; armed: bool; reason: int
    def feed(self, frame: bytes) -> bytes: ...          # returns the status byte(s)
    def poll(self) -> None: ...                         # apply timeouts at clock.monotonic_ns()
class FakeHeartbeatLink(HeartbeatLink): def __init__(self, mcu: FakeMcu) -> None
```

### F5. Safety layer — `src/so101_rl/safety/`

```python
# safety/types.py
class TripCause(IntFlag):
    NONE = 0; NAN = 1; RANGE = 2; TRACKING = 4; JOINT_LIMIT = 8; BOX = 16; THERMAL = 32
    PROTECTION_BIT = 64; STUCK = 128; STALE_CMD = 256; BUS_FAIL = 512; HEARTBEAT_FAIL = 1024
    OOD = 2048; USER_STOP = 4096; CALIB_MISMATCH = 8192; DEADMAN = 16384; BOX_PRESENT = 32768

class SmState(IntEnum): BOOT = 0; CHECK = 1; IDLE_TORQUE_OFF = 2; RUN = 3; HOLD = 4; COOLDOWN = 5; SAFE_STOPPING = 6; TORQUE_OFF = 7; LATCHED = 8

class ResumePolicy(Enum): AUTO = "auto"; FRESH_PROCESS = "fresh_process_and_checkpoint"; HUMAN_PHYSICAL_RESET = "human_physical_reset"

RESUME_BY_CAUSE: Final[dict[TripCause, ResumePolicy]] = {
    TripCause.THERMAL: ResumePolicy.AUTO,              # ≤3/h, then HUMAN
    TripCause.STALE_CMD: ResumePolicy.AUTO,            # HOLD→RUN on fresh seq; after SAFE_STOPPING → FRESH_PROCESS
    TripCause.NAN: ResumePolicy.FRESH_PROCESS, TripCause.STUCK: ResumePolicy.FRESH_PROCESS, TripCause.OOD: ResumePolicy.FRESH_PROCESS,
    TripCause.RANGE: ResumePolicy.FRESH_PROCESS, TripCause.BOX_PRESENT: ResumePolicy.FRESH_PROCESS,
    TripCause.DEADMAN: ResumePolicy.HUMAN_PHYSICAL_RESET, TripCause.HEARTBEAT_FAIL: ResumePolicy.HUMAN_PHYSICAL_RESET,
    TripCause.BUS_FAIL: ResumePolicy.HUMAN_PHYSICAL_RESET, TripCause.PROTECTION_BIT: ResumePolicy.HUMAN_PHYSICAL_RESET,
    TripCause.CALIB_MISMATCH: ResumePolicy.HUMAN_PHYSICAL_RESET, TripCause.USER_STOP: ResumePolicy.HUMAN_PHYSICAL_RESET,
}
# TRACKING, JOINT_LIMIT, BOX are *clamps* (reported in clamped_mask / trips), not state-machine trips.

@dataclass
class Verdict:                      # preallocated, filled by SafetyPipeline.apply
    target_ticks: np.ndarray        # int32[6], what will be written this tick
    goal_ticks: np.ndarray          # int32[6], interpolation end point (anti-windup: pulled back with the target)
    applied_delta_ticks: np.ndarray # float32[6]; goal - previous goal after clamps (the value that is LOGGED)
    clamped_mask: np.ndarray        # bool[6]
    trips: TripCause

# safety/clamp.py (pure numpy, no allocation, all take `out`)
def sanitize_delta(delta: np.ndarray, max_delta: np.ndarray, out: np.ndarray) -> TripCause      # NaN/inf → NAN + out=0; |Δ|>max → clip
def clamp_joint_limits(target: np.ndarray, limits: JointLimits, out: np.ndarray, mask: np.ndarray) -> None
def clamp_velocity(target: np.ndarray, prev_target: np.ndarray, vmax: np.ndarray, out: np.ndarray, mask: np.ndarray) -> None
def clamp_tracking(target: np.ndarray, present: np.ndarray, E: np.ndarray, out: np.ndarray, mask: np.ndarray) -> None
def check_range(goal: np.ndarray) -> bool               # dtype integer AND 0 ≤ goal ≤ 4095 for all; called immediately before write_goal

# safety/fk.py (pure numpy, chain from config/so101_kinematics.yaml — no placo, L7)
class So101FK:
    def __init__(self, chain: KinematicChain) -> None
    def gripper_xyz(self, q_rad: np.ndarray, out: np.ndarray) -> np.ndarray      # base frame, metres, 5 arm joints used
    def in_box(self, xyz: np.ndarray, box: WorkspaceBox) -> bool

# safety/thermal.py
class ThermalMonitor:
    def __init__(self, cfg: ThermalConfig, clock: Clock) -> None
    def update(self, temp_c: np.ndarray, status: np.ndarray, block_ok: bool) -> TripCause   # THERMAL at ≥ trip_c or stale temp; PROTECTION_BIT on any status bit
    def may_resume(self) -> bool                             # all temps ≤ resume_c for resume_dwell_s
    def trips_last_hour(self) -> int
    hottest_c: float; max_seen_c: np.ndarray                 # per joint, for the soak report

# safety/stuck.py  (port of robot_data_runner SafetyMonitor)
class StuckActionMonitor:
    def __init__(self, repeat_ticks: int = 90, epsilon_ticks: float = 0.5) -> None   # 3 s of identical non-zero deltas while not moving
    def update(self, applied_delta: np.ndarray, moving: np.ndarray) -> TripCause

# safety/safe_state.py
class SafeStateSequencer:               # RAMP_HOME → REST → TORQUE_OFF, bounded by cfg.bound_s; falls through to torque-off in place
    def __init__(self, cfg: SafeStateConfig, clock: Clock) -> None
    def start(self, present_ticks: np.ndarray) -> None
    def next(self, present_ticks: np.ndarray, target_out: np.ndarray) -> SafeStep    # SafeStep(write_target: bool, torque_off_ids: tuple[int, ...], done: bool, timed_out: bool)

# safety/state_machine.py
@dataclass
class SmInputs: block_ok: bool; cmd_fresh: bool; cmd_age_ns: int; consecutive_misses: int; trips: TripCause; hb_ok: bool; mcu: McuStatus | None; stop_requested: bool; may_resume_thermal: bool
@dataclass
class SmOutputs: state: SmState; use_policy: bool; run_safe_sequence: bool; torque_off_now: bool; resync_target: bool
class SafetyStateMachine:
    def __init__(self, cfg: So101Config, clock: Clock) -> None
    def step(self, inp: SmInputs, out: SmOutputs) -> SmOutputs
    state: SmState; fault_bits: TripCause

# safety/calib_check.py
@dataclass
class CalibReport: ok: bool; mismatches: dict[str, tuple[str, int, int]]     # motor → (field, file, register)
class CalibrationMismatch(RuntimeError): ...
def check_calibration(bus: Bus, expected: dict[str, MotorCalibration]) -> CalibReport
def require_calibration(bus: Bus, expected: dict[str, MotorCalibration]) -> None    # raises CalibrationMismatch; called before ANY torque enable

# safety/reenergize.py  — the frozen re-energise order (plan Phase 0 "On every re-energise")
def reenergize(bus: Bus, cfg: So101Config, calibration: dict[str, MotorCalibration], clock: Clock, block: BlockRead, *, timeout_s: float = 5.0) -> None:
    """1. wait until all 6 ids answer (ping_all)  2. require_calibration  3. write_registers(cfg.registers)
       4. read_block → write_goal(present) → read back Goal_Position == present for all 6  5. set_torque(True)  6. ramped home (via SafeStateSequencer RAMP_HOME)."""

# safety/pipeline.py — order fixed by ARCHITECTURE §6.3
class SafetyPipeline:
    def __init__(self, cfg: So101Config, fk: So101FK, thermal: ThermalMonitor, stuck: StuckActionMonitor) -> None
    def new_command(self, delta_ticks: np.ndarray, v: Verdict) -> TripCause          # sanitize_delta → goal = goal + Δ → clamp_joint_limits(goal)
    def apply(self, block: BlockRead, ticks_remaining: int, v: Verdict) -> Verdict     # interpolate → joint limits → velocity → tracking (anti-windup) → box via FK of target (BOX: revert to prev target; BOX_PRESENT: trip) → thermal → protection bits → stuck → check_range
```

### F6. Servo core and process — `src/so101_rl/servo/`

```python
# servo/timing.py
class Clock(Protocol):
    def monotonic_ns(self) -> int: ...
    def sleep_until(self, t_ns: int) -> None: ...
class RealClock(Clock): ...            # sleep(remaining-200µs) then spin
class FakeClock(Clock): def advance(self, ns: int) -> None    # sleep_until advances immediately
class TickStats: p50_ms, p99_ms, p999_ms, max_ms, overruns: int, samples: int; def record(self, jitter_ns: int, duration_ns: int) -> None

# servo/core.py
@dataclass
class TickResult: ok: bool; state: SmState; trips: TripCause; wrote: bool; hb_sent: bool; overrun_ns: int
class ServoCore:
    def __init__(self, cfg: So101Config, bus: Bus, ipc: ServoIpc, heartbeat: HeartbeatLink | None, pipeline: SafetyPipeline, sm: SafetyStateMachine, safe: SafeStateSequencer, clock: Clock, tick_log: RingWriter | None = None) -> None
    def start(self) -> None                      # reenergize(); state → RUN only after it completes
    def tick(self) -> TickResult                 # read_block → ipc.read_cmd (dedupe by seq) → sm.step → pipeline.apply → check_range → bus.write_goal → heartbeat.send → ipc.write_state
    def shutdown(self, reason: TripCause) -> None   # runs SafeStateSequencer to completion (bounded), then torque off, then disconnect. SIGINT/SIGTERM/exceptions all route here.

# servo/process.py
def run(cfg: So101Config, bus_factory: Callable[[], Bus], hb_factory: Callable[[], HeartbeatLink | None], *, require_rt: bool, max_ticks: int | None = None) -> int
#   sets SCHED_FIFO(50) + sched_setaffinity + mlockall (best effort unless require_rt), gc.collect/freeze/disable, absolute-deadline loop with resync-on-large-overrun, returns exit code
# servo/main.py  — CLI: so101-servo --backend {feetech,fake,mujoco} --port --heartbeat-port --shm-name --require-rt --max-ticks
# servo/soak.py  — soak harness: runs the servo with a random-policy client for --minutes, writes runs/soak-<ts>/report.json
@dataclass
class SoakReport: rt_active: bool; kernel: str; minutes: float; tick_stats: TickStats; hb_max_gap_ms: float; hb_gaps_over_200ms: int; thermal_trips: int; calib_mismatches: int; max_temp_c: dict[str, int]; rss_growth_mb: float; gc_collections: int; unexpected_safe_states: int
def check_soak_report(path: Path, *, strict: bool) -> tuple[bool, list[str]]    # gate: p999 < 5 ms, hb_gaps_over_200ms == 0, thermal_trips == 0, calib_mismatches == 0; strict=False (non-RT host) → advisory
```

### F7. Goal-conditioned Gym environment — `src/so101_rl/env/`

```python
# env/backend.py
class EnvBackend(Protocol):
    def reset(self, q0_ticks: np.ndarray | None) -> None: ...          # sim: set qpos + keyframe; real: ramped move via HOLD→RUN, waits until |target-present| small
    def send_delta(self, delta_ticks: np.ndarray, state_seq: int, episode: int) -> int: ...   # returns cmd seq
    def read_state(self, out: np.ndarray) -> bool: ...                  # STATE_DTYPE record
    def wait_policy_tick(self) -> None: ...                             # 100 ms cadence; sim advances 3 servo ticks
    def fk(self) -> So101FK: ...
class ShmBackend(EnvBackend): def __init__(self, cfg: So101Config, clock: Clock = RealClock()) -> None       # attaches ServoShm
class SimBackend(EnvBackend): def __init__(self, cfg: So101Config, scene_xml: Path, *, action_delay_ticks: int = 0, seed: int = 0) -> None
#   runs ServoCore(bus=MujocoBus, ipc=LocalIpc, heartbeat=FakeHeartbeatLink, clock=FakeClock) in-process: 3 core.tick() per policy step

# env/goal_env.py
ObsDict = dict[str, np.ndarray]      # keys exactly: "observation", "achieved_goal", "desired_goal"
class GoalSampler(Protocol): def sample(self, rng: np.random.Generator) -> np.ndarray   # xyz base frame, inside box ∩ radius ≤ 0.30 m ∩ camera FOV ∩ above kinematic floor
class So101ReachEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    def __init__(self, backend: EnvBackend, cfg: So101Config, goal_sampler: GoalSampler, *, episode_steps: int = 50, success_radius_m: float = 0.015, success_bonus: float = 1.0, render_mode: str | None = None) -> None
    observation_space = spaces.Dict({
        "observation":   spaces.Box(-np.inf, np.inf, (6 + 6 + 3,), np.float32),   # q_rad[6], dq (ticks/tick, scaled)[6], gripper_xyz[3]
        "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), np.float32),          # gripper_xyz from FK of PRESENT ticks
        "desired_goal":  spaces.Box(-np.inf, np.inf, (3,), np.float32)})
    action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)          # delta_ticks = round(action * cfg.action.max_delta_ticks); the ONLY scaling in the codebase
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[ObsDict, dict]
    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict]
    #   terminated = success; truncated = step limit OR any TripCause in state.fault_bits (trip → info["trip"] set, episode never counts as failure)
    #   info keys: "applied_delta_ticks", "target_ticks", "clamped_mask", "trips", "is_success", "state_seq", "cmd_seq"
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict | list[dict]) -> np.ndarray   # vectorised (SB3 HER); -‖a-g‖ + bonus·[‖a-g‖<r]
    def render(self) -> np.ndarray | None
```

### F8. Episode logging — `src/so101_rl/logging/`

Format: one HDF5 file per episode, written to a temp name and renamed on close (atomic), under `runs/<run_id>/episodes/ep_<000123>.h5`. The stdlib module is also called `logging`; the package uses absolute imports only and never puts `src/so101_rl` itself on `sys.path` (test enforces).

```python
@dataclass
class StepRecord:
    t_mono_ns: int; obs: ObsDict; action_cmd: np.ndarray; applied_delta_ticks: np.ndarray; target_ticks: np.ndarray; present_ticks: np.ndarray
    reward: float; terminated: bool; truncated: bool; trips: int; verifier: int        # verifier: 0 UNKNOWN, 1 SUCCESS, 2 FAILURE
    frame_small: np.ndarray | None = None      # uint8 [96,96,3]; world-model training resolution
    frame_full: np.ndarray | None = None       # uint8 [480,640,3]; kept only per policy below

class EpisodeLogger:
    def __init__(self, root: Path, run_id: str, *, keep_full_frames: Literal["never", "failures", "always"] = "failures", retention_days: int = 14, config_dump: dict | None = None) -> None
    def start_episode(self, meta: dict) -> int
    def log_step(self, rec: StepRecord) -> None
    def end_episode(self, outcome: Literal["success", "failure", "truncated", "trip"]) -> Path

@dataclass
class ArtifactReport: ok: bool; path: Path; n_steps: int; errors: list[str]
def check_episode(path: Path, cfg: So101Config | None = None) -> ArtifactReport   # lengths equal, no NaN, dtypes, |applied_delta| ≤ max_delta, 0 ≤ target ≤ 4095, frame counts match, file not truncated, attrs present
def check_run(root: Path) -> list[ArtifactReport]
def prune(root: Path, retention_days: int, *, dry_run: bool = True) -> list[Path]
def disk_budget_gb(root: Path) -> float

class RingWriter:                    # preallocated binary ring for the servo tick log; flushed by a low-priority thread
    def __init__(self, path: Path, dtype: np.dtype, capacity: int = 1_000_000) -> None
    def push(self, rec: np.ndarray) -> None          # no allocation
    def flush(self) -> int; def close(self) -> None
```

### F9. Perception (Stage A) — `src/so101_rl/perception/`

```python
class Intrinsics(NamedTuple): fx: float; fy: float; cx: float; cy: float; width: int; height: int; depth_scale: float
class Frame(NamedTuple): t_mono_ns: int; ok: bool
class Camera(Protocol):
    def start(self) -> Intrinsics: ...
    def read(self, rgb_out: np.ndarray, depth_m_out: np.ndarray) -> Frame: ...
    def stop(self) -> None: ...
class D435Camera(Camera): def __init__(self, serial: str = "819312071240", width: int = 640, height: int = 480, fps: int = 30, warmup_frames: int = 45) -> None
class FakeCamera(Camera): def __init__(self, rgb: np.ndarray, depth_m: np.ndarray, intr: Intrinsics) -> None

@dataclass
class Detection: u: float; v: float; area_px: int; xyz_cam: np.ndarray; confidence: float
class HsvDepthDieTracker:
    def __init__(self, hsv_lo: np.ndarray, hsv_hi: np.ndarray, intr: Intrinsics, *, min_area_px: int = 40, max_area_px: int = 4000, depth_window_m: float = 0.02) -> None
    def detect(self, rgb: np.ndarray, depth_m: np.ndarray) -> Detection | None      # None = not visible → verifier UNKNOWN, never SUCCESS (L11)

@dataclass(frozen=True)
class HandEye: T_base_cam: np.ndarray; residual_mm: float; n_poses: int; date: str; camera_serial: str
def load_hand_eye(path: Path = Path("calib/hand_eye.yaml")) -> HandEye        # refuses residual_mm > 3.0 or n_poses < 10
def fit_hand_eye(p_base: np.ndarray, p_cam: np.ndarray) -> tuple[np.ndarray, float]   # Umeyama rigid fit; returns (T 4x4, rms residual mm)
def pixel_to_cam(u: float, v: float, z_m: float, intr: Intrinsics) -> np.ndarray
def cam_to_base(xyz_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray

@dataclass
class FiducialReport: ok: bool; reproj_err_px: float; found: bool
class FiducialCheck:
    def __init__(self, marker_id: int, expected_corners_px: np.ndarray, *, dictionary: int = cv2.aruco.DICT_4X4_50, tol_px: float = 2.0) -> None
    def check(self, rgb: np.ndarray) -> FiducialReport                          # not found → ok=False (halts training, plan §Phase 0 camera-moved gate)

OBJECT_DTYPE = np.dtype([("label", "S16"), ("xyz_base", "<f4", (3,)), ("confidence", "<f4"), ("visible", "<u1"), ("t_mono_ns", "<i8")], align=True)
OBJECT_TABLE_DTYPE = np.dtype([("seq", "<u8"), ("t_mono_ns", "<i8"), ("n", "<u4"), ("objects", OBJECT_DTYPE, (8,)), ("fiducial_ok", "<u1"), ("crc32", "<u4")], align=True)
class ObjectTableShm:   # SeqlockBlock from servo/shm.py; perception writes, policy reads
    @classmethod
    def create(cls, name: str) -> "ObjectTableShm"; @classmethod
    def attach(cls, name: str) -> "ObjectTableShm"
    def write(self, rec: np.ndarray) -> None; def read(self, out: np.ndarray) -> bool
```

### F10. Sim — `src/so101_rl/sim/`

```python
def load_model(scene_xml: Path = ASSETS / "so101" / "so101_rl_scene.xml") -> tuple[mujoco.MjModel, mujoco.MjData]
class MujocoBus(Bus):      # F2; qpos ↔ ticks through the SAME calibration dict as the real arm
    def __init__(self, model, data, calibration: dict[str, MotorCalibration], cfg: So101Config, *, action_delay_ticks: int = 0, seed: int = 0) -> None
    def step_ticks(self, n: int = 1) -> None        # advances physics by n × (1/servo_hz); called by SimBackend between core.tick() calls
    written_goals: list[np.ndarray]                 # intercept for Test 1 / Test 3
def gripper_percent_to_rad(p: float) -> float; def gripper_rad_to_percent(q: float) -> float; def gripper_ticks_to_percent(t: int, calib) -> float
def rms_joint_error_pct(real_h5: Path, model, data, calibration) -> dict[str, float]    # sysid milestone check only (no fitting in Phase 0)
```

### F11. RL / WM scaffolding — `src/so101_rl/rl/`, `src/so101_rl/wm/`

```python
# rl/her.py — used by TD-MPC2 Buffer._prepare_batch hook (Phase 3) and unit-tested now on the toy env
def her_relabel(td: TensorDict, *, goal_slice: slice, achieved_slice: slice, reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], k: int = 4, strategy: Literal["future", "final"] = "future", generator: torch.Generator | None = None) -> TensorDict
# rl/tdmpc2_env.py — adapter from a gymnasium Dict goal env to the vendored TD-MPC2 env contract (state obs with goal appended)
class TdMpc2GoalEnv: obs_shape: dict[str, tuple[int, ...]]; action_dim: int; max_episode_steps: int; def reset(self) -> torch.Tensor; def step(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, dict]; def rand_act(self) -> torch.Tensor
def make_tdmpc2_cfg(overrides: dict) -> OmegaConf         # loads _vendor/tdmpc2/config.yaml, applies overrides, forces device auto-detect
# rl/toy_env.py (U0) — 2-D point-mass goal env, zero deps beyond numpy/gymnasium; used by the pixi solver spike and HER tests
class ToyReachEnv(gymnasium.Env): same Dict obs contract as F7
# rl/smoke_tdmpc2.py (U0) — `python -m so101_rl.rl.smoke_tdmpc2 --steps 1000 --device {auto,cpu,cuda}` → exit 0 and writes runs/smoke-tdmpc2/<ts>/{metrics.csv, model.pt}
# rl/train_sac_her.py — `train-reach-sac`: SB3 SAC + HerReplayBuffer on So101ReachEnv(SimBackend); --total-steps; writes runs/sac-her/<ts>/
# rl/train_tdmpc2.py  — `train-reach-tdmpc2`: vendored OnlineTrainer on TdMpc2GoalEnv(So101ReachEnv(SimBackend)); --steps
# wm/state_wm.py
class StateDynamicsModel(torch.nn.Module):
    def __init__(self, obs_dim: int = 15, act_dim: int = 6, hidden: int = 256) -> None
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor          # predicts next obs (delta parametrisation inside)
def rollout(model, obs0: torch.Tensor, acts: torch.Tensor) -> torch.Tensor           # [H, obs_dim]
def probe_gripper_error_m(model, episodes: EpisodeDataset, fk: So101FK, horizon_steps: int = 10) -> float   # Phase 2 milestone metric (1 s = 10 steps)
# wm/dataset.py
class EpisodeDataset(torch.utils.data.Dataset): def __init__(self, run_root: Path, *, require_ok: bool = True) -> None   # runs check_episode on every file first (L12)
# wm/train_state_wm.py — `train-wm-state`: --run-root --steps; writes runs/wm-state/<ts>/
```

## Tasks

Legend: **SW** = software, executable by an agent now, tests run without hardware. **HW** = hardware/human; has a checklist and names the software it needs. `Est.` is executor wall-clock (SW) or human time (HW). `Specialist` is who the executor consults. Acceptance commands assume the `cpu` environment until H1 lands: `pixi run -e cpu <task>`.

Tests marked `@pytest.mark.hardware` are skipped unless `SO101_HW=1`; `@pytest.mark.gpu` unless CUDA is available. `pixi run test` selects `-m "not hardware and not gpu"`.

---

### Unit U0 — Bootstrap and interface freeze (wave 0, sequential)

#### Task 0.1 (SW): pixi solver spike — `pyproject.toml` + `[tool.pixi]` + lock
- **Files:** `pyproject.toml`, `pixi.lock`, `src/so101_rl/__init__.py`, `src/so101_rl/py.typed`, `README.md`
- **Specialist:** pixi-workspace-maintainer (worker) for lock health; lerobot-specialist for the torch-mapping check
- **Dependencies:** None (git already initialised — do **not** `git init`)
- **Steps:**
  1. Write `pyproject.toml`: `[project] name="so101-rl", requires-python=">=3.12,<3.13", dependencies=[]` (runtime deps are declared in pixi; pip install support deferred to Stage B/C); `[build-system] hatchling`; `[tool.hatch.build.targets.wheel] packages=["src/so101_rl"]`; `[tool.pytest.ini_options] markers = ["hardware", "gpu", "slow"]`, `addopts="-ra --timeout=120"`, `testpaths=["tests"]`; `[tool.ruff] line-length=110, target-version="py312"`.
  2. `[tool.pixi.workspace] channels=["conda-forge"], platforms=["linux-64"]`. **Never** add the legacy `pytorch` channel.
  3. `[tool.pixi.dependencies]` (core): `python=3.12.*`, `numpy=2.2.*`, `pytorch>=2.7,<2.12` (variant chosen by the gpu/cpu features), `torchvision>=0.22,<0.27`, `torchcodec`, `pyserial>=3.5`, `pyyaml`, `h5py`, `gymnasium>=1.1.1,<2`, `psutil`, `pytest`, `pytest-timeout`, `hypothesis`, `ruff`. `[tool.pixi.pypi-dependencies]`: `so101-rl = { path=".", editable=true }`, `lerobot = { version="==0.6.1", extras=["feetech","grpcio-dep"] }`.
  4. Features: `sim` → `mujoco-python>=3.3` (conda-forge); `perception` → `opencv>=4.8` (conda-forge) + pypi `pyrealsense2>=2.57`; `train` → pypi `torchrl`, `tensordict`, `stable-baselines3>=2.7`, `hydra-core`, `omegaconf`, `termcolor`, `pandas`; `gpu` → `pytorch-gpu`, `cuda-version=12.8.*` (**re-pin to the driver's CUDA after H1**); `cpu` → `pytorch-cpu`. Environments: `default=["sim","perception","train","gpu"]`, `cpu=["sim","perception","train","cpu"]`, `robot=["perception"]` (small env for the servo host process — no torch beyond lerobot's own requirement).
  5. Tasks: `test`, `test-hw`, `lint`, `ci` (= lint + test), `smoke-tdmpc2`, `fetch-assets`, `extract-kinematics`, `servo`, `servo-fake`, `soak`, `calib-check`, `hand-eye`, `random-policy-sim`, `random-policy-real`, `train-reach-sac`, `train-reach-tdmpc2`, `train-wm-state`, `check-milestone`. Commands may point at modules that later units create; `pixi task list` must still succeed.
  6. `pixi install -e cpu` (and `-e default` if the driver is up). Resolve conflicts by adjusting **torchrl/tensordict** pins to the solved torch minor (pairing UNVERIFIED — check the torchrl README table; 2.7↔0.9, 2.8↔0.10, 2.9↔0.11, 2.10↔0.12, 2.11↔0.13 expected). Record the solved triple in `README.md` "Environment".
  7. Double-torch check: `pixi list -e cpu | grep -i -E "^(torch|pytorch)"` shows **one** `pytorch` (conda) and **no** PyPI `torch`. If PyPI torch appears, stop and report (fallback: `[tool.pixi.pypi-options]` override or `--no-deps` lerobot with hand-listed runtime deps).
- **Tests:** `tests/test_env_sanity.py::test_single_torch` (parses `pixi list --json`), `::test_import_lerobot_motors_no_torch_init` (importing `lerobot.motors.feetech` must not initialise CUDA), `::test_numpy_version` (2.2.x), `::test_python_version` (3.12).
- **Acceptance:** `pixi install -e cpu && pixi run -e cpu python -c "import lerobot.motors.feetech, torch, numpy; print(torch.__version__, numpy.__version__)"` → prints `2.x.y 2.2.z`, exit 0. `pixi run -e cpu pytest tests/test_env_sanity.py -q` → `4 passed`.
- **Est.:** 40 min (solver iterations dominate).

#### Task 0.2 (SW): Single-source configuration module
- **Files:** `src/so101_rl/config/__init__.py`, `config/schema.py`, `config/so101.yaml`, `config/loader.py`, `tests/test_config.py`, `tests/test_config_single_source.py`
- **Specialist:** python-best-practices
- **Dependencies:** 0.1
- **Steps:** implement F1 exactly; `load_config()` intersects `so101.yaml` joint limits with the calibration file's `range_min/max` (narrowest wins) and refuses to load if the calibration has a motor id ≠ 1–6 or a missing joint; `dump_env() -> dict` returns config + `os.environ` subset (`SO101_*`, `REQUIRE_RT`) for launch-diffing (L4); `rest_ticks`/`home_ticks` default from the MJCF keyframes converted via calibration (placeholder until H5 measures them — the yaml has a `verified_on_hardware: false` flag that `check_milestone` reads).
- **Tests:** frozen dataclasses reject mutation; limits intersection; NaN in yaml → `ValueError`; `test_config_single_source` greps `src/so101_rl` (excluding `config/`, `_vendor/`) for the literal values of `max_delta_ticks`, `tracking_error_ticks`, `trip_c`, `resume_c`, `stale_cmd_ms`, `bound_s` and fails on any hit; deliberately broken input (a temp copy of the yaml with `trip_c: 75` > firmware 70 °C) → loader raises.
- **Acceptance:** `pixi run -e cpu pytest tests/test_config.py tests/test_config_single_source.py -q` → all pass (≥ 10 tests).
- **Est.:** 30 min.

#### Task 0.3 (SW): Fetch official SO-101 MJCF assets (pinned) and extract the kinematic chain
- **Files:** `scripts/fetch_assets.py`, `scripts/extract_kinematics.py`, `assets/so101/upstream/` (fetched files, committed), `assets/so101/LICENSE` (Apache-2.0 copy), `assets/so101/NOTICE`, `assets/so101/UPSTREAM.json`, `src/so101_rl/config/so101_kinematics.yaml`, `tests/test_assets.py`
- **Specialist:** kinematics-specialist
- **Dependencies:** 0.1
- **Steps:**
  1. `fetch_assets.py --sha <commit>` downloads `Simulation/SO101/{so101_new_calib.xml, scene.xml, joints_properties.xml, assets/*.stl}` and the repo `LICENSE` from `raw.githubusercontent.com/TheRobotStudio/SO-ARM100/<sha>/…` into `assets/so101/upstream/`; writes `UPSTREAM.json` `{repo, sha, fetched, files: {path: sha256}}`; `NOTICE` states "SO-101 model © TheRobotStudio, Apache-2.0; modified: none (overrides live in so101_rl_scene.xml)". Upstream files are never edited.
  2. `extract_kinematics.py` loads `upstream/so101_new_calib.xml` with MuJoCo, walks body→joint for the 5 arm joints + gripper joint, and writes `so101_kinematics.yaml`: per joint `{name, mjcf_joint, parent_body_pos, parent_body_quat, axis, range_rad}` and the `gripper_tip` site offset (added if the MJCF lacks one: 0.0 m along the fixed jaw, documented). This yaml is the pure-numpy FK's only input (F5 `So101FK`).
  3. If total STL size > 25 MB, switch to fetch-on-demand (`pixi run fetch-assets`, gitignore `upstream/`) and say so in `README.md`.
- **Tests:** `test_assets.py`: sha256 of every fetched file matches `UPSTREAM.json`; MJCF loads; 6 joints found with names matching `JOINT_NAMES` order mapping in the yaml; LICENSE present and contains "Apache License".
- **Acceptance:** `pixi run -e cpu fetch-assets && pixi run -e cpu extract-kinematics && pixi run -e cpu pytest tests/test_assets.py -q` → pass.
- **Est.:** 30 min.

#### Task 0.4 (SW): Vendor TD-MPC2 (MIT) with reproducible patches and the state smoke train
- **Files:** `scripts/vendor_tdmpc2.py`, `src/so101_rl/_vendor/__init__.py`, `src/so101_rl/_vendor/tdmpc2/**` (copied `tdmpc2/tdmpc2/*` minus `envs/`), `src/so101_rl/_vendor/tdmpc2/LICENSE`, `src/so101_rl/_vendor/tdmpc2/PATCHES.md`, `src/so101_rl/_vendor/tdmpc2/UPSTREAM.json`, `src/so101_rl/rl/__init__.py`, `src/so101_rl/rl/toy_env.py`, `src/so101_rl/rl/smoke_tdmpc2.py`, `tests/test_vendor_tdmpc2.py`, `tests/test_toy_env.py`
- **Specialist:** lerobot-specialist (RL infra), python-best-practices
- **Dependencies:** 0.1
- **Steps:**
  1. `vendor_tdmpc2.py --sha <commit>` clones/downloads at the pinned sha, copies `tdmpc2/{tdmpc2.py, train.py, evaluate.py, config.yaml, common/, trainer/}` and `LICENSE`, then applies patches recorded in `PATCHES.md`: (a) rewrite `from common.` / `from trainer.` / `import common` to package-relative imports; (b) `common/buffer.py`: device from `cfg.device` (fallback `cuda:0` if available else `cpu`), skip `torch.cuda.mem_get_info()` on CPU (storage on CPU); (c) `tdmpc2.py`/`common/world_model.py`: `torch.compile` only when `cfg.compile` is true (default false); (d) `common/logger.py`: wandb import guarded, CSV always written; (e) `train.py`: `make_env` import replaced by a `cfg.env_factory` dotted path so no DMControl/ManiSkill import happens. Re-running the script must reproduce the tree byte-for-byte (`--check`).
  2. `rl/toy_env.py`: `ToyReachEnv` (2-D point mass, `observation`=[pos(2), vel(2)], goal-conditioned, 50 steps, same Dict contract as F7, `compute_reward` vectorised).
  3. `rl/smoke_tdmpc2.py`: builds `TdMpc2GoalEnv`-style flat state (obs ‖ goal, 6-D) around `ToyReachEnv` (a local minimal adapter; the full adapter lands in U9), `make_tdmpc2_cfg({"task": "toy-reach", "steps": N, "seed_steps": 200, "buffer_size": 10_000, "batch_size": 64, "horizon": 3, "model_size": 1, "eval_freq": 10**9, "save_video": False, "compile": False, "device": ...})`, runs `OnlineTrainer`, writes `runs/smoke-tdmpc2/<ts>/{metrics.csv, model.pt}`, exits non-zero if `metrics.csv` has < 3 rows or `model.pt` is missing (L12: check artifacts, not rc).
- **Tests:** `test_vendor_tdmpc2.py`: `import so101_rl._vendor.tdmpc2.tdmpc2` succeeds without CUDA; LICENSE present (MIT); `PATCHES.md` lists every diff hunk (`vendor_tdmpc2.py --check` passes); `@pytest.mark.slow` 200-step smoke on CPU finishes < 120 s and writes both artifacts. `test_toy_env.py`: `gymnasium.utils.env_checker.check_env` passes; a fixed policy reaches the goal.
- **Acceptance:** `pixi run -e cpu smoke-tdmpc2 --steps 1000 --device cpu` → exit 0, `runs/smoke-tdmpc2/<ts>/metrics.csv` has ≥ 3 rows and `model.pt` exists; < 10 min on CPU. `pixi run -e cpu pytest tests/test_vendor_tdmpc2.py tests/test_toy_env.py -q -m "not slow"` → pass. **Spike verdict recorded in `README.md`:** python 3.12 + numpy 2.2 + torch 2.x + torchrl/tensordict versions that worked.
- **Est.:** 45 min.

#### Task 0.5 (SW): Frozen contract files, test scaffolding, IPC primitive
- **Files:** `src/so101_rl/servo/__init__.py`, `servo/bus.py` (F2 Protocol + `BlockRead`), `servo/timing.py` (F6 `Clock`, `RealClock`, `FakeClock`, `TickStats`), `servo/shm.py` (F3 complete implementation), `src/so101_rl/safety/__init__.py`, `safety/types.py` (F5 enums/`Verdict`), `src/so101_rl/{env,perception,sim,wm,logging}/__init__.py` (empty), `tests/conftest.py`, `tests/test_shm.py`, `tests/test_timing.py`, `tests/test_types.py`
- **Specialist:** real-time-control-specialist
- **Dependencies:** 0.2
- **Steps:** implement F3 with `zlib.crc32` over a `memoryview` of the payload (no allocation per write beyond ints); `attach()` calls `multiprocessing.resource_tracker.unregister(shm._name, "shared_memory")` on 3.12; `conftest.py` registers markers, skips `hardware` unless `SO101_HW=1`, `gpu` unless `torch.cuda.is_available()`, provides `fake_clock`, `cfg` (loads `so101.yaml` + the real calibration JSON copied into `tests/data/so101_follower.calib.json`), and asserts `src/so101_rl` is not on `sys.path`.
- **Tests:** `test_shm.py`: dtype itemsizes are 64/232; header magic/version mismatch → `RuntimeError` at attach; **torn-read stress**: a writer *process* writes records whose payload is filled with `seq`, a reader hammers for 2 s and never accepts a record whose payload ≠ its seq or whose CRC fails; reader sees a repeated seq as "no new"; `FakeClock.sleep_until` advances exactly; `TickStats` percentiles on a known distribution.
- **Acceptance:** `pixi run -e cpu pytest tests/test_shm.py tests/test_timing.py tests/test_types.py -q` → pass.
- **Est.:** 40 min.

#### Task 0.6 (SW): `docs/ARCHITECTURE.md` sync after wave 0 (shared write)
- **Files:** `docs/ARCHITECTURE.md` (§4: `config/so101_kinematics.yaml`, `scripts/`, `assets/so101/upstream/`; §5: link to this plan's F-sections; §8: features `gpu`/`cpu`/`robot` and the task list; §10 changelog 0.2)
- **Dependencies:** 0.1–0.5
- **Acceptance:** `grep -c "so101_kinematics.yaml" docs/ARCHITECTURE.md` ≥ 1; version table has a 0.2 row.
- **Est.:** 10 min.

**Commit after 0.6:** `Bootstrap pixi package, vendored TD-MPC2, frozen interfaces`.

---

### Unit U1 — Servo backends and heartbeat (wave 1, parallel with U2, U4, U5, U6, U7)

#### Task 1.1 (SW): `FakeBus` with plant, latency and fault injection
- **Files:** `src/so101_rl/servo/fake_bus.py`, `tests/test_fake_bus.py`
- **Specialist:** real-time-control-specialist
- **Dependencies:** U0
- **Steps:** `FirstOrderPlant(tau_ticks_per_tick=…, max_speed)` moves `present` toward the last written goal per `clock` advance; obstacle model `block(joint, at_ticks)` makes present stop (for Test 2); `ThermalModel` heats with load² and cools with a time constant; `FaultInjector` with `fail_next_read(n)`, `fail_next_write(n)`, `set_status_bits(id, bits)`, `latency_ms(dist)`, `drop_motor(id)`; records `written_goals` (copies) and `torque_events`. `read_block` returns `False` and leaves arrays untouched on injected failure (never zeros).
- **Tests:** plant converges; injected read failure → `ok=False`, arrays unchanged; dropped motor → `ping_all` shows it; blocked joint never passes the obstacle; `read_calibration_regs` round-trips the calibration passed at construction.
- **Acceptance:** `pixi run -e cpu pytest tests/test_fake_bus.py -q` → pass.
- **Est.:** 35 min. `parallelizable-via-worktree: true`

#### Task 1.2 (SW): `FeetechBus` over lerobot 0.6.1 with one 11-byte block read
- **Files:** `src/so101_rl/servo/feetech_bus.py`, `tests/test_feetech_bus.py`
- **Specialist:** lerobot-specialist
- **Dependencies:** U0
- **Steps:** construct `FeetechMotorsBus(port, motors={name: Motor(id, "sts3215", MotorNormMode.RANGE_M100_100)}, calibration)`; on `connect()`: `bus.connect(handshake=True)`, build `scservo_sdk.GroupSyncRead(bus.port_handler, bus.packet_handler, 56, 11)` with all ids; `read_block` = `txRxPacket()` + `getData(id, addr, len)` per field, sign-decode bits 15/10 with `lerobot.motors.encoding_utils.decode_sign_magnitude`, set `ok=False` on `COMM_*` errors or a missing id; `write_goal` = `bus.sync_write("Goal_Position", {name: int}, normalize=False)`; `write_registers(cfg.registers)` writes every F1 register incl. `Protection_Current`, `Overload_Torque`, `Over_Current_Protection_Time` (plan: never written by LeRobot) and calls `configure_motors(return_delay_time=0, ...)`; `read_calibration_regs` uses `sync_read("Homing_Offset", normalize=False)` (decoded) + `Min/Max_Position_Limit`. Set `disable_torque_on_disconnect` semantics explicitly: our `disconnect()` never touches torque — the safe-state sequence does.
- **Tests:** unit tests with a monkeypatched `scservo_sdk` (a `MockPortHandler` returning canned packets): 11-byte packet decodes to expected ints incl. negative load (bit 10) and a Homing_Offset of −1647 (2048+1647 encoded → −1647 decoded); `Goal_Position` bytes are little-endian 2 bytes; comm error → `ok=False`. `@pytest.mark.hardware` `test_block_read_latency`: 300 reads, p99 < 6 ms, `Return_Delay_Time` reads back 0.
- **Acceptance:** `pixi run -e cpu pytest tests/test_feetech_bus.py -q -m "not hardware"` → pass.
- **Est.:** 40 min. `parallelizable-via-worktree: true`

#### Task 1.3 (SW): Heartbeat link, `FakeMcu`, protocol conformance
- **Files:** `src/so101_rl/servo/heartbeat.py`, `tests/test_heartbeat.py`
- **Specialist:** real-time-control-specialist; hardware-interface-specialist
- **Dependencies:** U0
- **Steps:** implement F4; `SerialHeartbeatLink` uses `serial.Serial(port, 115200, timeout=0, write_timeout=0.002)`, reads the echo with `in_waiting` (non-blocking), a missing/late echo → `send()` returns `False`; `FakeMcu` implements the firmware state machine on `FakeClock` (same invariants as `firmware/watchdog_mcu/PROTOCOL.md`, U7); `FakeHeartbeatLink` routes bytes to `FakeMcu`.
- **Tests:** ARM only from DISARMED; heartbeat with wrong seq → LATCHED reason 3; no heartbeat for 999 ms → still ARMED, 1001 ms → LATCHED reason 1; LATCHED ignores ARM; `reset_button(500 ms)` → DISARMED; write timeout → `send()` False; status byte bit layout; `test_drop` returns ms.
- **Acceptance:** `pixi run -e cpu pytest tests/test_heartbeat.py -q` → pass.
- **Est.:** 30 min. `parallelizable-via-worktree: true`

#### Task 1.4 (SW): U1 ARCHITECTURE sync (shared write, end of unit)
- **Files:** `docs/ARCHITECTURE.md` §5 (Bus protocol 56..66 note), §6 (heartbeat protocol pointer)
- **Est.:** 5 min.

**Commit after 1.4:** `Servo backends: FakeBus, FeetechBus, heartbeat link`.

---

### Unit U2 — Safety layer (wave 1)

#### Task 2.1 (SW): Clamps and range check with property tests
- **Files:** `src/so101_rl/safety/clamp.py`, `tests/test_clamp.py`
- **Specialist:** real-time-control-specialist; testing-specialist
- **Dependencies:** U0
- **Steps:** implement F5 `clamp.py`; anti-windup semantics documented in docstrings; no allocation (all `out=`).
- **Tests:** hypothesis: for random `delta` incl. NaN/±inf/huge, `sanitize_delta` output is finite and |out| ≤ max; `clamp_tracking` output satisfies |out − present| ≤ E; `check_range` rejects 4096, −1, float dtype, NaN; **Test 3 (action units)**: `check_range` is the last call before any write and an out-of-range goal is rejected before serialisation (asserted again end-to-end in 3.5).
- **Acceptance:** `pixi run -e cpu pytest tests/test_clamp.py -q` → pass.
- **Est.:** 30 min. `parallelizable-via-worktree: true`

#### Task 2.2 (SW): Pure-numpy FK from the extracted chain + workspace box
- **Files:** `src/so101_rl/safety/fk.py`, `tests/test_fk.py`
- **Specialist:** kinematics-specialist
- **Dependencies:** U0 (0.3 yaml)
- **Steps:** `KinematicChain.from_yaml(path)`; `So101FK.gripper_xyz` composes 5 rigid transforms + fixed tip offset (preallocated 4×4 buffers); `ticks_to_rad` via calibration `homing_offset` and the yaml's zero convention; `in_box`.
- **Tests:** rest pose gives a plausible xyz (z > table, x within box); FK is deterministic and allocation-free (`tracemalloc` delta 0 over 1000 calls); `in_box` boundaries; the MuJoCo cross-check lives in 3.6 (needs U4).
- **Acceptance:** `pixi run -e cpu pytest tests/test_fk.py -q` → pass.
- **Est.:** 30 min. `parallelizable-via-worktree: true`

#### Task 2.3 (SW): Thermal monitor and stuck-action monitor
- **Files:** `src/so101_rl/safety/thermal.py`, `src/so101_rl/safety/stuck.py`, `tests/test_thermal.py`, `tests/test_stuck.py`
- **Specialist:** real-time-control-specialist
- **Dependencies:** U0
- **Tests (fake clock):** ramp to 59 → no trip, 60 → THERMAL; cool to 51 → no resume, 50 held 29 s → no resume, 30 s → resume; oscillation 59↔61 → exactly one trip until resume; 3 trips in 3600 s → `trips_last_hour()==3` (state machine turns that into HUMAN); temperature 0 or 255 or stale > 1 s → trip; any `status` bit → PROTECTION_BIT. Stuck: identical non-zero delta for 90 ticks with `moving==0` → STUCK; changing delta resets.
- **Acceptance:** `pixi run -e cpu pytest tests/test_thermal.py tests/test_stuck.py -q` → pass.
- **Est.:** 30 min. `parallelizable-via-worktree: true`

#### Task 2.4 (SW): Safe-state sequencer (bounded 5 s) and re-energise order
- **Files:** `src/so101_rl/safety/safe_state.py`, `src/so101_rl/safety/reenergize.py`, `src/so101_rl/safety/calib_check.py`, `tests/test_safe_state.py`, `tests/test_reenergize.py`, `tests/test_calib_check.py`
- **Specialist:** lerobot-specialist (register order), real-time-control-specialist
- **Dependencies:** U0, 1.1 (`FakeBus` for tests — if U1 is not merged yet, test against a minimal in-test stub of `Bus`)
- **Steps:** port `ramped_home` semantics from `arm_motor_writer.py` into `SafeStateSequencer` RAMP_HOME (per-tick step ≤ `ramp_ticks_per_tick`, respects tracking clamp); REST phase moves to `rest_ticks`; TORQUE_OFF disables in `torque_off_order`; on `bound_s` expiry → `timed_out=True`, torque off in place. `reenergize()` follows the F5 docstring order exactly; `check_calibration` compares decoded values.
- **Tests:** sequence completes in < 5 s on a healthy plant; bus failing mid-ramp → falls through to torque-off at exactly `bound_s`; a policy command during SAFE_STOPPING is ignored (checked in 3.2); re-energise order asserted via `FakeBus` call log: `ping_all` → calibration regs → registers → `read_block` → `write_goal(present)` → `read_reg("Goal_Position")×6` → `set_torque(True)` (any other order fails); missing motor for > timeout → `TimeoutError`; **deliberately corrupted calibration** (homing_offset +1) → `CalibrationMismatch`, no `set_torque(True)` ever called; sign-magnitude case: register raw 2048+1647 vs file −1647 → **match** (decoded compare).
- **Acceptance:** `pixi run -e cpu pytest tests/test_safe_state.py tests/test_reenergize.py tests/test_calib_check.py -q` → pass.
- **Est.:** 45 min.

#### Task 2.5 (SW): Safety state machine + pipeline
- **Files:** `src/so101_rl/safety/state_machine.py`, `src/so101_rl/safety/pipeline.py`, `tests/test_state_machine.py`, `tests/test_pipeline.py`
- **Specialist:** real-time-control-specialist
- **Dependencies:** 2.1–2.4
- **Steps:** transitions per specialist §5 and `RESUME_BY_CAUSE`; `resync_target=True` on every entry to RUN; `SafetyPipeline.apply` in the ARCHITECTURE §6.3 order; BOX on the *commanded* target reverts to the previous target (clamp), BOX_PRESENT (present outside the box) is a trip; every clamp sets `clamped_mask`; `applied_delta_ticks` = post-clamp goal change (what gets logged, L4).
- **Tests:** table-driven transitions (RUN→HOLD at 151 ms, not 149; HOLD→RUN on fresh seq; N−1 misses stay HOLD, N → SAFE_STOPPING; THERMAL → COOLDOWN → RUN after resume with `resync_target`; 4th thermal trip in an hour → TORQUE_OFF (human); NaN → SAFE_STOPPING → TORQUE_OFF, no auto-resume; heartbeat fail / MCU latched → LATCHED and nothing leaves LATCHED); pipeline: interpolation reaches the goal in exactly 3 ticks; anti-windup keeps `goal` within E of present when blocked; order verified by injecting a value that would pass a later check but fail an earlier one.
- **Acceptance:** `pixi run -e cpu pytest tests/test_state_machine.py tests/test_pipeline.py -q` → pass (≥ 25 tests).
- **Est.:** 45 min.

#### Task 2.6 (SW): U2 ARCHITECTURE sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` §6 (check order confirmed, BOX vs BOX_PRESENT distinction, `RESUME_BY_CAUSE` table)
- **Est.:** 5 min.

**Commit after 2.6:** `Safety layer: clamps, FK box, thermal, safe-state, calibration check, state machine`.

---

### Unit U4 — MuJoCo sim (wave 1)

#### Task 4.1 (SW): Scene override with primitive jaw collisions, keyframes, die and zone
- **Files:** `assets/so101/so101_rl_scene.xml`, `assets/so101/README.md`, `src/so101_rl/sim/model.py`, `tests/test_sim_model.py`
- **Specialist:** simulation-to-hardware-bridge; kinematics-specialist
- **Dependencies:** U0 (0.3)
- **Steps:** `so101_rl_scene.xml` `<include file="upstream/so101_new_calib.xml"/>` plus: table plane at `TABLE_Z_M=-0.02`; `<default class="jaw_col">` with two box/capsule collision geoms on the fixed jaw and moving jaw (contype/conaffinity set so mesh-mesh contact is disabled for the jaws — plan §6); `<keyframe name="rest">` and `<keyframe name="home">` (joint values from `config/so101.yaml`, in radians via the kinematics yaml zero convention); a 16 mm die body (mass ≈ 4 g), a zone site 60 × 60 mm; an ArUco-sized fiducial site; sites `gripper_tip` and `base`. Do not modify upstream files. `load_model()` resolves paths relative to `assets/`.
- **Tests:** model compiles; keyframes exist; jaw geoms are primitives (`geom_type ∈ {box, capsule}`) and mesh jaw geoms have `contype=0`; stepping 300 steps from `rest` with `ctrl = qpos` moves each joint < 0.5° (rest is gravity-stable **in sim**; H5 verifies on hardware); `actuator` class `sts3215` present for 6 actuators.
- **Acceptance:** `pixi run -e cpu pytest tests/test_sim_model.py -q` → pass.
- **Est.:** 40 min. `parallelizable-via-worktree: true`

#### Task 4.2 (SW): `MujocoBus` (Bus protocol on the sim) + gripper mapping
- **Files:** `src/so101_rl/sim/mujoco_bus.py`, `src/so101_rl/sim/gripper_map.py`, `tests/test_mujoco_bus.py`, `tests/test_gripper_map.py`
- **Specialist:** simulation-to-hardware-bridge
- **Dependencies:** 4.1
- **Steps:** `write_goal(ticks)` → `ctrl = ticks_to_rad(ticks, calibration)` (with optional `action_delay_ticks` FIFO to model the measured 100–150 ms latency later); `read_block`: `position = rad_to_ticks(qpos)` clipped to 0..4095, `velocity` from `qvel`, `load` from `actuator_force` scaled to ±1000, `temperature_c` from a trivial thermal model (default constant 35), `status = 0`, `moving = |qvel| > eps`; `step_ticks(n)` runs `n × round((1/servo_hz)/model.opt.timestep)` `mj_step`s; `set_torque(False)` sets `ctrl` gains to zero via `actuator_gainprm` copy (arm falls under gravity — used by the sag-map sim preview only). Gripper: `0–100 % ↔ rad ↔ ticks` mapping using the calibration `range_min/max` of the gripper motor and the MJCF gripper joint range.
- **Tests:** implements `Bus` (`isinstance` via `runtime_checkable`); goal → present converges within 0.5° in 30 ticks; `written_goals` intercept records raw ints; gripper mapping round-trips (0 % ↔ closed joint range end, 100 % ↔ open) with |error| < 1 tick, and 50 % is the midpoint; out-of-range percent raises.
- **Acceptance:** `pixi run -e cpu pytest tests/test_mujoco_bus.py tests/test_gripper_map.py -q` → pass.
- **Est.:** 40 min.

#### Task 4.3 (SW): Sysid *check* script (no fitting) + U4 ARCHITECTURE sync
- **Files:** `src/so101_rl/sim/compare_real.py`, `tests/test_compare_real.py`, `docs/ARCHITECTURE.md` (§4 sim contents; §7 note "sysid fitting is the parallel track")
- **Dependencies:** 4.2, F8 format (read-only dependency on the HDF5 layout — implement against the F8 spec; U6 owns the writer)
- **Steps:** `rms_joint_error_pct(real_h5, …)` replays the logged `target_ticks` through `MujocoBus` and compares `present_ticks` per joint as % of the joint range; CLI prints a table and exits 1 if any joint > 5 %.
- **Tests:** a synthetic HDF5 (written by a test helper following F8) produced *from the sim itself* gives ≈ 0 %; a deliberately time-shifted copy gives > 5 % → exit 1 (gate shown to fail).
- **Acceptance:** `pixi run -e cpu pytest tests/test_compare_real.py -q` → pass.
- **Est.:** 25 min.

**Commit after 4.3:** `MuJoCo backend: scene override, MujocoBus, gripper mapping, sysid check`.

---

### Unit U5 — Perception (Stage A) (wave 1)

#### Task 5.1 (SW): Camera protocol, D435 wrapper, fake camera
- **Files:** `src/so101_rl/perception/camera.py`, `tests/test_camera.py`
- **Specialist:** robotics-vision-navigator
- **Dependencies:** U0
- **Steps:** F9 `Camera`; `D435Camera` selects the device by serial (never by `/dev/videoN` — plan §2), enables color 640×480 @30 + depth 640×480 @30 with align-to-color, discards `warmup_frames`, exposes `Intrinsics` incl. `depth_scale`; `read()` fills preallocated buffers, `ok=False` on `wait_for_frames` timeout. `pyrealsense2` import is lazy so the module imports without the SDK.
- **Tests:** `FakeCamera` returns frames and intrinsics; `D435Camera` import without pyrealsense2 → clear `ImportError` only at `start()`; `@pytest.mark.hardware` test: 45 warmup frames then 30 frames at ≥ 25 fps, depth at image centre within 0.3–1.5 m.
- **Acceptance:** `pixi run -e cpu pytest tests/test_camera.py -q -m "not hardware"` → pass.
- **Est.:** 25 min. `parallelizable-via-worktree: true`

#### Task 5.2 (SW): HSV + depth die tracker
- **Files:** `src/so101_rl/perception/die_tracker.py`, `src/so101_rl/perception/hsv_red_die.yaml`, `tests/test_die_tracker.py`, `tests/data/synthetic_die/*.npz` (generated by the test itself, not committed)
- **Specialist:** robotics-vision-navigator
- **Dependencies:** 5.1
- **Steps:** HSV threshold with the two red hue bands, morphological open, largest contour within `[min_area, max_area]`, centroid `(u,v)`, depth = median over a 5×5 window (reject if the window's depth spread > `depth_window_m`, i.e. edge/occlusion), `xyz_cam = pixel_to_cam`; `confidence` from area fit and depth consistency; returns `None` when no blob passes.
- **Tests:** synthetic frames (a red 16 px square on a grey table at z=0.614 m, 1 px ≈ 0.99 mm) → xyz within 2 mm; occluded (a grey rectangle over half the die) → `None`; two red blobs (fingertip look-alike smaller than `min_area`) → the die is chosen; red blob with inconsistent depth → `None`; hue wrap-around handled.
- **Acceptance:** `pixi run -e cpu pytest tests/test_die_tracker.py -q` → pass.
- **Est.:** 35 min.

#### Task 5.3 (SW): Frames, hand-eye artifact, fiducial camera-moved gate
- **Files:** `src/so101_rl/perception/frames.py`, `src/so101_rl/perception/fiducial.py`, `calib/README.md` (schemas for `hand_eye.yaml`, `fiducial.yaml`), `tests/test_frames.py`, `tests/test_fiducial.py`
- **Specialist:** robotics-vision-navigator; kinematics-specialist
- **Dependencies:** 5.1
- **Steps:** `fit_hand_eye` (Umeyama, no scale); `load_hand_eye` refuses residual > 3 mm or n < 10; `FiducialCheck` uses `cv2.aruco.ArucoDetector`, reprojection error = mean corner distance to `expected_corners_px` from `calib/fiducial.yaml`; not found → `ok=False`.
- **Tests:** synthetic 12-point correspondences with 1 mm noise → residual < 2 mm and T within 1e-3; `load_hand_eye` on a yaml with residual 3.5 → raises (gate fails on broken input); a rendered ArUco marker (cv2 `generateImageMarker`) at expected corners → ok; shifted by 3 px → `ok=False`; marker absent → `ok=False`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_frames.py tests/test_fiducial.py -q` → pass.
- **Est.:** 35 min.

#### Task 5.4 (SW): Object-table shared memory + hand-eye calibration driver (hardware-run later)
- **Files:** `src/so101_rl/perception/object_table.py`, `src/so101_rl/perception/calibrate_hand_eye.py`, `tests/test_object_table.py`
- **Specialist:** robotics-vision-navigator
- **Dependencies:** 5.2, 5.3, U0 (`SeqlockBlock`)
- **Steps:** F9 `ObjectTableShm`; `calibrate_hand_eye.py` (port of `LIT/scripts/calibrate_hand_eye.py` to the F7 `ShmBackend`: drives the arm to ≥ 10 farthest-point poses inside the box via joint deltas, detects the gripper tip with the tracker on a red marker, records `(FK xyz, cam xyz)`, calls `fit_hand_eye`, writes `calib/hand_eye.yaml` with `residual_mm, n_poses, date, camera_serial`; `--execute` required to move). This task only makes it runnable against `SimBackend` + `FakeCamera` for a dry run; H8 runs it for real.
- **Tests:** object table write/read round-trip, torn-read rejection; dry-run of `calibrate_hand_eye --dry-run` in sim with a synthetic camera model produces a yaml with residual < 1 mm.
- **Acceptance:** `pixi run -e cpu pytest tests/test_object_table.py -q` → pass; `pixi run -e cpu python -m so101_rl.perception.calibrate_hand_eye --dry-run` → writes `runs/hand-eye-dry/hand_eye.yaml`, exit 0. (Depends on U8 `SimBackend` for the dry run — if U8 is not merged, mark the dry-run test `xfail(strict=False)` and finish it in U10.)
- **Est.:** 40 min.

#### Task 5.5 (SW): U5 ARCHITECTURE sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` §4 (perception → `servo.shm` client arrow), §5 (object table dtype pointer)
- **Est.:** 5 min.

**Commit after 5.5:** `Perception Stage A: camera, HSV+depth die tracker, hand-eye, fiducial gate, object table`.

---

### Unit U6 — Logging (wave 1)

#### Task 6.1 (SW): Episode logger (applied action) with atomic writes
- **Files:** `src/so101_rl/logging/episode_logger.py`, `src/so101_rl/logging/schema.py`, `tests/test_episode_logger.py`
- **Specialist:** python-best-practices
- **Dependencies:** U0
- **Steps:** F8; datasets: `t_mono_ns[i8]`, `obs/observation[f4,N,15]`, `obs/achieved_goal`, `obs/desired_goal`, `action_cmd[f4,N,6]`, `applied_delta_ticks[f4,N,6]`, `target_ticks[i4,N,6]`, `present_ticks[i4,N,6]`, `reward[f4]`, `terminated[u1]`, `truncated[u1]`, `trips[u4]`, `verifier[u1]`, `frames_small[u1,N,96,96,3]` (gzip 1), `frames_full` optional; attrs: `run_id, episode, outcome, config_dump(json), git_sha, created`. Full frames kept per `keep_full_frames`; `end_episode` renames `.h5.tmp → .h5`.
- **Tests:** round-trip; the logged action equals `applied_delta_ticks` from the env `info`, **not** `action_cmd` (a test feeds different values and asserts the file's `applied_delta_ticks`); `keep_full_frames="failures"` writes full frames only for `outcome != "success"`; a crash before `end_episode` leaves only a `.tmp` (no `.h5`).
- **Acceptance:** `pixi run -e cpu pytest tests/test_episode_logger.py -q` → pass.
- **Est.:** 35 min. `parallelizable-via-worktree: true`

#### Task 6.2 (SW): Artifact checks, retention, disk budget, tick ring writer
- **Files:** `src/so101_rl/logging/checks.py`, `src/so101_rl/logging/ring_writer.py`, `tests/test_artifact_checks.py`, `tests/test_ring_writer.py`
- **Specialist:** python-best-practices
- **Dependencies:** 6.1
- **Tests:** `check_episode` fails on: a `.tmp` file, mismatched lengths, NaN reward, `applied_delta` > `max_delta`, `target` 4096, missing attrs, frame count ≠ steps — each as its own test (**every gate shown to fail on a broken input**); `prune(dry_run=True)` lists but does not delete; `disk_budget_gb`; `RingWriter.push` allocation-free (`tracemalloc`), flush thread drains, `close()` flushes the tail.
- **Acceptance:** `pixi run -e cpu pytest tests/test_artifact_checks.py tests/test_ring_writer.py -q` → pass.
- **Est.:** 30 min.

**Commit after 6.2:** `Episode logging with applied action, artifact checks, retention`.

---

### Unit U7 — Watchdog firmware + hardware checklist (wave 1)

#### Task 7.1 (SW): MCU firmware (MicroPython, RP2040) and protocol doc
- **Files:** `firmware/watchdog_mcu/PROTOCOL.md`, `firmware/watchdog_mcu/main.py` (~50 lines), `firmware/watchdog_mcu/README.md` (flash + wiring), `firmware/watchdog_mcu/test_vectors.json`, `tests/test_firmware_vectors.py`
- **Specialist:** hardware-interface-specialist; real-time-control-specialist
- **Dependencies:** U0 (protocol constants shared with 1.3 — both implement F4 verbatim)
- **Steps:** `main.py`: `machine.WDT(timeout=250)`, `relay = Pin(RELAY, Pin.OUT, value=0)`, `estop_in = Pin(ESTOP, Pin.IN, Pin.PULL_UP)`, `reset_btn`, `vbus = ADC(26)` (47k/10k divider), `sys.stdin.buffer` USB-CDC framing; state machine per F4; `0xE1` only if `test_jumper` pin is low. `test_vectors.json`: sequences of `(dt_ms, bytes_in) → expected status byte / state` that **both** `FakeMcu` (1.3) and the firmware must satisfy (the firmware run is H3).
- **Tests:** `test_firmware_vectors.py` runs the vectors against `FakeMcu` (so the vectors are proven executable); a `pyflakes`/`ruff` pass on `main.py` with a stub `machine` module.
- **Acceptance:** `pixi run -e cpu pytest tests/test_firmware_vectors.py -q` → pass; `wc -l firmware/watchdog_mcu/main.py` ≤ 80.
- **Est.:** 35 min. `parallelizable-via-worktree: true`

#### Task 7.2 (SW): Hardware checklist document
- **Files:** `docs/HARDWARE_CHECKLIST.md`
- **Specialist:** hardware-interface-specialist
- **Dependencies:** None
- **Steps:** one section per H-task below with: parts list, wiring diagram (ASCII), the test procedure, the artifact it must produce (`runs/hw/<test>-<ts>.json` schema), and the "shown to fail" variant. Includes the power-path topology from plan Phase 0 (relay NO 30/40 A, transistor + flyback, TVS 15–16 V, DC-13 e-stop, PSU ≥ 8–10 A, trip latching, no USB/GPIO relay, no AC SSR).
- **Acceptance:** every H-task id in this plan appears as a heading (`grep -c "^## H" docs/HARDWARE_CHECKLIST.md` ≥ 13).
- **Est.:** 30 min. `parallelizable-via-worktree: true`

**Commit after 7.2:** `Watchdog MCU firmware, protocol, hardware checklist`.

---

### Unit U3 — Servo core, process, soak harness, action-unit tests (wave 2; depends on U1, U2, U4, U6)

#### Task 3.1 (SW): `ServoCore.tick()`
- **Files:** `src/so101_rl/servo/core.py`, `tests/test_servo_core.py`
- **Specialist:** real-time-control-specialist
- **Dependencies:** U1, U2, U6 (`RingWriter`)
- **Steps:** F6; command dedupe by `seq`; `hb_sent` only when `read_block` and `write_goal` both succeeded this tick; `write_state` every tick even in fault states; `shutdown()` idempotent; `signal.signal(SIGINT/SIGTERM)` set a flag checked by `run()`, never do I/O in the handler.
- **Tests (FakeBus + FakeClock + FakeMcu + LocalIpc):** a command is applied exactly once across 3 ticks; late policy → HOLD, target held (no extrapolation), fresh seq → RUN; N misses → SAFE_STOPPING → TORQUE_OFF; read failure tick → no write, no heartbeat; 3 consecutive bus failures → LATCHED; `FakeMcu` latches when heartbeats stop (SIGSTOP model: stop calling `tick()` and advance the fake clock 1.1 s); thermal trip flow with resync; SIGINT flag → shutdown sequence runs and completes < 5 s of fake time; `tracemalloc` shows zero net allocation over 1000 ticks after warm-up.
- **Acceptance:** `pixi run -e cpu pytest tests/test_servo_core.py -q` → pass (≥ 15 tests).
- **Est.:** 45 min.

#### Task 3.2 (SW): Process runner, CLI, RT setup
- **Files:** `src/so101_rl/servo/process.py`, `src/so101_rl/servo/main.py`, `tests/test_servo_process.py`
- **Specialist:** real-time-control-specialist
- **Dependencies:** 3.1
- **Steps:** F6 `run()`; RT setup best-effort with `rt_active` in the state record (`hb_ok`/`sm_state` fields already exist; add `rt_active` to the soak report, not the dtype); deadline loop with resync; `--max-ticks` for tests; `--backend mujoco` constructs `MujocoBus` (sim feature) for a headless servo process.
- **Tests:** `subprocess` run with `--backend fake --max-ticks 300` exits 0 in < 15 s real time and its state record via `ServoShm.attach` shows `tick == 300`, `overruns` small; `kill -9` of the subprocess → `FakeMcu` in the test harness (fed through a pty pair) reports `latched` within 1.5 s; `REQUIRE_RT=1` without privileges → exit code 3 and a clear message.
- **Acceptance:** `pixi run -e cpu pytest tests/test_servo_process.py -q` → pass; `pixi run -e cpu servo-fake --max-ticks 300` → exit 0.
- **Est.:** 40 min.

#### Task 3.3 (SW): Soak harness and gate
- **Files:** `src/so101_rl/servo/soak.py`, `tests/test_soak.py`
- **Specialist:** real-time-control-specialist; performance-optimization-specialist
- **Dependencies:** 3.2
- **Steps:** runs the servo process plus a random-policy shm client (uniform deltas ≤ max) for `--minutes`; collects `TickStats`, heartbeat gaps (from the tick log), `max_temp_c`, thermal trips, calibration mismatches (from `CHECK` state entries), RSS growth (`psutil`), `gc.get_count()` deltas, kernel + `rt_active`; writes `report.json`; `check_soak_report(strict=rt_active)`.
- **Tests:** a 5-second fake soak writes a valid report; `check_soak_report` **fails** on a hand-edited report with `p999_ms: 5.1`, with `hb_gaps_over_200ms: 1`, with `thermal_trips: 1` — each its own test; `strict=False` turns those into warnings.
- **Acceptance:** `pixi run -e cpu pytest tests/test_soak.py -q` → pass; `pixi run -e cpu soak --backend fake --minutes 0.1` → `runs/soak-*/report.json` exists.
- **Est.:** 35 min.

#### Task 3.4 (SW): Action-units Test 1 and Test 2 (both backends)
- **Files:** `tests/test_action_units.py`
- **Specialist:** real-time-control-specialist
- **Dependencies:** 3.1, U4
- **Tests:**
  - **Test 1:** for each backend in `{FakeBus, MujocoBus}` and each joint j: send unit action `+1` on j (delta = `max_delta_ticks[j]`) from rest; after 3 ticks, `written_goals[-1][j] − written_goals[0][j] == max_delta_ticks[j]` exactly (integer ticks, 0.0879°/tick), other joints unchanged; both backends produce **identical** goal sequences tick by tick.
  - **Test 2:** block joint j (FakeBus obstacle at rest+30 ticks; MujocoBus: a wall geom added programmatically 30 ticks away) and command `+1` for 300 ticks: `|target − present| ≤ E[j]` on **every** tick and the stored goal never exceeds `present + E[j]` (no wind-up); on removing the obstacle the arm moves at most `E[j]` further before new commands.
  - **Test 3 (end-to-end):** a policy command producing goal 4096 or NaN never reaches `write_goal` (intercept records nothing out of range) and the state record shows `RANGE`/`NAN` trips.
- **Acceptance:** `pixi run -e cpu pytest tests/test_action_units.py -q` → pass (3 × 6 × 2 parametrised cases + 2).
- **Est.:** 40 min.

#### Task 3.5 (SW): FK cross-check against MuJoCo and latency measurement hooks
- **Files:** `tests/test_fk_vs_mujoco.py`, `src/so101_rl/servo/latency.py`, `tests/test_latency.py`
- **Specialist:** kinematics-specialist
- **Dependencies:** 2.2, 4.2
- **Tests:** `So101FK.gripper_xyz(q)` vs `mj_kinematics` site position at 500 random in-range q → max |Δ| < 1e-6 m; `latency.measure_bus(bus, n=300) -> dict(p50, p99, max)` and `measure_obs_to_cmd()` (policy timestamp vs servo apply timestamp) work on FakeBus; `@pytest.mark.hardware` variants record to `runs/hw/latency-<ts>.json` (p99 read < 6 ms, per plan table).
- **Acceptance:** `pixi run -e cpu pytest tests/test_fk_vs_mujoco.py tests/test_latency.py -q -m "not hardware"` → pass.
- **Est.:** 30 min. `parallelizable-via-worktree: true` (with 3.3/3.4)

#### Task 3.6 (SW): U3 ARCHITECTURE sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` §3 (servo process CLI, rt_active, soak gate), §6 (final check order incl. `check_range` as the last gate)
- **Est.:** 5 min.

**Commit after 3.6:** `Servo process: core tick, RT runner, soak gate, action-unit tests`.

---

### Unit U8 — Goal-conditioned env (wave 3; depends on U3, U4, U5, U6)

#### Task 8.1 (SW): Backends: `SimBackend` (in-process ServoCore + MujocoBus) and `ShmBackend`
- **Files:** `src/so101_rl/env/backend.py`, `src/so101_rl/env/local_ipc.py`, `tests/test_env_backend.py`
- **Specialist:** real-time-control-specialist; simulation-to-hardware-bridge
- **Dependencies:** U3, U4
- **Steps:** F7; `LocalIpc` implements `ServoIpc` with two in-memory records (same dtypes); `SimBackend.wait_policy_tick()` = 3 × (`core.tick()`, `bus.step_ticks(1)`, `clock.advance(period)`); `ShmBackend.wait_policy_tick()` sleeps to the next 100 ms boundary and then polls `read_state` until `seq` advances (bounded); `reset()` in real mode issues a HOLD then small deltas toward `q0` (never a jump), waits until `|target − present| < 5 ticks` on all joints.
- **Tests:** `SimBackend` reset → state record populated, `sm_state == RUN`; 10 policy steps advance `tick` by 30; `ShmBackend` against `servo-fake` subprocess: attach, 20 steps, `cmd_seq` monotone, state `last_cmd_seq` follows; `reset()` never produces a per-tick jump > `max_ticks_per_tick`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_env_backend.py -q` → pass.
- **Est.:** 40 min.

#### Task 8.2 (SW): `So101ReachEnv` + goal sampler + Gym/SB3 checkers
- **Files:** `src/so101_rl/env/goal_env.py`, `src/so101_rl/env/goals.py`, `src/so101_rl/env/__init__.py` (gymnasium `register("So101Reach-v0")`), `tests/test_goal_env.py`
- **Specialist:** lerobot-specialist (SB3/HER contract), testing-specialist
- **Dependencies:** 8.1, 2.2
- **Steps:** F7; `BoxGoalSampler(cfg.box, radius_m=0.30, z_min=kinematic floor, fov=None)` rejection-samples; `truncated=True` + `info["trip"]` on any `fault_bits`; `render()` in sim via `mujoco.Renderer` (rgb_array) — optional import.
- **Tests:** `gymnasium.utils.env_checker.check_env` and `stable_baselines3.common.env_checker.check_env` pass on `SimBackend`; `compute_reward` vectorised matches scalar; an episode with a scripted "move toward goal" policy reaches `is_success` on ≥ 8/10 goals in sim (kinematic sanity, not RL); a NaN action → `truncated=True`, `info["trip"] & TripCause.NAN`, and the arm target is unchanged; sampled goals all inside the box and within 0.30 m of the base; `info["applied_delta_ticks"]` equals the state record's value (what the logger stores).
- **Acceptance:** `pixi run -e cpu pytest tests/test_goal_env.py -q` → pass.
- **Est.:** 45 min.

#### Task 8.3 (SW): U8 ARCHITECTURE sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` §5 (env obs/action shapes, truncation semantics), §4 (`env/local_ipc.py`)
- **Est.:** 5 min.

**Commit after 8.3:** `Goal-conditioned reach env with sim and shm backends`.

---

### Unit U9 — RL and world-model scaffolding (wave 4; depends on U8)

#### Task 9.1 (SW): HER relabel + TD-MPC2 goal-env adapter
- **Files:** `src/so101_rl/rl/her.py`, `src/so101_rl/rl/tdmpc2_env.py`, `src/so101_rl/rl/cfg.py`, `tests/test_her.py`, `tests/test_tdmpc2_env.py`
- **Specialist:** lerobot-specialist
- **Dependencies:** U8, 0.4
- **Tests:** `her_relabel` on a TensorDict of `ToyReachEnv` trajectories: relabelled `desired_goal` equals a future `achieved_goal` of the same episode (never crosses episode boundaries), rewards recomputed by `reward_fn`, `k=0` is identity; `TdMpc2GoalEnv(So101ReachEnv(SimBackend))` exposes `obs_shape["state"] == (18,)`, `rand_act()` in [-1,1], `step` returns tensors; 20 random steps run.
- **Acceptance:** `pixi run -e cpu pytest tests/test_her.py tests/test_tdmpc2_env.py -q` → pass.
- **Est.:** 40 min. `parallelizable-via-worktree: true`

#### Task 9.2 (SW): Entry points `train-reach-sac`, `train-reach-tdmpc2` (smoke-level)
- **Files:** `src/so101_rl/rl/train_sac_her.py`, `src/so101_rl/rl/train_tdmpc2.py`, `tests/test_train_entrypoints.py`
- **Specialist:** lerobot-specialist
- **Dependencies:** 9.1
- **Steps:** SB3 `SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"), learning_starts=…)`; artifacts `runs/sac-her/<ts>/{model.zip, monitor.csv}`; TD-MPC2 entry uses `make_tdmpc2_cfg` + `OnlineTrainer` on the adapter, `runs/tdmpc2/<ts>/{metrics.csv, model.pt}`; both exit non-zero if artifacts are missing (L12). The ≥ 90 % MuJoCo gate is **Phase 3**, not here.
- **Tests:** `@pytest.mark.slow` 300-step runs of each on CPU finish < 3 min and produce the artifacts; a run with `--total-steps 0` exits non-zero (artifact check shown to fail).
- **Acceptance:** `pixi run -e cpu train-reach-sac --total-steps 300` and `pixi run -e cpu train-reach-tdmpc2 --steps 300` → exit 0 with artifacts.
- **Est.:** 40 min.

#### Task 9.3 (SW): State world-model skeleton + dataset + probe metric
- **Files:** `src/so101_rl/wm/state_wm.py`, `src/so101_rl/wm/dataset.py`, `src/so101_rl/wm/train_state_wm.py`, `tests/test_state_wm.py`
- **Specialist:** lerobot-specialist (world-model bridge patterns)
- **Dependencies:** U6 (format), 2.2 (FK)
- **Tests:** `EpisodeDataset` refuses a run containing one failing `check_episode` (L12) unless `require_ok=False`; model fits a synthetic linear system to MSE < 1e-3 in 300 steps on CPU; `probe_gripper_error_m` on a perfect model → ≈ 0, on an untrained model → large (metric shown to discriminate).
- **Acceptance:** `pixi run -e cpu pytest tests/test_state_wm.py -q` → pass; `pixi run -e cpu train-wm-state --run-root <synthetic> --steps 100` → `runs/wm-state/<ts>/model.pt`.
- **Est.:** 40 min. `parallelizable-via-worktree: true` (with 9.1)

**Commit after 9.3:** `RL/WM scaffolding: HER, TD-MPC2 adapter, SAC+HER and TD-MPC2 entry points, state WM`.

---

### Unit U10 — Milestone script and gate checker (wave 4)

#### Task 10.1 (SW): `run_random_policy.py` — the same script for sim and real
- **Files:** `scripts/run_random_policy.py`, `tests/test_run_random_policy.py`
- **Specialist:** testing-specialist
- **Dependencies:** U8, U6
- **Steps:** `--backend {sim,real} --episodes N --run-id …`; builds `So101ReachEnv`, `EpisodeLogger`, optional `FiducialCheck` (real), runs uniform-random actions, logs every step with `applied_delta_ticks` from `info`, runs `check_episode` on every file at the end and exits non-zero on any failure; real mode refuses to start unless `runs/hw/tier.json` says `tier ≥ 0 attended` and a human is flagged present (`--attended` flag required until Tier 1).
- **Tests:** sim run of 3 episodes → 3 `.h5` files, all `check_episode` ok; a monkeypatched logger that drops a dataset → exit non-zero.
- **Acceptance:** `pixi run -e cpu random-policy-sim --episodes 3` → exit 0; `ls runs/<run-id>/episodes/*.h5 | wc -l` → 3.
- **Est.:** 30 min.

#### Task 10.2 (SW): `check_milestone_phase0.py` — every gate must be able to fail
- **Files:** `scripts/check_milestone_phase0.py`, `tests/test_check_milestone.py`
- **Specialist:** testing-specialist
- **Dependencies:** 10.1, 3.3
- **Steps:** reads: latest `runs/soak-*/report.json` (gate via `check_soak_report(strict=True)`), `runs/hw/estop-*.json` (`drop_ms ≤ 50`, `v_after < 1.0`), `runs/hw/deadman-*.json` (`kill9_ms ≤ 1500`, `usb_unplug_ms ≤ 1500`, `reboot: true`), `runs/hw/safe-stop-*.json` (`goal_frozen: true`, `vel_below_thresh_ms ≤ 100`), `runs/hw/calib-check-*.json` (`ok: true`, `broken_copy_rejected: true`), `runs/hw/sysid-*.json` (`rms_pct ≤ 5` per joint), `calib/hand_eye.yaml` (`residual_mm ≤ 3`, `n_poses ≥ 10`), `config/so101.yaml: verified_on_hardware: true`, `nvidia-smi` healthy, `runs/<random-policy>` sim and real both with all episodes ok. Prints a table PASS / FAIL / NOT-RUN and exits 0 only when everything passes.
- **Tests:** a fixture tree with all-good artifacts → exit 0; **each artifact broken in turn** → that row FAIL and exit 1 (parametrised); missing artifact → NOT-RUN, exit 1.
- **Acceptance:** `pixi run -e cpu pytest tests/test_check_milestone.py -q` → pass; `pixi run -e cpu check-milestone` on the current tree → exit 1 with NOT-RUN rows for hardware items (expected until the H-tasks land).
- **Est.:** 30 min.

**Commit after 10.2:** `Phase 0 milestone script and gate checker`.

---

### Hardware / human tasks (H-series)

Each has a checklist section in `docs/HARDWARE_CHECKLIST.md` (7.2) and produces an artifact under `runs/hw/` that `check_milestone_phase0.py` reads. **Tier 0 (attended, hand on the e-stop) for every item until H13.**

| Id | Task | Needs (software) | Checklist (abridged) | Artifact / acceptance | Est. |
|---|---|---|---|---|---|
| **H1** | Restore the NVIDIA driver; laptop-vs-desktop; GPU thermals | 0.1 (re-pin `cuda-version`), 0.4 (`smoke-tdmpc2 --device cuda`) | `dkms status`; `sudo dkms autoinstall` for kernel 7.0.0-34; else boot the previous kernel or switch driver branch; `hostnamectl chassis` / `/sys/class/dmi/id/chassis_type`; 20-min `torch.matmul` loop while logging `nvidia-smi -q -d TEMPERATURE,POWER` | `nvidia-smi` shows RTX 3080 10 GB; `runs/hw/gpu-<ts>.json {driver, cuda, chassis, max_temp_c, throttled: false}`; `pixi install -e default` solves; `pixi run smoke-tdmpc2 --device cuda` exit 0 | 1–3 h |
| **H2** | Power path: PSU ≥ 8–10 A @ 12 V, relay module (12 V automotive NO 30/40 A; transistor + flyback), TVS 15–16 V bidirectional across the servo bus, latching e-stop (DC-13 ≥ supply current, or in the control circuit with a mechanical break in the load path) | 7.2 wiring diagram | Wire per diagram; measure continuity with the relay open; e-stop opens the load path; no USB/GPIO relay, no AC SSR | Photo + `runs/hw/power-path-<ts>.json {psu_a, relay_part, tvs_part, estop_rating}`; **e-stop test** (with MCU ADC via `0xE1`, or a scope): V_bus < 1 V within 50 ms → `runs/hw/estop-<ts>.json {drop_ms, v_after}` | 3–4 h + parts |
| **H3** | Watchdog MCU bring-up | 7.1 firmware, 1.3 link, 3.2 process | Flash `main.py` to the RP2040; wire relay pin, e-stop input, reset button, ADC divider, TEST jumper; run `test_vectors.json` against the real MCU over serial | `runs/hw/deadman-<ts>.json`: no-heartbeat → relay opens at 1.0 ± 0.1 s; `kill -9` servo → open ≤ 1.5 s; USB unplug ≤ 1.5 s; host reboot → open; latch persists until the physical reset; a serial "reset" attempt does **not** clear the latch (shown to fail) | 2–3 h |
| **H4** | Servo fan + thermal duty-cycle study | 3.3 soak (`max_temp_c`), 2.3 | Mount a small fan on the servos; run 20-on/10-off duty at random motion; also a 10-min *hold-still* segment (plan: holding is worse than moving) | `runs/hw/thermal-<ts>.json {duty, max_temp_c per joint, time_to_60c_s or null}`; update `so101.yaml duty_on_s/duty_off_s` if needed | 1.5 h |
| **H5** | Rest pose, torque-off verification, sag map, stale-goal snap test | 2.4 safe-state, 3.2 `servo --backend feetech`, 4.1 keyframe | Foam under the arm; from `rest_ticks` disable torque → no motion (video); sample limp falls from the box corners and highest poses (camera recording); none touch the D435 mount; deliberately leave a stale goal and re-energise **with foam** to see the snap that step 3 of re-energise prevents, then repeat with the real sequence | `runs/hw/rest-pose-<ts>.json {rest_ticks, moved_ticks_after_torque_off ≤ 5}`, `runs/hw/sag-map-<ts>/` videos; set `verified_on_hardware: true` in `so101.yaml` | 2 h |
| **H6** | Register configuration proof and thermal-trip race | 1.2 `write_registers`, 2.3 | Read back all F1 registers after `configure`; stall one joint against a stop (foam) → firmware fold-back visible (`Status` bit or load drop) **and** the 60 °C software trip fires before any firmware 70 °C event during H4 | `runs/hw/registers-<ts>.json {readback, foldback_observed: true, sw_trip_first: true}`; choose arm `protection_current` / `over_current_protection_time` values and commit them to `so101.yaml` | 1.5 h |
| **H7** | Calibration read-back on the real arm | 2.4 `calib-check` task | Run `pixi run calib-check` (all match); then run against a **deliberately corrupted copy** of the JSON (homing_offset +1) → refuses, no torque enable | `runs/hw/calib-check-<ts>.json {ok: true, broken_copy_rejected: true}` | 20 min |
| **H8** | Hand-eye calibration + table fiducial | 5.4 driver, 5.3 loader/fiducial, 3.2 servo | Red marker on the gripper tip; `pixi run hand-eye --execute` (≥ 10 poses; attended); place the ArUco marker on the table in view; record `expected_corners_px`; nudge the camera 2 mm → the gate must fail, put it back | `calib/hand_eye.yaml` (residual ≤ 3 mm, n ≥ 10) and `calib/fiducial.yaml` committed; `runs/hw/camera-moved-<ts>.json {nudged_err_px > 2, restored_err_px ≤ 2}` | 1.5 h |
| **H9** | Gravity-return tray for the 16 mm die | — (plan Phase 4 says "tested in Phase 0") | Build a tilted tray with a start well inside the box and in camera view; 20 drops from random points | `runs/hw/tray-<ts>.json {returns: ≥ 19/20}`; photos | 2 h + build |
| **H10** | 30-minute attended random-motion soak (Tier 0) | 3.3 soak, H2–H7 | `REQUIRE_RT=1 pixi run soak --backend feetech --minutes 30` with the camera + a dummy MPPI load process running (`stress-ng` on the other cores); a human at the e-stop | `runs/soak-<ts>/report.json` passes `check_soak_report(strict=True)`: p99.9 overrun < 5 ms, zero heartbeat gaps > 200 ms, zero thermal trips, zero calibration mismatches, max temp per servo logged | 45 min |
| **H11** | Software safe-stop timing | 3.1 (`USER_STOP` via `CmdMode.STOP_REQ`), tick log | While moving, send `STOP_REQ`; from the tick log: `goal_ticks` frozen on the next tick and joint velocity below threshold within 100 ms; repeat with an injected fake temperature (`--inject-temp 61` debug flag on the fake thermal input, hardware bus otherwise real) | `runs/hw/safe-stop-<ts>.json {goal_frozen: true, vel_below_thresh_ms ≤ 100}` | 45 min |
| **H12** | Sysid check against ≥ 10 real trajectories | 4.3 `compare_real`, 10.1 logs | Use the H10 soak or 10 random-policy episodes; run `python -m so101_rl.sim.compare_real runs/<id>` | `runs/hw/sysid-<ts>.json {rms_pct per joint}`; ≤ 5 % passes; > 5 % is **recorded, not fixed** here (fitting is the parallel sysid track) | 30 min |
| **H13** | Tier promotion review (0 → 1) | 10.2 `check-milestone` | Human reviews every artifact row; signs `runs/hw/tier.json {tier: 1, by, date, milestone_sha}`; the 2-hour goal-directed soak stays a Phase 2 precondition | `pixi run check-milestone` exit 0 | 30 min |

Rough hardware total: **≈ 20 human-hours + parts lead time**, spread across the software waves (H1 can start immediately; H2/H3 as soon as 7.1 lands; H5–H12 after U3).

## Execution Strategy

- **Wave 0 (sequential):** U0 tasks 0.1 → 0.2 → (0.3 ‖ 0.4 ‖ 0.5) → 0.6. Everything else codes against U0's frozen files.
- **Wave 1 (parallel, 6 units):** U1, U2, U4, U5, U6, U7. Inside U2, 2.4/2.5 wait for 2.1–2.3. No two units write the same file; `docs/ARCHITECTURE.md` is updated only by each unit's final sync task and merges are append-only (§4 lists, §5/§6 tables, §10 changelog rows).
- **Wave 2:** U3 (needs all of U1, U2, U4, U6 merged). Tasks 3.3, 3.4, 3.5 can run concurrently after 3.2.
- **Wave 3:** U8. **Wave 4 (parallel):** U9, U10.
- **Hardware track:** H1 immediately; H2, H3 after 7.1/7.2; H4–H12 after U3 (servo process exists); H13 last.
- **Always-parallel:** quality review + test generation per unit (executor's standard step).

Rough software effort: 42 SW tasks (36 substantive × 25–45 min + 6 `ARCHITECTURE.md` sync tasks × 5–10 min) ≈ **20–24 executor-hours** across 5 waves (critical path U0 → U2 → U3 → U8 → U9/U10 ≈ 9 h of that). Add ≈ 20 human-hours of hardware work.

## Git Commits

One commit per unit (listed after each unit above), plus:
- After H5/H6/H7: `Hardware-verified rest pose, register values, calibration check` (edits `config/so101.yaml`, adds `runs/hw/*.json` — note `runs/` is gitignored; the milestone script reads them in place, and `docs/HARDWARE_CHECKLIST.md` gets a results table).
- After H8: `Add hand-eye and fiducial calibration artifacts`.
- After H13: `Phase 0 milestone: tier 1 approved`.

Every commit that changes a module, process, interface, safety-check order or pixi feature touches `docs/ARCHITECTURE.md` in the same commit (its §10 rule; enforced by a `tests/test_architecture_sync.py` that fails if `src/so101_rl/*/` directories are not all listed in §4).

## How to Run

Single-session run (default):
1. Open this plan in the executor and proceed unit by unit, task by task; run each task's acceptance command before moving on.

Parallel run (wave 1 and wave 4 tasks carry `parallelizable-via-worktree: true`):
1. For each parallel unit, spawn a worktree:
   ```bash
   /worktree create feature/phase0-u1-servo-backends
   /worktree create feature/phase0-u2-safety
   /worktree create feature/phase0-u4-sim
   /worktree create feature/phase0-u5-perception
   /worktree create feature/phase0-u6-logging
   /worktree create feature/phase0-u7-firmware
   ```
2. Open a terminal per worktree, `cd` into the printed path, run `claude`, and execute that unit's tasks (each unit's file list is exclusive, so merges are conflict-free except the append-only `docs/ARCHITECTURE.md` sync lines).
3. Merge all wave-1 branches into `main`, then start U3 from `main`; repeat for later waves.

This section is suggestive — the user owns the decision. See `commands/worktree.md` and `skills/worktree/SKILL.md`.

## execution_shape

```yaml
execution_shape:
  plan: plans/phase0-implementation-plan.md
  phase: "Phase 0: Foundation + Stage A scaffolding"
  units:
    - id: U0
      name: bootstrap-and-interface-freeze
      wave: 0
      depends_on: []
      parallelizable_via_worktree: false
      tasks: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
      owns_files:
        - pyproject.toml
        - pixi.lock
        - README.md
        - scripts/fetch_assets.py
        - scripts/extract_kinematics.py
        - scripts/vendor_tdmpc2.py
        - assets/so101/upstream/**
        - assets/so101/LICENSE
        - assets/so101/NOTICE
        - assets/so101/UPSTREAM.json
        - src/so101_rl/__init__.py
        - src/so101_rl/py.typed
        - src/so101_rl/config/**
        - src/so101_rl/_vendor/**
        - src/so101_rl/rl/__init__.py
        - src/so101_rl/rl/toy_env.py
        - src/so101_rl/rl/smoke_tdmpc2.py
        - src/so101_rl/servo/__init__.py
        - src/so101_rl/servo/bus.py
        - src/so101_rl/servo/timing.py
        - src/so101_rl/servo/shm.py
        - src/so101_rl/safety/__init__.py
        - src/so101_rl/safety/types.py
        - src/so101_rl/env/__init__.py
        - src/so101_rl/perception/__init__.py
        - src/so101_rl/sim/__init__.py
        - src/so101_rl/wm/__init__.py
        - src/so101_rl/logging/__init__.py
        - tests/conftest.py
        - tests/data/so101_follower.calib.json
        - tests/test_env_sanity.py
        - tests/test_config.py
        - tests/test_config_single_source.py
        - tests/test_assets.py
        - tests/test_vendor_tdmpc2.py
        - tests/test_toy_env.py
        - tests/test_shm.py
        - tests/test_timing.py
        - tests/test_types.py
        - tests/test_architecture_sync.py
    - id: U1
      name: servo-backends-and-heartbeat
      wave: 1
      depends_on: [U0]
      parallelizable_via_worktree: true
      tasks: [1.1, 1.2, 1.3, 1.4]
      owns_files:
        - src/so101_rl/servo/fake_bus.py
        - src/so101_rl/servo/feetech_bus.py
        - src/so101_rl/servo/heartbeat.py
        - tests/test_fake_bus.py
        - tests/test_feetech_bus.py
        - tests/test_heartbeat.py
    - id: U2
      name: safety-layer
      wave: 1
      depends_on: [U0]
      parallelizable_via_worktree: true
      tasks: [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
      owns_files:
        - src/so101_rl/safety/clamp.py
        - src/so101_rl/safety/fk.py
        - src/so101_rl/safety/thermal.py
        - src/so101_rl/safety/stuck.py
        - src/so101_rl/safety/safe_state.py
        - src/so101_rl/safety/reenergize.py
        - src/so101_rl/safety/calib_check.py
        - src/so101_rl/safety/state_machine.py
        - src/so101_rl/safety/pipeline.py
        - tests/test_clamp.py
        - tests/test_fk.py
        - tests/test_thermal.py
        - tests/test_stuck.py
        - tests/test_safe_state.py
        - tests/test_reenergize.py
        - tests/test_calib_check.py
        - tests/test_state_machine.py
        - tests/test_pipeline.py
    - id: U4
      name: mujoco-sim
      wave: 1
      depends_on: [U0]
      parallelizable_via_worktree: true
      tasks: [4.1, 4.2, 4.3]
      owns_files:
        - assets/so101/so101_rl_scene.xml
        - assets/so101/README.md
        - src/so101_rl/sim/model.py
        - src/so101_rl/sim/mujoco_bus.py
        - src/so101_rl/sim/gripper_map.py
        - src/so101_rl/sim/compare_real.py
        - tests/test_sim_model.py
        - tests/test_mujoco_bus.py
        - tests/test_gripper_map.py
        - tests/test_compare_real.py
    - id: U5
      name: perception-stage-a
      wave: 1
      depends_on: [U0]
      parallelizable_via_worktree: true
      tasks: [5.1, 5.2, 5.3, 5.4, 5.5]
      owns_files:
        - src/so101_rl/perception/camera.py
        - src/so101_rl/perception/die_tracker.py
        - src/so101_rl/perception/hsv_red_die.yaml
        - src/so101_rl/perception/frames.py
        - src/so101_rl/perception/fiducial.py
        - src/so101_rl/perception/object_table.py
        - src/so101_rl/perception/calibrate_hand_eye.py
        - calib/README.md
        - tests/test_camera.py
        - tests/test_die_tracker.py
        - tests/test_frames.py
        - tests/test_fiducial.py
        - tests/test_object_table.py
    - id: U6
      name: episode-logging
      wave: 1
      depends_on: [U0]
      parallelizable_via_worktree: true
      tasks: [6.1, 6.2]
      owns_files:
        - src/so101_rl/logging/episode_logger.py
        - src/so101_rl/logging/schema.py
        - src/so101_rl/logging/checks.py
        - src/so101_rl/logging/ring_writer.py
        - tests/test_episode_logger.py
        - tests/test_artifact_checks.py
        - tests/test_ring_writer.py
    - id: U7
      name: watchdog-firmware-and-hw-checklist
      wave: 1
      depends_on: [U0]
      parallelizable_via_worktree: true
      tasks: [7.1, 7.2]
      owns_files:
        - firmware/watchdog_mcu/**
        - docs/HARDWARE_CHECKLIST.md
        - tests/test_firmware_vectors.py
    - id: U3
      name: servo-core-process-soak
      wave: 2
      depends_on: [U1, U2, U4, U6]
      parallelizable_via_worktree: false
      tasks: [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]
      owns_files:
        - src/so101_rl/servo/core.py
        - src/so101_rl/servo/process.py
        - src/so101_rl/servo/main.py
        - src/so101_rl/servo/soak.py
        - src/so101_rl/servo/latency.py
        - tests/test_servo_core.py
        - tests/test_servo_process.py
        - tests/test_soak.py
        - tests/test_action_units.py
        - tests/test_fk_vs_mujoco.py
        - tests/test_latency.py
    - id: U8
      name: goal-conditioned-env
      wave: 3
      depends_on: [U3, U4, U5, U6]
      parallelizable_via_worktree: false
      tasks: [8.1, 8.2, 8.3]
      owns_files:
        - src/so101_rl/env/backend.py
        - src/so101_rl/env/local_ipc.py
        - src/so101_rl/env/goal_env.py
        - src/so101_rl/env/goals.py
        - tests/test_env_backend.py
        - tests/test_goal_env.py
    - id: U9
      name: rl-wm-scaffolding
      wave: 4
      depends_on: [U8]
      parallelizable_via_worktree: true
      tasks: [9.1, 9.2, 9.3]
      owns_files:
        - src/so101_rl/rl/her.py
        - src/so101_rl/rl/tdmpc2_env.py
        - src/so101_rl/rl/cfg.py
        - src/so101_rl/rl/train_sac_her.py
        - src/so101_rl/rl/train_tdmpc2.py
        - src/so101_rl/wm/state_wm.py
        - src/so101_rl/wm/dataset.py
        - src/so101_rl/wm/train_state_wm.py
        - tests/test_her.py
        - tests/test_tdmpc2_env.py
        - tests/test_train_entrypoints.py
        - tests/test_state_wm.py
    - id: U10
      name: milestone-scripts
      wave: 4
      depends_on: [U8]
      parallelizable_via_worktree: true
      tasks: [10.1, 10.2]
      owns_files:
        - scripts/run_random_policy.py
        - scripts/check_milestone_phase0.py
        - tests/test_run_random_policy.py
        - tests/test_check_milestone.py
    - id: H
      name: hardware-human-track
      wave: "1..4 (interleaved; H1 immediately, H2-H3 after U7, H4-H12 after U3, H13 last)"
      depends_on: [U7, U3, U10]
      parallelizable_via_worktree: false
      tasks: [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13]
      owns_files:
        - calib/hand_eye.yaml
        - calib/fiducial.yaml
        - runs/hw/**          # gitignored artifacts read by check_milestone_phase0.py
        - src/so101_rl/config/so101.yaml   # ONLY the measured values (rest_ticks, registers, duty, verified_on_hardware) — written after U0, never concurrently with a software unit
  waves:
    - {wave: 0, units: [U0]}
    - {wave: 1, units: [U1, U2, U4, U5, U6, U7]}
    - {wave: 2, units: [U3]}
    - {wave: 3, units: [U8]}
    - {wave: 4, units: [U9, U10]}
  shared_writes:
    - path: docs/ARCHITECTURE.md
      rule: "Updated at the END of every unit that changes a module, process, interface, safety-check order or pixi feature (tasks 0.6, 1.4, 2.6, 4.3, 5.5, 3.6, 8.3; U6/U7/U9/U10 add a §4/§8 line if they add a module or task). Append-only edits to §4 lists, §5/§6 tables and the §10 changelog; bump the version line. Never edited inside a wave-1 task other than the unit's final sync task, to keep worktree merges conflict-free."
    - path: README.md
      rule: "Environment section written by U0 (spike verdict). Later units append one line under 'Entry points' for each new pixi task, after their unit completes."
    - path: pyproject.toml
      rule: "Owned by U0. Units that need a new pixi task or dependency report it back instead of editing; U0 (or the integrator) applies it between waves."
    - path: docs/HARDWARE_CHECKLIST.md
      rule: "Owned by U7 for the procedure text; the H-track appends a results table row per artifact after each H-task."
```

## Gaps and notes for the master orchestrator

- ~~`safety_limits.py` does not exist~~ **Corrected:** it exists at `LIT/src/robot-data-runner/src/robot_data_runner/safety_limits.py`, and the plan now gives that path. `safety/clamp.py` re-implements it, using it as a reference.
- **Block read length:** 11 bytes (56..66) instead of the plan's 8 so `Status (65)` is checked every tick; recorded above with a measured fallback.
- **Firmware language:** MicroPython on RP2040 (keeps "pure Python" and lets `FakeMcu` share test vectors); the protocol is language-neutral if Arduino C is preferred later.
- **`So101Config.rest_ticks` and the arm `Protection_Current` value are placeholders until H5/H6** — `so101.yaml` carries `verified_on_hardware: false` and the milestone checker fails until it is flipped by a human.
- **No MuJoCo specialist and no pixi-packaging specialist agent** exist in the harness; `simulation-to-hardware-bridge`, `kinematics-specialist` and the `pixi-workspace-maintainer` worker were assigned instead.
- The `lerobot-specialist` inspected 0.5.1 locally; every 0.6.x fact used above was re-verified against the installed 0.6.0 tables/API and the PyPI 0.6.1 metadata.

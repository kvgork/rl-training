"""Three-layer parity: ServoCore + MujocoBus (the real servo path on CPU MuJoCo) vs the batched sim (frozen test
definition: ``plans/sim-track-implementation-plan.md`` "Parity test definition").

* Layer A (CPU, ``pixi run test``): the present-tick stream recorded from the CPU rig, fed to a fresh numpy
  ``SafetyPipeline`` and to ``BatchedServoChain`` (torch CPU, N=1 and N=4 with world 2 carrying the sequence):
  targets, goals, clamped masks, the clamp bits of the trips and the applied deltas are bit-identical.
* Layer B (``gpu``): both rigs closed loop (CPU float64 MuJoCo vs MuJoCo Warp float32).
* Layer C (``gpu``): grasp contact parity (close the gripper on the die, lift, hold).
* STUCK (plan §10 2026-10-08): a wall press that trips the CPU servo's stuck-action monitor; the recorded stream
  (present + Moving flags) trips the chain on the same tick (CPU), the batched env closed loop within +-1 policy
  step (``gpu``). Layer A also feeds the recorded Moving flags, so its trip comparison includes STUCK.

Every layer has a "shown to fail" twin. Layer B/C live further down this file.

CPU rig: booted like ``tests/test_action_units.py::Rig`` (button + ARM + re-energise), then for every sequence the
physics is reset to the ``home`` keyframe (sim time kept monotonic) and the pipeline resynced to the present ticks,
exactly what ``BatchedGoalEnv.reset`` does. One rig per module (boot once).

NaN rows (corpus f): the CPU rig receives a ZERO delta on those steps so it stays in RUN (a NaN would end its RUN
for good: FRESH_PROCESS); the chains under test receive the NaN. Layer A only needs a valid present stream.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from so101_rl.config import N_JOINTS, load_calibration, load_config
from so101_rl.env.tasks import action_to_delta_np
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.pipeline import SafetyPipeline
from so101_rl.safety.stuck import StuckActionMonitor
from so101_rl.safety.thermal import ThermalMonitor
from so101_rl.safety.types import SmState, TripCause, Verdict
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.core import build_servo_core
from so101_rl.servo.heartbeat import FakeHeartbeatLink, FakeMcu
from so101_rl.servo.shm import CmdMode, LocalIpc, new_cmd, new_state
from so101_rl.servo.timing import FakeClock
from so101_rl.sim.batched_servo import CHAIN_TRIP_BITS, STUCK_REPEAT_TICKS, BatchedServoChain, ChainOut
from so101_rl.sim.model import jaw_collision_geom_ids, load_model
from so101_rl.sim.mujoco_bus import MujocoBus

NS = 1_000_000_000
DATA = Path(__file__).parent / "data"
STEPS = 200
TICKS = 3 * STEPS
N_SEQ = 20
# corpus kinds per sequence index (plan: a uniform, b saturating/sign flips, c joint limit, d box wall/floor,
# e alternating +-1, f NaN rows)
KIND = ["a"] * 6 + ["b"] * 3 + ["c"] * 2 + ["d"] * 3 + ["e"] * 3 + ["f"] * 3
NAN_STEPS = (50, 100, 150)
CPU = torch.device("cpu")


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def mcal():
    return load_calibration(DATA / "so101_follower.calib.json")


@pytest.fixture(scope="module")
def mcfg(mcal):
    return load_config(calibration=mcal)


@pytest.fixture(scope="module")
def kchain(mcfg):
    return KinematicChain.from_yaml(mcfg.kinematics_path)


# ------------------------------------------------------------------------------------------------ corpus
def _plan_box_sequence(cfg, cal, chain, seed: int) -> np.ndarray:
    """(d): drive the tip +x into the box wall at x = 0.34 (beyond it, so the BOX clamp is active), then down
    through the box floor. Open loop: computed once on an ideal kinematic plant (present = target) through the
    batched chain itself, with a numeric Jacobian; the result is a fixed action sequence."""
    rng = np.random.default_rng(seed)
    bc = BatchedServoChain(cfg, cal, chain, 1, device=CPU)
    out = ChainOut.empty(1, CPU)
    home = torch.as_tensor(np.asarray(cfg.safe_state.home_ticks, np.int64))[None]
    bc.resync(home)
    md = np.asarray(cfg.action.max_delta_ticks, np.float64)
    xyz = torch.zeros(1, 3, dtype=torch.float64)
    acts = np.zeros((STEPS, N_JOINTS), np.float32)
    present = home.clone()
    split = 50 + int(rng.integers(0, 20))
    for i in range(STEPS):
        g = bc.goal.clone()
        x0 = bc.fk_tip(g, xyz).clone()[0].numpy()
        J = np.zeros((3, 4))
        for k, j in enumerate((0, 1, 2, 3)):
            gp = g.clone()
            gp[0, j] += 10
            J[:, k] = (bc.fk_tip(gp, xyz)[0].numpy() - x0) / 10
        y = 0.06 * (seed - 12)  # the three (d) sequences approach the wall at y = -0.06, 0, +0.06
        if i < split:
            want = np.array([0.36 - x0[0], y - x0[1], 0.0])
        else:
            want = np.array([0.325 - x0[0], y - x0[1], -0.03 - x0[2]])
        want = np.clip(want, -0.02, 0.02)
        dq = np.linalg.lstsq(J, want, rcond=None)[0]
        a = np.zeros(N_JOINTS)
        a[[0, 1, 2, 3]] = dq / md[[0, 1, 2, 3]]
        a /= max(1.0, float(np.abs(a).max()))  # keep the direction: scale, do not clip per joint
        a[5] = 0.3 * np.sin(i / 7.0)  # some gripper motion
        acts[i] = np.clip(a, -1.0, 1.0)
        bc.new_command(torch.as_tensor(np.rint(acts[i] * md.astype(np.float32)))[None])
        for rem in (3, 2, 1):
            bc.apply(present, rem, out)
            present = out.target_ticks.clone()
    return acts


def corpus(cfg, cal, chain, idx: int) -> np.ndarray:
    """Action sequence ``idx`` (0..19): float32[STEPS, 6] in [-1, 1] (NaN rows for kind f). Seeded generator, so
    the test data stays small."""
    kind = KIND[idx]
    rng = np.random.default_rng(1234 + idx)
    if kind == "a":
        return rng.uniform(-1, 1, size=(STEPS, N_JOINTS)).astype(np.float32)
    if kind == "b":  # saturating +-1 bursts, sign flip every 3 steps, random joint subset per burst
        acts = np.zeros((STEPS, N_JOINTS), np.float32)
        sign = rng.choice([-1.0, 1.0], size=N_JOINTS)
        for i in range(STEPS):
            if i % 3 == 0:
                sign = -sign
                on = rng.uniform(size=N_JOINTS) < 0.6
            acts[i] = sign * on
        return acts
    if kind == "c":  # shoulder_lift ramped into its lower joint limit and held (with a STUCK-breaking wobble)
        acts = rng.uniform(-0.15, 0.15, size=(STEPS, N_JOINTS)).astype(np.float32) * (idx == 10)
        acts[:, 1] = np.where(np.arange(STEPS) % 21 == 20, -0.9, -1.0) * (0.5 if idx == 10 else 1.0)
        acts[:, 1] = np.minimum(acts[:, 1], 0)
        return acts
    if kind == "d":
        return _plan_box_sequence(cfg, cal, chain, idx)
    if kind == "e":  # alternating +-1 on all joints for 30 steps, then uniform
        acts = rng.uniform(-0.5, 0.5, size=(STEPS, N_JOINTS)).astype(np.float32)
        start = int(rng.integers(0, 40))
        alt = np.where(np.arange(30) % 2 == 0, 1.0, -1.0)[:, None]
        acts[start : start + 30] = alt
        return acts
    acts = rng.uniform(-1, 1, size=(STEPS, N_JOINTS)).astype(np.float32)  # f
    for s, j in zip(NAN_STEPS, rng.integers(0, N_JOINTS, size=3), strict=True):
        acts[s, j] = np.nan
    return acts


def action_to_delta(acts: np.ndarray, cfg) -> np.ndarray:
    """F7 rule, the one both envs use (``tasks.action_to_delta_np`` / ``_torch``)."""
    return action_to_delta_np(acts, cfg.action.max_delta_ticks)


# ------------------------------------------------------------------------------------------------ CPU rig
class SimClock:
    """MuJoCo-time clock (tests/test_action_units.py): ``sleep_until`` steps the physics."""

    def __init__(self, bus: MujocoBus) -> None:
        self.bus = bus
        self.period_ns = round(bus.steps_per_tick * bus.model.opt.timestep * NS)
        self._offset = 1_000 * NS

    def monotonic_ns(self) -> int:
        return round(self.bus.data.time * NS) + self._offset

    def sleep_until(self, t_ns: int) -> None:
        n = -(-(t_ns - self.monotonic_ns()) // self.period_ns)
        if n > 0:
            self.bus.step_ticks(n)


@dataclasses.dataclass
class CpuRun:
    present: np.ndarray  # int64[TICKS, 6] read at the start of each tick
    target: np.ndarray  # int64[TICKS, 6]
    goal: np.ndarray
    applied: np.ndarray  # float32[TICKS, 6]
    mask: np.ndarray  # bool[TICKS, 6]
    trips: np.ndarray  # int64[TICKS]
    in_run: np.ndarray  # bool[TICKS]: the core applied the pipeline on this tick (state RUN)
    qpos: np.ndarray  # float64[TICKS, nq] after the tick's physics
    jaw_table: np.ndarray  # int64[TICKS] contacts jaw_col <-> table after the tick's physics
    deltas: np.ndarray  # float32[STEPS, 6] what the chains get (NaN rows kept)
    moving: np.ndarray  # uint8[TICKS, 6] the Moving flags read with ``present`` (MujocoBus: |qvel| > 1e-3 rad/s)


class CpuRig:
    """ServoCore + MujocoBus + LocalIpc + FakeHeartbeatLink(FakeMcu) booted to RUN once; reusable per sequence."""

    def __init__(self, cfg, cal, *, die_mass_scale: float = 1.0) -> None:
        self.cfg = cfg
        model, data = load_model()
        if die_mass_scale != 1.0:
            b = model.body("die").id
            model.body_mass[b] *= die_mass_scale
            model.body_inertia[b] *= die_mass_scale
        self.model, self.data = model, data
        self.bus = MujocoBus(model, data, cal, cfg)
        self.bus.connect()
        self.bus.reset_to_keyframe("rest")
        self.bus.set_torque(False)
        self.clock = SimClock(self.bus)
        self.mcu = FakeMcu(self.clock)
        self.ipc = LocalIpc()
        self.core = build_servo_core(cfg, cal, self.bus, self.ipc, FakeHeartbeatLink(self.mcu), self.clock, log=lambda m: None)
        self.period = self.core.period_ns
        self.cmd = new_cmd()
        self.st = new_state()
        self.mcu.set_button(True)
        self.core.start(arm_timeout_s=5.0)
        self.mcu.set_button(False)
        assert self.core.state is SmState.RUN
        self.jaw = set(jaw_collision_geom_ids(model).tolist())
        self.table = model.geom("table").id
        self.die_geom = model.geom("die_geom").id

    def reset(self, key: str = "home", *, qpos: np.ndarray | None = None, ctrl: np.ndarray | None = None) -> np.ndarray:
        """Physics to ``key`` (or explicit qpos/ctrl), zero velocity, sim time kept; pipeline resynced to present."""
        m, d = self.model, self.data
        t = d.time
        mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
        if qpos is not None:
            d.qpos[:] = qpos
        if ctrl is not None:
            d.ctrl[:] = ctrl
        d.time = t
        mujoco.mj_forward(m, d)
        assert self.core.state is SmState.RUN, "CPU rig left RUN in an earlier sequence"
        present = self.bus.q_to_ticks(d.qpos[self.bus.qpos_adr])
        self.core.pipeline.resync(present)
        return present.astype(np.int64)

    def contacts(self, set_a: set[int], b: int) -> int:
        d = self.data
        n = 0
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            n += (g1 in set_a and g2 == b) or (g2 in set_a and g1 == b)
        return n

    def send(self, delta: np.ndarray) -> None:
        self.ipc.read_state(self.st)
        self.cmd["state_seq"] = self.st["seq"][0]
        self.cmd["t_mono_ns"] = self.clock.monotonic_ns()
        self.cmd["mode"] = int(CmdMode.RUN)
        self.cmd["delta_ticks"][0] = np.asarray(delta, dtype=np.float32)
        self.ipc.write_cmd(self.cmd)

    def run(self, acts: np.ndarray, *, die_contacts: bool = False) -> CpuRun:
        deltas = action_to_delta(acts, self.cfg)
        n = len(acts) * 3
        r = CpuRun(
            present=np.zeros((n, 6), np.int64),
            target=np.zeros((n, 6), np.int64),
            goal=np.zeros((n, 6), np.int64),
            applied=np.zeros((n, 6), np.float32),
            mask=np.zeros((n, 6), bool),
            trips=np.zeros(n, np.int64),
            in_run=np.zeros(n, bool),
            qpos=np.zeros((n, self.model.nq)),
            jaw_table=np.zeros(n, np.int64),
            deltas=deltas,
            moving=np.zeros((n, 6), np.uint8),
        )
        other = self.die_geom if die_contacts else self.table
        for i, dl in enumerate(deltas):
            self.send(np.zeros(N_JOINTS, np.float32) if np.isnan(dl).any() else dl)
            for k in range(3):
                t = 3 * i + k
                t0 = self.clock.monotonic_ns()
                was_run = self.core.state is SmState.RUN
                res = self.core.tick()
                self.clock.sleep_until(t0 + self.period)
                self.ipc.read_state(self.st)
                s = self.st[0]
                r.present[t] = s["present_ticks"]
                r.moving[t] = s["moving"]
                r.target[t] = s["target_ticks"]
                r.goal[t] = s["goal_ticks"]
                r.applied[t] = s["applied_delta_ticks"]
                r.mask[t] = s["clamped_mask"].astype(bool)
                r.trips[t] = int(res.trips)
                r.in_run[t] = was_run and res.state is SmState.RUN
                r.qpos[t] = self.data.qpos
                r.jaw_table[t] = self.contacts(self.jaw, other)
        return r


@pytest.fixture(scope="module")
def cpu_rig(mcfg, mcal):
    return CpuRig(mcfg, mcal)


_RUNS: dict[int, CpuRun] = {}


def cpu_run(rig: CpuRig, cfg, cal, chain, idx: int) -> CpuRun:
    """Closed-loop CPU run of sequence ``idx`` from ``home`` (cached per module)."""
    if idx not in _RUNS:
        rig.reset("home")
        _RUNS[idx] = rig.run(corpus(cfg, cal, chain, idx))
    return _RUNS[idx]


# ------------------------------------------------------------------------------------------------ Layer A
class NpChain:
    """A fresh numpy SafetyPipeline driven by an explicit present stream (the reference for Layer A)."""

    def __init__(self, cfg, cal, chain, present0: np.ndarray) -> None:
        clock = FakeClock(start_ns=100 * NS)
        self.clock = clock
        self.pipe = SafetyPipeline(cfg, So101FK(chain, cal), ThermalMonitor(cfg.thermal, clock), StuckActionMonitor())
        self.block = BlockRead()
        self.block.temperature_c[...] = 35
        self.block.moving[...] = 1  # overwritten per tick by ``tick(..., moving)`` (STUCK emulated, 2026-10-08)
        self.block.ok = True
        self.block.position[...] = present0
        self.v = Verdict()
        self.pipe.resync(self.block.position)

    def command(self, delta: np.ndarray) -> None:
        self.pipe.new_command(np.asarray(delta, np.float32), self.v)

    def tick(self, present: np.ndarray, rem: int, moving: np.ndarray) -> Verdict:
        self.clock.advance(33_333_333)
        self.block.position[...] = present
        self.block.moving[...] = moving
        return self.pipe.apply(self.block, rem, self.v)


def layer_a(cfg, cal, chain, run: CpuRun, *, nworld: int, world: int, vmax_offset: int = 0) -> dict:
    """Feed the recorded stream to NpChain and BatchedServoChain; count mismatches per field."""
    ref = NpChain(cfg, cal, chain, run.present[0])
    bc = BatchedServoChain(cfg, cal, chain, nworld, device=CPU)
    if vmax_offset:
        bc.max_ticks_per_tick += vmax_offset
    out = ChainOut.empty(nworld, CPU)
    rng = np.random.default_rng(7)
    # the other worlds get their own (different) streams: a cross-world leak would show as a mismatch
    other_p = np.clip(run.present[0] + np.cumsum(rng.integers(-20, 21, size=(len(run.present), 6)), 0), 0, 4095)
    other_d = rng.uniform(-60, 60, size=(len(run.deltas), 6)).astype(np.float32)
    bc.resync(torch.as_tensor(np.tile(run.present[0], (nworld, 1))))
    counts = {k: 0 for k in ("target", "goal", "mask", "trips", "applied", "nan_trips", "ticks", "core_mismatch")}
    for i, dl in enumerate(run.deltas):
        ref.command(dl)
        d = np.tile(other_d[i], (nworld, 1))
        d[world] = dl
        bc.new_command(torch.as_tensor(d))
        for k, rem in enumerate((3, 2, 1)):
            t = 3 * i + k
            v = ref.tick(run.present[t], rem, run.moving[t])
            p = np.tile(other_p[t], (nworld, 1))
            p[world] = run.present[t]
            mv = np.ones((nworld, N_JOINTS), np.uint8)  # the other worlds move (their streams are random walks)
            mv[world] = run.moving[t]
            bc.apply(torch.as_tensor(p), rem, out, moving=torch.as_tensor(mv))
            w = world
            counts["ticks"] += 1
            counts["target"] += int(not np.array_equal(v.target_ticks, out.target_ticks[w].numpy()))
            counts["goal"] += int(not np.array_equal(v.goal_ticks, out.goal_ticks[w].numpy()))
            counts["mask"] += int(not np.array_equal(v.clamped_mask, out.clamped_mask[w].numpy()))
            counts["trips"] += int((int(v.trips) & CHAIN_TRIP_BITS) != int(out.trips[w]))
            counts["applied"] += int(not np.array_equal(v.applied_delta_ticks, out.applied_delta_ticks[w].numpy()))
            if np.isnan(dl).any() and k == 0:
                # both report NAN; the goal left untouched is covered by the goal comparison
                counts["nan_trips"] += int(bool(int(v.trips) & TripCause.NAN) and bool(int(out.trips[w]) & TripCause.NAN))
            # harness sanity: while the CPU core was in RUN the fresh pipeline reproduces its records
            if run.in_run[t] and not np.isnan(dl).any():
                counts["core_mismatch"] += int(not np.array_equal(v.target_ticks, run.target[t]))
    return counts


@pytest.mark.parametrize("idx", range(N_SEQ), ids=[f"seq{i}{KIND[i]}" for i in range(N_SEQ)])
def test_layer_a_chain_bit_identical(cpu_rig, mcfg, mcal, kchain, idx):
    run = cpu_run(cpu_rig, mcfg, mcal, kchain, idx)
    for nworld, world in ((1, 0), (4, 2)):
        c = layer_a(mcfg, mcal, kchain, run, nworld=nworld, world=world)
        assert c["ticks"] == TICKS
        bad = {k: c[k] for k in ("target", "goal", "mask", "trips", "applied")}
        assert not any(bad.values()), f"N={nworld}: mismatching ticks {bad}"
        assert c["core_mismatch"] == 0, "harness: the fresh pipeline does not reproduce the CPU core's targets"
        if KIND[idx] == "f":
            assert c["nan_trips"] == len(NAN_STEPS)


def test_layer_a_exercises_every_stage(cpu_rig, mcfg, mcal, kchain):
    """The corpus is only adversarial if it activates every stage the chain has (else 0 mismatches is vacuous)."""
    seen = 0
    totals = []
    for idx in range(N_SEQ):
        run = cpu_run(cpu_rig, mcfg, mcal, kchain, idx)
        bc = BatchedServoChain(mcfg, mcal, kchain, 1, device=CPU)
        out = ChainOut.empty(1, CPU)
        bc.resync(torch.as_tensor(run.present[:1]))
        for i, dl in enumerate(run.deltas):
            bc.new_command(torch.as_tensor(dl[None]))
            for k, rem in enumerate((3, 2, 1)):
                bc.apply(torch.as_tensor(run.present[3 * i + k][None]), rem, out)
                seen |= int(out.trips[0])
        frac = bc.clamp_active_fraction
        if KIND[idx] == "c":
            assert frac["joint_limit"] > 0 or seen & TripCause.JOINT_LIMIT
        if KIND[idx] == "d":
            assert frac["box"] > 0, f"seq{idx}: the box sequence never hit the box"
        totals.append(frac)
    for cause in (TripCause.NAN, TripCause.JOINT_LIMIT, TripCause.BOX):
        assert seen & cause, f"no sequence activated {cause!r}"
    # TRACKING / velocity: the simulated servos (kp 998) follow within E on this corpus, so these stages are
    # covered bit-exactly by the adversarial present streams of tests/test_batched_servo.py instead (reported).
    print("clamp-active fraction per sequence:", [{k: round(v, 4) for k, v in f.items()} for f in totals])


def test_layer_a_gripper_floor_is_exercised(cpu_rig, mcfg, mcal, kchain):
    """Layer A covers the gripper floor (plan §10 2026-10-06 / 2026-10-07): on the box sequences (d) a chain with the
    gripper floor moved out of the way emits different targets, i.e. the point floor decided some ticks (and Layer A
    above is bit-identical WITH it)."""
    no_floor = dataclasses.replace(mcfg, box=dataclasses.replace(mcfg.box, gripper_floor_z_m=-1.0))
    differ = 0
    for idx in [i for i in range(N_SEQ) if KIND[i] == "d"]:
        run = cpu_run(cpu_rig, mcfg, mcal, kchain, idx)
        chains = [BatchedServoChain(c, mcal, kchain, 1, device=CPU) for c in (mcfg, no_floor)]
        outs = [ChainOut.empty(1, CPU) for _ in chains]
        for bc in chains:
            bc.resync(torch.as_tensor(run.present[:1]))
        for i, dl in enumerate(run.deltas):
            for bc in chains:
                bc.new_command(torch.as_tensor(dl[None]))
            for k, rem in enumerate((3, 2, 1)):
                p = torch.as_tensor(run.present[3 * i + k][None])
                for bc, out in zip(chains, outs, strict=True):
                    bc.apply(p, rem, out)
                differ += int(not torch.equal(outs[0].target_ticks, outs[1].target_ticks))
    print(f"ticks decided by the gripper floor on the (d) sequences: {differ}")
    assert differ > 0


def test_layer_a_shown_to_fail_with_perturbed_vmax(cpu_rig, mcfg, mcal, kchain):
    """The gate can fail: the torch chain with ``max_ticks_per_tick`` perturbed (only there) mismatches."""
    total = 0
    for idx in (6, 7, 14):  # saturating / alternating sequences
        run = cpu_run(cpu_rig, mcfg, mcal, kchain, idx)
        for off in (+1, -1):
            c = layer_a(mcfg, mcal, kchain, run, nworld=1, world=0, vmax_offset=off)
            total += c["target"]
    assert total > 0


# ------------------------------------------------------------------------------------------------ Layer B
gpu = pytest.mark.gpu
CONTACT_FREE = ("a", "b", "e")
_B_RESULTS: dict[int, dict] = {}


def _parity_env(cfg, cal, *, nworld: int = 4, substep_offset: int = 0, task=None, model=None):
    """BatchedGoalEnv for the parity rig: REACH with an endless episode and an unreachable goal (no reset during
    the 200 steps), no start noise, delay 0, no tracker."""
    from so101_rl.env.batched_env import BatchedGoalEnv, model_sim_params
    from so101_rl.env.tasks import REACH

    model = model if model is not None else load_model()[0]
    task = task if task is not None else dataclasses.replace(REACH, episode_steps=10**6, trip_penalty=-1e7)
    env = BatchedGoalEnv(cfg, cal, task, nworld, sim_params=model_sim_params(model), model=model, seed=0, allow_provisional_camera=True)
    env.start_noise_ticks = 0
    env.steps_per_tick += substep_offset
    return env


def _run_batched(env, acts: np.ndarray, *, die_contacts: bool = False, start=None) -> dict[str, np.ndarray]:
    """Closed loop from home (or ``start = (qpos, ctrl)``); every world gets the same sequence. Per tick (world
    0): present, target, qpos (before the tick's physics), jaw contacts; per step: the obs present and trips.
    Stops at the first truncation."""
    from so101_rl.env.obs import CENTER_TICK

    env.reset(phase_randomise=False)
    if start is not None:
        env.reset_to_state(*start)
    env.goal.fill_(5.0)
    env.trace = []
    n = env.nworld
    q_norm = env.task.obs.q_norm
    obs_present, step_trips = [], []
    for a in acts:
        st = env.step(torch.as_tensor(np.tile(a, (n, 1)), device=env.device))
        obs_present.append(torch.round(st.obs[0, q_norm].double() * 2048 + CENTER_TICK).long().cpu().numpy())
        step_trips.append(int(st.info["trips"][0]))
        if bool(st.truncated[0]):
            break
    tr = env.trace
    env.trace = None
    key = "jaw_die" if die_contacts else "jaw_table"
    return {
        "present": torch.stack([r["present"][0] for r in tr]).cpu().numpy(),
        "target": torch.stack([r["target"][0] for r in tr]).cpu().numpy(),
        "qpos_all": torch.stack([r["qpos"] for r in tr]).double().cpu().numpy(),  # [T, N, nq] before the physics
        "contacts": torch.stack([r[key][0] for r in tr]).cpu().numpy(),  # of the step before the read
        "obs_present": np.array(obs_present),
        "step_trips": np.array(step_trips),
    }


def _layer_b(cpu: CpuRun, gpu_r: dict, kind: str) -> dict:
    """Compare per tick while both rigs are valid. qpos / contacts: the CPU records them AFTER the tick's physics,
    the batched trace BEFORE it, so CPU tick t pairs with batched tick t + 1."""
    T = len(gpu_r["present"])
    valid = int(np.argmin(cpu.in_run)) if not cpu.in_run.all() else len(cpu.in_run)
    if kind == "f":
        valid = min(valid, 3 * NAN_STEPS[0])
    T = min(T, valid)
    pc, pw = cpu.present[:T], gpu_r["present"][:T]
    tc, tw = cpu.target[:T], gpu_r["target"][:T]
    same_p = (pc == pw).all(axis=1)
    target_diff = np.abs(tc - tw)
    qn = T - 1
    q_cpu = cpu.qpos[:qn, :6]
    q_gpu = gpu_r["qpos_all"][1 : qn + 1, 0, :6]
    present_diff = np.abs(pc - pw)
    steps = T // 3
    obs_cpu_k3 = cpu.present[2 : 3 * steps : 3]  # the 3rd tick's record of every step
    obs_cpu_next = cpu.present[3 : 3 * steps + 1 : 3]  # the next step's first record (after 3 full ticks)
    obs_w = gpu_r["obs_present"][: len(obs_cpu_k3)]
    k = min(len(obs_w), len(obs_cpu_next))
    nan_ok = None
    if kind == "f" and len(gpu_r["step_trips"]) > NAN_STEPS[0]:
        nan_ok = bool(gpu_r["step_trips"][NAN_STEPS[0]] & int(TripCause.NAN))
    return {
        "ticks": T,
        "target_mismatch_on_equal_present": int((target_diff[same_p] != 0).any(axis=1).sum()),
        "max_target_diff": int(target_diff.max()) if T else 0,
        "target_diff_frac": float((target_diff != 0).mean()) if T else 0.0,
        "max_present_diff": int(present_diff.max()) if T else 0,
        "present_diff_hist": {int(a): int(b) for a, b in zip(*np.unique(present_diff, return_counts=True), strict=True)},
        "qpos_max": float(np.abs(q_cpu - q_gpu).max()) if qn > 0 else 0.0,
        "qpos_max_per_joint": np.abs(q_cpu - q_gpu).max(axis=0).round(7).tolist() if qn > 0 else [],
        "qpos_argmax_tick": int(np.abs(q_cpu - q_gpu).max(axis=1).argmax()) if qn > 0 else 0,
        "contacts_equal": bool((cpu.jaw_table[:qn] == gpu_r["contacts"][1 : qn + 1]).all()),
        "contacts_cpu_max": int(cpu.jaw_table[:T].max()) if T else 0,
        # only steps whose tick-3 present agrees between the rigs (a +-1 tick quantisation difference is rule (2))
        "obs_vs_tick3_mismatch": int(
            ((obs_w != obs_cpu_k3[: len(obs_w)]).any(axis=1) & same_p[2 : 3 * len(obs_w) : 3]).sum()
        ),
        "obs_vs_next_record_mismatch": int((obs_w[:k] != obs_cpu_next[:k]).any(axis=1).sum()),
        "obs_steps": len(obs_w),
        "nan_trip_at_first_nan": nan_ok,
    }


@pytest.fixture(scope="module")
def parity_env(mcfg, mcal):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    env = _parity_env(mcfg, mcal)
    yield env
    env.close()


def _write_report(name: str, payload: dict) -> Path:
    out = Path("runs/parity")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


@gpu
@pytest.mark.parametrize("idx", range(N_SEQ), ids=[f"seq{i}{KIND[i]}" for i in range(N_SEQ)])
def test_layer_b_closed_loop(cpu_rig, parity_env, mcfg, mcal, kchain, idx):
    kind = KIND[idx]
    cpu = cpu_run(cpu_rig, mcfg, mcal, kchain, idx)
    r = _layer_b(cpu, _run_batched(parity_env, corpus(mcfg, mcal, kchain, idx)), kind)
    _B_RESULTS[idx] = r
    print(f"seq{idx}{kind}: {r}")
    assert r["ticks"] >= (3 * NAN_STEPS[0] if kind == "f" else TICKS - 3)
    assert r["target_mismatch_on_equal_present"] == 0  # (1)
    assert r["max_target_diff"] <= 1 and r["target_diff_frac"] <= 0.001  # (2)
    if kind == "d":
        assert r["contacts_equal"]  # (4)
    # observation timing: the policy sees the 3rd tick's record (present after 2 ticks of the command)
    assert r["obs_vs_tick3_mismatch"] == 0
    if kind == "f":
        assert r["nan_trip_at_first_nan"]


# (3) fails on these sequences (measured 2026-10-06, CPU f64 vs Warp f32; targets stayed IDENTICAL on every tick):
# seq6b 9.8e-4 rad and seq7b 1.1e-3 rad (shoulder vs lower_arm mesh self-contact plus the MJCF joint-range limit of
# shoulder_lift: MJCF min = tick 909, calibrated limit 869), seq15e 1.7e-3 rad (alternating +-1: the actuators sit
# in their force saturation +-3.35 N m on almost every tick). Smooth sequences agree to 1e-6..8e-5 rad. Not loosened:
# a user decision (plan §10 2026-10-06 "Layer B parity tolerance"): strict xfails, only these three, each capped
# at _B3_XFAIL_CEILING_RAD (2e-3 rad) so worse drift fails outright. With the gripper floor check (2026-10-07)
# seq7b no longer touches the table (it drifted 4.9e-3 rad with jaw-table contact under the tip-only floor).
_B3_XFAIL = {6: "self-contact + MJCF joint limit", 7: "self-contact + MJCF joint limit", 15: "actuator force saturation"}
_B3_SEQ = [i for i in range(N_SEQ) if KIND[i] in (*CONTACT_FREE, "d")]
# Ceiling for the xfail sequences: the drift may exceed 1e-4 rad (strict xfail) but never this (else a hard failure).
_B3_XFAIL_CEILING_RAD = 2e-3


@gpu
@pytest.mark.parametrize(
    "idx",
    [
        pytest.param(
            i,
            marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"Layer B (3) > 1e-4 rad: {_B3_XFAIL[i]}"),
        )
        if i in _B3_XFAIL
        else i
        for i in _B3_SEQ
    ],
    ids=[f"seq{i}{KIND[i]}" for i in _B3_SEQ],
)
def test_layer_b_trajectory(cpu_rig, parity_env, mcfg, mcal, kchain, idx):
    """(3): arm qpos within 1e-4 rad on the contact-free sequences (a, b, e), 5e-4 rad on the box sequences (d)."""
    if idx not in _B_RESULTS:
        cpu = cpu_run(cpu_rig, mcfg, mcal, kchain, idx)
        _B_RESULTS[idx] = _layer_b(cpu, _run_batched(parity_env, corpus(mcfg, mcal, kchain, idx)), KIND[idx])
    r = _B_RESULTS[idx]
    # the documented ceiling of the accepted xfails (plan §10 2026-10-06, max measured 1.7e-3 rad): worse drift is a
    # real failure, raised as RuntimeError so the strict xfail(raises=AssertionError) does NOT absorb it
    if idx in _B3_XFAIL and not r["qpos_max"] <= _B3_XFAIL_CEILING_RAD:
        raise RuntimeError(f"seq{idx}: qpos drift {r['qpos_max']} rad exceeds the xfail ceiling {_B3_XFAIL_CEILING_RAD}")
    bound = 5e-4 if KIND[idx] == "d" else 1e-4
    assert r["qpos_max"] <= bound, (r["qpos_max"], r["qpos_max_per_joint"])


@gpu
def test_layer_b_report(cpu_rig, parity_env, mcfg, mcal, kchain):
    """Report (not assert) the per-sequence numbers and the present-difference histogram (runs/parity/)."""
    for idx in range(N_SEQ):
        if idx not in _B_RESULTS:
            cpu = cpu_run(cpu_rig, mcfg, mcal, kchain, idx)
            _B_RESULTS[idx] = _layer_b(cpu, _run_batched(parity_env, corpus(mcfg, mcal, kchain, idx)), KIND[idx])
    path = _write_report("layer-b", {f"seq{i}{KIND[i]}": _B_RESULTS[i] for i in range(N_SEQ)})
    print(f"Layer B report: {path}")
    # the 2-vs-3 tick observation offset is detectable on this corpus (else the timing check is vacuous)
    assert sum(r["obs_vs_next_record_mismatch"] for r in _B_RESULTS.values()) > 100


@gpu
def test_layer_b_shown_to_fail_substeps(cpu_rig, mcfg, mcal, kchain):
    """The gate can fail: the batched env with 18 instead of 17 substeps per tick exceeds the trajectory bound."""
    env = _parity_env(mcfg, mcal, substep_offset=1)
    try:
        cpu = cpu_run(cpu_rig, mcfg, mcal, kchain, 0)
        r = _layer_b(cpu, _run_batched(env, corpus(mcfg, mcal, kchain, 0)), KIND[0])
    finally:
        env.close()
    print(f"substeps 18: {r}")
    assert r["qpos_max"] > 1e-4


# ------------------------------------------------------------------------------------------------ Layer C
# Grasp pose (found by a kinematic search, S1.3): gripper pointing down over the table, jaw tips 1-2 mm above it,
# the die resting between the open jaws (6 mm nearer the fixed jaw than the midpoint, faces aligned with the
# closing direction). Arm ticks pan..roll, gripper ticks.
GRASP_ARM_TICKS = (2047, 2110, 2410, 2640, 2047)
GRASP_GRIPPER_OPEN = 2350
GRASP_DIE_OFFSET_M = 0.006
DIE_REST_Z = -0.012


def _grasp_state(model, cal) -> tuple[np.ndarray, np.ndarray]:
    from so101_rl.sim.gripper_map import gripper_ticks_to_rad

    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("home").id)
    qa = (np.asarray(GRASP_ARM_TICKS, np.float64) - 2047) * (2 * np.pi / 4096)
    qg = gripper_ticks_to_rad(GRASP_GRIPPER_OPEN, cal)
    d.qpos[:5], d.qpos[5] = qa, qg
    mujoco.mj_kinematics(model, d)
    pf = d.geom_xpos[model.geom("jaw_fixed_tip").id].copy()
    pm = d.geom_xpos[model.geom("jaw_moving_tip").id].copy()
    u = (pm - pf) / np.linalg.norm(pm - pf)
    mid = (pf + pm) / 2 - GRASP_DIE_OFFSET_M * u
    yaw = np.arctan2(u[1], u[0])
    q = d.qpos.copy()
    dq = model.jnt_qposadr[model.joint("die_free").id]
    q[dq : dq + 3] = (mid[0], mid[1], DIE_REST_Z)
    q[dq + 3 : dq + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    ctrl = np.concatenate([qa, [qg]])
    return q, ctrl


def _grasp_actions() -> np.ndarray:
    """Close the gripper over 20 steps, lift 10 steps (shoulder_lift), hold 20 steps (gripper kept closing)."""
    acts = np.zeros((50, N_JOINTS), np.float32)
    acts[:, 5] = -0.3
    acts[20:30, 1] = -0.5
    return acts


def _die_tip_dist(model, qpos: np.ndarray) -> float:
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(model, d)
    dq = model.jnt_qposadr[model.joint("die_free").id]
    return float(np.linalg.norm(d.qpos[dq : dq + 3] - d.site_xpos[model.site("gripper_tip").id]))


def _layer_c(cfg, cal, *, cpu_die_mass_scale: float = 1.0) -> dict:
    # the SHIPPED config: since plan §10 2026-10-06 the box floor is table + 5 mm (-0.015), so the grasp pose
    # (FK tip z ~ -0.011) is inside the box (the test-local floor override of S1.3 is gone)
    gcfg = cfg
    rig = CpuRig(gcfg, cal, die_mass_scale=cpu_die_mass_scale)
    q, ctrl = _grasp_state(rig.model, cal)
    rig.reset("home", qpos=q, ctrl=ctrl)
    acts = _grasp_actions()
    cpu = rig.run(acts, die_contacts=True)
    env = _parity_env(gcfg, cal)
    try:
        g = _run_batched(env, acts, die_contacts=True, start=(q, ctrl))
        g_final = env.ws.qpos[0].double().cpu().numpy()
    finally:
        env.close()
    model = rig.model
    dq = int(model.jnt_qposadr[model.joint("die_free").id])
    T = min(len(cpu.present), len(g["present"]))
    die_cpu = cpu.qpos[: T - 1, dq : dq + 3]
    die_gpu = g["qpos_all"][1:T, 0, dq : dq + 3]
    c_cpu = cpu.jaw_table[: T - 1] > 0  # (jaw_table holds jaw-die counts with die_contacts=True)
    c_gpu = g["contacts"][1:T] > 0
    first_cpu = int(np.argmax(c_cpu)) if c_cpu.any() else -1
    first_gpu = int(np.argmax(c_gpu)) if c_gpu.any() else -1
    lo, hi = sorted((first_cpu, first_gpu))
    differ = c_cpu != c_gpu
    differ[max(lo - 1, 0) : hi + 2] = False  # +-1 tick tolerance around first contact
    return {
        "ticks": T,
        "cpu_in_run": bool(cpu.in_run.all()),
        "die_tip_cpu_m": _die_tip_dist(model, cpu.qpos[-1]),
        "die_tip_gpu_m": _die_tip_dist(model, g_final),
        "die_z_end_cpu_m": float(cpu.qpos[-1, dq + 2]),
        "die_z_end_gpu_m": float(g_final[dq + 2]),
        "die_pos_max_diff_m": float(np.abs(die_cpu - die_gpu).max()),
        "die_pos_diff_at_end_m": float(np.abs(cpu.qpos[-1, dq : dq + 3] - g_final[dq : dq + 3]).max()),
        "first_contact_cpu": first_cpu,
        "first_contact_gpu": first_gpu,
        "contact_status_mismatch_ticks": int(differ.sum()),
        "contact_ticks_cpu": int(c_cpu.sum()),
        "contact_ticks_gpu": int(c_gpu.sum()),
        "gripper_present_end": (int(cpu.present[-1, 5]), int(g["present"][-1, 5])),
        "arm_qpos_max_diff": float(np.abs(cpu.qpos[: T - 1, :6] - g["qpos_all"][1:T, 0, :6]).max()),
    }


def _layer_c_pass(r: dict) -> bool:
    return (
        r["die_tip_cpu_m"] < 0.02
        and r["die_tip_gpu_m"] < 0.02
        and r["die_pos_max_diff_m"] <= 2e-3
        and r["first_contact_cpu"] >= 0
        and abs(r["first_contact_cpu"] - r["first_contact_gpu"]) <= 1
        and r["contact_status_mismatch_ticks"] == 0
    )


@gpu
def test_layer_c_grasp_contact_parity(mcfg, mcal):
    """Grasp contact parity (the S0 spike's open item). A failure is a gate failure of S0 (plan §7: the
    primitive-jaw contact model differs between engines), reported to the user, not loosened."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    r = _layer_c(mcfg, mcal)
    print(f"Layer C: {r}")
    print(f"Layer C report: {_write_report('layer-c', r)}")
    assert r["cpu_in_run"]
    assert r["die_z_end_cpu_m"] > DIE_REST_Z + 0.02, "the CPU grasp itself failed (die not lifted)"
    assert _layer_c_pass(r), r


@gpu
def test_layer_c_shown_to_fail_die_mass(mcfg, mcal):
    """The gate can fail: die mass x10 on the CPU side only."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    r = _layer_c(mcfg, mcal, cpu_die_mass_scale=10.0)
    print(f"Layer C, CPU die x10: {r}")
    assert not _layer_c_pass(r)


# ------------------------------------------------------------------------------------------------ STUCK
# Plan §10 2026-10-08 "STUCK emulated in batched sim": the M7 CPU-servo zero-shot run found the reach policy pressing
# the tip into the box's y wall (seed 7 goal #32) until the servo's stuck-action monitor tripped. Scenario: approach
# (0.254, -0.20, 0.032) on an ideal plant, then hold ONE saturated action that pushes the tip through y = -0.22.
STUCK_APPROACH, STUCK_HOLD = 40, 60


def _wall_press_actions(cfg, cal, chain) -> np.ndarray:
    """float32[STUCK_APPROACH + STUCK_HOLD, 6]: Jacobian approach to the wall, then one constant action (the policy's
    saturated output) pressing -y. Open loop, planned once on an ideal kinematic plant (present = target)."""
    bc = BatchedServoChain(cfg, cal, chain, 1, device=CPU)
    out = ChainOut.empty(1, CPU)
    present = torch.as_tensor(np.asarray(cfg.safe_state.home_ticks, np.int64))[None]
    bc.resync(present)
    md = np.asarray(cfg.action.max_delta_ticks, np.float64)
    xyz = torch.zeros(1, 3, dtype=torch.float64)
    acts = np.zeros((STUCK_APPROACH + STUCK_HOLD, N_JOINTS), np.float32)
    a = np.zeros(N_JOINTS)
    for i in range(len(acts)):
        if i <= STUCK_APPROACH:
            g = bc.goal.clone()
            x0 = bc.fk_tip(g, xyz).clone()[0].numpy()
            J = np.zeros((3, 4))
            for k in range(4):
                gp = g.clone()
                gp[0, k] += 10
                J[:, k] = (bc.fk_tip(gp, xyz)[0].numpy() - x0) / 10
            want = np.array([0.254, -0.20, 0.032]) - x0 if i < STUCK_APPROACH else np.array([0.0, -0.02, 0.0])
            dq = np.linalg.lstsq(J, np.clip(want, -0.015, 0.015), rcond=None)[0]
            a = np.zeros(N_JOINTS)
            a[:4] = dq / md[:4]
            a /= max(1.0, float(np.abs(a).max()))
            if i == STUCK_APPROACH:
                a /= float(np.abs(a).max())  # saturated, like a tanh policy pressing on
        acts[i] = np.clip(a, -1.0, 1.0)
        bc.new_command(torch.as_tensor(np.rint(acts[i] * md.astype(np.float32)))[None])
        for rem in (3, 2, 1):
            bc.apply(present, rem, out)
            present = out.target_ticks.clone()
    return acts


_STUCK_RUN: list[CpuRun] = []


def stuck_cpu_run(cfg, cal, chain) -> CpuRun:
    """The wall press on a FRESH CPU rig (a STUCK trip ends RUN for good: FRESH_PROCESS), cached."""
    if not _STUCK_RUN:
        rig = CpuRig(cfg, cal)
        rig.reset("home")
        _STUCK_RUN.append(rig.run(_wall_press_actions(cfg, cal, chain)))
    return _STUCK_RUN[0]


def _first(trips: np.ndarray, cause: TripCause) -> int:
    hit = (np.asarray(trips) & int(cause)) != 0
    return int(np.argmax(hit)) if hit.any() else -1


def test_stuck_cpu_servo_trips_on_the_wall_press(mcfg, mcal, kchain):
    """Harness: the real servo path (ServoCore + StuckActionMonitor + MujocoBus) trips STUCK on this scenario, after
    the box clamp held the tip at the wall, and only after STUCK_REPEAT_TICKS ticks of the held action."""
    run = stuck_cpu_run(mcfg, mcal, kchain)
    t_stuck, t_box = _first(run.trips, TripCause.STUCK), _first(run.trips, TripCause.BOX)
    assert t_box >= 0 and t_stuck > t_box, (t_box, t_stuck)
    assert t_stuck >= 3 * STUCK_APPROACH + STUCK_REPEAT_TICKS - 1
    assert run.in_run[t_stuck - 1] and not run.in_run[t_stuck + 1]  # the trip ended RUN
    assert not run.moving[t_stuck - STUCK_REPEAT_TICKS + 1 : t_stuck + 1].any()


def test_stuck_layer_a_same_tick(mcfg, mcal, kchain):
    """The recorded present / Moving stream fed to a fresh SafetyPipeline and to BatchedServoChain (N=4, world 2):
    the chain trips STUCK on exactly the tick the CPU servo did; targets / trips agree up to it."""
    run = stuck_cpu_run(mcfg, mcal, kchain)
    t_cpu = _first(run.trips, TripCause.STUCK)
    ref = NpChain(mcfg, mcal, kchain, run.present[0])
    bc = BatchedServoChain(mcfg, mcal, kchain, 4, device=CPU)
    out = ChainOut.empty(4, CPU)
    bc.resync(torch.as_tensor(np.tile(run.present[0], (4, 1))))
    ref_trips, chain_trips = [], []
    for i, dl in enumerate(run.deltas):
        ref.command(dl)
        bc.new_command(torch.as_tensor(np.tile(dl, (4, 1))))
        for k, rem in enumerate((3, 2, 1)):
            t = 3 * i + k
            if t > t_cpu:
                break
            v = ref.tick(run.present[t], rem, run.moving[t])
            mv = np.ones((4, N_JOINTS), np.uint8)
            mv[2] = run.moving[t]
            bc.apply(torch.as_tensor(np.tile(run.present[t], (4, 1))), rem, out, moving=torch.as_tensor(mv))
            assert np.array_equal(v.target_ticks, out.target_ticks[2].numpy()), f"tick {t}"
            ref_trips.append(int(v.trips) & CHAIN_TRIP_BITS)
            chain_trips.append(int(out.trips[2]))
            assert not int(out.trips[0]) & TripCause.STUCK  # the moving worlds never trip
    assert ref_trips == chain_trips
    assert _first(np.array(chain_trips), TripCause.STUCK) == t_cpu


def test_stuck_shown_to_fail_with_moving_always_set(mcfg, mcal, kchain):
    """The emulation needs the Moving flags: fed ``moving = 1`` (the pre-2026-10-08 Layer A reference) the chain never
    trips on the same stream, so the same-tick result above is not vacuous."""
    run = stuck_cpu_run(mcfg, mcal, kchain)
    bc = BatchedServoChain(mcfg, mcal, kchain, 1, device=CPU)
    out = ChainOut.empty(1, CPU)
    bc.resync(torch.as_tensor(run.present[:1]))
    seen = 0
    for i, dl in enumerate(run.deltas):
        bc.new_command(torch.as_tensor(dl[None]))
        for k, rem in enumerate((3, 2, 1)):
            bc.apply(torch.as_tensor(run.present[3 * i + k][None]), rem, out, moving=torch.ones(1, N_JOINTS))
            seen |= int(out.trips[0])
    assert not seen & TripCause.STUCK


@gpu
def test_stuck_layer_b_same_policy_step(mcfg, mcal, kchain):
    """Closed loop: the batched env (MuJoCo Warp, moving from qvel) truncates on STUCK within +-1 policy step of the
    CPU servo's STUCK tick, and counts it in its trip stats."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    run = stuck_cpu_run(mcfg, mcal, kchain)
    step_cpu = _first(run.trips, TripCause.STUCK) // 3
    env = _parity_env(mcfg, mcal)
    try:
        r = _run_batched(env, _wall_press_actions(mcfg, mcal, kchain))
        stats = env.stats
    finally:
        env.close()
    step_gpu = _first(r["step_trips"], TripCause.STUCK)
    print(f"STUCK: CPU servo step {step_cpu}, batched env step {step_gpu}")
    assert step_gpu >= 0, "the batched env never tripped STUCK"
    assert abs(step_gpu - step_cpu) <= 1
    assert len(r["step_trips"]) == step_gpu + 1  # _run_batched stops at the first truncation: STUCK truncated
    assert stats["trips_per_1000_steps/stuck"] > 0

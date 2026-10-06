"""Three-layer parity: ServoCore + MujocoBus (the real servo path on CPU MuJoCo) vs the batched sim (frozen test
definition: ``plans/sim-track-implementation-plan.md`` "Parity test definition").

* Layer A (CPU, ``pixi run test``): the present-tick stream recorded from the CPU rig, fed to a fresh numpy
  ``SafetyPipeline`` and to ``BatchedServoChain`` (torch CPU, N=1 and N=4 with world 2 carrying the sequence):
  targets, goals, clamped masks, the clamp bits of the trips and the applied deltas are bit-identical.
* Layer B (``gpu``): both rigs closed loop (CPU float64 MuJoCo vs MuJoCo Warp float32).
* Layer C (``gpu``): grasp contact parity (close the gripper on the die, lift, hold).

Every layer has a "shown to fail" twin. Layer B/C live further down this file.

CPU rig: booted like ``tests/test_action_units.py::Rig`` (button + ARM + re-energise), then for every sequence the
physics is reset to the ``home`` keyframe (sim time kept monotonic) and the pipeline resynced to the present ticks,
exactly what ``BatchedGoalEnv.reset`` does. One rig per module (boot once).

NaN rows (corpus f): the CPU rig receives a ZERO delta on those steps so it stays in RUN (a NaN would end its RUN
for good: FRESH_PROCESS); the chains under test receive the NaN. Layer A only needs a valid present stream.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from so101_rl.config import N_JOINTS, load_calibration, load_config
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
from so101_rl.sim.batched_servo import CHAIN_TRIP_BITS, BatchedServoChain, ChainOut
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
    """F7 rule, in float32 like the batched env: ``rint(action * max_delta_ticks)``."""
    return np.rint(acts.astype(np.float32) * np.asarray(cfg.action.max_delta_ticks, np.float32))


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
        self.block.moving[...] = 1  # STUCK is not emulated (D-TRIPS); keep it out of the reference
        self.block.ok = True
        self.block.position[...] = present0
        self.v = Verdict()
        self.pipe.resync(self.block.position)

    def command(self, delta: np.ndarray) -> None:
        self.pipe.new_command(np.asarray(delta, np.float32), self.v)

    def tick(self, present: np.ndarray, rem: int) -> Verdict:
        self.clock.advance(33_333_333)
        self.block.position[...] = present
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
            v = ref.tick(run.present[t], rem)
            p = np.tile(other_p[t], (nworld, 1))
            p[world] = run.present[t]
            bc.apply(torch.as_tensor(p), rem, out)
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


def test_layer_a_shown_to_fail_with_perturbed_vmax(cpu_rig, mcfg, mcal, kchain):
    """The gate can fail: the torch chain with ``max_ticks_per_tick`` perturbed (only there) mismatches."""
    total = 0
    for idx in (6, 7, 14):  # saturating / alternating sequences
        run = cpu_run(cpu_rig, mcfg, mcal, kchain, idx)
        for off in (+1, -1):
            c = layer_a(mcfg, mcal, kchain, run, nworld=1, world=0, vmax_offset=off)
            total += c["target"]
    assert total > 0

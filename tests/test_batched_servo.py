"""FS5 BatchedServoChain (sim track S1.1): the torch clamp chain against ``safety/pipeline.py`` on CPU torch.

The closed-loop / recorded-stream parity layers live in ``tests/test_batched_parity.py``; here: FK, the individual
stage semantics, and a randomised open-loop comparison on synthetic (adversarial) present streams.
"""

import numpy as np
import pytest
import torch

from so101_rl.config import N_JOINTS
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.pipeline import SafetyPipeline
from so101_rl.safety.stuck import StuckActionMonitor
from so101_rl.safety.thermal import ThermalMonitor
from so101_rl.safety.types import TripCause, Verdict
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.timing import FakeClock
from so101_rl.sim.batched_servo import CHAIN_TRIP_BITS, STAGES, BatchedServoChain, ChainOut

T = TripCause
CPU = torch.device("cpu")
# tip at (0.284, 0, 0.198): inside the box (tests/test_pipeline.py)
POSE = np.array([2047, 1395, 2699, 2047, 2047, 2047], dtype=np.int64)
# shoulder_lift 1930: tip z ~ 0.016, +45 ticks of lift leaves the box floor (tests/test_pipeline.py)
LOW_POSE = np.array([2047, 1930, 2699, 2047, 2047, 2047], dtype=np.int64)


@pytest.fixture
def chain(cfg):
    return KinematicChain.from_yaml(cfg.kinematics_path)


def _batched(cfg, calibration, chain, n, pose=POSE):
    bc = BatchedServoChain(cfg, calibration, chain, n, device=CPU)
    bc.resync(torch.as_tensor(np.tile(pose, (n, 1))))
    return bc, ChainOut.empty(n, CPU)


class NpRef:
    """The numpy SafetyPipeline fed an explicit present stream (no plant)."""

    def __init__(self, cfg, calibration, chain, pose):
        clock = FakeClock(start_ns=100 * 10**9)
        self.clock = clock
        self.fk = So101FK(chain, calibration)
        self.pipe = SafetyPipeline(cfg, self.fk, ThermalMonitor(cfg.thermal, clock), StuckActionMonitor())
        self.block = BlockRead()
        self.block.temperature_c[...] = 35
        self.block.moving[...] = 1  # never STUCK (not emulated by the batched chain, D-TRIPS)
        self.block.ok = True
        self.block.position[...] = pose
        self.v = Verdict()
        self.pipe.resync(self.block.position)

    def command(self, delta):
        return self.pipe.new_command(np.asarray(delta, dtype=np.float32), self.v)

    def tick(self, present, rem):
        self.clock.advance(33_333_333)
        self.block.position[...] = present
        self.pipe.apply(self.block, rem, self.v)
        return self.v


def _same(v: Verdict, out: ChainOut, w: int) -> list[str]:
    bad = []
    if not np.array_equal(v.target_ticks, out.target_ticks[w].numpy()):
        bad.append(f"target {v.target_ticks} vs {out.target_ticks[w].numpy()}")
    if not np.array_equal(v.goal_ticks, out.goal_ticks[w].numpy()):
        bad.append(f"goal {v.goal_ticks} vs {out.goal_ticks[w].numpy()}")
    if not np.array_equal(v.applied_delta_ticks, out.applied_delta_ticks[w].numpy()):
        bad.append("applied")
    if not np.array_equal(v.clamped_mask, out.clamped_mask[w].numpy()):
        bad.append(f"mask {v.clamped_mask} vs {out.clamped_mask[w].numpy()}")
    if (int(v.trips) & CHAIN_TRIP_BITS) != int(out.trips[w]):
        bad.append(f"trips {T(int(v.trips) & CHAIN_TRIP_BITS)!r} vs {T(int(out.trips[w]))!r}")
    return bad


# ---- FK -----------------------------------------------------------------------------------------------------------
def test_fk_matches_so101fk(cfg, calibration, chain):
    rng = np.random.default_rng(0)
    ticks = rng.integers(0, 4096, size=(500, N_JOINTS))
    bc = BatchedServoChain(cfg, calibration, chain, 500, device=CPU)
    got = bc.fk_tip(torch.as_tensor(ticks), torch.zeros(500, 3, dtype=torch.float64)).numpy()
    fk = So101FK(chain, calibration)
    want = np.stack([fk.gripper_xyz_from_ticks(t.astype(np.int32), np.zeros(3)) for t in ticks])
    assert np.abs(got - want).max() < 1e-9


# ---- stage semantics --------------------------------------------------------------------------------------------
def test_interp_lands_on_goal_in_3_ticks(cfg, calibration, chain):
    bc, out = _batched(cfg, calibration, chain, 2)
    delta = torch.tensor([[45, -45, 30, -7, 1, 60], [0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    bc.new_command(delta)
    goal = torch.as_tensor(POSE) + delta.to(torch.int64)
    targets = []
    for rem in (3, 2, 1):
        bc.apply(torch.as_tensor(np.tile(POSE, (2, 1))), rem, out)  # present held: a stiff-but-slow plant
        targets.append(out.target_ticks.clone())
    assert torch.equal(targets[-1][0], goal[0])
    assert torch.equal(targets[-1][1], torch.as_tensor(POSE))
    assert not torch.equal(targets[0][0], goal[0])  # spread, not jumped
    assert torch.equal(out.applied_delta_ticks[0], delta[0])
    assert int(out.trips[0]) == 0


def test_tracking_pullback_bounds_goal(cfg, calibration, chain):
    """A blocked joint: present never moves; |goal - present| <= E on every tick, TRACKING reported."""
    bc, out = _batched(cfg, calibration, chain, 1)
    E = torch.as_tensor(np.asarray(cfg.action.tracking_error_ticks, np.int64))
    present = torch.as_tensor(POSE)[None]
    saw_tracking = False
    for _ in range(10):
        bc.new_command(torch.tensor([[45.0, 0, 0, 0, 0, 0]]))
        for rem in (3, 2, 1):
            bc.apply(present, rem, out)
            assert ((bc.goal - present).abs() <= E).all()
            assert ((out.target_ticks - present).abs() <= E).all()
            saw_tracking |= bool(int(out.trips[0]) & int(T.TRACKING))
    assert saw_tracking
    assert int(bc.goal[0, 0] - present[0, 0]) == int(E[0])  # wound up to E, not beyond (anti-windup)


def test_box_revert(cfg, calibration, chain):
    """Commanded target below the box floor: target and goal revert to prev_target, BOX + mask; present inside."""
    bc, out = _batched(cfg, calibration, chain, 1, pose=LOW_POSE)
    bc.new_command(torch.tensor([[0.0, 45, 0, 0, 0, 0]]))
    present = torch.as_tensor(LOW_POSE)[None]
    reverted = False
    for rem in (3, 2, 1):
        prev = bc.prev_target.clone()
        bc.apply(present, rem, out)
        if int(out.trips[0]) & int(T.BOX):
            reverted = True
            assert torch.equal(out.target_ticks, prev)
            assert torch.equal(bc.goal, prev)
            assert bool(out.clamped_mask[0, 1])
        assert not int(out.trips[0]) & int(T.BOX_PRESENT)
    assert reverted


def test_box_present_trips_and_holds(cfg, calibration, chain):
    bc, out = _batched(cfg, calibration, chain, 1, pose=LOW_POSE)
    bc.new_command(torch.zeros(1, 6))
    deep = torch.as_tensor(LOW_POSE + np.array([0, 200, 0, 0, 0, 0]))[None]  # present well below the floor
    prev = bc.prev_target.clone()
    bc.apply(deep, 3, out)
    assert int(out.trips[0]) & int(T.BOX_PRESENT)
    assert torch.equal(out.target_ticks, prev)


def test_nan_row_zeroed_whole_row(cfg, calibration, chain):
    bc, out = _batched(cfg, calibration, chain, 3)
    delta = torch.tensor(
        [[45.0, float("nan"), 0, 0, 0, 0], [10.0, 0, 0, 0, 0, float("inf")], [10.0, 0, 0, 0, 0, 0]]
    )
    nan = bc.new_command(delta)
    assert nan.tolist() == [True, True, False]
    present = torch.as_tensor(np.tile(POSE, (3, 1)))
    bc.apply(present, 3, out)
    for w in (0, 1):
        assert int(out.trips[w]) & int(T.NAN)
        assert torch.equal(bc.goal[w], present[w])  # the finite 45 / 10 of the same row was NOT applied
        assert (out.applied_delta_ticks[w] == 0).all()
    assert not int(out.trips[2]) & int(T.NAN)
    bc.apply(present, 2, out)
    assert int(out.trips[0]) == 0  # reported once


def test_no_cross_world_leak(cfg, calibration, chain):
    """World k saturates every stage; world j (zero commands, present = target) never sees a clamp or a change."""
    n = 4
    bc, out = _batched(cfg, calibration, chain, n)
    rng = np.random.default_rng(3)
    for step in range(30):
        delta = torch.zeros(n, 6)
        delta[1] = torch.as_tensor(rng.choice([-1e6, 1e6, float("nan") if step == 7 else 1e6], size=6))
        delta[3] = 45.0 * torch.as_tensor(rng.choice([-1.0, 1.0], size=6))
        bc.new_command(delta)
        for rem in (3, 2, 1):
            present = torch.as_tensor(np.tile(POSE, (n, 1)))
            present[3] = torch.as_tensor(rng.integers(0, 4096, size=6))  # adversarial present stream
            bc.apply(present, rem, out)
            for j in (0, 2):
                assert torch.equal(out.target_ticks[j], torch.as_tensor(POSE))
                assert not out.clamped_mask[j].any()
                assert int(out.trips[j]) == 0


def test_clamp_active_fraction_counts_stages(cfg, calibration, chain):
    bc, out = _batched(cfg, calibration, chain, 2)
    assert bc.clamp_active_fraction == dict.fromkeys(STAGES, 0.0)
    present = torch.as_tensor(np.tile(POSE, (2, 1)))
    for _ in range(5):
        bc.new_command(torch.tensor([[45.0, 0, 0, 0, 0, 0], [0.0] * 6]))
        for rem in (3, 2, 1):
            bc.apply(present, rem, out)
    frac = bc.clamp_active_fraction
    assert 0.0 < frac["tracking"] <= 0.5  # only world 0 is ever clamped
    assert frac["box"] == 0.0


# ---- randomised open-loop equivalence with the numpy pipeline --------------------------------------------------
KINDS = ("uniform", "saturating", "nan", "pushes", "low")


def _stream(rng, kind, pose, steps):
    """(deltas[steps,6] float32, presents[steps*3,6]) adversarial streams; presents depend only on the seed."""
    deltas = rng.uniform(-1.3, 1.3, size=(steps, 6)) * 45
    if kind == "saturating":
        deltas = np.sign(rng.uniform(-1, 1, size=(steps, 6))) * 1e4
    if kind == "nan":
        deltas[rng.integers(0, steps, 3), rng.integers(0, 6, 3)] = np.nan
    walk = np.cumsum(rng.integers(-25, 26, size=(steps * 3, 6)), axis=0)
    presents = pose + walk
    if kind == "pushes":  # external pushes: present jumps far from the target and comes back
        for t in rng.integers(0, steps * 3 - 5, 8):
            presents[t : t + 4] += rng.integers(-400, 401, size=6)
    if kind == "low":  # around the box floor
        presents = LOW_POSE + np.cumsum(rng.integers(-15, 16, size=(steps * 3, 6)), axis=0)
    return deltas.astype(np.float32), np.clip(presents, 0, 4095)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(3))
def test_random_streams_bit_identical_to_pipeline(cfg, calibration, chain, kind, seed):
    rng = np.random.default_rng(1000 * seed + KINDS.index(kind))
    pose = LOW_POSE if kind == "low" else POSE
    steps = 80
    deltas, presents = _stream(rng, kind, pose, steps)
    ref = NpRef(cfg, calibration, chain, pose)
    bc, out = _batched(cfg, calibration, chain, 3, pose=pose)
    stages_seen = 0
    for i in range(steps):
        ref.command(deltas[i])
        bc.new_command(torch.as_tensor(np.stack([deltas[i], np.zeros(6, np.float32), deltas[i]])))
        for k, rem in enumerate((3, 2, 1)):
            p = presents[3 * i + k]
            v = ref.tick(p, rem)
            pt = torch.as_tensor(np.stack([p, POSE if kind != "low" else LOW_POSE, p]))
            bc.apply(pt, rem, out)
            for w in (0, 2):
                bad = _same(v, out, w)
                assert not bad, f"step {i} tick {k} world {w}: {bad}"
            stages_seen |= int(v.trips)
    assert stages_seen  # the stream exercised at least one clamp

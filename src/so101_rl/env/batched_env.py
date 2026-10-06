"""Batched goal environment on MuJoCo Warp with the real servo loop's tick rules (FS9, sim track S1.3).

One policy step (10 Hz) of every world, in the order of the real ``ServoCore`` path (``servo/core.py``):

1. ``delta = rint(action * max_delta_ticks)`` in float32 (the F7 rule, the only [-1, 1] -> ticks scaling) and
   ``BatchedServoChain.new_command`` (sanitize -> rint -> goal += delta -> joint limits; NaN rows trip).
2. Three servo ticks, ``ticks_remaining`` = 3, 2, 1. Each tick: ``present = q_to_ticks(qpos)`` (the read at the
   START of the tick, MujocoBus quantisation: arm ``rint(sign q / rad_per_tick + 2047)``, gripper through the
   0-100 % map, clipped 0..4095) -> tracker frame (die task) -> ``chain.apply(present, rem)`` -> the servo's
   register clip to the calibration range -> per-world delay FIFO -> ``ctrl = ticks_to_q`` clipped to
   ``ctrlrange`` -> ``WarpSim.step(steps_per_tick)`` (17 substeps of 1/510 s = 1/30 s, the same count as
   ``MujocoBus.step_ticks``).
3. Observation timing = the real arm's: the policy sees the state record of the 3rd tick, whose ``present`` was
   read at the START of tick 3, i.e. after only TWO ticks of the new command (the 3rd tick's write still happens
   and acts during the next step). ``obs``, ``priv``, ``achieved`` and the reward all use that instant; the
   latest tracker frame is taken at the same instant. (Master/SR1 parity requirement 2026-10-06; pinned by
   ``tests/test_batched_parity.py`` Layer B and ``tests/test_batched_env.py``.)
4. Reward / success / debounce through ``env/tasks.py`` (the one definition for sim and real, FS4), episode
   bookkeeping, ``final_*`` (pre-reset copies) and auto-reset of the done worlds.

Episode semantics (F7/FS9): ``terminated`` = debounced success; ``truncated`` = step limit or an emulated trip
(NAN, RANGE, BOX_PRESENT; decision D-TRIPS), never both. Clamp bits (JOINT_LIMIT, TRACKING, BOX) never end an
episode. ``info["trips"]`` / ``info["clamped_mask"]`` are ORed over the step's three ticks; ``applied_delta_ticks``
is the command's complete post-clamp delta (the LOGGED action, L4).

Resets (``reset``): the arm at ``home`` plus uniform +-``start_noise_ticks`` (30) on the five arm joints, FK
checked inside the box (8 tries, else exact home; no host sync); reach goals from ``task.goal_bounds`` by a torch
rejection sampler (32 candidates, fallback (0.20, 0, 0.15)) with an FK-reachability voxel table; the die task places the die at rest in the start well
(``die_start``: xy ~ N(centre, sigma) clipped so the die footprint stays on the flat well floor for its yaw,
yaw uniform, z = table + 8 mm, zero velocity; decision D-TRAY) and the goal is the zone centre at resting height.
The chain is resynced to the reset's present ticks (``resync_target`` semantics), the delay FIFO pre-filled with
them, the tracker restarted and primed. ``phase_randomise`` starts every world at a random ``episode_step`` on the
first full reset only (decorrelated auto-resets).

Sim parameters: ``sim_params`` is any object with the FS7 ``SimParams`` fields (``kp kv damping frictionloss
armature forcerange`` [6], ``delay_ticks``, ``die_mass_kg die_friction table_friction``); :func:`model_sim_params`
reads them from the compiled model (= the MJCF prior, what ``SimParams.from_mjcf`` returns; SU2 owns that class).
``randomizer`` (FS8) hooks land with SU3 (wave-2 hand-off of this file): passing one raises for now.

Host syncs: none per step except the overflow check every ``cfg.sim_warp.overflow_check_every_steps`` steps
(one copy) and ``close()``; ``stats`` and ``render_worlds`` sync when called.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import mujoco
import numpy as np
import torch

from so101_rl.config import CENTER_TICK, JOINT_NAMES, N_JOINTS, TICKS_PER_REV, MotorCalibration, So101Config
from so101_rl.env.obs import PRIV_OBS, DieObsT, build_policy_obs_torch
from so101_rl.env.tasks import DIE_HALF_EDGE_M, TaskSpec, debounce_torch, reward_torch, success_torch
from so101_rl.safety.fk import KinematicChain
from so101_rl.safety.types import TripCause
from so101_rl.sim.batched_servo import BatchedServoChain, ChainOut
from so101_rl.sim.gripper_map import gripper_range_rad, gripper_ticks_per_rad
from so101_rl.sim.model import TABLE_Z_M, jaw_collision_geom_ids, load_model
from so101_rl.sim.sim_tracker import CameraModel, Occluders, SimDieTracker, TrackerNoise, TrackerOut
from so101_rl.sim.warp_model import WarpSim

__all__ = ["TRUNCATING_TRIPS", "BatchedGoalEnv", "BatchedStep", "ModelSimParams", "model_sim_params"]

_RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_REV
TRUNCATING_TRIPS = int(TripCause.NAN | TripCause.RANGE | TripCause.BOX_PRESENT)  # D-TRIPS
GRIPPER = N_JOINTS - 1
_GOAL_CANDIDATES = 32
_VOXEL_M = 0.01  # reachability table resolution
_START_TRIES = 8
_FALLBACK_GOAL = (0.20, 0.0, 0.15)
_WELL_HALF_M = 0.0115  # flat floor of the start well (assets/so101/so101_rl_scene.xml tray), half width


@dataclass
class BatchedStep:
    obs: torch.Tensor
    priv: torch.Tensor
    achieved: torch.Tensor
    desired: torch.Tensor  # [N, ...]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor  # [N] float / bool / bool
    final_obs: torch.Tensor
    final_priv: torch.Tensor
    final_achieved: torch.Tensor  # pre-reset observation of the worlds that reset this step (others: copies of obs)
    info: dict[str, torch.Tensor]


@dataclass(frozen=True, eq=False)
class ModelSimParams:
    """The FS7 ``SimParams`` fields read from a compiled model (duck-typed stand-in until SU2's class merges)."""

    kp: np.ndarray
    kv: np.ndarray
    damping: np.ndarray
    frictionloss: np.ndarray
    armature: np.ndarray
    forcerange: np.ndarray
    delay_ticks: int
    deadband_ticks: np.ndarray
    die_mass_kg: float
    die_friction: float
    table_friction: float


def model_sim_params(model: mujoco.MjModel) -> ModelSimParams:
    """The model's own servo/die/table values (the MJCF prior): applying them is a no-op."""
    act = np.array([model.actuator(n).id for n in JOINT_NAMES])
    dof = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINT_NAMES])
    return ModelSimParams(
        kp=model.actuator_gainprm[act, 0].astype(np.float64),
        kv=-model.actuator_biasprm[act, 2].astype(np.float64),
        damping=model.dof_damping[dof].astype(np.float64),
        frictionloss=model.dof_frictionloss[dof].astype(np.float64),
        armature=model.dof_armature[dof].astype(np.float64),
        forcerange=np.abs(model.actuator_forcerange[act]).max(axis=1).astype(np.float64),
        delay_ticks=0,
        deadband_ticks=np.zeros(N_JOINTS, np.int64),
        die_mass_kg=float(model.body_mass[model.body("die").id]),
        die_friction=float(model.geom_friction[model.geom("die_geom").id, 0]),
        table_friction=float(model.geom_friction[model.geom("table").id, 0]),
    )


class _WorldParams:
    """Per-world tensors with the attribute names ``WarpSim.set_world_params`` reads (FS7 BatchedSimParams)."""

    def __init__(self, p: Any, n: int) -> None:
        def rows(x: Any) -> torch.Tensor:
            return torch.as_tensor(np.asarray(x, np.float64)).float().expand(n, N_JOINTS).clone()

        self.kp, self.kv, self.damping = rows(p.kp), rows(p.kv), rows(p.damping)
        self.frictionloss, self.armature, self.forcerange = rows(p.frictionloss), rows(p.armature), rows(p.forcerange)
        self.die_mass = torch.full((n,), float(p.die_mass_kg))
        self.die_friction = torch.full((n,), float(p.die_friction))
        self.table_friction = torch.full((n,), float(p.table_friction))


class BatchedGoalEnv:
    """See the module docstring."""

    act_dim: int = 6

    def __init__(
        self,
        cfg: So101Config,
        calibration: dict[str, MotorCalibration],
        task: TaskSpec,
        nworld: int,
        *,
        sim_params: Any,
        randomizer: Any | None = None,
        tracker_noise: TrackerNoise | None = None,
        model: mujoco.MjModel | None = None,
        seed: int = 0,
        device: str = "cuda:0",
    ) -> None:
        if randomizer is not None:
            raise NotImplementedError("BatchedGoalEnv: domain-randomisation hooks land with SU3 (FS8, wave 2)")
        self.cfg, self.task, self.nworld = cfg, task, int(nworld)
        self.device = dev = torch.device(device)
        self.model = model if model is not None else load_model()[0]
        m = self.model
        self.ws = WarpSim(m, cfg.sim_warp, self.nworld, device=device)
        self.chain = BatchedServoChain(cfg, calibration, KinematicChain.from_yaml(cfg.kinematics_path), self.nworld, device=dev)
        self.obs_dim = task.obs.dim
        self.priv_dim = PRIV_OBS.dim
        self.start_noise_ticks = 30  # reset: uniform +- on the five arm joints (parity tests set 0)
        self.trace: list[dict[str, torch.Tensor]] | None = None  # per-tick record for the parity tests
        n = self.nworld
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(int(seed))

        # ---- indices
        self._q_adr = torch.tensor([int(m.jnt_qposadr[m.joint(j).id]) for j in JOINT_NAMES], device=dev)
        self._v_adr = torch.tensor([int(m.jnt_dofadr[m.joint(j).id]) for j in JOINT_NAMES], device=dev)
        self._act = torch.tensor([m.actuator(j).id for j in JOINT_NAMES], device=dev)
        die_j = m.joint("die_free").id
        self._die_q = int(m.jnt_qposadr[die_j])
        self._die_v = int(m.jnt_dofadr[die_j])
        self._jaw = jaw_collision_geom_ids(m).tolist()
        self._die_geom = [m.geom("die_geom").id]
        self._table_geom = [m.geom("table").id]
        self.steps_per_tick = max(1, round((1.0 / cfg.rates.servo_hz) / m.opt.timestep))

        # ---- calibration mapping (MujocoBus, float64)
        cal = [calibration[j] for j in JOINT_NAMES]
        self._sign = torch.tensor([-1.0 if c.drive_mode else 1.0 for c in cal], dtype=torch.float64, device=dev)
        g = calibration["gripper"]
        self._g_min, self._g_span, self._g_flip = float(g.range_min), float(g.range_max - g.range_min), bool(g.drive_mode)
        self._g_lo, self._g_hi = gripper_range_rad()
        tpr = torch.full((N_JOINTS,), 4096.0 / (2.0 * math.pi), dtype=torch.float64, device=dev)
        tpr[GRIPPER] = gripper_ticks_per_rad(calibration)
        self._vel_scale = tpr  # velocity = rint(qd * ticks_per_rad * sign)
        self._reg_lo = torch.tensor([c.range_min for c in cal], dtype=torch.int64, device=dev)
        self._reg_hi = torch.tensor([c.range_max for c in cal], dtype=torch.int64, device=dev)
        cr = m.actuator_ctrlrange[[m.actuator(j).id for j in JOINT_NAMES]]
        self._ctrl_lo = torch.tensor(cr[:, 0], dtype=torch.float64, device=dev)
        self._ctrl_hi = torch.tensor(cr[:, 1], dtype=torch.float64, device=dev)
        self._max_delta32 = torch.tensor(np.asarray(cfg.action.max_delta_ticks, np.float32), device=dev)
        self._max_delta64 = self._max_delta32.double()

        # ---- sim parameters (servo, die, table) and the delay FIFO
        self.sim_params = sim_params
        self.ws.set_world_params(_WorldParams(sim_params, n))
        self._centre = torch.cat(
            [torch.as_tensor(np.asarray(getattr(sim_params, k), np.float64)) for k in ("kp", "kv", "damping", "frictionloss", "armature")]
        ).float().to(dev)
        self.delay_ticks = torch.full((n,), int(sim_params.delay_ticks), dtype=torch.int64, device=dev)
        self._fifo_len = int(sim_params.delay_ticks) + 1
        self._fifo = torch.zeros((n, self._fifo_len, N_JOINTS), dtype=torch.int64, device=dev)
        self._fifo_head = 0
        self._arange = torch.arange(n, device=dev)

        # ---- home pose
        home = m.key("home")
        self._home_qpos = torch.as_tensor(home.qpos, dtype=torch.float32, device=dev)
        self._home_ctrl = torch.as_tensor(home.ctrl, dtype=torch.float32, device=dev)

        # ---- goals
        gb = task.goal_bounds
        self._goal_lo = torch.as_tensor(gb.lo, dtype=torch.float64, device=dev)
        self._goal_hi = torch.as_tensor(gb.hi, dtype=torch.float64, device=dev)
        self._goal_r = float(gb.radius_max_m)
        self._reach_vox: torch.Tensor | None = None
        self._vox_dims: torch.Tensor | None = None
        self._zone_goal = (
            torch.as_tensor(task.zone.goal_xyz(), dtype=torch.float32, device=dev) if task.zone is not None else None
        )

        # ---- tracker (die tasks)
        self.tracker: SimDieTracker | None = None
        self._occl: Occluders | None = None
        if task.obs.has_die:
            noise = tracker_noise if tracker_noise is not None else TrackerNoise()
            self.tracker = SimDieTracker(CameraModel.default(), noise, n, device=dev, seed=seed + 7919)
            self._occl = Occluders(m, dev)
            self._trk = TrackerOut.empty(n, dev)
            self._trk_obs = TrackerOut.empty(n, dev)

        # ---- per-world episode state
        self.goal = torch.zeros((n, 3), dtype=torch.float32, device=dev)
        self.episode_step = torch.zeros(n, dtype=torch.int64, device=dev)
        self._count = torch.zeros(n, dtype=torch.int64, device=dev)
        self._prev_achieved = torch.zeros((n, 3), dtype=torch.float32, device=dev)
        self._applied = torch.zeros((n, N_JOINTS), dtype=torch.float32, device=dev)
        self._out = ChainOut.empty(n, dev)
        self._obs = torch.zeros((n, self.obs_dim), dtype=torch.float32, device=dev)
        self._priv = torch.zeros((n, self.priv_dim), dtype=torch.float32, device=dev)
        self._achieved = torch.zeros((n, 3), dtype=torch.float32, device=dev)
        self._xyz = torch.zeros((n, 3), dtype=torch.float64, device=dev)
        self._first_reset = True
        self._steps = 0
        self._closed = False
        # stats (device counters)
        self._n_trips = torch.zeros(3, dtype=torch.int64, device=dev)  # NAN, RANGE, BOX_PRESENT
        self._n_visible = torch.zeros((), dtype=torch.int64, device=dev)
        self._n_resets = torch.zeros((), dtype=torch.int64, device=dev)
        self._renderer: dict[tuple[int, int], mujoco.Renderer] = {}
        self._cpu_data: mujoco.MjData | None = None

    # ------------------------------------------------------------------ conversions (MujocoBus semantics)
    def q_to_ticks(self, q: torch.Tensor) -> torch.Tensor:
        """qpos [N,6] -> present ticks int64 [N,6] (``MujocoBus.q_to_ticks``: float64 math, half-to-even)."""
        q = q.double()
        t = torch.round(self._sign * q / _RAD_PER_TICK + CENTER_TICK)
        p = ((q[:, GRIPPER] - self._g_lo) / (self._g_hi - self._g_lo) * 100.0).clamp(0.0, 100.0)
        frac = p / 100.0
        if self._g_flip:
            frac = 1.0 - frac
        t[:, GRIPPER] = torch.round(self._g_min + frac * self._g_span)
        return t.clamp(0, TICKS_PER_REV - 1).to(torch.int64)

    def ticks_to_q(self, ticks: torch.Tensor) -> torch.Tensor:
        """ticks [N,6] -> joint angles float64 (``MujocoBus.ticks_to_q``)."""
        t = ticks.double()
        q = self._sign * (t - CENTER_TICK) * _RAD_PER_TICK
        frac = (t[:, GRIPPER] - self._g_min) / self._g_span
        if self._g_flip:
            frac = 1.0 - frac
        p = (frac * 100.0).clamp(0.0, 100.0)
        q[:, GRIPPER] = self._g_lo + p / 100.0 * (self._g_hi - self._g_lo)
        return q

    def velocity_ticks(self, qd: torch.Tensor) -> torch.Tensor:
        return torch.round(qd.double() * self._vel_scale * self._sign).to(torch.int64)

    def _present(self) -> torch.Tensor:
        return self.q_to_ticks(self.ws.qpos[:, self._q_adr])

    # ------------------------------------------------------------------ geometry helpers
    def _bodies(self, names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.stack([self.ws.body_xpos(b) for b in names], dim=1),
            torch.stack([self.ws.body_xquat(b) for b in names], dim=1),
        )

    def _tracker_frame(self, frames: int = 1) -> None:
        assert self.tracker is not None and self._occl is not None
        pos, quat = self._bodies(self._occl.bodies)
        c, h, rot = self._occl.compute(pos, quat)
        die = self.ws.body_xpos("die")
        tip = self.ws.site_xpos("gripper_tip")
        self.tracker.update(die, tip, c, h, frames, self._trk, jaw_rot=rot)

    # ------------------------------------------------------------------ observation at the current instant
    def _observe(self, present: torch.Tensor) -> None:
        """Fill obs / priv / achieved from ``present`` (the read at the start of the current tick) and the current
        physics state (same instant)."""
        cfg, ws = self.cfg, self.ws
        qd = ws.qvel[:, self._v_adr]
        vel = self.velocity_ticks(qd)
        tip = self.chain.fk_tip(present, self._xyz).float()
        die_t = None
        if self.tracker is not None:
            self._trk_obs.xyz.copy_(self._trk.xyz)
            self._trk_obs.visible.copy_(self._trk.visible)
            self._trk_obs.age_frames.copy_(self._trk.age_frames)
            die_t = DieObsT(xyz=self._trk.xyz, visible=self._trk.visible, age_frames=self._trk.age_frames)
        build_policy_obs_torch(
            self.task.obs, present_ticks=present, velocity=vel, applied_delta_ticks=self._applied, tip_xyz=tip,
            cfg=cfg, die=die_t, out=self._obs,
        )  # fmt: skip
        self._achieved.copy_(self._trk.xyz if self.tracker is not None else tip)
        # privileged (critic only)
        P, pr = PRIV_OBS, self._priv
        pr[:, P.q_true] = ws.qpos[:, self._q_adr]
        pr[:, P.dq_true] = qd
        pr[:, P.tip_xyz_true] = ws.site_xpos("gripper_tip")
        pr[:, P.die_pos] = ws.qpos[:, self._die_q : self._die_q + 3]
        pr[:, P.die_quat] = ws.qpos[:, self._die_q + 3 : self._die_q + 7]
        pr[:, P.die_vel] = ws.qvel[:, self._die_v : self._die_v + 3]
        pr[:, P.die_in_gripper] = (ws.contact_count(self._jaw, self._die_geom) > 0).float()[:, None]
        pr[:, P.die_on_table] = (ws.contact_count(self._die_geom, self._table_geom) > 0).float()[:, None]
        dr = torch.ones((self.nworld, 33), dtype=torch.float32, device=self.device)  # no DR: every ratio 1
        dr[:, 30] = self.delay_ticks.float() / 5.0
        pr[:, P.dr_params] = dr
        lat = self.tracker.latency.float() / 3.0 if self.tracker is not None else torch.zeros(self.nworld, device=self.device)
        pr[:, P.tracker_delay] = lat[:, None]

    # ------------------------------------------------------------------ reset
    def _reachable_table(self) -> torch.Tensor:
        """bool voxel grid (``_VOXEL_M``) over the goal bounds: True where the FK of a joint configuration within
        the joint limits lands (2e5 uniform joint samples, deterministic). Interim FK-reachability filter until
        SR1's ``BoxGoalSampler`` table is shared (FS12 cross-test, open item)."""
        if self._reach_vox is None:
            g = torch.Generator(device=self.device).manual_seed(12345)
            lo = torch.as_tensor(np.asarray(self.cfg.joints.min_ticks, np.int64), device=self.device)
            hi = torch.as_tensor(np.asarray(self.cfg.joints.max_ticks, np.int64), device=self.device)
            dims = torch.ceil((self._goal_hi - self._goal_lo) / _VOXEL_M).to(torch.int64) + 1
            vox = torch.zeros(int(dims.prod()), dtype=torch.bool, device=self.device)
            xyz = torch.zeros((50_000, 3), dtype=torch.float64, device=self.device)
            for _ in range(4):
                u = torch.rand((50_000, N_JOINTS), generator=g, device=self.device, dtype=torch.float64)
                t = (lo + u * (hi - lo).double()).round().to(torch.int64)
                p = self.chain.fk_tip(t, xyz)
                inside = ((p >= self._goal_lo) & (p <= self._goal_hi)).all(dim=1)
                idx = ((p - self._goal_lo) / _VOXEL_M).floor().to(torch.int64)
                flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
                vox[flat[inside]] = True
            self._reach_vox, self._vox_dims = vox, dims
        return self._reach_vox

    def _sample_goals(self) -> torch.Tensor:
        n, g = self.nworld, self._gen
        u = torch.rand((n, _GOAL_CANDIDATES, 3), generator=g, device=self.device, dtype=torch.float64)
        cand = self._goal_lo + u * (self._goal_hi - self._goal_lo)
        vox, dims = self._reachable_table(), self._vox_dims
        idx = ((cand - self._goal_lo) / _VOXEL_M).floor().to(torch.int64)
        flat = (idx[..., 0] * dims[1] + idx[..., 1]) * dims[2] + idx[..., 2]
        ok = (cand.norm(dim=-1) <= self._goal_r) & vox[flat]
        first = torch.argmax(ok.to(torch.int8), dim=1)
        pick = cand[self._arange, first]
        fb = torch.tensor(_FALLBACK_GOAL, dtype=torch.float64, device=self.device)
        return torch.where(ok.any(dim=1)[:, None], pick, fb).float()

    def _start_qpos(self) -> torch.Tensor:
        """home + uniform +-start_noise_ticks on the arm joints, FK inside the box (fixed tries, no host sync)."""
        n, k = self.nworld, int(self.start_noise_ticks)
        q = self._home_qpos.expand(n, -1).clone()
        if k <= 0:
            return q
        home_t = self.q_to_ticks(q[:, self._q_adr])
        jl_lo = torch.as_tensor(np.asarray(self.cfg.joints.min_ticks, np.int64), device=self.device)
        jl_hi = torch.as_tensor(np.asarray(self.cfg.joints.max_ticks, np.int64), device=self.device)
        box = self.cfg.box
        lo = torch.tensor([box.x[0], box.y[0], box.z[0]], dtype=torch.float64, device=self.device)
        hi = torch.tensor([box.x[1], box.y[1], box.z[1]], dtype=torch.float64, device=self.device)
        chosen = home_t.clone()
        found = torch.zeros(n, dtype=torch.bool, device=self.device)
        for _ in range(_START_TRIES):
            off = torch.randint(-k, k + 1, (n, N_JOINTS), generator=self._gen, device=self.device)
            off[:, GRIPPER] = 0
            t = torch.maximum(torch.minimum(home_t + off, jl_hi), jl_lo)  # never start beyond a joint limit
            xyz = self.chain.fk_tip(t, self._xyz)
            ok = ((xyz >= lo) & (xyz <= hi)).all(dim=1) & ~found
            chosen = torch.where(ok[:, None], t, chosen)
            found |= ok
        qa = self.ticks_to_q(chosen).float()
        arm = self._q_adr[: N_JOINTS - 1]
        q[:, arm] = torch.where(found[:, None], qa[:, : N_JOINTS - 1], q[:, arm])  # no valid draw: exact home
        return q

    def _place_die(self, q: torch.Tensor) -> None:
        ds = self.task.die_start
        assert ds is not None
        n, g, dev = self.nworld, self._gen, self.device
        yaw = torch.rand(n, generator=g, device=dev) * (2 * math.pi)
        reach = DIE_HALF_EDGE_M * (torch.cos(yaw).abs() + torch.sin(yaw).abs())  # footprint half extent
        slack = (_WELL_HALF_M - reach).clamp_min(0.0)[:, None]
        xy = torch.randn((n, 2), generator=g, device=dev) * ds.sigma_xy_m
        xy = torch.maximum(torch.minimum(xy, slack), -slack)
        c = torch.tensor(ds.well_center_xy_m, dtype=torch.float32, device=dev)
        q[:, self._die_q : self._die_q + 2] = c + xy
        q[:, self._die_q + 2] = TABLE_Z_M + DIE_HALF_EDGE_M
        q[:, self._die_q + 3] = torch.cos(yaw / 2)
        q[:, self._die_q + 4 : self._die_q + 6] = 0.0
        q[:, self._die_q + 6] = torch.sin(yaw / 2)

    def reset(
        self, mask: torch.Tensor | None = None, *, phase_randomise: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset the worlds in ``mask`` (all if None); returns ``(obs, priv, achieved, desired)`` for ALL worlds
        (views of internal buffers: copy what you keep)."""
        self._check_open()
        n, dev = self.nworld, self.device
        full = mask is None
        mask = torch.ones(n, dtype=torch.bool, device=dev) if full else mask.to(device=dev, dtype=torch.bool)
        q = self._start_qpos()
        if self.task.die_start is not None:
            self._place_die(q)
        ctrl = self._home_ctrl.expand(n, -1).clone()
        ctrl[:, self._act] = q[:, self._q_adr]  # hold the start pose (== the keyframe ctrl without noise)
        self.ws.reset_worlds(mask, q, torch.zeros_like(self.ws.qvel), ctrl)
        present = self._present()
        self.chain.resync(present, mask)
        held = present.clamp(self._reg_lo, self._reg_hi)
        self._fifo.copy_(torch.where(mask[:, None, None], held[:, None, :], self._fifo))
        self._applied.masked_fill_(mask[:, None], 0.0)
        self._count.masked_fill_(mask, 0)
        goals = self._sample_goals() if self._zone_goal is None else self._zone_goal.expand(n, -1)
        self.goal.copy_(torch.where(mask[:, None], goals, self.goal))
        if self.tracker is not None:
            die = self.ws.body_xpos("die")
            self.tracker.reset(mask, die)
            self.tracker.prime(mask, die, self._trk)
        steps = torch.zeros(n, dtype=torch.int64, device=dev)
        if full and phase_randomise and self._first_reset:
            steps = torch.randint(0, self.task.episode_steps, (n,), generator=self._gen, device=dev)
        if full:
            self._first_reset = False
        self.episode_step.copy_(torch.where(mask, steps, self.episode_step))
        self._n_resets += mask.sum()
        self._observe_masked(present, mask)
        self._prev_achieved.copy_(torch.where(mask[:, None], self._achieved, self._prev_achieved))
        return self._obs, self._priv, self._achieved, self.goal

    def _observe_masked(self, present: torch.Tensor, mask: torch.Tensor) -> None:
        """``_observe`` for the worlds in ``mask`` only; the other worlds keep their (step) observation."""
        old = (self._obs.clone(), self._priv.clone(), self._achieved.clone())
        self._observe(present)
        m = mask[:, None]
        for buf, prev in zip((self._obs, self._priv, self._achieved), old, strict=True):
            buf.copy_(torch.where(m, buf, prev))

    def reset_to_state(
        self, qpos: torch.Tensor, ctrl: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Start the worlds in ``mask`` (all if None) from an explicit state (``qpos [nq]`` or ``[N, nq]``,
        ``ctrl [nu]`` or ``[N, nu]``, zero velocity) with the same bookkeeping as :meth:`reset` (chain resync,
        delay FIFO, tracker restart, episode step 0) but the CURRENT goals. For tests and scripted scenarios
        (parity Layer C, debugging)."""
        self._check_open()
        n, dev = self.nworld, self.device
        mask = torch.ones(n, dtype=torch.bool, device=dev) if mask is None else mask.to(device=dev, dtype=torch.bool)
        q = torch.as_tensor(qpos, dtype=torch.float32, device=dev).expand(n, -1)
        c = torch.as_tensor(ctrl, dtype=torch.float32, device=dev).expand(n, -1)
        self.ws.reset_worlds(mask, q, torch.zeros_like(self.ws.qvel), c)
        present = self._present()
        self.chain.resync(present, mask)
        held = present.clamp(self._reg_lo, self._reg_hi)
        self._fifo.copy_(torch.where(mask[:, None, None], held[:, None, :], self._fifo))
        self._applied.masked_fill_(mask[:, None], 0.0)
        self._count.masked_fill_(mask, 0)
        self.episode_step.masked_fill_(mask, 0)
        if self.tracker is not None:
            die = self.ws.body_xpos("die")
            self.tracker.reset(mask, die)
            self.tracker.prime(mask, die, self._trk)
        self._observe_masked(present, mask)
        self._prev_achieved.copy_(torch.where(mask[:, None], self._achieved, self._prev_achieved))
        return self._obs, self._priv, self._achieved, self.goal

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor) -> BatchedStep:
        """``action`` float32[N,6] in [-1, 1] -> one policy step of every world (module docstring)."""
        self._check_open()
        n, dev, ws, chain, out = self.nworld, self.device, self.ws, self.chain, self._out
        a = action.to(device=dev, dtype=torch.float32)
        if a.shape != (n, N_JOINTS):
            raise ValueError(f"action must be [{n}, {N_JOINTS}], got {tuple(a.shape)}")
        chain.new_command(torch.round(a * self._max_delta32))
        trips = torch.zeros(n, dtype=torch.int64, device=dev)
        clamped = torch.zeros((n, N_JOINTS), dtype=torch.bool, device=dev)
        for k, rem in enumerate((3, 2, 1)):
            present = self._present()
            if self.tracker is not None:
                self._tracker_frame(1)
            chain.apply(present, rem, out)
            trips |= out.trips
            clamped |= out.clamped_mask
            if k == 2:
                # The state record the policy reads: present (and the physics) at the START of tick 3, the
                # command's applied delta after tick 3's apply. apply() does not touch the physics, so the
                # world state here is still that of the read.
                self._applied.copy_(out.applied_delta_ticks)
                self._observe(present)
            goal = out.target_ticks.clamp(self._reg_lo, self._reg_hi)  # servo Min/Max_Position_Limit
            self._fifo_head = (self._fifo_head + 1) % self._fifo_len
            self._fifo[:, self._fifo_head] = goal
            delayed = self._fifo[self._arange, (self._fifo_head - self.delay_ticks) % self._fifo_len]
            q = torch.maximum(torch.minimum(self.ticks_to_q(delayed), self._ctrl_hi), self._ctrl_lo)
            ws.ctrl[:, self._act] = q.float()
            if self.trace is not None:
                self.trace.append(self._trace_row(present, out, k))
            ws.step(self.steps_per_tick)
        return self._finish_step(trips, clamped)

    def _trace_row(self, present: torch.Tensor, out: ChainOut, k: int) -> dict[str, torch.Tensor]:
        return {
            "present": present.clone(),
            "target": out.target_ticks.clone(),
            "goal": out.goal_ticks.clone(),
            "trips": out.trips.clone(),
            "qpos": self.ws.qpos.clone(),
            "jaw_table": self.ws.contact_count(self._jaw, self._table_geom),
            "jaw_die": self.ws.contact_count(self._jaw, self._die_geom),
        }

    def _finish_step(self, trips: torch.Tensor, clamped: torch.Tensor) -> BatchedStep:
        task, dev = self.task, self.device
        vis = self._trk_obs.visible if self.tracker is not None else None
        info_r = {
            "action_clamped": clamped.any(dim=1).to(torch.float32),
            "applied_action": (self._applied.double() / self._max_delta64).clamp(-1.0, 1.0).float(),
        }
        reward = reward_torch(self._achieved, self.goal, task, visible=vis, info=info_r)
        pred = success_torch(self._achieved, self.goal, task, visible=vis)
        success, self._count = debounce_torch(
            pred, self._count, task, achieved=self._achieved, prev_achieved=self._prev_achieved
        )
        self._prev_achieved.copy_(self._achieved)
        self.episode_step += 1
        tripped = (trips & TRUNCATING_TRIPS) != 0
        terminated = success
        truncated = ((self.episode_step >= task.episode_steps) | tripped) & ~terminated
        done = terminated | truncated
        info = {
            "applied_delta_ticks": self._applied.clone(),
            "clamped_mask": clamped,
            "trips": trips,
            "success": success,
            "tracker_visible": vis.clone() if vis is not None else torch.ones(self.nworld, dtype=torch.bool, device=dev),
            "episode_step": self.episode_step.clone(),
            "dr_stage": torch.zeros(self.nworld, dtype=torch.int64, device=dev),
        }
        final_obs, final_priv, final_ach = self._obs.clone(), self._priv.clone(), self._achieved.clone()
        desired = self.goal.clone()
        self._n_trips += torch.stack([((trips & int(c)) != 0).sum() for c in (TripCause.NAN, TripCause.RANGE, TripCause.BOX_PRESENT)])
        if vis is not None:
            self._n_visible += vis.sum()
        self._steps += 1
        if self._steps % max(1, self.cfg.sim_warp.overflow_check_every_steps) == 0:
            self.ws.check_overflow()
        # auto-reset (masked; no host sync)
        self.reset(done, phase_randomise=False)
        return BatchedStep(
            obs=self._obs.clone(), priv=self._priv.clone(), achieved=self._achieved.clone(), desired=desired,
            reward=reward, terminated=terminated, truncated=truncated,
            final_obs=final_obs, final_priv=final_priv, final_achieved=final_ach, info=info,
        )  # fmt: skip

    # ------------------------------------------------------------------ misc
    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("BatchedGoalEnv is closed")

    @property
    def stats(self) -> dict[str, float]:
        steps = max(self._steps, 1)
        trips = self._n_trips.cpu().tolist()
        out = {f"clamp_active/{k}": v for k, v in self.chain.clamp_active_fraction.items()}
        out["tracker_visible_frac"] = float(self._n_visible) / (steps * self.nworld) if self.tracker else 1.0
        for name, v in zip(("nan", "range", "box_present"), trips, strict=True):
            out[f"trips_per_1000_steps/{name}"] = 1000.0 * v / (steps * self.nworld)
        out["resets"] = float(self._n_resets)
        out["overflow_checks"] = float(self.ws.overflow_checks)
        out["max_nefc_seen"] = float(self.ws.max_nefc_seen)
        return out

    def render_worlds(
        self, worlds: Sequence[int], *, width: int = 320, height: int = 240, camera: str = "overhead"
    ) -> np.ndarray:
        """uint8[k,H,W,3] via ``mujoco.Renderer`` on a CPU MjData copy of each world (pixel-track hook)."""
        self._check_open()
        if self._cpu_data is None:
            self._cpu_data = mujoco.MjData(self.model)
        r = self._renderer.get((width, height))
        if r is None:
            r = self._renderer[(width, height)] = mujoco.Renderer(self.model, height, width)
        frames = np.zeros((len(worlds), height, width, 3), np.uint8)
        for i, w in enumerate(worlds):
            self.ws.copy_world_to(int(w), self._cpu_data)
            mujoco.mj_forward(self.model, self._cpu_data)
            r.update_scene(self._cpu_data, camera=camera)
            frames[i] = r.render()
        return frames

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.ws.close()
        finally:
            for r in self._renderer.values():
                r.close()
            self._renderer.clear()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _bench(argv: list[str] | None = None) -> int:
    from so101_rl.config import load_calibration, load_config
    from so101_rl.env.tasks import DIE_TO_ZONE, REACH

    ap = argparse.ArgumentParser(prog="python -m so101_rl.env.batched_env", description="Batched env benchmark")
    ap.add_argument("--task", choices=("reach", "die_to_zone"), default="reach")
    ap.add_argument("--nworld", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--calibration", default="tests/data/so101_follower.calib.json")
    ap.add_argument("--bench", action="store_true", help="random actions, report policy-steps/s and clamp stats")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cal = load_calibration(args.calibration)
    cfg = load_config(calibration=cal)
    task = REACH if args.task == "reach" else DIE_TO_ZONE
    model = load_model()[0]
    env = BatchedGoalEnv(cfg, cal, task, args.nworld, sim_params=model_sim_params(model), model=model, seed=args.seed)
    gen = torch.Generator(device=env.device)
    gen.manual_seed(args.seed)
    env.reset()
    for _ in range(5):  # warm-up: graph capture
        env.step(torch.rand((env.nworld, 6), generator=gen, device=env.device) * 2 - 1)
    torch.cuda.synchronize(env.device)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(torch.rand((env.nworld, 6), generator=gen, device=env.device) * 2 - 1)
    torch.cuda.synchronize(env.device)
    dt = time.perf_counter() - t0
    sps = args.steps * env.nworld / dt
    stats = env.stats
    env.close()
    print(f"task={args.task} nworld={args.nworld} steps={args.steps} wall={dt:.2f}s policy_steps_per_s={sps:.0f}")
    print("clamp_active:", {k.split('/')[1]: round(v, 4) for k, v in stats.items() if k.startswith("clamp_active/")})
    print("stats:", {k: round(v, 4) for k, v in stats.items() if not k.startswith("clamp_active/")})
    return 0


if __name__ == "__main__":
    sys.exit(_bench())

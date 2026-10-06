"""Multiple-shooting replay of 30 Hz tick-log segments in simulation (FS7, sim-first track S2.3).

For each :class:`~so101_rl.sim.sysid.dataset.TickSegment` the logged target stream is replayed through the servo
side of the model exactly as ``MujocoBus.write_goal`` applies it (hysteresis ``deadband_ticks`` -> the
Min/Max_Position_Limit register clip -> the ``delay_ticks`` FIFO -> ticks to radians -> ``ctrlrange`` clip), one
servo tick = ``steps_per_tick`` mj_steps, and the simulated present position after tick ``k`` is compared with the
logged ``present[k + 1]`` (tick-log convention, ``dataset`` module docstring).

**Multiple shooting:** every ``shooting_window_ticks`` (60 = 2 s) the simulation is re-initialised from the logged
state (``qpos`` from ``present[w]``, ``qvel`` from a central difference of the logged positions, the rest of the scene
at the ``home`` keyframe, solver warm-start cleared), so friction or delay errors cannot integrate into a drift that
swamps every other parameter's signal. The delay FIFO is pre-filled from the log (``history_targets``, then
``present[0]`` = the servo holding where it is).

CPU (``mujoco``, float64) and MuJoCo Warp (float32, one candidate parameter set per world) run the identical
procedure; ``tests/test_sysid_fit.py::test_warp_and_cpu_objectives_agree`` pins them together.

``rms_pct`` is the compare-real definition: per joint, RMS(sim - real) over rows ``1..T-1`` as % of the joint's
calibrated range (``range_max - range_min``). ``quantize=False`` returns the simulated position as unrounded ticks,
which gives the fit a smooth objective; gates and reports use the quantised (``int32``) form.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from so101_rl.config import CENTER_TICK, JOINT_NAMES, N_JOINTS, TICKS_PER_REV, MotorCalibration, So101Config
from so101_rl.sim.sysid.params import MAX_DELAY_TICKS, SimParams, apply_deadband

if TYPE_CHECKING:
    import mujoco
    import torch

    from so101_rl.sim.sysid.dataset import TickSegment
    from so101_rl.sim.sysid.params import BatchedSimParams
    from so101_rl.sim.warp_model import WarpSim

SHOOTING_WINDOW_TICKS = 60
RESET_KEYFRAME = "home"
_GRIPPER = N_JOINTS - 1
_RAD_PER_TICK = 2.0 * np.pi / TICKS_PER_REV


class TickMap:
    """Vectorised ``MujocoBus`` conversions (``ticks_to_q`` / ``q_to_ticks``) for ``[..., 6]`` arrays and tensors."""

    def __init__(self, calibration: dict[str, MotorCalibration], model: mujoco.MjModel) -> None:
        from so101_rl.sim.gripper_map import gripper_range_rad

        self.sign = np.array([-1.0 if calibration[n].drive_mode else 1.0 for n in JOINT_NAMES])
        g = calibration["gripper"]
        self.g_min, self.g_max = float(g.range_min), float(g.range_max)
        self.g_flip = bool(g.drive_mode)
        self.g_lo, self.g_hi = gripper_range_rad()
        self.reg_lo = np.array([calibration[n].range_min for n in JOINT_NAMES], dtype=np.int64)
        self.reg_hi = np.array([calibration[n].range_max for n in JOINT_NAMES], dtype=np.int64)
        act = [model.actuator(n).id for n in JOINT_NAMES]
        self.ctrl_lo, self.ctrl_hi = np.asarray(model.actuator_ctrlrange[act], dtype=np.float64).T.copy()
        self.span = (self.reg_hi - self.reg_lo).astype(np.float64)

    # numpy ---------------------------------------------------------------------------------------
    def ticks_to_q(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        q = self.sign * (t - CENTER_TICK) * _RAD_PER_TICK
        frac = np.clip((t[..., _GRIPPER] - self.g_min) / (self.g_max - self.g_min), 0.0, 1.0)
        if self.g_flip:
            frac = 1.0 - frac
        q[..., _GRIPPER] = self.g_lo + frac * (self.g_hi - self.g_lo)
        return q

    def q_to_ticks(self, q: np.ndarray, *, quantize: bool = True) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        t = self.sign * q / _RAD_PER_TICK + CENTER_TICK
        frac = np.clip((q[..., _GRIPPER] - self.g_lo) / (self.g_hi - self.g_lo), 0.0, 1.0)
        if self.g_flip:
            frac = 1.0 - frac
        t[..., _GRIPPER] = self.g_min + frac * (self.g_max - self.g_min)
        if quantize:
            return np.clip(np.rint(t), 0, TICKS_PER_REV - 1).astype(np.int32)
        return np.clip(t, 0.0, TICKS_PER_REV - 1.0)

    def ctrl_from_ticks(self, goal: np.ndarray) -> np.ndarray:
        """The servo register clip, then ``ctrlrange``: what ``MujocoBus.write_goal`` puts into ``data.ctrl``."""
        g = np.clip(np.asarray(goal, dtype=np.int64), self.reg_lo, self.reg_hi)
        return np.clip(self.ticks_to_q(g), self.ctrl_lo, self.ctrl_hi)

    # torch ---------------------------------------------------------------------------------------
    def q_to_ticks_torch(self, q: torch.Tensor, *, quantize: bool = True) -> torch.Tensor:
        import torch

        sign = torch.as_tensor(self.sign, dtype=q.dtype, device=q.device)
        t = sign * q / _RAD_PER_TICK + CENTER_TICK
        frac = ((q[..., _GRIPPER] - self.g_lo) / (self.g_hi - self.g_lo)).clamp(0.0, 1.0)
        if self.g_flip:
            frac = 1.0 - frac
        t = t.clone()
        t[..., _GRIPPER] = self.g_min + frac * (self.g_max - self.g_min)
        if quantize:
            return torch.round(t).clamp(0, TICKS_PER_REV - 1).to(torch.int32)
        return t.clamp(0.0, TICKS_PER_REV - 1.0)


def steps_per_tick(model: mujoco.MjModel, cfg: So101Config) -> int:
    """The ``MujocoBus`` substep count (17 at the 1/510 s timestep)."""
    return max(1, round((1.0 / cfg.rates.servo_hz) / model.opt.timestep))


def effective_ctrl_ticks(seg: TickSegment, delay_ticks: int, deadband_ticks: np.ndarray) -> np.ndarray:
    """The goal (ticks) the actuator holds during each tick of ``seg``: ``[T, 6]`` int64.

    Hysteresis on the logged target stream (history included), then the FIFO: tick ``k`` sees the target written
    ``delay_ticks`` ticks earlier; before the log starts, the servo holds ``present[0]``."""
    if not 0 <= delay_ticks <= MAX_DELAY_TICKS:
        raise ValueError(f"delay_ticks {delay_ticks} outside 0..{MAX_DELAY_TICKS}")
    hist = seg.history_targets if seg.history_targets is not None else np.zeros((0, N_JOINTS), np.int32)
    pad = np.repeat(seg.present_ticks[:1], MAX_DELAY_TICKS - len(hist), axis=0)
    ext = np.concatenate([pad, hist, seg.target_ticks]).astype(np.int64)  # MAX_DELAY_TICKS rows before row 0
    ext = apply_deadband(ext, deadband_ticks)
    T = len(seg)
    return ext[MAX_DELAY_TICKS - delay_ticks : MAX_DELAY_TICKS - delay_ticks + T]


def initial_states(seg: TickSegment, tmap: TickMap, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """``(q, qd)`` of the six joints at every row, from the logged present positions (central differences)."""
    q = tmap.ticks_to_q(seg.present_ticks)
    qd = np.zeros_like(q)
    if len(q) > 2:
        qd[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    qd[0] = (q[1] - q[0]) / dt
    qd[-1] = (q[-1] - q[-2]) / dt
    return q, qd


def _window_starts(T: int, window: int) -> range:
    if window < 1:
        raise ValueError("shooting_window_ticks must be >= 1")
    return range(0, T - 1, window)


def replay_segment_cpu(
    seg: TickSegment,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    calibration: dict[str, MotorCalibration],
    cfg: So101Config,
    params: SimParams,
    *,
    shooting_window_ticks: int = SHOOTING_WINDOW_TICKS,
    quantize: bool = True,
    tmap: TickMap | None = None,
    apply_params: bool = True,
) -> np.ndarray:
    """Simulated present ticks ``[T, 6]`` (int32, or float64 unrounded with ``quantize=False``); row 0 = the log."""
    import mujoco

    if apply_params:
        params.apply_cpu(model)
    tmap = tmap if tmap is not None else TickMap(calibration, model)
    nsub = steps_per_tick(model, cfg)
    dt = 1.0 / cfg.rates.servo_hz
    ctrl = tmap.ctrl_from_ticks(effective_ctrl_ticks(seg, params.delay_ticks, params.deadband_ticks))
    q, qd = initial_states(seg, tmap, dt)
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES])
    dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINT_NAMES])
    act = np.array([model.actuator(n).id for n in JOINT_NAMES])
    key = model.key(RESET_KEYFRAME)
    T = len(seg)
    out = np.empty((T, N_JOINTS), dtype=np.int32 if quantize else np.float64)
    out[0] = seg.present_ticks[0]
    for w in _window_starts(T, shooting_window_ticks):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = key.qpos
        data.qpos[qadr] = q[w]
        data.qvel[dadr] = qd[w]
        data.ctrl[act] = ctrl[w]
        mujoco.mj_forward(model, data)
        for k in range(w, min(w + shooting_window_ticks, T - 1)):
            data.ctrl[act] = ctrl[k]
            mujoco.mj_step(model, data, nsub)
            out[k + 1] = tmap.q_to_ticks(data.qpos[qadr], quantize=quantize)
    return out


def replay_segments_warp(
    segs: Sequence[TickSegment],
    ws: WarpSim,
    candidates: BatchedSimParams,
    calibration: dict[str, MotorCalibration],
    cfg: So101Config,
    *,
    shooting_window_ticks: int = SHOOTING_WINDOW_TICKS,
    quantize: bool = True,
    tmap: TickMap | None = None,
) -> torch.Tensor:
    """``[n_candidates, sum(T), 6]`` simulated present ticks, one candidate parameter set per world (segments are
    concatenated along the time axis; each segment's row 0 is the log)."""
    import torch

    n = ws.sizes.nworld
    if candidates.nworld != n:
        raise ValueError(f"{candidates.nworld} candidates for {n} worlds")
    model = ws.model
    tmap = tmap if tmap is not None else TickMap(calibration, model)
    ws.set_world_params(candidates)
    nsub = steps_per_tick(model, cfg)
    dt = 1.0 / cfg.rates.servo_hz
    dev = ws.device
    qadr = torch.as_tensor([int(model.jnt_qposadr[model.joint(j).id]) for j in JOINT_NAMES], device=dev)
    dadr = torch.as_tensor([int(model.jnt_dofadr[model.joint(j).id]) for j in JOINT_NAMES], device=dev)
    act = torch.as_tensor([model.actuator(j).id for j in JOINT_NAMES], device=dev)
    key_q = torch.as_tensor(model.key(RESET_KEYFRAME).qpos, dtype=torch.float32, device=dev)
    mask = torch.ones(n, dtype=torch.bool, device=dev)
    delays = candidates.delay_ticks.cpu().numpy()
    bands = candidates.deadband_ticks.cpu().numpy()
    combos: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for i in range(n):
        combos.setdefault((int(delays[i]), tuple(int(b) for b in bands[i])), []).append(i)
    outs = []
    for seg in segs:
        T = len(seg)
        ctrl_np = np.empty((n, T, N_JOINTS), dtype=np.float32)
        for (d, b), worlds in combos.items():
            ctrl_np[worlds] = tmap.ctrl_from_ticks(effective_ctrl_ticks(seg, d, np.asarray(b)))
        ctrl = torch.as_tensor(ctrl_np, device=dev)
        q_np, qd_np = initial_states(seg, tmap, dt)
        q = torch.as_tensor(q_np, dtype=torch.float32, device=dev)
        qd = torch.as_tensor(qd_np, dtype=torch.float32, device=dev)
        qpos_log = torch.empty((n, T, N_JOINTS), dtype=torch.float32, device=dev)
        qpos_log[:, 0] = q[0]
        qpos0 = key_q.clone()
        qvel0 = torch.zeros(model.nv, dtype=torch.float32, device=dev)
        ctrl0 = torch.zeros(model.nu, dtype=torch.float32, device=dev)
        for w in _window_starts(T, shooting_window_ticks):
            qpos0[qadr] = q[w]
            qvel0[dadr] = qd[w]
            ctrl0[act] = ctrl[0, w]
            ws.reset_worlds(mask, qpos0, qvel0, ctrl0)
            for k in range(w, min(w + shooting_window_ticks, T - 1)):
                ws.ctrl[:, act] = ctrl[:, k]
                ws.step(nsub)
                qpos_log[:, k + 1] = ws.qpos[:, qadr]
        ticks = tmap.q_to_ticks_torch(qpos_log.double(), quantize=quantize)
        ticks[:, 0] = torch.as_tensor(seg.present_ticks[0], device=dev).to(ticks.dtype)
        outs.append(ticks)
    return torch.cat(outs, dim=1)


def rms_pct(sim_present: np.ndarray, real_present: np.ndarray, calibration: dict[str, MotorCalibration]) -> np.ndarray:
    """Per joint ``RMS(sim - real)`` over rows ``1..T-1`` (row 0 is the shared start), % of the calibrated range."""
    sim = np.asarray(sim_present, dtype=np.float64)
    real = np.asarray(real_present, dtype=np.float64)
    if sim.shape != real.shape or sim.shape[-1] != N_JOINTS or sim.shape[-2] < 2:
        raise ValueError(f"sim {sim.shape} / real {real.shape} must be matching [..., T >= 2, 6]")
    span = np.array([calibration[n].range_max - calibration[n].range_min for n in JOINT_NAMES], dtype=np.float64)
    err = sim[..., 1:, :] - real[..., 1:, :]
    return np.sqrt(np.mean(err**2, axis=-2)) / span * 100.0


def pooled_rms_pct(
    sims: Sequence[np.ndarray], segs: Sequence[TickSegment], calibration: dict[str, MotorCalibration]
) -> np.ndarray:
    """``rms_pct`` over the rows ``1..T-1`` of all segments pooled (the per-joint figure that is gated)."""
    span = np.array([calibration[n].range_max - calibration[n].range_min for n in JOINT_NAMES], dtype=np.float64)
    sq = np.zeros(N_JOINTS)
    cnt = 0
    for sim, seg in zip(sims, segs, strict=True):
        e = np.asarray(sim, dtype=np.float64)[1:] - seg.present_ticks[1:].astype(np.float64)
        sq += (e**2).sum(axis=0)
        cnt += len(e)
    return np.sqrt(sq / max(cnt, 1)) / span * 100.0


def as_joint_dict(x: np.ndarray) -> dict[str, float]:
    return {n: float(v) for n, v in zip(JOINT_NAMES, np.asarray(x, dtype=np.float64), strict=True)}


__all__ = [
    "RESET_KEYFRAME",
    "SHOOTING_WINDOW_TICKS",
    "TickMap",
    "as_joint_dict",
    "effective_ctrl_ticks",
    "initial_states",
    "pooled_rms_pct",
    "replay_segment_cpu",
    "replay_segments_warp",
    "rms_pct",
    "steps_per_tick",
]

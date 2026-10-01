"""``MujocoBus``: the F2 ``Bus`` protocol on top of MuJoCo (plan F10).

The servo core (``ServoCore.tick``) talks to a ``Bus``; here the "servos" are the six position actuators of the
scene. qpos <-> ticks goes through the SAME calibration dict as the real arm: arm joints through
``ticks_to_rad`` / ``rad_to_ticks`` (mid-range tick 2047 = 0 rad), the gripper through the 0-100 % mapping in
``gripper_map``. The caller alternates ``write_goal`` / ``step_ticks`` / ``read_block`` exactly like the
real 30 Hz loop.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import mujoco
import numpy as np

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, So101Config, rad_to_ticks, ticks_to_rad
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.sim.gripper_map import gripper_rad_to_ticks, gripper_ticks_per_rad, gripper_ticks_to_rad

GRIPPER = N_JOINTS - 1
DEFAULT_TEMPERATURE_C = 35  # trivial thermal model: constant, overridable via ``MujocoBus.temperature_c``
SUPPLY_VOLTAGE_DV = 120  # 12.0 V in 0.1 V units
MOVING_EPS_RAD_S = 1e-3
LOAD_FULL_SCALE = 1000  # 1000 = 100 % of the actuator force range
# MuJoCo warnings that mean the state is no longer finite/sane. MuJoCo silently resets such a state and writes
# MUJOCO_LOG.TXT into the cwd; ``step_ticks`` turns them into a hard error instead.
_FATAL_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)


class MujocoBus(Bus):
    """Simulated servo bus. Single-threaded; not safe to share between processes."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        calibration: dict[str, MotorCalibration],
        cfg: So101Config,
        *,
        action_delay_ticks: int = 0,
        seed: int = 0,
    ) -> None:
        if action_delay_ticks < 0:
            raise ValueError("action_delay_ticks must be >= 0")
        self.model = model
        self.data = data
        self.calibration = calibration
        self.cfg = cfg
        self.action_delay_ticks = action_delay_ticks
        self.rng = np.random.default_rng(seed)

        self._jnt = [model.joint(n) for n in JOINT_NAMES]
        self.qpos_adr = np.array([j.qposadr[0] for j in self._jnt], dtype=np.intp)
        self.dof_adr = np.array([j.dofadr[0] for j in self._jnt], dtype=np.intp)
        act_ids = []
        for n in JOINT_NAMES:
            a = model.actuator(n)
            if int(model.actuator_trnid[a.id, 0]) != model.joint(n).id:
                raise ValueError(f"actuator {n!r} does not drive joint {n!r}")
            act_ids.append(a.id)
        self.act_ids = np.array(act_ids, dtype=np.intp)
        self._ids = [calibration[n].id for n in JOINT_NAMES]
        self._force_scale = np.maximum(np.abs(model.actuator_forcerange[self.act_ids]).max(axis=1), 1e-9)
        self._ticks_per_rad = np.full(N_JOINTS, 4096.0 / (2.0 * np.pi))
        self._ticks_per_rad[GRIPPER] = gripper_ticks_per_rad(calibration)
        # d(ticks)/d(rad) is negative for drive_mode = 1 (arm joints and gripper alike), so velocity follows the sign.
        self._vel_sign = np.array([-1.0 if calibration[n].drive_mode else 1.0 for n in JOINT_NAMES])
        self._steps_per_tick = max(1, round((1.0 / cfg.rates.servo_hz) / model.opt.timestep))

        self.temperature_c = np.full(N_JOINTS, DEFAULT_TEMPERATURE_C, dtype=np.int32)
        self.written_goals: list[np.ndarray] = []
        self._connected = False
        self._torque = np.ones(N_JOINTS, dtype=bool)
        self._saved_gain = model.actuator_gainprm[self.act_ids].copy()
        self._saved_bias = model.actuator_biasprm[self.act_ids].copy()
        self._delay: deque[np.ndarray] = deque()
        self._goal: np.ndarray | None = None  # last accepted goal (raw ticks); None until the first write
        self._regs: dict[tuple[str, int], int] = {}
        for n in JOINT_NAMES:
            c = calibration[n]
            self._regs[("Homing_Offset", c.id)] = c.homing_offset
            self._regs[("Min_Position_Limit", c.id)] = c.range_min
            self._regs[("Max_Position_Limit", c.id)] = c.range_max
            self._regs[("Torque_Enable", c.id)] = 1

    # ---- conversions -------------------------------------------------------------------------------
    def ticks_to_q(self, ticks: np.ndarray) -> np.ndarray:
        q = ticks_to_rad(ticks, self.calibration)
        q[GRIPPER] = gripper_ticks_to_rad(float(ticks[GRIPPER]), self.calibration)
        return q

    def q_to_ticks(self, q: np.ndarray) -> np.ndarray:
        t = rad_to_ticks(q, self.calibration)
        t[GRIPPER] = gripper_rad_to_ticks(float(q[GRIPPER]), self.calibration)
        return np.clip(t, 0, 4095).astype(np.int32)

    @property
    def steps_per_tick(self) -> int:
        return self._steps_per_tick

    # ---- Bus protocol ------------------------------------------------------------------------------
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        """Never changes torque implicitly."""
        self._connected = False

    def ping_all(self) -> dict[int, bool]:
        return dict.fromkeys(self._ids, self._connected)

    def read_block(self, out: BlockRead) -> bool:
        if not self._connected:
            out.ok = False
            return False
        d = self.data
        q = d.qpos[self.qpos_adr]
        qd = d.qvel[self.dof_adr]
        out.position[:] = self.q_to_ticks(q)
        out.velocity[:] = np.rint(qd * self._ticks_per_rad * self._vel_sign).astype(np.int32)
        force = d.actuator_force[self.act_ids]
        out.load[:] = np.clip(np.rint(force / self._force_scale * LOAD_FULL_SCALE), -LOAD_FULL_SCALE, LOAD_FULL_SCALE)
        out.voltage_dv[:] = SUPPLY_VOLTAGE_DV
        out.temperature_c[:] = self.temperature_c
        out.status[:] = 0
        out.moving[:] = np.abs(qd) > MOVING_EPS_RAD_S
        out.t_mono_ns = round(d.time * 1e9)
        out.ok = True
        return True

    def write_goal(self, goal_ticks: np.ndarray) -> bool:
        """Raw ticks in, ``ctrl`` radians out (through the optional delay FIFO). False on bad input."""
        if not self._connected:
            return False
        g = np.asarray(goal_ticks)
        if g.shape != (N_JOINTS,) or not np.issubdtype(g.dtype, np.integer):
            return False
        if g.min() < 0 or g.max() > 4095:
            return False
        raw = g.astype(np.int32, copy=True)
        self.written_goals.append(raw)
        self._goal = raw.copy()
        # The servo clamps its goal to its Min/Max_Position_Limit registers (calibrated range_min/max by default).
        lo_t = np.array([self._regs[("Min_Position_Limit", i)] for i in self._ids], dtype=np.int32)
        hi_t = np.array([self._regs[("Max_Position_Limit", i)] for i in self._ids], dtype=np.int32)
        goal = np.clip(raw, lo_t, hi_t)
        if self.action_delay_ticks:
            if not self._delay:  # first write: until the first goal arrives the arm holds where it is
                hold = self.q_to_ticks(self.data.qpos[self.qpos_adr])
                self._delay.extend(hold.copy() for _ in range(self.action_delay_ticks))
            self._delay.append(goal)
            goal = self._delay.popleft()
        ctrl = self.ticks_to_q(goal)
        lo, hi = self.model.actuator_ctrlrange[self.act_ids].T
        self.data.ctrl[self.act_ids] = np.clip(ctrl, lo, hi)
        return True

    def set_torque(self, enabled: bool, ids: Sequence[int] | None = None) -> bool:
        """Disabling zeroes the actuator gain/bias (copy kept), so the arm falls under gravity."""
        if not self._connected:
            return False
        idx = range(N_JOINTS) if ids is None else [self._ids.index(i) for i in ids]
        for i in idx:
            a = self.act_ids[i]
            if enabled:
                self.model.actuator_gainprm[a] = self._saved_gain[i]
                self.model.actuator_biasprm[a] = self._saved_bias[i]
            else:
                self.model.actuator_gainprm[a] = 0.0
                self.model.actuator_biasprm[a] = 0.0
            self._torque[i] = bool(enabled)
            self._regs[("Torque_Enable", self._ids[i])] = int(enabled)
        return True

    def read_reg(self, name: str, motor_id: int) -> int:
        """Live registers as on ``FakeBus``: position, goal and torque come from the simulation state."""
        if motor_id not in self._ids:
            raise ValueError(f"unknown motor id {motor_id}")
        j = self._ids.index(motor_id)
        if name == "Present_Position":
            return int(self.q_to_ticks(self.data.qpos[self.qpos_adr])[j])
        if name == "Goal_Position":
            if self._goal is None:  # nothing written yet: the servo holds where it is
                return int(self.q_to_ticks(self.data.qpos[self.qpos_adr])[j])
            return int(self._goal[j])
        if name == "Torque_Enable":
            return int(self._torque[j])
        if name == "Present_Temperature":
            return int(self.temperature_c[j])
        if name == "Present_Voltage":
            return SUPPLY_VOLTAGE_DV
        try:
            return self._regs[(name, motor_id)]
        except KeyError:
            raise KeyError(f"MujocoBus: register {name!r} of motor {motor_id} was never written") from None

    def write_reg(self, name: str, motor_id: int, value: int) -> None:
        if motor_id not in self._ids:
            raise ValueError(f"unknown motor id {motor_id}")
        if name == "Torque_Enable":
            self.set_torque(bool(value), [motor_id])
            return
        self._regs[(name, motor_id)] = int(value)

    def read_calibration_regs(self) -> dict[str, MotorCalibration]:
        """The sim 'servo EEPROM' always agrees with the calibration it was built from."""
        return dict(self.calibration)

    # ---- sim extras --------------------------------------------------------------------------------
    @property
    def torque_enabled(self) -> np.ndarray:
        return self._torque.copy()

    def step_ticks(self, n: int = 1) -> None:
        """Advance physics by ``n`` servo ticks (``n x round((1/servo_hz)/timestep)`` mj_steps)."""
        before = [self.data.warning[w].number for w in _FATAL_WARNINGS]
        for _ in range(int(n) * self._steps_per_tick):
            mujoco.mj_step(self.model, self.data)
        for w, b in zip(_FATAL_WARNINGS, before, strict=True):
            if self.data.warning[w].number > b:
                raise RuntimeError(f"MuJoCo simulation unstable: {mujoco.mju_warningText(w, 0)}")

    def reset_to_keyframe(self, name: str = "rest") -> None:
        """Reset physics to a keyframe, clear the delay FIFO and the intercept, restore torque."""
        self._restore_gains()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(name).id)
        mujoco.mj_forward(self.model, self.data)
        self._delay.clear()
        self.written_goals.clear()
        self._goal = None

    def _restore_gains(self) -> None:
        self.model.actuator_gainprm[self.act_ids] = self._saved_gain
        self.model.actuator_biasprm[self.act_ids] = self._saved_bias
        self._torque[:] = True
        for i in self._ids:
            self._regs[("Torque_Enable", i)] = 1

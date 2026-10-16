"""CPU MuJoCo stand-in for :class:`~so101_rl.sim.warp_model.WarpSim` (D2b, scripted die demos).

``BatchedGoalEnv(device="cpu")`` builds this instead of ``WarpSim`` (which needs CUDA), so the batched env -- servo
chain, simulated tracker with its occluders and visibility rule, reward, success, debounce, resets, the die start
stages -- runs unchanged on the CPU with a handful of worlds: the scripted die controller is measured there
(``scripts/scripted_die_eval.py``) and the CPU tests drive a real die episode without a GPU.

One ``mujoco.MjModel`` copy and one ``MjData`` per world, stepped one after the other with ``mj_step`` (float64).
The torch tensors ``qpos`` / ``qvel`` / ``ctrl`` / ``actuator_force`` are float32 ``[N, ...]`` mirrors with WarpSim's
names: ``ctrl`` is written by the env and copied into every ``MjData`` before a step; the state is copied back after
it. Derived quantities (``body_xpos``, ``site_xpos``, contacts) are those of the last ``mj_step`` (computed at the
start of the step, as MJWarp's ``step`` leaves them) or of the ``mj_forward`` of a reset.

Not bitwise equal to MJWarp (float64 CPU MuJoCo vs float32 Warp; the S0 CPU/Warp parity tests bound the
difference on the arm); a CPU scripted-success number is a sim estimate, not a GPU-env measurement. No overflow
buffers exist on the CPU: :meth:`check_overflow` only counts. ``set_world_params`` applies the same field mapping as
``WarpSim.set_world_params`` to each world's own model copy and runs ``mj_setConst``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import mujoco
import numpy as np
import torch

from so101_rl.config.schema import JOINT_NAMES

__all__ = ["CpuSim"]


class CpuSim:
    """``WarpSim``'s interface (the part ``BatchedGoalEnv`` uses) on CPU MuJoCo for ``nworld`` worlds."""

    def __init__(self, model: mujoco.MjModel, cfg: Any, nworld: int, *, device: str = "cpu") -> None:
        if nworld < 1:
            raise ValueError(f"nworld must be >= 1, got {nworld}")
        if torch.device(device).type != "cpu":
            raise ValueError(f"CpuSim runs on the CPU, got device {device!r}")
        self.device = torch.device("cpu")
        self.model = model
        self._cfg = cfg
        self.nworld = int(nworld)
        self._models = [_copy_model(model) for _ in range(self.nworld)]
        self._datas = [mujoco.MjData(m) for m in self._models]
        f32 = {"dtype": torch.float32}
        n = self.nworld
        self._qpos = torch.zeros((n, model.nq), **f32)
        self._qvel = torch.zeros((n, model.nv), **f32)
        self._ctrl = torch.zeros((n, model.nu), **f32)
        self._actuator_force = torch.zeros((n, model.nu), **f32)
        self._xpos = torch.zeros((n, model.nbody, 3), **f32)
        self._xquat = torch.zeros((n, model.nbody, 4), **f32)
        self._site_xpos = torch.zeros((n, model.nsite, 3), **f32)
        self._nefc = torch.zeros(n, dtype=torch.int32)
        self._arm_dofs = [int(model.jnt_dofadr[model.joint(j).id]) for j in JOINT_NAMES]
        self._actuators = [model.actuator(j).id for j in JOINT_NAMES]
        self._nominal_die_mass = float(model.body_mass[model.body("die").id])
        self._nominal_die_inertia = model.body_inertia[model.body("die").id].copy()
        self.max_nefc_seen = 0
        self.max_nacon_seen = 0
        self.solver_limit_worlds = 0
        self.overflow_checks = 0
        self._closed = False
        for w, d in enumerate(self._datas):
            mujoco.mj_forward(self._models[w], d)
            self._pull(w)

    # ------------------------------------------------------------------ views (WarpSim names)
    @property
    def qpos(self) -> torch.Tensor:
        return self._qpos

    @property
    def qvel(self) -> torch.Tensor:
        return self._qvel

    @property
    def ctrl(self) -> torch.Tensor:
        return self._ctrl

    @property
    def actuator_force(self) -> torch.Tensor:
        return self._actuator_force

    @property
    def nefc(self) -> torch.Tensor:
        return self._nefc

    @property
    def nacon(self) -> torch.Tensor:
        return torch.tensor([sum(int(d.ncon) for d in self._datas)], dtype=torch.int32)

    @property
    def vram_bytes(self) -> int:
        return 0

    def site_xpos(self, site: str) -> torch.Tensor:
        return self._site_xpos[:, self.model.site(site).id]

    def body_xpos(self, body: str) -> torch.Tensor:
        return self._xpos[:, self.model.body(body).id]

    def body_xquat(self, body: str) -> torch.Tensor:
        return self._xquat[:, self.model.body(body).id]

    # ------------------------------------------------------------------ internals
    def _pull(self, w: int) -> None:
        d = self._datas[w]
        self._qpos[w] = torch.from_numpy(d.qpos)
        self._qvel[w] = torch.from_numpy(d.qvel)
        self._actuator_force[w] = torch.from_numpy(d.actuator_force)
        self._xpos[w] = torch.from_numpy(d.xpos)
        self._xquat[w] = torch.from_numpy(d.xquat)
        self._site_xpos[w] = torch.from_numpy(d.site_xpos)
        self._nefc[w] = int(d.nefc)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("CpuSim is closed")

    # ------------------------------------------------------------------ stepping
    def step(self, nsteps: int) -> None:
        """``nsteps`` ``mj_step`` calls for every world with the current ``ctrl``."""
        self._check_open()
        nsteps = int(nsteps)
        if nsteps < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")
        ctrl = self._ctrl.double().numpy()
        for w, (m, d) in enumerate(zip(self._models, self._datas, strict=True)):
            d.ctrl[:] = ctrl[w]
            for _ in range(nsteps):
                mujoco.mj_step(m, d)
            self._pull(w)

    def check_overflow(self) -> None:
        """No contact buffers to overflow on the CPU: counts the check and the largest constraint count."""
        self._check_open()
        self.overflow_checks += 1
        self.max_nefc_seen = max(self.max_nefc_seen, int(self._nefc.max()))
        self.max_nacon_seen = max(self.max_nacon_seen, int(self.nacon[0]))

    # ------------------------------------------------------------------ resets and parameters
    def reset_worlds(self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor) -> None:
        """``WarpSim.reset_worlds``: state of the masked worlds (full ``[N, ...]`` or one broadcast row), then
        ``mj_forward``."""
        self._check_open()
        mask = mask.to(device="cpu", dtype=torch.bool)
        if mask.shape != (self.nworld,):
            raise ValueError(f"mask must be bool[{self.nworld}], got {tuple(mask.shape)}")
        n = self.nworld
        q = torch.as_tensor(qpos, dtype=torch.float64, device="cpu").expand(n, -1).numpy()
        v = torch.as_tensor(qvel, dtype=torch.float64, device="cpu").expand(n, -1).numpy()
        c = torch.as_tensor(ctrl, dtype=torch.float64, device="cpu").expand(n, -1).numpy()
        for w in torch.nonzero(mask).flatten().tolist():
            m, d = self._models[w], self._datas[w]
            mujoco.mj_resetData(m, d)
            d.qpos[:] = q[w]
            d.qvel[:] = v[w]
            d.ctrl[:] = c[w]
            mujoco.mj_forward(m, d)
            self._ctrl[w] = torch.from_numpy(d.ctrl)
            self._pull(w)

    def set_world_params(self, params: Any, *, validate: bool = True) -> None:
        """``WarpSim.set_world_params`` field mapping, world by world (``kp kv damping frictionloss armature
        forcerange`` [N,6], ``die_mass die_friction table_friction`` [N])."""
        self._check_open()
        n = self.nworld
        raw = {k: torch.as_tensor(getattr(params, k)).detach().to("cpu", torch.float64).numpy()
               for k in ("kp", "kv", "damping", "frictionloss", "armature", "forcerange", "die_mass", "die_friction",
                         "table_friction")}  # fmt: skip
        for k, x in raw.items():
            want = (n, 6) if x.ndim == 2 else (n,)
            if x.shape != want:
                raise ValueError(f"{k}: expected shape {want}, got {x.shape}")
            if validate and not np.isfinite(x).all():
                raise ValueError(f"{k}: non-finite value")
        act, dofs = self._actuators, self._arm_dofs
        die_b, die_g, table_g = self.model.body("die").id, self.model.geom("die_geom").id, self.model.geom("table").id
        for w, (m, d) in enumerate(zip(self._models, self._datas, strict=True)):
            kp, kv = raw["kp"][w], raw["kv"][w]
            m.actuator_gainprm[act, 0] = kp
            m.actuator_biasprm[act, 1] = -kp
            m.actuator_biasprm[act, 2] = -kv
            m.actuator_forcerange[act, 0] = -raw["forcerange"][w]
            m.actuator_forcerange[act, 1] = raw["forcerange"][w]
            m.dof_damping[dofs] = raw["damping"][w]
            m.dof_armature[dofs] = raw["armature"][w]
            m.dof_frictionloss[dofs] = raw["frictionloss"][w]
            mass = float(raw["die_mass"][w])
            m.body_mass[die_b] = mass
            m.body_inertia[die_b] = self._nominal_die_inertia * (mass / self._nominal_die_mass)
            m.geom_friction[die_g, 0] = raw["die_friction"][w]
            m.geom_friction[table_g, 0] = raw["table_friction"][w]
            qpos = d.qpos.copy()
            mujoco.mj_setConst(m, d)
            d.qpos[:] = qpos
            mujoco.mj_forward(m, d)
            self._pull(w)

    # ------------------------------------------------------------------ queries
    def contact_count(self, geoms_a: Sequence[int], geoms_b: Sequence[int]) -> torch.Tensor:
        """int32[N]: contacts between geom set A and geom set B (either order) in each world, from the last step."""
        a, b = {int(g) for g in geoms_a}, {int(g) for g in geoms_b}
        out = torch.zeros(self.nworld, dtype=torch.int32)
        for w, d in enumerate(self._datas):
            k = 0
            for i in range(int(d.ncon)):
                g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
                k += int((g1 in a and g2 in b) or (g1 in b and g2 in a))
            out[w] = k
        return out

    def copy_world_to(self, world: int, data: mujoco.MjData) -> None:
        """Copy one world's state into ``data`` (qpos, qvel, ctrl; the caller runs ``mj_forward``)."""
        if not 0 <= world < self.nworld:
            raise IndexError(f"world {world} outside 0..{self.nworld - 1}")
        src = self._datas[world]
        data.qpos[:] = src.qpos
        data.qvel[:] = src.qvel
        data.ctrl[:] = src.ctrl

    def world_data(self, world: int) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """The live ``(MjModel, MjData)`` of one world (CPU only: tests and the scripted controller's diagnostics)."""
        return self._models[world], self._datas[world]

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _copy_model(model: mujoco.MjModel) -> mujoco.MjModel:
    import copy

    return copy.copy(model)

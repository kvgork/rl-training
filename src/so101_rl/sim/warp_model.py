"""MuJoCo Warp wrapper: one MJWarp ``Model`` + ``Data`` for N worlds of the RL scene (FS2, sim-first track S0.3).

Torch sees the device state through zero-copy views (``wp.to_torch``). Stepping replays one captured CUDA graph per
distinct ``nsteps`` value. Buffer overflows RAISE :class:`ContactBufferOverflow` (MJWarp itself only prints a
warning and silently drops constraints/contacts).

MJWarp 3.14.0 API, verified against the installed package (``inspect.signature`` and the source, 2026-10-05); the
plan's names came from the 3.5 package and all still hold, plus the additions marked NEW:

- ``put_model(mjm, batch_sizes: dict[str, int] | None = None) -> Model`` - NEW ``batch_sizes``: per-field leading
  (world) dimension for Model fields whose array spec starts with ``*``; kernels index ``field[worldid % shape[0]]``.
  All FS2/FS7 parameter fields batch (``dof_damping/armature/frictionloss``, ``actuator_gainprm/biasprm/forcerange``,
  ``geom_friction``, ``geom_size``, ``body_mass/inertia``) and so do the derived ``body_subtreemass``,
  ``dof_invweight0``, ``body_invweight0``, ``actuator_acc0`` that ``set_const`` recomputes.
- ``make_data(mjm, nworld=1, nconmax=None, nccdmax=None, njmax=None, njmax_nnz=None, naconmax=None, naccdmax=None,
  nvmax=None) -> Data``; ``njmax`` is per world, ``nconmax`` per world with ``naconmax = nconmax * nworld`` shared by
  all worlds (one contact pool).
- ``put_data``, ``get_data_into(result: MjData, mjm, d, world_id=0)``, ``reset_data(m, d, reset=None)``,
  ``step(m, d)``, ``forward(m, d)``, ``kinematics(m, d)``, ``set_const(m, d, restore=True)`` (NEW: recomputes the
  qpos0-dependent constants after mass/armature changes; restores ``d.qpos`` in place).
- ``Data.nefc`` int32 ``(nworld,)``, ``Data.nacon`` int32 ``(1,)``, ``Data.njmax`` / ``Data.naconmax`` ints.
- NEW ``Data.overflow`` int32 ``(nworld,)``: a STICKY per-world ``OverflowType`` bitmask (NEFC, NJMAX_NNZ,
  BROADPHASE, NARROWPHASE, CCD, ..., ITERATIONS, LS_ITERATIONS) set every step and cleared only by ``reset_data``.
  This wrapper never calls ``reset_data`` and folds the mask into its own accumulator, so an overflow on any step
  between two checks is caught. ITERATIONS / LS_ITERATIONS are solver-convergence flags, not buffer overflows:
  counted, not raised. ``Model.opt.warn_overflow`` (compile-time bitmask) selects which overflows ``wp.printf``.
- ``wp.ScopedCapture`` around ``nsteps`` calls of ``step`` captures without a warm-up call and does not advance
  the state; ``wp.capture_launch(graph, stream)`` replays it.

Unbatched fields and ``stat.meaninertia`` (verified in the 3.14 source, ``io.put_model`` / ``set_const`` /
``solver``): ``batch_sizes`` only accepts ``Model`` fields, and ``Model.stat`` (a ``Statistic``) is created with a
leading dimension of 1 regardless. ``stat.meaninertia`` is still declared per-world (``array("*", float)``): the
solver reads ``meaninertia[worldid % shape[0]]`` (it scales the convergence tolerance) and ``set_const`` launches
one thread per row of it. This wrapper therefore replaces it with an ``[nworld]`` array right after ``put_model``
(before any graph capture), so ``set_world_params`` -> ``set_const`` gives each world its own mean inertia instead
of world 0's. Every other ``Model`` field not in :data:`BATCHED_FIELDS` is shared by all worlds.

Determinism: NOT bitwise reproducible. All worlds share one contact pool filled with atomics (``nacon``), so the
contact (and constraint-row) order within a world can change between runs and the solver sums in that order; the
float32 differences are then amplified by contact dynamics. Measured 2026-10-05 (two identical runs, 256 worlds,
1020 mj_steps of random ctrl from ``home``): every world differed, max |dqpos| 0.041 rad. Short smooth runs agree
to float32 rounding (graph replay vs eager over 102 steps: 1e-6, test). Seeded training runs are therefore
reproducible statistically, not bitwise; per-world results never depend on the other worlds' parameters.

Streams: Warp launches on its own (non-blocking) device stream, torch on its current stream. Every public call
fences both ways with events (no host sync), so torch writes to ``ctrl``/``qpos`` happen before the graph and torch
reads of ``qpos``/``site_xpos`` after it. Graphs bake in array pointers: nothing here rebinds a Model/Data array
after construction (all writes are in place through torch views).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Self

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from so101_rl.config.schema import JOINT_NAMES, SimWarpConfig

if TYPE_CHECKING:  # FS7 (SU2) defines BatchedSimParams; set_world_params only needs its attributes
    from so101_rl.sim.sysid.params import BatchedSimParams

__all__ = ["BATCHED_FIELDS", "BUFFER_OVERFLOW_MASK", "PARAM_RULES", "ContactBufferOverflow", "WarpSim", "WarpSizes"]

# Model fields allocated with a leading world dimension (per-world sysid / domain randomisation, FS7/FS8).
BATCHED_FIELDS: Final[tuple[str, ...]] = (
    "dof_damping",
    "dof_armature",
    "dof_frictionloss",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "geom_friction",
    "geom_size",
    "body_mass",
    "body_inertia",
    # derived by set_const (must be batched too, or every world writes into the same row)
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "actuator_acc0",
)

_SOLVER_FLAGS: Final[int] = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
# Every capacity overflow: data was dropped. Raised, never tolerated (FS1 overflow_policy "fail").
BUFFER_OVERFLOW_MASK: Final[int] = int(mjw.OverflowType.ALL) & ~_SOLVER_FLAGS

DIE_BODY: Final[str] = "die"
DIE_GEOM: Final[str] = "die_geom"
TABLE_GEOM: Final[str] = "table"


# set_world_params value rules: True = strictly positive, False = non-negative (all must be finite).
PARAM_RULES: Final[dict[str, bool]] = {
    "kp": True,
    "kv": False,
    "damping": False,
    "frictionloss": False,
    "armature": False,
    "forcerange": True,
    "die_mass": True,
    "die_friction": False,
    "table_friction": False,
}


def _validate_params(raw: dict[str, torch.Tensor]) -> None:
    """Raise ValueError naming every out-of-range field. Host tensors are checked on the host; device tensors are
    reduced on the device and read back with ONE copy for all of them."""
    names = list(raw)
    flags = [
        (~torch.isfinite(raw[k].double()) | ((raw[k] <= 0) if PARAM_RULES[k] else (raw[k] < 0))).any() for k in names
    ]
    bad: list[str] = [k for k, f in zip(names, flags, strict=True) if f.device.type == "cpu" and bool(f)]
    dev = [(k, f) for k, f in zip(names, flags, strict=True) if f.device.type != "cpu"]
    if dev:
        host = torch.stack([f for _, f in dev]).cpu()  # the one device->host sync
        bad += [k for (k, _), b in zip(dev, host.tolist(), strict=True) if b]
    if bad:
        rules = ", ".join(f"{k} {'> 0' if PARAM_RULES[k] else '>= 0'}" for k in bad)
        raise ValueError(f"invalid world params (non-finite or out of range): {rules}")


class ContactBufferOverflow(RuntimeError):
    """njmax / nconmax / naconmax (or another MJWarp capacity) was exceeded in at least one world."""


@dataclass(frozen=True)
class WarpSizes:
    nworld: int
    njmax: int
    nconmax: int  # per world
    naconmax: int  # all worlds (one shared contact pool)


def _flag_names(mask: int) -> list[str]:
    return [f.name for f in mjw.OverflowType if f.value and not (f.value & (f.value - 1)) and mask & f.value]


class WarpSim:
    """One MJWarp Model + Data for N worlds of the RL scene, torch views zero-copy on cuda."""

    def __init__(
        self,
        model: mujoco.MjModel,
        cfg: SimWarpConfig,
        nworld: int,
        *,
        device: str = "cuda:0",
        capture_graph: bool = True,
    ) -> None:
        if cfg.overflow_policy != "fail":  # the loader already refuses anything else; keep the invariant local
            raise ValueError(f"overflow_policy must be 'fail', got {cfg.overflow_policy!r}")
        if nworld < 1:
            raise ValueError(f"nworld must be >= 1, got {nworld}")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"WarpSim needs a CUDA device, got {device!r} (cuda available: {torch.cuda.is_available()})")
        wp.init()
        self._wp_device = wp.get_device(device)
        self._wp_stream = self._wp_device.stream
        self._torch_streams: dict[int, Any] = {}
        self._capture = bool(capture_graph)
        self._graphs: dict[Any, Any] = {}
        self._closed = False
        self.model = model
        self._cfg = cfg

        mem0 = self._mempool_used()
        with wp.ScopedDevice(self._wp_device):
            self.m = mjw.put_model(model, batch_sizes=dict.fromkeys(BATCHED_FIELDS, nworld))
            # stat is not covered by batch_sizes; meaninertia is per-world in every kernel that reads it (docstring)
            mi = wp.array(np.full(nworld, model.stat.meaninertia, dtype=np.float32), dtype=wp.float32)
            mi._is_batched = True
            self.m.stat.meaninertia = mi
            # print only real capacity overflows; solver-iteration limits are counted in stats instead
            self.m.opt.warn_overflow = BUFFER_OVERFLOW_MASK
            self.d = mjw.make_data(model, nworld=nworld, njmax=cfg.njmax, nconmax=cfg.nconmax_per_world)
        self.sizes = WarpSizes(nworld=nworld, njmax=int(self.d.njmax), nconmax=cfg.nconmax_per_world, naconmax=int(self.d.naconmax))

        t = wp.to_torch
        self._qpos, self._qvel, self._ctrl = t(self.d.qpos), t(self.d.qvel), t(self.d.ctrl)
        self._actuator_force = t(self.d.actuator_force)
        self._xpos, self._xquat, self._site_xpos = t(self.d.xpos), t(self.d.xquat), t(self.d.site_xpos)
        self._time, self._qacc, self._qacc_ws = t(self.d.time), t(self.d.qacc), t(self.d.qacc_warmstart)
        self._act = t(self.d.act) if model.na else None
        self._qfrc_applied, self._xfrc_applied = t(self.d.qfrc_applied), t(self.d.xfrc_applied)
        self._overflow, self._nefc, self._nacon = t(self.d.overflow), t(self.d.nefc), t(self.d.nacon)
        self._contact_geom, self._contact_world = t(self.d.contact.geom), t(self.d.contact.worldid)
        self._overflow_acc = torch.zeros(nworld, dtype=torch.int32, device=self.device)
        self._contact_idx = torch.arange(self.sizes.naconmax, device=self.device, dtype=torch.int32)
        self._geom_sets: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]] = {}

        # Per-world parameter views (shape [N, ...]); in-place writes keep the graph pointers valid.
        self._p = {name: t(getattr(self.m, name)) for name in BATCHED_FIELDS}
        self._meaninertia = t(self.m.stat.meaninertia)  # [N]
        self._arm_dofs = torch.tensor(
            [int(model.jnt_dofadr[model.joint(n).id]) for n in JOINT_NAMES], device=self.device, dtype=torch.long
        )
        self._actuators = torch.tensor([model.actuator(n).id for n in JOINT_NAMES], device=self.device, dtype=torch.long)
        self._nominal_body_mass = self._p["body_mass"][0].clone()
        self._nominal_body_inertia = self._p["body_inertia"][0].clone()

        # Diagnostics, updated by check_overflow().
        self.max_nefc_seen = 0
        self.max_nacon_seen = 0
        self.solver_limit_worlds = 0  # worlds that hit the solver iteration / linesearch limit since construction
        self.overflow_checks = 0

        self._vram = max(0, self._mempool_used() - mem0)
        self._graph_mem = 0
        self._forward()  # derived quantities (xpos, site_xpos, ...) for the initial qpos0 state

    # ------------------------------------------------------------------ views
    @property
    def qpos(self) -> torch.Tensor:  # [N, nq] float32, zero-copy
        return self._qpos

    @property
    def qvel(self) -> torch.Tensor:  # [N, nv]
        return self._qvel

    @property
    def ctrl(self) -> torch.Tensor:  # [N, nu]
        return self._ctrl

    @property
    def actuator_force(self) -> torch.Tensor:  # [N, nu]
        return self._actuator_force

    @property
    def nefc(self) -> torch.Tensor:  # int32[N] constraint rows of the last step (diagnostics)
        return self._nefc

    @property
    def nacon(self) -> torch.Tensor:  # int32[1] contacts of the last step, all worlds (diagnostics)
        return self._nacon

    @property
    def vram_bytes(self) -> int:
        """Warp mempool bytes allocated by this instance (Model, Data, graphs). Excludes the CUDA context and torch."""
        return self._vram + self._graph_mem

    def site_xpos(self, site: str) -> torch.Tensor:  # [N, 3] world frame (== base frame in the scene)
        return self._site_xpos[:, self.model.site(site).id]

    def body_xpos(self, body: str) -> torch.Tensor:  # [N, 3]
        return self._xpos[:, self.model.body(body).id]

    def body_xquat(self, body: str) -> torch.Tensor:  # [N, 4] wxyz
        return self._xquat[:, self.model.body(body).id]

    # ------------------------------------------------------------------ streams
    def _torch_stream(self) -> Any:
        ts = torch.cuda.current_stream(self.device)
        ws = self._torch_streams.get(ts.cuda_stream)
        if ws is None:
            ws = self._torch_streams[ts.cuda_stream] = wp.stream_from_torch(ts)
        return ws

    def _warp_after_torch(self) -> None:
        self._wp_stream.wait_stream(self._torch_stream())

    def _torch_after_warp(self) -> None:
        self._torch_stream().wait_stream(self._wp_stream)

    def _mempool_used(self) -> int:
        if wp.is_mempool_enabled(self._wp_device):
            return int(wp.get_mempool_used_mem_current(self._wp_device))
        free, total = torch.cuda.mem_get_info(self.device)  # mempool off: device-wide figure (includes others)
        return int(total - free)

    def _run(self, key: Any, fn) -> None:
        """Run ``fn`` on the warp stream, through a graph captured on first use when capture is on."""
        self._check_open()
        self._warp_after_torch()
        if self._capture:
            graph = self._graphs.get(key)
            if graph is None:
                before = self._mempool_used()
                with wp.ScopedCapture(device=self._wp_device, stream=self._wp_stream) as cap:
                    fn()
                graph = self._graphs[key] = cap.graph
                self._graph_mem += max(0, self._mempool_used() - before)
            wp.capture_launch(graph, stream=self._wp_stream)
        else:
            with wp.ScopedStream(self._wp_stream):
                fn()
        self._torch_after_warp()

    def _forward(self) -> None:
        self._run("forward", lambda: mjw.forward(self.m, self.d))

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("WarpSim is closed")

    # ------------------------------------------------------------------ stepping
    def step(self, nsteps: int) -> None:
        """``nsteps`` mj_steps for every world (one captured CUDA graph per distinct ``nsteps``)."""
        nsteps = int(nsteps)
        if nsteps < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")

        def body() -> None:
            for _ in range(nsteps):
                mjw.step(self.m, self.d)

        self._run(("step", nsteps), body)

    def check_overflow(self) -> None:
        """Raise :class:`ContactBufferOverflow` if any world overflowed a buffer since construction.

        One device->host copy. Uses MJWarp's sticky per-world bitmask (so overflows on steps between two checks
        are caught) plus the current ``nefc`` / ``nacon`` counters against ``njmax`` / ``naconmax``. The wrapper's
        accumulator is never cleared: once a world overflowed, every later check (and ``close``) raises again.
        """
        self._check_open()
        self._torch_after_warp()
        self._overflow_acc.bitwise_or_(self._overflow)
        packed = torch.cat((self._overflow_acc, self._nefc, self._nacon)).cpu().numpy()
        n = self.sizes.nworld
        acc, nefc, nacon = packed[:n], packed[n : 2 * n], int(packed[2 * n])
        self.overflow_checks += 1
        self.max_nefc_seen = max(self.max_nefc_seen, int(nefc.max()))
        self.max_nacon_seen = max(self.max_nacon_seen, nacon)
        self.solver_limit_worlds = int(np.count_nonzero(acc & _SOLVER_FLAGS))
        problems: list[str] = []
        bad = np.flatnonzero(acc & BUFFER_OVERFLOW_MASK)
        for w in bad[:5]:
            problems.append(f"world {int(w)}: {'|'.join(_flag_names(int(acc[w])))}")
        over = np.flatnonzero(nefc > self.sizes.njmax)
        for w in over[:5]:
            problems.append(f"world {int(w)}: nefc {int(nefc[w])} > njmax {self.sizes.njmax}")
        if nacon > self.sizes.naconmax:
            problems.append(f"nacon {nacon} > naconmax {self.sizes.naconmax} (nconmax {self.sizes.nconmax}/world)")
        if problems:
            more = max(len(bad), len(over)) - 5
            tail = f" (+{more} more worlds)" if more > 0 else ""
            raise ContactBufferOverflow(
                f"MuJoCo Warp buffer overflow in {max(len(bad), len(over), 1)} world(s): "
                + "; ".join(problems)
                + tail
                + f". Raise sim_warp.njmax ({self.sizes.njmax}) / nconmax_per_world ({self.sizes.nconmax}) in so101.yaml."
            )

    # ------------------------------------------------------------------ resets and parameters
    def reset_worlds(self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor) -> None:
        """Set the state of the worlds in ``mask`` (bool[N]) and recompute their derived quantities (mj_forward).

        ``qpos`` / ``qvel`` / ``ctrl`` are full-size ``[N, nq]`` / ``[N, nv]`` / ``[N, nu]`` (rows outside ``mask`` are
        ignored) or one row broadcast to every masked world. Fixed shapes keep this free of host syncs. Unmasked worlds
        are untouched: ``mjw.forward`` does not write ``qacc_warmstart``, and ``step`` recomputes the rest.
        """
        self._check_open()
        self._torch_after_warp()
        mask = mask.to(device=self.device, dtype=torch.bool)
        if mask.shape != (self.sizes.nworld,):
            raise ValueError(f"mask must be bool[{self.sizes.nworld}], got {tuple(mask.shape)}")
        m2 = mask[:, None]
        # Keep the masked worlds' overflow history, then clear their sticky bits (the accumulator remembers them).
        self._overflow_acc.bitwise_or_(self._overflow)
        self._overflow.masked_fill_(mask, 0)
        self._qpos.copy_(torch.where(m2, qpos.to(self._qpos), self._qpos))
        self._qvel.copy_(torch.where(m2, qvel.to(self._qvel), self._qvel))
        self._ctrl.copy_(torch.where(m2, ctrl.to(self._ctrl), self._ctrl))
        for view in (self._qacc, self._qacc_ws, self._qfrc_applied):
            view.masked_fill_(m2, 0.0)
        self._xfrc_applied.masked_fill_(mask.view(-1, *([1] * (self._xfrc_applied.dim() - 1))), 0.0)
        self._time.masked_fill_(mask, 0.0)
        if self._act is not None:
            self._act.masked_fill_(m2, 0.0)
        self._forward()

    def set_world_params(self, params: BatchedSimParams, *, validate: bool = True) -> None:
        """Per-world servo, die and table parameters (FS7 ``BatchedSimParams``; attributes used, no import).

        ``kp``/``kv`` -> position actuator ``gainprm[0] = kp``, ``biasprm[1] = -kp``, ``biasprm[2] = -kv``;
        ``damping``/``armature``/``frictionloss`` -> the six arm/gripper dofs; ``forcerange`` -> ``±F``;
        ``die_mass`` -> die ``body_mass`` (inertia scaled by the same ratio); ``die_friction`` /
        ``table_friction`` -> sliding friction of ``die_geom`` / ``table``. MuJoCo combines a contact's friction as
        the max of the two geoms (the jaw boxes have 1.2), so randomise ``die_friction`` above the partner geoms'
        values or lower those (FS8 decides; this method only writes the per-geom values). ``delay_ticks`` / ``deadband_ticks`` are
        servo-chain parameters (env side), not physics. Runs ``mjw.set_const`` so the derived constants
        (subtree mass, invweight0, actuator_acc0, meaninertia) follow mass and armature.

        Writes are in place, so captured step graphs pick the new values up on their next replay (test-pinned).

        ``validate`` (default on) rejects non-finite values and out-of-range ones (:data:`PARAM_RULES`: ``kp``,
        ``forcerange``, ``die_mass`` > 0; the rest >= 0) BEFORE anything is written, naming the field. Inputs on the
        host (numpy, CPU tensors — the FS7/FS8 samplers' output) are checked on the host: no device sync. Inputs
        already on the GPU cost exactly one device->host sync for all fields together; a caller that validated at
        its sampler passes ``validate=False`` to keep resets sync-free.
        """
        self._check_open()
        n = self.sizes.nworld
        shapes = {"kp": (n, 6), "kv": (n, 6), "damping": (n, 6), "frictionloss": (n, 6), "armature": (n, 6),
                  "forcerange": (n, 6), "die_mass": (n,), "die_friction": (n,), "table_friction": (n,)}
        raw: dict[str, torch.Tensor] = {}
        for name, shape in shapes.items():
            x = torch.as_tensor(getattr(params, name))
            if tuple(x.shape) != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {tuple(x.shape)}")
            raw[name] = x
        if validate:
            _validate_params(raw)
        self._torch_after_warp()
        v = {k: x.to(device=self.device, dtype=torch.float32) for k, x in raw.items()}
        p = self._p
        kp, kv = v["kp"], v["kv"]
        act = self._actuators
        p["actuator_gainprm"][:, act, 0] = kp
        p["actuator_biasprm"][:, act, 1] = -kp
        p["actuator_biasprm"][:, act, 2] = -kv
        fr = v["forcerange"]
        p["actuator_forcerange"][:, act, 0] = -fr
        p["actuator_forcerange"][:, act, 1] = fr
        dofs = self._arm_dofs
        p["dof_damping"][:, dofs] = v["damping"]
        p["dof_armature"][:, dofs] = v["armature"]
        p["dof_frictionloss"][:, dofs] = v["frictionloss"]
        die_b, die_g, table_g = self.model.body(DIE_BODY).id, self.model.geom(DIE_GEOM).id, self.model.geom(TABLE_GEOM).id
        mass = v["die_mass"]
        p["body_mass"][:, die_b] = mass
        p["body_inertia"][:, die_b] = self._nominal_body_inertia[die_b] * (mass / self._nominal_body_mass[die_b])[:, None]
        p["geom_friction"][:, die_g, 0] = v["die_friction"]
        p["geom_friction"][:, table_g, 0] = v["table_friction"]
        self._warp_after_torch()
        with wp.ScopedDevice(self._wp_device), wp.ScopedStream(self._wp_stream):
            mjw.set_const(self.m, self.d)  # not captured: allocates scratch, runs rarely (episode resets)
        self._torch_after_warp()

    # ------------------------------------------------------------------ queries
    def contact_count(self, geoms_a: Sequence[int], geoms_b: Sequence[int]) -> torch.Tensor:
        """int32[N]: contacts between geom set A and geom set B (either order) in each world, from the last step."""
        self._check_open()
        self._torch_after_warp()
        ng = self.model.ngeom
        key = (tuple(int(g) for g in geoms_a), tuple(int(g) for g in geoms_b))
        sets = self._geom_sets.get(key)
        if sets is None:  # built once per geom-set pair: no host->device copy on later calls
            for g in (*key[0], *key[1]):
                if not 0 <= g < ng:
                    raise IndexError(f"geom id {g} outside 0..{ng - 1}")
            in_a = torch.zeros(ng + 1, dtype=torch.bool, device=self.device)  # index ng = "no geom" (-1, flex)
            in_b = torch.zeros_like(in_a)
            in_a[torch.as_tensor(key[0], dtype=torch.long, device=self.device)] = True
            in_b[torch.as_tensor(key[1], dtype=torch.long, device=self.device)] = True
            sets = self._geom_sets[key] = (in_a, in_b)
        in_a, in_b = sets
        g = self._contact_geom.long()
        g = torch.where(g < 0, ng, g)
        hit = (in_a[g[:, 0]] & in_b[g[:, 1]]) | (in_b[g[:, 0]] & in_a[g[:, 1]])
        hit &= self._contact_idx < self._nacon.clamp(max=self.sizes.naconmax)
        world = self._contact_world.long().clamp(0, self.sizes.nworld - 1)
        out = torch.zeros(self.sizes.nworld, dtype=torch.int32, device=self.device)
        out.scatter_add_(0, world, hit.to(torch.int32))
        return out

    def copy_world_to(self, world: int, data: mujoco.MjData) -> None:
        """Copy one world's state into a CPU ``MjData`` (rendering / debugging); synchronises."""
        self._check_open()
        if not 0 <= world < self.sizes.nworld:
            raise IndexError(f"world {world} outside 0..{self.sizes.nworld - 1}")
        self._warp_after_torch()
        wp.synchronize_stream(self._wp_stream)
        mjw.get_data_into(data, self.model, self.d, world_id=int(world))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        if exc is None:
            self.close()
            return
        # Leaving with an exception: never replace it. A failing final check (overflow, or a CUDA error) is
        # attached to the original exception as a note, and the original propagates.
        try:
            self.close()
        except Exception as final:  # noqa: BLE001 - reported on the original exception, not swallowed
            exc.add_note(f"WarpSim.close() final overflow check also failed: {type(final).__name__}: {final}")

    def close(self) -> None:
        """Check for overflows one last time (a run cannot end with an undetected overflow), then release graphs.

        Raises :class:`ContactBufferOverflow` like :meth:`check_overflow`; the instance is closed either way. Inside
        ``with WarpSim(...)`` an exception already in flight is never masked (see ``__exit__``)."""
        if self._closed:
            return
        try:
            self.check_overflow()
        finally:
            self._graphs.clear()
            self._geom_sets.clear()
            self._closed = True

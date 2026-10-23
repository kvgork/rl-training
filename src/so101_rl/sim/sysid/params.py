"""Simulator parameters for sysid and domain randomisation (FS7, sim-first track S2.1).

:class:`SimParams` is one parameter set (the sysid result, or the MJCF prior). :class:`BatchedSimParams` holds one
set per world for :meth:`so101_rl.sim.warp_model.WarpSim.set_world_params` and the randomiser (FS8).

``calib/sim_params.yaml`` (committed, like ``calib/hand_eye.yaml``; schema in ``calib/README.md``)::

    version: 2            # SIM_PARAMS_VERSION: the file SCHEMA version; any other value is refused
    meta:                 # SimParamsMeta
      version: 0          # the parameter RELEASE: 0 = MJCF prior (shipped by SU2), >= 1 = fitted on the arm (HS3)
      heldout_rms_pct: null | {joint: pct}   # null = never verified against real data
      unidentified: [...] # parameters that must NOT be read as precise (always includes frictionloss)
      ...
    params: {kp: [6], kv: [6], damping: [6], frictionloss: [6], armature: [6], forcerange: [6],
             delay_ticks: int, deadband_ticks: [6], die_mass_kg: float, die_friction: float, table_friction: float}
    servo_group:          # ServoGroup: kp, kv, armature = prior x scale per joint (one ratio group)
      prior: {kp: [6], kv: [6], armature: [6]}
      scale: [6]
    dr:                   # DrRanges: the randomisation ranges the S3 randomiser draws from (plan §10 2026-10-06)
      servo_group_rel: 0.4            # the group (kp, kv, armature jointly) x U(1 - rel, 1 + rel)
      frictionloss_range: [lo, hi]    # N m, absolute, uniform (frictionloss is not identified)
      frictionloss_source: "..."

**Servo parameter semantics (plan §10 2026-10-06, "Servo parameter identification"; schema 2).** Through the clamp
chain the MJCF position servo's kp, kv and armature act almost only through their ratios, and frictionloss is
invisible under ±1-tick noise (SU2 findings). The sysid therefore fits ONE scale per joint applied to kp, kv and
armature jointly (the ratio group; ``servo_group.scale``, ratios kept at the prior's) plus the integer delay;
frictionloss stays at the prior and is listed in ``meta.unidentified``. Domain randomisation varies the group by
``dr.servo_group_rel`` (±40 %) around the fit and frictionloss over ``dr.frictionloss_range``.

:meth:`SimParams.from_yaml` refuses a file whose schema version differs, or whose held-out RMS is missing (``null``)
or above the compare-real gate (``MAX_RMS_PCT`` = 1 % of range since plan §10 2026-10-06; was 5 %) on any joint,
unless ``allow_unverified=True``.
The shipped version-0 file is the MJCF prior, so the batched env always has a centre; consumers that train on it
must opt in with ``allow_unverified`` (plan D-DIE-PARAMS: die-to-zone waits for release >= 1).

Mapping onto MuJoCo (identical on the CPU model and in MuJoCo Warp, see ``WarpSim.set_world_params``):
position actuator ``gainprm[0] = kp``, ``biasprm[1] = -kp``, ``biasprm[2] = -kv``; ``actuator_forcerange = ±F``;
``dof_damping`` / ``dof_armature`` / ``dof_frictionloss`` of the six joint dofs; die ``body_mass`` (inertia scaled
by the same ratio); sliding friction of ``die_geom`` / ``table``. ``delay_ticks`` (command-to-motion delay in whole
servo ticks) and ``deadband_ticks`` (target hysteresis, decision D-BACKLASH) are servo-side, applied to the target
stream by the replay / env, never written into the model.

**Order matters on the CPU:** ``MujocoBus`` caches the actuator gains when it is constructed (to zero and restore them
on torque off/on), so call :meth:`SimParams.apply_cpu` BEFORE building a ``MujocoBus`` on that model.
"""

from __future__ import annotations

import hashlib
import math
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self

import numpy as np
import yaml

from so101_rl.config import JOINT_NAMES, N_JOINTS

if TYPE_CHECKING:
    import mujoco
    import torch

__all__ = [
    "DIE_BODY",
    "DIE_GEOM",
    "FRICTIONLOSS_RANGE",
    "INT_FIELDS",
    "MAX_DELAY_TICKS",
    "MEASURED_DEADBAND_TICKS",
    "PER_JOINT_FIELDS",
    "SCALAR_FIELDS",
    "SERVO_GROUP_FIELDS",
    "SERVO_GROUP_REL",
    "SIM_PARAMS_VERSION",
    "TABLE_GEOM",
    "BatchedSimParams",
    "DrRanges",
    "ServoGroup",
    "SimParams",
    "SimParamsFile",
    "SimParamsMeta",
    "apply_deadband",
    "mjcf_sha256",
    "read_sim_params_file",
    "read_sim_params_yaml",
]

SIM_PARAMS_VERSION: Final[int] = 2  # 2 (2026-10-06): + servo_group, dr, meta.unidentified
MAX_DELAY_TICKS: Final[int] = 6  # the sysid delay grid is 0..MAX_DELAY_TICKS (FS7)
MAX_DEADBAND_TICKS: Final[int] = 64  # sanity bound only (measured 9..32 ticks on the arm, plan §10 2026-10-11)
# the measured STS3215 deadband per joint (pan, lift, elbow, wrist_flex, wrist_roll, gripper), plan §10 2026-10-11:
# the feasibility twin's default (scripts/feasibility_sim_servo.py) and the scripted controller's compensation bound
MEASURED_DEADBAND_TICKS: Final[tuple[int, ...]] = (32, 14, 16, 12, 9, 9)

PER_JOINT_FIELDS: Final[tuple[str, ...]] = ("kp", "kv", "damping", "frictionloss", "armature", "forcerange")
SCALAR_FIELDS: Final[tuple[str, ...]] = ("die_mass_kg", "die_friction", "table_friction")
INT_FIELDS: Final[tuple[str, ...]] = ("delay_ticks", "deadband_ticks")
# value rule per float field: True = strictly positive, False = non-negative (all finite)
_POSITIVE: Final[dict[str, bool]] = {
    "kp": True,
    "kv": False,
    "damping": False,
    "frictionloss": False,
    "armature": False,
    "forcerange": True,
    "die_mass_kg": True,
    "die_friction": False,
    "table_friction": False,
}

# The servo ratio group (plan §10 2026-10-06): fitted and randomised as ONE scale per joint.
SERVO_GROUP_FIELDS: Final[tuple[str, ...]] = ("kp", "kv", "armature")
SERVO_GROUP_REL: Final[float] = 0.40  # DR: the group x U(0.6, 1.4) around the fit
# DR range of the (unidentified) Coulomb friction, N m, every joint. Lower bound 0: the fit cannot tell zero from the
# prior (frictionloss x0 .. x4 moves the training objective by < 0.003 % of range on synthetic data, SU2/integration
# 2026-10-06). Upper bound 0.21 N m = 4 x the upstream MJCF prior (0.052 N m, sts3215 class of TheRobotStudio's
# so101_new_calib.xml) = 7 % of the STS3215 stall torque at 12 V (30 kg cm = 2.94 N m, the class forcerange). This
# is a CHOICE bounded by those two sourced numbers, not a measurement; HS3 may narrow it.
FRICTIONLOSS_RANGE: Final[tuple[float, float]] = (0.0, 0.21)
FRICTIONLOSS_SOURCE: Final[str] = (
    "0 .. 4 x the upstream MJCF prior 0.052 N m (TheRobotStudio so101_new_calib.xml, class sts3215) = 7 % of the "
    "STS3215 12 V stall torque 2.94 N m (30 kg cm, the MJCF class forcerange); a bounded choice, not a measurement"
)

DIE_BODY: Final[str] = "die"
DIE_GEOM: Final[str] = "die_geom"
TABLE_GEOM: Final[str] = "table"
_PARAM_KEYS: Final[tuple[str, ...]] = (*PER_JOINT_FIELDS, *INT_FIELDS, *SCALAR_FIELDS)
_META_KEYS: Final[tuple[str, ...]] = (
    "version",
    "date",
    "git_sha",
    "mjcf_sha256",
    "excitation_run",
    "fit_rms_pct",
    "heldout_rms_pct",
    "spread",
    "backend",
    "unidentified",
)


def _vec6(name: str, value: Any, *, dtype: type) -> np.ndarray:
    a = np.asarray(value)
    if a.shape != (N_JOINTS,):
        raise ValueError(f"{name}: expected shape ({N_JOINTS},), got {a.shape}")
    if dtype is int:
        if a.dtype.kind == "b" or not (a.dtype.kind in "iu" or (a.dtype.kind == "f" and np.all(a == np.rint(a)))):
            raise TypeError(f"{name}: expected integers, got {a.tolist()}")
        out = a.astype(np.int64)
    else:
        if a.dtype.kind not in "iuf" or a.dtype.kind == "b":
            raise TypeError(f"{name}: expected numbers, got dtype {a.dtype}")
        out = a.astype(np.float64)
    out = out.copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class SimParams:
    """One simulator parameter set. Arrays are float64 ``[6]`` in ``JOINT_NAMES`` order (read-only copies)."""

    kp: np.ndarray
    kv: np.ndarray
    damping: np.ndarray
    frictionloss: np.ndarray
    armature: np.ndarray
    forcerange: np.ndarray  # [6] symmetric |F| max
    delay_ticks: int  # whole servo ticks of command-to-motion delay (one value for the bus)
    deadband_ticks: np.ndarray  # [6] int, 0 = none (target hysteresis, D-BACKLASH)
    die_mass_kg: float = 0.004
    die_friction: float = 1.0
    table_friction: float = 0.8

    def __post_init__(self) -> None:
        for name in PER_JOINT_FIELDS:
            v = _vec6(name, getattr(self, name), dtype=float)
            bad = ~np.isfinite(v) | ((v <= 0) if _POSITIVE[name] else (v < 0))
            if bad.any():
                rule = "> 0" if _POSITIVE[name] else ">= 0"
                raise ValueError(f"{name}: every value must be finite and {rule}, got {v.tolist()}")
            object.__setattr__(self, name, v)
        db = _vec6("deadband_ticks", self.deadband_ticks, dtype=int)
        if (db < 0).any() or (db > MAX_DEADBAND_TICKS).any():
            raise ValueError(f"deadband_ticks must be within 0..{MAX_DEADBAND_TICKS}, got {db.tolist()}")
        object.__setattr__(self, "deadband_ticks", db)
        d = self.delay_ticks
        if isinstance(d, bool) or not isinstance(d, int | np.integer) or not 0 <= int(d) <= MAX_DELAY_TICKS:
            raise ValueError(f"delay_ticks must be an int within 0..{MAX_DELAY_TICKS}, got {d!r}")
        object.__setattr__(self, "delay_ticks", int(d))
        for name in SCALAR_FIELDS:
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, int | float | np.integer | np.floating):
                raise TypeError(f"{name}: expected a number, got {x!r}")
            x = float(x)
            if not math.isfinite(x) or (x <= 0 if _POSITIVE[name] else x < 0):
                raise ValueError(f"{name}: must be finite and {'> 0' if _POSITIVE[name] else '>= 0'}, got {x}")
            object.__setattr__(self, name, x)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_mjcf(cls, model: mujoco.MjModel) -> SimParams:
        """The upstream ``sts3215`` values of a freshly loaded model (the version-0 prior).

        Must be called before a ``MujocoBus`` is built on ``model``: the bus zeroes the gains while torque is off.
        """
        act, dof = _actuator_and_dof_ids(model)
        kp = np.array(model.actuator_gainprm[act, 0], dtype=np.float64)
        if (kp <= 0).any():
            raise ValueError(
                f"actuator gainprm[0] (kp) is {kp.tolist()}: read the model before a MujocoBus de-energised it"
            )
        biased = np.array(model.actuator_biasprm[act, 1], dtype=np.float64)
        if not np.allclose(biased, -kp):
            raise ValueError("actuators are not MuJoCo position servos (biasprm[1] != -gainprm[0])")
        fr = np.abs(np.array(model.actuator_forcerange[act], dtype=np.float64)).max(axis=1)
        if not np.all(model.actuator_forcelimited[act]):
            raise ValueError("every joint actuator must be force-limited (forcerange is part of the servo model)")
        return cls(
            kp=kp,
            kv=-np.array(model.actuator_biasprm[act, 2], dtype=np.float64),
            damping=np.array(model.dof_damping[dof], dtype=np.float64),
            frictionloss=np.array(model.dof_frictionloss[dof], dtype=np.float64),
            armature=np.array(model.dof_armature[dof], dtype=np.float64),
            forcerange=fr,
            delay_ticks=0,
            deadband_ticks=np.zeros(N_JOINTS, dtype=np.int64),
            die_mass_kg=float(model.body(DIE_BODY).mass[0]),
            die_friction=float(model.geom(DIE_GEOM).friction[0]),
            table_friction=float(model.geom(TABLE_GEOM).friction[0]),
        )

    @classmethod
    def from_yaml(cls, path: Path = Path("calib/sim_params.yaml"), *, allow_unverified: bool = False) -> SimParams:
        """Load ``calib/sim_params.yaml``. Refuses a schema version other than :data:`SIM_PARAMS_VERSION`, and an
        unverified file (``heldout_rms_pct`` null, incomplete, or above ``MAX_RMS_PCT`` = 1 % on any joint) unless
        ``allow_unverified``."""
        params, _ = read_sim_params_yaml(path, allow_unverified=allow_unverified)
        return params

    def to_yaml(
        self, path: Path, *, meta: SimParamsMeta, servo_group: ServoGroup | None = None, dr: DrRanges | None = None
    ) -> None:
        """Write the schema-``SIM_PARAMS_VERSION`` file atomically (``.tmp`` + rename).

        ``servo_group`` defaults to :meth:`ServoGroup.identity` (scale 1 around these values), ``dr`` to the
        defaults of :class:`DrRanges`. The group must reproduce ``kp``/``kv``/``armature`` (checked)."""
        path = Path(path)
        group = servo_group if servo_group is not None else ServoGroup.identity(self)
        group.check(self)
        doc = {
            "version": SIM_PARAMS_VERSION,
            "meta": meta.to_dict(),
            "params": self.to_dict(),
            "servo_group": group.to_dict(),
            "dr": (dr if dr is not None else DrRanges()).to_dict(),
        }
        header = (
            "# Simulator parameters (FS7). Schema: calib/README.md. Written by so101_rl.sim.sysid "
            "(params.SimParams.to_yaml).\n"
            "# meta.version 0 = the MJCF prior (unverified: heldout_rms_pct null); >= 1 = fitted on the real arm (HS3).\n"
            "# kp, kv, armature = servo_group.prior x servo_group.scale (one ratio group per joint); frictionloss is\n"
            "# not identified (meta.unidentified): use the dr ranges, not the point value (plan §10 2026-10-06).\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(tmp, "w") as fh:
            fh.write(header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=120))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in _PARAM_KEYS:
            v = getattr(self, name)
            if isinstance(v, np.ndarray):
                out[name] = [int(x) for x in v] if v.dtype.kind in "iu" else [float(x) for x in v]
            else:
                out[name] = v
        return out

    @classmethod
    def from_dict(cls, doc: dict[str, Any], *, where: str = "params") -> SimParams:
        if not isinstance(doc, dict):
            raise TypeError(f"{where}: expected a mapping")
        keys = set(doc)
        missing, unknown = set(_PARAM_KEYS) - keys, keys - set(_PARAM_KEYS)
        if missing or unknown:
            raise ValueError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
        try:
            return cls(**{k: doc[k] for k in _PARAM_KEYS})
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{where}: {exc}") from None

    # ------------------------------------------------------------------ application
    def apply_cpu(self, model: mujoco.MjModel) -> None:
        """Write every physics parameter into the CPU ``model`` (same mapping as ``WarpSim.set_world_params``).

        ``delay_ticks`` / ``deadband_ticks`` are not physics: the caller applies them to the target stream (the
        ``MujocoBus(action_delay_ticks=...)`` FIFO and :func:`apply_deadband`). Call before building a MujocoBus.
        """
        act, dof = _actuator_and_dof_ids(model)
        model.actuator_gainprm[act, 0] = self.kp
        model.actuator_biasprm[act, 1] = -self.kp
        model.actuator_biasprm[act, 2] = -self.kv
        model.actuator_forcerange[act, 0] = -self.forcerange
        model.actuator_forcerange[act, 1] = self.forcerange
        model.dof_damping[dof] = self.damping
        model.dof_armature[dof] = self.armature
        model.dof_frictionloss[dof] = self.frictionloss
        b = model.body(DIE_BODY).id
        old = float(model.body_mass[b])
        model.body_mass[b] = self.die_mass_kg
        model.body_inertia[b] = model.body_inertia[b] * (self.die_mass_kg / old)
        model.geom_friction[model.geom(DIE_GEOM).id, 0] = self.die_friction
        model.geom_friction[model.geom(TABLE_GEOM).id, 0] = self.table_friction
        import mujoco

        # derived constants (invweight0, subtree mass, actuator_acc0, ...) follow mass and armature, as
        # WarpSim.set_world_params does with mjw.set_const
        mujoco.mj_setConst(model, mujoco.MjData(model))

    # ------------------------------------------------------------------ vectors (CMA-ES)
    @staticmethod
    def vector_size(names: Sequence[str]) -> int:
        n = 0
        for name in names:
            if name in PER_JOINT_FIELDS or name == "deadband_ticks":
                n += N_JOINTS
            elif name in SCALAR_FIELDS or name == "delay_ticks":
                n += 1
            else:
                raise KeyError(f"unknown SimParams field {name!r}")
        return n

    def as_vector(self, names: Sequence[str]) -> np.ndarray:
        """float64 concatenation of the named fields (per-joint fields contribute 6 entries, in JOINT_NAMES order)."""
        self.vector_size(names)  # validates the names
        parts = [np.atleast_1d(np.asarray(getattr(self, n), dtype=np.float64)) for n in names]
        return np.concatenate(parts) if parts else np.zeros(0)

    @classmethod
    def from_vector(cls, vec: np.ndarray, names: Sequence[str], *, base: SimParams) -> SimParams:
        """Inverse of :meth:`as_vector`: ``base`` with the named fields replaced (integer fields are rounded)."""
        vec = np.asarray(vec, dtype=np.float64)
        if vec.shape != (cls.vector_size(names),):
            raise ValueError(f"vector of shape {vec.shape} does not match {list(names)} ({cls.vector_size(names)})")
        upd: dict[str, Any] = {}
        i = 0
        for name in names:
            if name in PER_JOINT_FIELDS:
                upd[name] = vec[i : i + N_JOINTS].copy()
                i += N_JOINTS
            elif name == "deadband_ticks":
                upd[name] = np.rint(vec[i : i + N_JOINTS]).astype(np.int64)
                i += N_JOINTS
            elif name == "delay_ticks":
                upd[name] = int(np.rint(vec[i]))
                i += 1
            else:
                upd[name] = float(vec[i])
                i += 1
        return replace(base, **upd)

    def scaled(self, **factors: float | Sequence[float]) -> SimParams:
        """A copy with float fields multiplied (``p.scaled(kp=0.7, kv=1.4)``); used for synthetic ground truth."""
        upd: dict[str, Any] = {}
        for name, f in factors.items():
            if name not in PER_JOINT_FIELDS and name not in SCALAR_FIELDS:
                raise KeyError(f"cannot scale {name!r}")
            upd[name] = getattr(self, name) * (np.asarray(f, dtype=np.float64) if name in PER_JOINT_FIELDS else float(f))
        return replace(self, **upd)

    def close_to(self, other: SimParams, *, rtol: float = 1e-9) -> bool:
        """Field-wise equality within ``rtol`` (the dataclass is ``eq=False`` because of the arrays)."""
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if not np.allclose(a, b, rtol=rtol, atol=0.0):
                return False
        return True


def _actuator_and_dof_ids(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    act = np.array([model.actuator(n).id for n in JOINT_NAMES], dtype=np.intp)
    dof = np.array([int(model.jnt_dofadr[model.joint(n).id]) for n in JOINT_NAMES], dtype=np.intp)
    for a, n in zip(act, JOINT_NAMES, strict=True):
        if int(model.actuator_trnid[a, 0]) != model.joint(n).id:
            raise ValueError(f"actuator {n!r} does not drive joint {n!r}")
    return act, dof


@dataclass(frozen=True)
class SimParamsMeta:
    version: int  # parameter release: 0 = MJCF prior, >= 1 = fitted on real data
    date: str
    git_sha: str
    mjcf_sha256: str
    excitation_run: str
    fit_rms_pct: dict[str, float]
    heldout_rms_pct: dict[str, float] | None  # None = never verified (the shipped prior)
    spread: dict[str, float]
    backend: str  # "mjcf-prior", "warp", "cpu"
    notes: tuple[str, ...] = field(default=())
    unidentified: tuple[str, ...] = field(default=("frictionloss",))  # never to be read as precise

    def to_dict(self) -> dict[str, Any]:
        def fdict(d: dict[str, float] | None) -> dict[str, float] | None:
            return None if d is None else {str(k): float(v) for k, v in d.items()}

        out: dict[str, Any] = {
            "version": int(self.version),
            "date": self.date,
            "git_sha": self.git_sha,
            "mjcf_sha256": self.mjcf_sha256,
            "excitation_run": self.excitation_run,
            "fit_rms_pct": fdict(self.fit_rms_pct),
            "heldout_rms_pct": fdict(self.heldout_rms_pct),
            "spread": fdict(self.spread),
            "backend": self.backend,
            "unidentified": list(self.unidentified),
        }
        if self.notes:
            out["notes"] = list(self.notes)
        return out

    @classmethod
    def from_dict(cls, doc: Any, *, where: str = "meta") -> SimParamsMeta:
        if not isinstance(doc, dict):
            raise TypeError(f"{where}: expected a mapping")
        keys = set(doc)
        missing, unknown = set(_META_KEYS) - keys, keys - set(_META_KEYS) - {"notes"}
        if missing or unknown:
            raise ValueError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
        v = doc["version"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"{where}.version must be an int >= 0, got {v!r}")
        un = doc["unidentified"]
        if not isinstance(un, list) or not all(isinstance(x, str) for x in un):
            raise TypeError(f"{where}.unidentified: expected a list of parameter names")

        def fdict(key: str, *, nullable: bool = False) -> dict[str, float] | None:
            d = doc[key]
            if d is None and nullable:
                return None
            if not isinstance(d, dict):
                raise TypeError(f"{where}.{key}: expected a mapping{' or null' if nullable else ''}")
            return {str(k): float(x) for k, x in d.items()}

        return cls(
            version=v,
            date=str(doc["date"]),
            git_sha=str(doc["git_sha"]),
            mjcf_sha256=str(doc["mjcf_sha256"]),
            excitation_run=str(doc["excitation_run"]),
            fit_rms_pct=fdict("fit_rms_pct") or {},
            heldout_rms_pct=fdict("heldout_rms_pct", nullable=True),
            spread=fdict("spread") or {},
            backend=str(doc["backend"]),
            notes=tuple(str(n) for n in doc.get("notes") or ()),
            unidentified=tuple(un),
        )

    def verification_problems(self, max_pct: float) -> list[str]:
        """Why this release does not pass the held-out gate ([] = verified)."""
        if self.heldout_rms_pct is None:
            return ["heldout_rms_pct is null (never verified against real data)"]
        problems = [f"{n}: no held-out RMS" for n in JOINT_NAMES if n not in self.heldout_rms_pct]
        for n in JOINT_NAMES:
            e = self.heldout_rms_pct.get(n)
            if e is not None and not (math.isfinite(e) and e <= max_pct):
                problems.append(f"{n}: held-out RMS {e:.3f} % > {max_pct:g} %")
        return problems


@dataclass(frozen=True, eq=False)
class ServoGroup:
    """kp, kv and armature as ONE ratio group per joint: ``value = prior_value x scale`` (plan §10 2026-10-06).

    ``prior`` holds the three fields' reference values (the fit's prior), ``scale`` the fitted per-joint factor
    (1 = the prior). The ratios kp : kv : armature are those of ``prior`` by construction."""

    prior: dict[str, np.ndarray]  # {kp, kv, armature}: float64[6]
    scale: np.ndarray  # float64[6], > 0

    def __post_init__(self) -> None:
        if set(self.prior) != set(SERVO_GROUP_FIELDS):
            raise ValueError(f"servo_group.prior: keys must be exactly {list(SERVO_GROUP_FIELDS)}, got {sorted(self.prior)}")
        pr = {}
        for k in SERVO_GROUP_FIELDS:
            v = _vec6(f"servo_group.prior.{k}", self.prior[k], dtype=float)
            ok = (v > 0) if _POSITIVE[k] else (v >= 0)  # the SimParams rule: kp > 0, kv and armature >= 0
            if not (np.isfinite(v).all() and ok.all()):
                rule = "> 0" if _POSITIVE[k] else ">= 0"
                raise ValueError(f"servo_group.prior.{k}: every value must be finite and {rule}, got {v.tolist()}")
            pr[k] = v
        object.__setattr__(self, "prior", pr)
        sc = _vec6("servo_group.scale", self.scale, dtype=float)
        if not (np.isfinite(sc).all() and (sc > 0).all()):
            raise ValueError(f"servo_group.scale: every value must be finite and > 0, got {sc.tolist()}")
        object.__setattr__(self, "scale", sc)

    @classmethod
    def identity(cls, params: SimParams) -> ServoGroup:
        """Scale 1 around ``params`` itself (the MJCF prior, or any set that was not group-fitted)."""
        return cls(prior={k: getattr(params, k) for k in SERVO_GROUP_FIELDS}, scale=np.ones(N_JOINTS))

    @classmethod
    def from_fit(cls, params: SimParams, prior: SimParams) -> ServoGroup:
        """The group of a fitted set: ``scale = params.kp / prior.kp`` (checked to hold for kv and armature)."""
        g = cls(prior={k: getattr(prior, k) for k in SERVO_GROUP_FIELDS}, scale=params.kp / prior.kp)
        g.check(params)
        return g

    def apply(self, base: SimParams, scale: np.ndarray | None = None) -> SimParams:
        """``base`` with kp, kv, armature = prior x ``scale`` (default: this group's scale)."""
        s = self.scale if scale is None else np.asarray(scale, dtype=np.float64)
        return replace(base, **{k: self.prior[k] * s for k in SERVO_GROUP_FIELDS})

    def check(self, params: SimParams, *, rtol: float = 1e-9) -> None:
        for k in SERVO_GROUP_FIELDS:
            want = self.prior[k] * self.scale
            if not np.allclose(getattr(params, k), want, rtol=rtol, atol=0.0):
                raise ValueError(
                    f"servo_group: params.{k} {getattr(params, k).tolist()} != prior x scale {want.tolist()} "
                    "(kp, kv and armature are one ratio group)"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"prior": {k: [float(x) for x in self.prior[k]] for k in SERVO_GROUP_FIELDS},
                "scale": [float(x) for x in self.scale]}

    @classmethod
    def from_dict(cls, doc: Any, *, where: str = "servo_group") -> ServoGroup:
        if not isinstance(doc, dict) or set(doc) != {"prior", "scale"} or not isinstance(doc["prior"], dict):
            raise ValueError(f"{where}: expected a mapping with exactly prior: {{kp, kv, armature}} and scale")
        try:
            return cls(prior=dict(doc["prior"]), scale=doc["scale"])
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{where}: {exc}") from None


@dataclass(frozen=True)
class DrRanges:
    """Domain-randomisation ranges recorded with the release (plan §10 2026-10-06; consumed by FS8 / S3)."""

    servo_group_rel: float = SERVO_GROUP_REL
    frictionloss_range: tuple[float, float] = FRICTIONLOSS_RANGE
    frictionloss_source: str = FRICTIONLOSS_SOURCE

    def __post_init__(self) -> None:
        r = self.servo_group_rel
        if isinstance(r, bool) or not isinstance(r, int | float) or not 0.0 <= float(r) < 1.0:
            raise ValueError(f"dr.servo_group_rel must be within [0, 1), got {r!r}")
        lo, hi = (float(x) for x in self.frictionloss_range)
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo <= hi):
            raise ValueError(f"dr.frictionloss_range must be 0 <= lo <= hi, got {self.frictionloss_range!r}")
        if not str(self.frictionloss_source).strip():
            raise ValueError("dr.frictionloss_source: a source/justification is required")
        object.__setattr__(self, "servo_group_rel", float(r))
        object.__setattr__(self, "frictionloss_range", (lo, hi))

    def servo_group_bounds(self, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint group scale range ``scale x (1 -+ rel)``."""
        s = np.asarray(scale, dtype=np.float64)
        return s * (1.0 - self.servo_group_rel), s * (1.0 + self.servo_group_rel)

    def to_dict(self) -> dict[str, Any]:
        return {"servo_group_rel": self.servo_group_rel, "frictionloss_range": list(self.frictionloss_range),
                "frictionloss_source": self.frictionloss_source}

    @classmethod
    def from_dict(cls, doc: Any, *, where: str = "dr") -> DrRanges:
        keys = {"servo_group_rel", "frictionloss_range", "frictionloss_source"}
        if not isinstance(doc, dict) or set(doc) != keys:
            raise ValueError(f"{where}: expected a mapping with exactly {sorted(keys)}")
        fr = doc["frictionloss_range"]
        if not isinstance(fr, list | tuple) or len(fr) != 2:
            raise ValueError(f"{where}.frictionloss_range: expected [lo, hi]")
        try:
            return cls(servo_group_rel=doc["servo_group_rel"], frictionloss_range=(fr[0], fr[1]),
                       frictionloss_source=str(doc["frictionloss_source"]))
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{where}: {exc}") from None


@dataclass(frozen=True)
class SimParamsFile:
    """Everything in ``calib/sim_params.yaml`` (schema 2)."""

    params: SimParams
    meta: SimParamsMeta
    servo_group: ServoGroup
    dr: DrRanges


def read_sim_params_yaml(
    path: Path = Path("calib/sim_params.yaml"), *, allow_unverified: bool = False
) -> tuple[SimParams, SimParamsMeta]:
    """Load ``(params, meta)`` strictly (see :meth:`SimParams.from_yaml` for the refusals)."""
    f = read_sim_params_file(path, allow_unverified=allow_unverified)
    return f.params, f.meta


def read_sim_params_file(path: Path = Path("calib/sim_params.yaml"), *, allow_unverified: bool = False) -> SimParamsFile:
    """Load the whole file strictly: params, meta, the servo ratio group (consistent with params) and the DR ranges."""
    from so101_rl.sim.compare_real import MAX_RMS_PCT  # the one gate value (compare-real owns it)

    path = Path(path)
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict):
        raise TypeError(f"{path}: expected a mapping")
    if doc.get("version") != SIM_PARAMS_VERSION or isinstance(doc.get("version"), bool):
        raise ValueError(
            f"{path}: schema version {doc.get('version')!r} is not supported (this code reads version "
            f"{SIM_PARAMS_VERSION}; regenerate release 0 with `python -m so101_rl.sim.sysid.params --write-prior`)"
        )
    keys = set(doc)
    want = {"version", "meta", "params", "servo_group", "dr"}
    if keys != want:
        raise ValueError(f"{path}: top-level keys must be exactly {sorted(want)}; got {sorted(keys)}")
    meta = SimParamsMeta.from_dict(doc["meta"], where=f"{path}: meta")
    params = SimParams.from_dict(doc["params"], where=f"{path}: params")
    group = ServoGroup.from_dict(doc["servo_group"], where=f"{path}: servo_group")
    try:
        group.check(params, rtol=1e-6)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    dr = DrRanges.from_dict(doc["dr"], where=f"{path}: dr")
    if meta.backend != "mjcf-prior" or meta.version > 0:
        try:
            current = mjcf_sha256()
        except OSError:
            current = meta.mjcf_sha256
        if current != meta.mjcf_sha256:
            warnings.warn(
                f"{path}: fitted on an MJCF with sha256 {meta.mjcf_sha256[:12]}..., the scene is now {current[:12]}...; "
                "re-run compare-real before trusting these parameters",
                stacklevel=2,
            )
    problems = meta.verification_problems(MAX_RMS_PCT)
    if problems and not allow_unverified:
        raise ValueError(
            f"{path}: unverified sim parameters (release {meta.version}): {'; '.join(problems)}. "
            "Pass allow_unverified=True to use them anyway (e.g. the MJCF prior for reach on v0)."
        )
    return SimParamsFile(params=params, meta=meta, servo_group=group, dr=dr)


def mjcf_sha256(scene_xml: Path | None = None) -> str:
    """sha256 over the RL scene MJCF and every upstream ``*.xml`` it can include (sorted by name)."""
    from so101_rl.sim.model import SCENE_XML

    scene = Path(scene_xml) if scene_xml is not None else SCENE_XML
    h = hashlib.sha256()
    files = [scene, *sorted((scene.parent / "upstream").glob("*.xml"))]
    for p in files:
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def apply_deadband(targets: np.ndarray, band: np.ndarray, *, initial: np.ndarray | None = None) -> np.ndarray:
    """D-BACKLASH target hysteresis: the written value moves only when the target leaves ``±band`` around the
    last written value, and then follows it to the band edge.

    ``targets`` int ``[..., T, 6]`` (leading batch dims allowed), ``band`` int broadcastable to ``[..., 6]``;
    ``initial`` = the value held before row 0 (default: row 0 itself). ``band == 0`` is the identity.
    """
    t = np.asarray(targets, dtype=np.int64)
    b = np.broadcast_to(np.asarray(band, dtype=np.int64), t.shape[:-2] + t.shape[-1:])
    if not b.any():
        return t.copy()
    out = np.empty_like(t)
    held = (t[..., 0, :] if initial is None else np.broadcast_to(np.asarray(initial, np.int64), b.shape)).copy()
    for k in range(t.shape[-2]):
        x = t[..., k, :]
        held = np.where(x > held + b, x - b, np.where(x < held - b, x + b, held))
        out[..., k, :] = held
    return out


class BatchedSimParams:
    """Per-world parameters (FS7) for ``WarpSim.set_world_params`` and domain randomisation (FS8)."""

    FLOAT6: ClassVar[tuple[str, ...]] = PER_JOINT_FIELDS
    SCALARS: ClassVar[dict[str, str]] = {"die_mass": "die_mass_kg", "die_friction": "die_friction",
                                         "table_friction": "table_friction"}

    kp: torch.Tensor
    kv: torch.Tensor
    damping: torch.Tensor
    frictionloss: torch.Tensor
    armature: torch.Tensor
    forcerange: torch.Tensor
    die_mass: torch.Tensor
    die_friction: torch.Tensor
    table_friction: torch.Tensor
    delay_ticks: torch.Tensor
    deadband_ticks: torch.Tensor

    def __init__(self, nworld: int, device: torch.device | str) -> None:
        import torch

        if nworld < 1:
            raise ValueError(f"nworld must be >= 1, got {nworld}")
        self.nworld = int(nworld)
        self.device = torch.device(device)
        f32 = {"dtype": torch.float32, "device": self.device}
        for name in self.FLOAT6:
            setattr(self, name, torch.zeros(nworld, N_JOINTS, **f32))
        for name in self.SCALARS:
            setattr(self, name, torch.zeros(nworld, **f32))
        self.delay_ticks = torch.zeros(nworld, dtype=torch.int64, device=self.device)
        self.deadband_ticks = torch.zeros(nworld, N_JOINTS, dtype=torch.int64, device=self.device)

    @classmethod
    def broadcast(cls, p: SimParams, nworld: int, device: torch.device | str) -> BatchedSimParams:
        return cls.stack([p] * int(nworld), device) if nworld > 0 else cls(nworld, device)

    @classmethod
    def stack(cls, ps: Sequence[SimParams], device: torch.device | str) -> BatchedSimParams:
        """One world per parameter set (CMA-ES candidates)."""
        import torch

        out = cls(len(ps), device)
        for name in cls.FLOAT6:
            getattr(out, name).copy_(torch.as_tensor(np.stack([getattr(p, name) for p in ps]), dtype=torch.float32))
        for name, src in cls.SCALARS.items():
            getattr(out, name).copy_(torch.as_tensor([getattr(p, src) for p in ps], dtype=torch.float32))
        out.delay_ticks.copy_(torch.as_tensor([p.delay_ticks for p in ps], dtype=torch.int64))
        out.deadband_ticks.copy_(torch.as_tensor(np.stack([p.deadband_ticks for p in ps]), dtype=torch.int64))
        return out

    def world(self, i: int) -> SimParams:
        """World ``i`` as a :class:`SimParams` (float32 values widened to float64)."""
        return SimParams(
            **{n: getattr(self, n)[i].detach().cpu().double().numpy() for n in self.FLOAT6},
            delay_ticks=int(self.delay_ticks[i]),
            deadband_ticks=self.deadband_ticks[i].detach().cpu().numpy(),
            **{src: float(getattr(self, n)[i]) for n, src in self.SCALARS.items()},
        )

    def to(self, device: torch.device | str) -> Self:
        import torch

        out = type(self)(self.nworld, device)
        for name in (*self.FLOAT6, *self.SCALARS, "delay_ticks", "deadband_ticks"):
            getattr(out, name).copy_(getattr(self, name).to(torch.device(device)))
        return out

    def to_gainprm_biasprm(self) -> tuple[torch.Tensor, torch.Tensor]:
        """MuJoCo position actuator parameters ``[N, 6, 10]``: ``gainprm[0] = kp``, ``biasprm[1] = -kp``,
        ``biasprm[2] = -kv`` (every other entry 0)."""
        import torch

        gain = torch.zeros(self.nworld, N_JOINTS, 10, dtype=torch.float32, device=self.device)
        bias = torch.zeros_like(gain)
        gain[..., 0] = self.kp
        bias[..., 1] = -self.kp
        bias[..., 2] = -self.kv
        return gain, bias


def prior_meta(*, date: str, git_sha: str) -> SimParamsMeta:
    """Metadata of the release-0 file: the MJCF prior, never verified against the arm."""
    return SimParamsMeta(
        version=0,
        date=date,
        git_sha=git_sha,
        mjcf_sha256=mjcf_sha256(),
        excitation_run="none (MJCF prior)",
        fit_rms_pct={},
        heldout_rms_pct=None,
        spread={},
        backend="mjcf-prior",
        notes=("upstream sts3215 class values (so101_new_calib.xml); replaced by the HS3 fit (release >= 1)",),
        unidentified=("servo_group", "delay_ticks", "frictionloss"),  # nothing is identified in the prior
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m so101_rl.sim.sysid.params --write-prior PATH``: regenerate the release-0 (MJCF prior) file."""
    import argparse
    import datetime as dt
    import subprocess

    from so101_rl.sim.model import load_model

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--write-prior", type=Path, required=True, metavar="PATH")
    args = ap.parse_args(argv)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = "unknown"
    model, _ = load_model()
    SimParams.from_mjcf(model).to_yaml(args.write_prior, meta=prior_meta(date=dt.datetime.now(dt.UTC).date().isoformat(), git_sha=sha))
    print(f"wrote {args.write_prior}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

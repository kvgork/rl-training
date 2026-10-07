"""Excitation plans for system identification (FS7, sim-first track S2.2).

A plan is a sequence of :class:`Segment` s, turned by :func:`generate_deltas` into 10 Hz **pre-clamp** joint deltas in
ticks (``float32[T, 6]``, integral values). The runner feeds them to ``ServoCore`` through the normal command path, so
the excitation passes the real clamp chain (the logged tick stream is what the arm did, L4).

Every segment is an offset trajectory relative to the pose the arm is in when the plan starts (``home`` after the
boot re-energise): a slow cosine ramp to the segment's ``centre_ticks``, the excitation body (which starts and ends at
the centre), a ramp back and a hold. Each segment therefore returns to the start pose, so segments are independent
and :func:`check_plan_in_box` can preview the whole plan kinematically.

Kinds (plan S2.2, specialist guidance "Sysid order"):

* ``multisine``: 3-5 sines in 0.2-4 Hz (below the 5 Hz Nyquist of the 10 Hz command), amplitudes ~ 1/f (flat
  velocity spectrum), seeded phases, a 0.5 s taper at both ends; a multi-joint segment gives every joint its own
  frequency set.
* ``chirp``: 0.1 -> 3 Hz linear sweep (armature), velocity-limited envelope.
* ``step``: ±10 / ±20 / ±45 tick steps with 1 s holds (delay, dead-band).
* ``ramp``: slow ±5 ticks/step triangles (friction).
* ``random``: the soak random policy (uniform deltas pulled back to the centre), then a return ramp.
* ``gripper``: open/close cycles at three speeds (``with_gripper_die``: closing on the die in the start well, HS2).

A body whose per-step change would exceed ``max_delta_ticks`` (the command clamp, invisible in the clamp mask) is
scaled down per joint, so every delta is a real request the chain may then clamp at the velocity / tracking / box
stages. The clamp-active fraction those stages produce is measured by the runner, not predicted here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, So101Config

SegmentKind = Literal["multisine", "chirp", "step", "ramp", "random", "gripper"]
KINDS: Final[tuple[str, ...]] = ("multisine", "chirp", "step", "ramp", "random", "gripper")
GRIPPER: Final[int] = N_JOINTS - 1
ARM: Final[tuple[int, ...]] = tuple(range(N_JOINTS - 1))
ALL: Final[tuple[int, ...]] = tuple(range(N_JOINTS))
MULTISINE_BAND_HZ: Final[tuple[float, float]] = (0.2, 4.0)
DELTA_HEADROOM: Final[float] = 0.95  # bodies are scaled so |delta| <= 0.95 x max_delta_ticks
_LEAD_FRACTION: Final[float] = 0.5  # lead/tail ramps move at most this fraction of max_delta per step
_TAPER_S: Final[float] = 0.5

# Per-joint excitation centre (offset from home) and amplitude [ticks], from the FK in-box preview around the home
# placeholder with a 20 mm margin (shoulder_lift: -90..+610 ticks; elbow_flex: -300..+320; ...). The gripper is
# opened to mid-range (home is ~closed). check_plan_in_box re-verifies every plan; NEEDS H5 when home moves.
CENTRE_TICKS: Final[tuple[int, ...]] = (0, 260, 0, 0, 0, 700)
AMPLITUDE_TICKS: Final[tuple[int, ...]] = (400, 220, 220, 300, 600, 500)


@dataclass(frozen=True)
class Segment:
    kind: SegmentKind
    joints: tuple[int, ...]  # indices into JOINT_NAMES
    duration_s: float  # body duration (lead/tail ramps and the hold come on top)
    amplitude_ticks: float | tuple[float, ...]  # scalar or one per entry of ``joints``
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown segment kind {self.kind!r}")
        if not self.joints or any(not 0 <= j < N_JOINTS for j in self.joints) or len(set(self.joints)) != len(self.joints):
            raise ValueError(f"joints must be distinct indices 0..{N_JOINTS - 1}, got {self.joints}")
        if not (math.isfinite(self.duration_s) and self.duration_s > 0):
            raise ValueError(f"duration_s must be > 0, got {self.duration_s}")
        amp = np.broadcast_to(np.asarray(self.amplitude_ticks, dtype=np.float64), (len(self.joints),))
        if not (np.isfinite(amp).all() and (amp >= 0).all()):
            raise ValueError(f"amplitude_ticks must be finite and >= 0, got {self.amplitude_ticks}")

    def amplitudes(self) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.amplitude_ticks, dtype=np.float64), (len(self.joints),)).copy()

    @property
    def label(self) -> str:
        names = "all" if self.joints == ALL else ("arm" if self.joints == ARM else "+".join(JOINT_NAMES[j] for j in self.joints))
        return f"{self.kind}:{names}"


@dataclass(frozen=True)
class ExcitationPlan:
    segments: tuple[Segment, ...]
    policy_hz: float
    seed: int
    name: str = "custom"
    hold_s: float = 1.0  # settle at the start pose after every segment

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a plan needs at least one segment")
        if not self.policy_hz > 0:
            raise ValueError("policy_hz must be > 0")


@dataclass(frozen=True)
class SegmentSpan:
    """Where a segment sits in the generated delta array (policy steps, half-open ``[start, stop)``)."""

    index: int
    kind: str
    label: str
    start: int
    stop: int


@dataclass
class PlanCheck:
    ok: bool
    n_ticks: int
    min_box_margin_m: float  # smallest signed distance of the previewed tip to a box face (< 0 = outside)
    min_table_clearance_m: float
    joint_limit_ticks: int  # previewed ticks on which the joint-limit clamp would act
    first_violation: tuple[int, tuple[float, float, float]] | None
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------- bodies
def _taper(n: int, hz: float) -> np.ndarray:
    """Raised-cosine taper (0 at both ends, 1 in the middle) over ``_TAPER_S``."""
    w = np.ones(n)
    m = min(n // 2, max(1, round(_TAPER_S * hz)))
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
    w[:m] = ramp
    w[n - m :] = ramp[::-1]
    w[0] = w[-1] = 0.0
    return w


def _multisine(n: int, hz: float, amp: float, freqs: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / hz
    f = np.asarray(freqs, dtype=np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=len(f))
    x = (np.sin(2.0 * np.pi * f[None, :] * t[:, None] + phase[None, :]) / f[None, :]).sum(axis=1)
    x *= _taper(n, hz)
    peak = np.abs(x).max()
    return amp * x / peak if peak > 0 else x


def _chirp(n: int, hz: float, amp: float, f0: float, f1: float, vmax: float) -> np.ndarray:
    t = np.arange(n) / hz
    dur = n / hz
    f = f0 + (f1 - f0) * t / dur
    env = np.minimum(amp, vmax / (2.0 * np.pi * np.maximum(f, 1e-6)))
    return env * np.sin(2.0 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2.0 * dur))) * _taper(n, hz)


def _steps(hz: float, sizes: Sequence[float], hold_s: float) -> np.ndarray:
    h = max(1, round(hold_s * hz))
    out: list[float] = []
    for s in sizes:
        out += [s] * h + [0.0] * h + [-s] * h + [0.0] * h
    return np.asarray(out)


def _triangle(amp: float, rate: float) -> np.ndarray:
    k = max(1, math.ceil(amp / rate))
    up = np.linspace(0.0, amp, k + 1)[1:]
    return np.concatenate([up, up[::-1][1:], [0.0], -up, -up[::-1][1:], [0.0]])


def _random_walk(n: int, amp: float, max_d: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    p = np.zeros(n)
    x = 0.0
    for k in range(1, n):
        step = float(np.clip(rng.uniform(-1.0, 1.0) * scale * max_d - 0.1 * x, -max_d, max_d))
        x = float(np.clip(x + step, -amp, amp))
        p[k] = x
    return p


def _gripper_cycles(amp: float, rates: Sequence[float], cycles: int) -> np.ndarray:
    out: list[np.ndarray] = []
    for r in rates:
        k = max(1, math.ceil(amp / r))
        up = np.linspace(0.0, amp, k + 1)[1:]
        for _ in range(cycles):
            out += [up, np.full(5, amp), up[::-1][1:], np.zeros(6)]
    return np.concatenate(out)


def _return_ramp(x_end: np.ndarray, max_d: np.ndarray) -> np.ndarray:
    """[k, n] linear ramp from ``x_end`` to 0 at <= _LEAD_FRACTION x max_delta per step (empty if already 0)."""
    k = int(np.ceil((np.abs(x_end) / (_LEAD_FRACTION * max_d)).max())) if np.any(x_end) else 0
    if k == 0:
        return np.zeros((0, len(x_end)))
    s = np.arange(1, k + 1)[:, None] / k
    return x_end[None, :] * (1.0 - s)


def _cosine_ramp(a: np.ndarray, b: np.ndarray, max_d: np.ndarray) -> np.ndarray:
    """[k, n] raised-cosine move from ``a`` to ``b`` (exclusive of ``a``), peak step <= _LEAD_FRACTION x max_d."""
    dist = np.abs(b - a)
    if not dist.any():
        return np.zeros((0, len(a)))
    # peak step of a raised-cosine over k steps is (pi / 2) * dist / k
    k = int(np.ceil((np.pi / 2.0 * dist / (_LEAD_FRACTION * max_d)).max()))
    s = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, k + 1) / k)
    return a[None, :] + (b - a)[None, :] * s[:, None]


def _segment_body(seg: Segment, plan: ExcitationPlan, max_d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Body offsets ``[n, len(seg.joints)]`` relative to the segment centre (0 at both ends, except random which is
    followed by its own return ramp)."""
    hz = plan.policy_hz
    n = max(2, round(seg.duration_s * hz))
    amps = seg.amplitudes()
    md = max_d[list(seg.joints)]
    p = seg.params
    cols: list[np.ndarray] = []
    if seg.kind == "multisine":
        base = np.asarray(p.get("freqs_hz", (0.2, 0.55, 1.3, 2.6, 4.0)), dtype=np.float64)
        for i, a in enumerate(amps):
            f = np.clip(base * (1.0 + 0.13 * i), *MULTISINE_BAND_HZ) if len(seg.joints) > 1 else base
            cols.append(_multisine(n, hz, a, f, rng))
    elif seg.kind == "chirp":
        f0, f1 = float(p.get("f0_hz", 0.1)), float(p.get("f1_hz", 3.0))
        for a, m in zip(amps, md, strict=True):
            cols.append(_chirp(n, hz, a, f0, f1, DELTA_HEADROOM * m * hz))
    elif seg.kind == "step":
        x = _steps(hz, p.get("sizes", (10, 20, 45)), float(p.get("hold_s", 1.0)))
        cols = [x.copy() for _ in amps]
    elif seg.kind == "ramp":
        rate = float(p.get("rate_ticks", 5.0))
        cols = [_triangle(a, rate) for a in amps]
        L = max(len(c) for c in cols)
        cols = [np.pad(c, (0, L - len(c))) for c in cols]
    elif seg.kind == "random":
        scale = float(p.get("scale", 0.6))
        cols = [_random_walk(n, a, m, scale, rng) for a, m in zip(amps, md, strict=True)]
    elif seg.kind == "gripper":
        cyc = _gripper_cycles(float(amps[0]), p.get("rates_ticks", (10, 25, 45)), int(p.get("cycles", 2)))
        cols = [cyc.copy() for _ in amps]
    body = np.stack(cols, axis=1)
    if seg.kind in ("multisine", "chirp", "random"):
        # scale per joint so the 10 Hz change never exceeds the command clamp (it would be invisible in the mask)
        d = np.abs(np.diff(body, axis=0)).max(axis=0)
        body *= np.minimum(1.0, DELTA_HEADROOM * md / np.maximum(d, 1e-9))[None, :]
    if seg.kind == "random":
        body = np.concatenate([body, _return_ramp(body[-1], md)], axis=0)
    return body


def plan_offsets(plan: ExcitationPlan, cfg: So101Config) -> tuple[np.ndarray, list[SegmentSpan]]:
    """Absolute offset trajectory ``float64[T + 1, 6]`` (row 0 = 0, the start pose) and the segment spans."""
    rng = np.random.default_rng(plan.seed)
    max_d = np.asarray(cfg.action.max_delta_ticks, dtype=np.float64)
    rows: list[np.ndarray] = [np.zeros((1, N_JOINTS))]
    spans: list[SegmentSpan] = []
    hold = np.zeros((max(1, round(plan.hold_s * plan.policy_hz)), N_JOINTS))
    t = 0
    for i, seg in enumerate(plan.segments):
        centre = np.zeros(N_JOINTS)
        c = seg.params.get("centre_ticks")
        js = list(seg.joints)
        centre[js] = np.asarray(c, dtype=np.float64) if c is not None else np.asarray(CENTRE_TICKS, dtype=np.float64)[js]
        lead = _cosine_ramp(np.zeros(N_JOINTS), centre, max_d)
        body_j = _segment_body(seg, plan, max_d, rng)
        body = np.repeat(centre[None, :], len(body_j), axis=0)
        body[:, js] += body_j
        tail = _cosine_ramp(body[-1], np.zeros(N_JOINTS), max_d)
        block = np.concatenate([lead, body, tail], axis=0)
        spans.append(SegmentSpan(i, seg.kind, seg.label, t, t + len(block)))
        rows += [block, hold]
        t += len(block) + len(hold)
    return np.concatenate(rows, axis=0), spans


def generate_deltas(plan: ExcitationPlan, cfg: So101Config) -> np.ndarray:
    """``float32[T, 6]`` 10 Hz deltas in ticks (pre-clamp, integral values): ``diff(rint(offsets))``."""
    p, _ = plan_offsets(plan, cfg)
    return np.diff(np.rint(p), axis=0).astype(np.float32)


def segment_spans(plan: ExcitationPlan, cfg: So101Config) -> list[SegmentSpan]:
    return plan_offsets(plan, cfg)[1]


# --------------------------------------------------------------------------------------------- plans
def _per_joint(kind: SegmentKind, j: int, duration_s: float, amp: float, **params: Any) -> Segment:
    return Segment(kind, (j,), duration_s, amp, params)


def default_plan(cfg: So101Config, *, with_gripper_die: bool, seed: int = 0) -> ExcitationPlan:
    """~10 min of arm time: per joint (gripper in air included) multisine, chirp, steps and ramps, then 60 s of the
    random policy, a 60 s all-joint multisine, and (``with_gripper_die``) gripper cycles at three speeds.

    Amplitudes are the in-box values of :data:`AMPLITUDE_TICKS` (multisine/chirp are additionally limited by the
    command clamp). The clamp-active fraction target on the real arm is [0.05, 0.4] for the multisine/chirp segments
    (HS1: lower the amplitudes and repeat when it is outside); the runner measures and reports it.
    """
    hz = cfg.rates.policy_hz
    segs: list[Segment] = []
    for j in ALL:
        a = AMPLITUDE_TICKS[j]
        segs += [
            _per_joint("multisine", j, 20.0, a),
            _per_joint("chirp", j, 20.0, a, f0_hz=0.1, f1_hz=3.0),
            _per_joint("step", j, 12.0, 0.0, sizes=(10, 20, 45), hold_s=1.0),
            _per_joint("ramp", j, 10.0, min(a, 120), rate_ticks=5.0),
        ]
    segs.append(Segment("random", ALL, 60.0, tuple(0.5 * a for a in AMPLITUDE_TICKS), {"scale": 0.6}))
    segs.append(Segment("multisine", ALL, 60.0, tuple(0.4 * a for a in AMPLITUDE_TICKS)))
    if with_gripper_die:
        segs.append(Segment("gripper", (GRIPPER,), 30.0, AMPLITUDE_TICKS[GRIPPER], {"rates_ticks": (10, 25, 45), "cycles": 2}))
    return ExcitationPlan(tuple(segs), hz, seed, name="default")


def short_plan(cfg: So101Config, *, with_gripper_die: bool = False, seed: int = 0) -> ExcitationPlan:
    """~20 s dry-run plan (``--plan short``): one segment of each main kind (the gripper stays put)."""
    hz = cfg.rates.policy_hz
    arm_centre = {"centre_ticks": (0, 120, 0, 0, 0)}  # a short lift offset keeps the lead/tail ramps short
    amp = tuple(0.3 * a for a in AMPLITUDE_TICKS[:GRIPPER])
    segs = [
        _per_joint("multisine", 0, 4.0, 250.0),
        _per_joint("chirp", 3, 3.0, 150.0, f0_hz=0.2, f1_hz=2.0),
        _per_joint("step", 2, 2.0, 0.0, sizes=(20,), hold_s=0.5),
        Segment("random", ARM, 3.0, amp, {"scale": 0.6, **arm_centre}),
        Segment("multisine", ARM, 3.0, amp, arm_centre),
    ]
    if with_gripper_die:
        segs.append(Segment("gripper", (GRIPPER,), 3.0, 300.0, {"rates_ticks": (45,), "cycles": 1, "centre_ticks": (0,)}))
    return ExcitationPlan(tuple(segs), hz, seed, name="short", hold_s=0.5)


def fit40_plan(cfg: So101Config, *, with_gripper_die: bool = False, seed: int = 0) -> ExcitationPlan:
    """~40 s all-joint plan for the synthetic ground-truth recovery test (every joint moves in every segment)."""
    hz = cfg.rates.policy_hz
    amp = tuple(0.45 * a for a in AMPLITUDE_TICKS)
    c = {"centre_ticks": (0, 200, 0, 0, 0, 400)}  # shorter lead/tail ramps than the default centres
    segs = [
        Segment("multisine", ALL, 12.0, amp, c),
        Segment("step", ALL, 4.0, 0.0, {"sizes": (45,), "hold_s": 1.0, **c}),
        Segment("ramp", ALL, 4.0, 60.0, {"rate_ticks": 5.0, **c}),
        Segment("chirp", ALL, 6.0, amp, {"f0_hz": 0.2, "f1_hz": 3.0, **c}),
        Segment("random", ALL, 5.0, amp, {"scale": 0.7, **c}),
    ]
    return ExcitationPlan(tuple(segs), hz, seed, name="fit40", hold_s=0.5)


PLANS: Final[dict[str, Any]] = {"default": default_plan, "short": short_plan, "fit40": fit40_plan}


def named_plan(name: str, cfg: So101Config, *, with_gripper_die: bool = False, seed: int = 0) -> ExcitationPlan:
    try:
        factory = PLANS[name]
    except KeyError:
        raise ValueError(f"unknown plan {name!r}; known: {sorted(PLANS)}") from None
    return factory(cfg, with_gripper_die=with_gripper_die, seed=seed)


def plan_table(plan: ExcitationPlan, cfg: So101Config) -> list[dict[str, Any]]:
    """JSON-able segment table (``plan.json``)."""
    out = []
    for span, seg in zip(segment_spans(plan, cfg), plan.segments, strict=True):
        out.append(
            {
                "index": span.index,
                "kind": seg.kind,
                "label": span.label,
                "joints": [JOINT_NAMES[j] for j in seg.joints],
                "duration_s": seg.duration_s,
                "amplitude_ticks": [float(a) for a in seg.amplitudes()],
                "params": {k: (list(v) if isinstance(v, tuple | list) else v) for k, v in seg.params.items()},
                "steps": [span.start, span.stop],
            }
        )
    return out


# --------------------------------------------------------------------------------------------- kinematic preview
def check_plan_in_box(
    plan: ExcitationPlan,
    cfg: So101Config,
    calibration: dict[str, MotorCalibration],
    chain: Any,
    start_ticks: np.ndarray,
    *,
    margin_m: float = 0.0,
) -> PlanCheck:
    """Kinematic preview of the clamped path, assuming perfect tracking: per command ``goal = clip(goal +
    clip(delta, ±max_delta), joint limits)``, interpolated over ``interp_ticks`` like the pipeline; every tick's FK
    tip must stay inside ``cfg.box`` shrunk by ``margin_m`` and above ``table_z_m + table_clearance_m``, and the lowest
    wrist/gripper point (``So101FK.tip_and_floor_z``) ``margin_m`` above ``box.gripper_floor_z_m`` (the pipeline's
    gripper floor, plan §10 2026-10-06 / 2026-10-07; ``min_box_margin_m`` includes that face).

    The box clamp is deliberately NOT applied: a plan must not rely on it (it would also wreck the excitation).
    ``chain`` is a ``safety.fk.KinematicChain``.
    """
    from so101_rl.safety.fk import So101FK

    fk = So101FK(chain, calibration)
    deltas = generate_deltas(plan, cfg)
    max_d = np.asarray(cfg.action.max_delta_ticks, dtype=np.float64)
    lo = np.asarray(cfg.joints.min_ticks, dtype=np.int64)
    hi = np.asarray(cfg.joints.max_ticks, dtype=np.int64)
    interp = int(cfg.action.interp_ticks)
    goal = np.asarray(start_ticks, dtype=np.int64).copy()
    prev = goal.copy()
    box = cfg.box
    table = cfg.safe_state.table_z_m + cfg.safe_state.table_clearance_m
    xyz = np.zeros(3)
    min_margin = math.inf
    min_clear = math.inf
    jl = 0
    first: tuple[int, tuple[float, float, float]] | None = None
    reasons: list[str] = []
    tick = 0
    for d in deltas:
        if not np.isfinite(d).all():
            reasons.append(f"non-finite delta at tick {tick}")
            break
        raw = goal + np.rint(np.clip(d, -max_d, max_d)).astype(np.int64)
        goal = np.clip(raw, lo, hi)
        cmd_clamped = bool((goal != raw).any())
        for rem in range(interp, 0, -1):
            target = prev + np.rint((goal - prev) / rem).astype(np.int64)
            floor_z = fk.tip_and_floor_z_from_ticks(target, xyz)
            m = min(xyz[0] - box.x[0], box.x[1] - xyz[0], xyz[1] - box.y[0], box.y[1] - xyz[1],
                    xyz[2] - box.z[0], box.z[1] - xyz[2], floor_z - box.gripper_floor_z_m)
            min_margin = min(min_margin, m)
            min_clear = min(min_clear, xyz[2] - table)
            if (m < margin_m or xyz[2] < table) and first is None:
                first = (tick, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
            jl += int(cmd_clamped and rem == interp)
            prev = target
            tick += 1
    if first is not None:
        reasons.append(
            f"tip leaves the box (or the lowest gripper point its floor; margin {margin_m} m) or goes below the table "
            f"at tick {first[0]}: {first[1]}"
        )
    if not math.isfinite(min_margin):
        reasons.append("empty plan")
    return PlanCheck(
        ok=not reasons,
        n_ticks=tick,
        min_box_margin_m=float(min_margin),
        min_table_clearance_m=float(min_clear),
        joint_limit_ticks=jl,
        first_violation=first,
        reasons=reasons,
    )


def scaled_plan(plan: ExcitationPlan, factor: float) -> ExcitationPlan:
    """Every amplitude (and step size) multiplied by ``factor`` (tests: amplitude x10 must be rejected)."""
    segs = []
    for s in plan.segments:
        params = dict(s.params)
        if "sizes" in params:
            params["sizes"] = tuple(float(x) * factor for x in params["sizes"])
        amp = tuple(float(a) * factor for a in s.amplitudes())
        segs.append(Segment(s.kind, s.joints, s.duration_s, amp, params))
    return ExcitationPlan(tuple(segs), plan.policy_hz, plan.seed, name=f"{plan.name}x{factor:g}", hold_s=plan.hold_s)


__all__ = [
    "ALL",
    "AMPLITUDE_TICKS",
    "ARM",
    "CENTRE_TICKS",
    "GRIPPER",
    "KINDS",
    "PLANS",
    "ExcitationPlan",
    "PlanCheck",
    "Segment",
    "SegmentSpan",
    "check_plan_in_box",
    "default_plan",
    "fit40_plan",
    "generate_deltas",
    "named_plan",
    "plan_offsets",
    "plan_table",
    "scaled_plan",
    "segment_spans",
    "short_plan",
]

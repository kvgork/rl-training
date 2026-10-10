"""Numerical joint-zero adjustment (plan §10 2026-10-10 "Joint zeros set numerically from the 2-pose camera fit").

``pixi run zero-adjust --from-report runs/camera-calib/<ts>`` (or ``--joint shoulder_lift=-4.94 ...`` in degrees)
shifts the zero of the chosen arm joints so that their reading equals the TRUE angle measured by
``camera-calibrate --poses 2`` (``diagnostics.tilt_vs_offsets``). Dry-run by default; ``--apply --attended`` writes.

Sign (verified against lerobot 0.6.1 and the STS3215):

* the offset is ``q_true = q_read + offset`` (``perception/camera_calibrate.observations``: the model is posed at
  ``qpos_from_ticks(ticks) + delta``, i.e. ``config.ticks_to_rad``, so the fitted ``delta`` is what the reading lacks);
* ``ticks_to_rad`` = ``s * (Present_Position - 2047) * 2pi/4096`` with ``s = -1`` for ``drive_mode`` 1, else +1, so the
  reading must move by ``dread = round(s * offset_deg * 4096 / 360)`` ticks;
* the servo reports ``Present_Position = Actual_Position - Homing_Offset`` (lerobot
  ``FeetechMotorsBus._get_half_turn_homings``: homing = actual - 2047 puts the reading at 2047), so
  ``new Homing_Offset = old - dread``. ``drive_mode`` is not a servo register: the servo-side shift is always
  ``dread``;
* ``Min/Max_Position_Limit`` are in reading coordinates (lerobot records them from ``Present_Position`` after the
  homing write), so both shift by ``dread`` too and the PHYSICAL range of motion is unchanged.

Refusals (exit 2, nothing written): a report whose verdict is not ``joint_offsets``, a used offset with
sigma > ``--max-sigma-deg`` or |offset| > 10 deg, a calibration JSON modified after the report (the offsets were
measured against another calibration, and this is what stops a second apply of the same report), the gripper, a
new Homing_Offset beyond the 12-bit sign-magnitude range (+-2047) or limits outside 0..4095. ``--apply`` also
refuses without ``--attended``, while the servo process holds the port lock, with torque on on ANY motor, when the
servo registers differ from the JSON (run ``calib-check``), or when the arm moves during the reads.

``--apply --attended``: back up the JSON (``<json>.bak-<ts>``), per chosen motor ``Lock`` 0 (EEPROM unlock, as
lerobot's ``disable_torque`` / ``safety.reenergize`` do), ``Homing_Offset``, ``Min_Position_Limit``,
``Max_Position_Limit`` (lerobot ``write_calibration`` order), ``Lock`` back to its previous value; read everything
back, check the present reading moved by the predicted ``dread`` (+-2 ticks, the other joints by 0), write the JSON
atomically (only after every read-back passed), run the calib-check comparison against the new file and write
``runs/hw/zero-adjust-<ts>.json`` (``runs/hw/failed/`` on a failure). A failure after a write prints every write
done, the backup and the command that restores it: ``--restore <backup> --apply --attended`` writes the backup's
registers to every joint that differs and puts the backup back as the JSON.

Only this module (servo package) writes; the dry-run reads through the read-only probe (``servo.probe``).
Exit: 0 ok, 2 refused (nothing written), 3 failed after a write (restore printed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from so101_rl.config import JOINT_NAMES, TICKS_PER_REV, MotorCalibration, load_calibration
from so101_rl.safety.calib_check import check_calibration
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.servo.shm import SerialPortLock, ServoAlreadyRunning

ADJUSTABLE = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")  # never the gripper
FROM_REPORT_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")  # wrist_roll: barely visible, only via --joint
MAX_ABS_OFFSET_DEG = 10.0
DEFAULT_MIN_DEG = 0.5
DEFAULT_MAX_SIGMA_DEG = 0.5
HOMING_MAX = 2047  # Homing_Offset: sign-magnitude, sign bit 11
TICK_MAX = 4095
MOVE_TOL_TICKS = 2
STILL_SPREAD_TICKS = 3
N_READS = 5
EXIT_OK, EXIT_REFUSED, EXIT_WRITTEN_FAILED = 0, 2, 3
TICKS_PER_DEG = TICKS_PER_REV / 360.0


class ZeroAdjustRefused(RuntimeError):
    """Nothing was written."""


@dataclass(frozen=True)
class JointShift:
    joint: str
    offset_deg: float | None  # None for a restore
    delta_ticks: int  # predicted change of Present_Position = old.homing_offset - new.homing_offset
    old: MotorCalibration
    new: MotorCalibration


# --------------------------------------------------------------------------------------------- planning (pure)
def delta_ticks(offset_deg: float, drive_mode: int) -> int:
    """Reading change that makes the reading equal the true angle (``q_true = q_read + offset``)."""
    return round((-1 if drive_mode else 1) * offset_deg * TICKS_PER_DEG)


def _check_target(joint: str, new: MotorCalibration) -> None:
    if abs(new.homing_offset) > HOMING_MAX:
        raise ZeroAdjustRefused(f"{joint}: new Homing_Offset {new.homing_offset} outside +-{HOMING_MAX}")
    if not 0 <= new.range_min < new.range_max <= TICK_MAX:
        raise ZeroAdjustRefused(f"{joint}: new limits {new.range_min}..{new.range_max} outside 0..{TICK_MAX}")


def shifts_from_offsets(
    offsets_deg: dict[str, float], calibration: dict[str, MotorCalibration], *, min_deg: float = DEFAULT_MIN_DEG
) -> tuple[list[JointShift], dict[str, str]]:
    """The register changes for ``offsets_deg`` (deg, ``q_true = q_read + offset``), JOINT_NAMES order, and the
    skipped joints with the reason. Raises :class:`ZeroAdjustRefused` on a bad joint, offset or target."""
    shifts, skipped = [], {}
    for joint, off in offsets_deg.items():
        if joint not in ADJUSTABLE:
            raise ZeroAdjustRefused(f"{joint}: not adjustable (arm joints only: {', '.join(ADJUSTABLE)})")
        if not math.isfinite(off) or abs(off) > MAX_ABS_OFFSET_DEG:
            raise ZeroAdjustRefused(f"{joint}: offset {off} deg outside +-{MAX_ABS_OFFSET_DEG:g} deg")
    for joint in (n for n in JOINT_NAMES if n in offsets_deg):
        off = float(offsets_deg[joint])
        old = calibration[joint]
        d = delta_ticks(off, old.drive_mode)
        if abs(off) < min_deg or d == 0:
            skipped[joint] = f"{off:+.2f} deg below --min-deg {min_deg:g}" if abs(off) < min_deg else "rounds to 0 ticks"
            continue
        new = replace(old, homing_offset=old.homing_offset - d, range_min=old.range_min + d, range_max=old.range_max + d)
        _check_target(joint, new)
        shifts.append(JointShift(joint, off, d, old, new))
    return shifts, skipped


def shifts_to_restore(current: dict[str, MotorCalibration], backup: dict[str, MotorCalibration]) -> list[JointShift]:
    """Register changes that put every joint whose registers differ from ``backup`` back (``--restore``)."""
    out = []
    for joint in JOINT_NAMES:
        cur, want = current[joint], backup[joint]
        if (cur.homing_offset, cur.range_min, cur.range_max) == (want.homing_offset, want.range_min, want.range_max):
            continue
        if joint not in ADJUSTABLE or cur.id != want.id:
            raise ZeroAdjustRefused(f"{joint}: registers {cur} differ from the backup {want}; not a zero-adjust change")
        _check_target(joint, want)
        out.append(JointShift(joint, None, cur.homing_offset - want.homing_offset, cur, want))
    return out


def offsets_from_report(
    run_dir: Path, calibration_path: Path, *, max_sigma_deg: float = DEFAULT_MAX_SIGMA_DEG
) -> tuple[dict[str, float], dict]:
    """The offsets of :data:`FROM_REPORT_JOINTS` from ``<run_dir>/report.json`` ``diagnostics.tilt_vs_offsets``,
    refused unless the verdict is ``joint_offsets``, every sigma <= ``max_sigma_deg``, |offset| <= 10 deg and the
    calibration JSON is older than the report."""
    path = Path(run_dir) / "report.json"
    try:
        tvo = json.loads(path.read_text())["diagnostics"]["tilt_vs_offsets"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ZeroAdjustRefused(f"{path}: no diagnostics.tilt_vs_offsets ({type(e).__name__}: {e})") from e
    if not tvo or tvo.get("verdict") != "joint_offsets":
        raise ZeroAdjustRefused(f"{path}: verdict {None if not tvo else tvo.get('verdict')!r}, not 'joint_offsets'")
    offsets, sigmas = {}, tvo["offsets_sigma_deg"]
    for j in FROM_REPORT_JOINTS:
        off, sd = float(tvo["offsets_deg"][j]), float(sigmas[j])
        if not sd <= max_sigma_deg:
            raise ZeroAdjustRefused(f"{path}: {j} sigma {sd:.3f} deg > --max-sigma-deg {max_sigma_deg:g}")
        if not abs(off) <= MAX_ABS_OFFSET_DEG:
            raise ZeroAdjustRefused(f"{path}: {j} offset {off:+.2f} deg outside +-{MAX_ABS_OFFSET_DEG:g} deg")
        offsets[j] = off
    if calibration_path.stat().st_mtime > path.stat().st_mtime:
        raise ZeroAdjustRefused(
            f"{calibration_path} was modified after {path}: the offsets were measured against another calibration "
            "(already adjusted?). Re-run camera-calibrate --poses 2, or pass --joint explicitly"
        )
    return offsets, tvo


def parse_joint_args(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, sep, val = item.partition("=")
        try:
            if not sep:
                raise ValueError
            out[name.strip()] = float(val)
        except ValueError:
            raise ZeroAdjustRefused(f"--joint {item!r}: expected NAME=DEGREES") from None
    return out


def format_table(shifts: list[JointShift], present: dict[str, int] | None) -> str:
    head = (
        f"{'joint':<14}{'offset deg':>11}{'dticks':>8}{'homing old -> new':>20}{'min old -> new':>17}"
        f"{'max old -> new':>17}{'present -> predicted':>23}"
    )
    lines = [head, "-" * len(head)]
    for s in shifts:
        off = "restore" if s.offset_deg is None else f"{s.offset_deg:+.2f}"
        pres = "-" if present is None else f"{present[s.joint]} -> {present[s.joint] + s.delta_ticks}"
        lines.append(
            f"{s.joint:<14}{off:>11}{s.delta_ticks:>+8d}"
            f"{f'{s.old.homing_offset} -> {s.new.homing_offset}':>20}"
            f"{f'{s.old.range_min} -> {s.new.range_min}':>17}{f'{s.old.range_max} -> {s.new.range_max}':>17}"
            f"{pres:>23}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- bus side
def _present(bus: Bus, n: int = N_READS) -> np.ndarray:
    reads = []
    for _ in range(n):
        b = BlockRead()
        if not bus.read_block(b):
            raise ConnectionError("block read failed")
        reads.append(b.position.astype(np.int64))
    pos = np.stack(reads)
    spread = pos.max(axis=0) - pos.min(axis=0)
    if int(spread.max()) > STILL_SPREAD_TICKS:
        raise ZeroAdjustRefused(f"the arm moved during the reads (spread {spread.tolist()} ticks); let it rest")
    return np.rint(np.median(pos, axis=0)).astype(np.int64)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(json.dumps(data, indent=4))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclass
class ApplyLog:
    written: list[dict]  # every acked register write, in order
    backup: Path | None = None
    json_replaced: bool = False
    pending: dict | None = None  # the write in flight: it may have reached the servo even if it raised


def apply_shifts(
    bus: Bus,
    shifts: list[JointShift],
    calibration: dict[str, MotorCalibration],
    calibration_path: Path,
    log: ApplyLog,
    *,
    restore_json: Path | None = None,
    stamp: str,
) -> dict:
    """See the module docstring. ``bus`` is connected; the caller holds the port lock. Raises
    :class:`ZeroAdjustRefused` before the first write; any other exception may come after writes (see ``log``)."""
    ids = [calibration[n].id for n in JOINT_NAMES]
    silent = [i for i, ok in bus.ping_all().items() if not ok]
    if silent:
        raise ZeroAdjustRefused(f"motors {silent} do not answer a ping (servo power off?)")
    torque = {n: int(bus.read_reg("Torque_Enable", calibration[n].id)) for n in JOINT_NAMES}
    if any(torque.values()):
        raise ZeroAdjustRefused(f"torque must be off on every motor (Torque_Enable {torque})")
    regs = bus.read_calibration_regs()
    for s in shifts:  # the plan was made from these values: the servo must still hold them
        r = regs[s.joint]
        if (r.id, r.homing_offset, r.range_min, r.range_max) != (s.old.id, s.old.homing_offset, s.old.range_min, s.old.range_max):
            raise ZeroAdjustRefused(f"{s.joint}: servo registers {r} differ from the planned old values {s.old}")
    if restore_json is None:
        mism = check_calibration(bus, calibration)
        if not mism.ok:
            raise ZeroAdjustRefused(f"servo registers differ from the JSON ({mism.mismatches}); run calib-check")
    before = _present(bus)
    locks = {s.joint: int(bus.read_reg("Lock", s.new.id)) for s in shifts}

    log.backup = calibration_path.with_name(f"{calibration_path.name}.bak-{stamp}")
    shutil.copy2(calibration_path, log.backup)

    def write(joint: str, reg: str, old: int, new: int) -> None:
        log.pending = {"joint": joint, "id": calibration[joint].id, "register": reg, "old": old, "new": new}
        bus.write_reg(reg, calibration[joint].id, new)
        log.written.append(log.pending)
        log.pending = None

    for s in shifts:
        write(s.joint, "Lock", locks[s.joint], 0)
    for s in shifts:
        write(s.joint, "Homing_Offset", s.old.homing_offset, s.new.homing_offset)
        write(s.joint, "Min_Position_Limit", s.old.range_min, s.new.range_min)
        write(s.joint, "Max_Position_Limit", s.old.range_max, s.new.range_max)
    for s in shifts:
        write(s.joint, "Lock", 0, locks[s.joint])

    after_regs = bus.read_calibration_regs()
    want = dict(regs)
    want.update({s.joint: replace(s.new, drive_mode=regs[s.joint].drive_mode) for s in shifts})
    bad = {n: (asdict(want[n]), asdict(after_regs[n])) for n in JOINT_NAMES if after_regs[n] != want[n]}
    lock_back = {s.joint: int(bus.read_reg("Lock", s.new.id)) for s in shifts}
    if bad or lock_back != locks:
        raise RuntimeError(f"read-back differs: registers {bad}, Lock {lock_back} (expected {locks})")
    after = _present(bus)
    predicted = before.copy()
    for s in shifts:
        predicted[JOINT_NAMES.index(s.joint)] += s.delta_ticks
    if np.abs(after - predicted).max() > MOVE_TOL_TICKS:
        raise RuntimeError(f"present reading {after.tolist()} != predicted {predicted.tolist()} (+-{MOVE_TOL_TICKS})")

    if restore_json is not None:
        new_raw = json.loads(restore_json.read_text())
    else:
        new_raw = json.loads(calibration_path.read_text())
        for s in shifts:
            new_raw[s.joint].update(homing_offset=s.new.homing_offset, range_min=s.new.range_min, range_max=s.new.range_max)
    _atomic_write_json(calibration_path, new_raw)
    log.json_replaced = True
    check = check_calibration(bus, load_calibration(calibration_path))
    if not check.ok:
        raise RuntimeError(f"calib-check after the write failed: {check.mismatches}")
    return {
        "ids": ids,
        "torque_enabled": torque,
        "lock_before": locks,
        "present_before": before.tolist(),
        "present_after": after.tolist(),
        "present_predicted": predicted.tolist(),
        "registers_before": {n: asdict(regs[n]) for n in JOINT_NAMES},
        "registers_after": {n: asdict(after_regs[n]) for n in JOINT_NAMES},
        "calib_check": {"ok": True, "mismatches": {}},
    }


def _failure_text(log: ApplyLog, calibration_path: Path, shifts: list[JointShift], err: BaseException) -> str:
    lines = [f"zero-adjust: FAILED AFTER WRITING ({type(err).__name__}: {err})", "registers written (in order):"]
    lines += [f"  {w['joint']} (id {w['id']}) {w['register']}: {w['old']} -> {w['new']}" for w in log.written]
    if log.pending:
        w = log.pending
        lines.append(f"  MAYBE (it raised): {w['joint']} (id {w['id']}) {w['register']}: {w['old']} -> {w['new']}")
    lines.append("old values (what the servo held before):")
    lines += [
        f"  {s.joint}: homing_offset {s.old.homing_offset}, range_min {s.old.range_min}, range_max {s.old.range_max}"
        for s in shifts
    ]
    state = "now holds the NEW values" if log.json_replaced else "was NOT changed"
    lines.append(f"calibration JSON {calibration_path} {state}; backup of the old file: {log.backup}")
    lines.append(f"restore: pixi run zero-adjust --restore {log.backup} --apply --attended   (then pixi run calib-check)")
    return "\n".join(lines)


def _write_report(result: dict, out_dir: Path, ok: bool, stamp: str) -> Path:
    root = out_dir if ok else out_dir / "failed"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"zero-adjust-{stamp}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    return path


def _rest_note(cfg, shifts: list[JointShift]) -> str | None:
    ss = getattr(cfg, "safe_state", None)
    if ss is None or ss.rest_ticks is None:
        return None
    parts = [
        f"{s.joint}: the pose that read rest {int(ss.rest_ticks[JOINT_NAMES.index(s.joint)])} will read "
        f"{int(ss.rest_ticks[JOINT_NAMES.index(s.joint)]) + s.delta_ticks}"
        for s in shifts
    ]
    return (
        "note: so101.yaml rest_ticks / home_ticks are tick values and are NOT changed; the physical pose they name moves: "
        f"{'; '.join(parts)} (rest_tolerance_ticks {ss.rest_tolerance_ticks})"
    )


# --------------------------------------------------------------------------------------------- CLI
def main(
    argv: list[str] | None = None,
    *,
    bus_factory: Callable[[str, dict[str, MotorCalibration]], Bus] | None = None,
    link_factory=None,
) -> int:
    """See the module docstring. ``bus_factory(port, calibration)`` (apply) and ``link_factory`` (the probe's
    read-only link, dry-run) are test hooks; the defaults are ``FeetechBus`` and the probe's ``ScsReadOnlyLink``."""
    from so101_rl.config import load_config
    from so101_rl.servo.probe import DEFAULT_TIMEOUT_S, ProbeRefused, probe_joints, read_calibration

    ap = argparse.ArgumentParser(prog="python -m so101_rl.servo.zero_adjust", description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-report", type=Path, help="runs/camera-calib/<ts> (diagnostics.tilt_vs_offsets)")
    src.add_argument("--joint", action="append", help="NAME=DEGREES (q_true = q_read + offset); repeatable")
    src.add_argument("--restore", type=Path, help="a <json>.bak-<ts> backup: write its registers back")
    ap.add_argument("--min-deg", type=float, default=DEFAULT_MIN_DEG, help="leave joints below this alone")
    ap.add_argument("--max-sigma-deg", type=float, default=DEFAULT_MAX_SIGMA_DEG, help="--from-report only")
    ap.add_argument("--apply", action="store_true", help="write (default: dry-run, nothing written)")
    ap.add_argument("--attended", action="store_true", help="required with --apply: you are at the arm")
    ap.add_argument("--port", default=None, help="default: so101.yaml port")
    ap.add_argument("--calibration", type=Path, default=None, help="default: so101.yaml calibration_path")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/hw"))
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="dry-run probe bound, s")
    args = ap.parse_args(argv)

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    try:
        if args.apply and not args.attended:
            raise ZeroAdjustRefused("--apply writes servo EEPROM: it needs --attended (you at the arm, servo stopped)")
        cfg = load_config(calibration=load_calibration(args.calibration) if args.calibration else None)
        cal_path = Path(args.calibration or cfg.calibration_path).expanduser()
        calibration = load_calibration(cal_path)
        port = args.port or cfg.port
        source: dict = {}
        skipped: dict[str, str] = {}
        if args.restore is not None:
            backup = load_calibration(args.restore)
            source = {"restore": str(args.restore)}
            shifts = None  # needs the registers: planned below
        else:
            if args.from_report is not None:
                offsets, tvo = offsets_from_report(args.from_report, cal_path, max_sigma_deg=args.max_sigma_deg)
                source = {"from_report": str(args.from_report), "tilt_vs_offsets": tvo}
            else:
                offsets = parse_joint_args(args.joint)
                source = {"joint": offsets}
            shifts, skipped = shifts_from_offsets(offsets, calibration, min_deg=args.min_deg)
    except (ZeroAdjustRefused, OSError, ValueError, KeyError) as e:
        print(f"zero-adjust: refused: {e}", file=sys.stderr)
        return EXIT_REFUSED

    print(f"calibration: {cal_path}\nport: {port}")
    if not args.apply:  # ---------------------------------------------------------------- dry-run: read only
        present = None
        try:
            if shifts is None:
                shifts = shifts_to_restore(
                    read_calibration(calibration, port=port, link_factory=link_factory, timeout_s=args.timeout), backup
                )
            res = probe_joints(calibration, port=port, link_factory=link_factory, timeout_s=args.timeout)
            present = {n: int(res.ticks[j]) for j, n in enumerate(JOINT_NAMES)}
            if res.homing_matches_calibration is False and args.restore is None:
                print(f"warning: servo homing offsets {res.homing_offset} differ from the JSON: --apply would refuse")
            if res.torque_enabled and any(res.torque_enabled.values()):
                print(f"warning: torque is on ({res.torque_enabled}): --apply would refuse")
        except (ProbeRefused, ZeroAdjustRefused, OSError) as e:
            if shifts is None:
                print(f"zero-adjust: refused: cannot read the registers for --restore ({e})", file=sys.stderr)
                return EXIT_REFUSED
            print(f"(no present reading: {e})")
        print(format_table(shifts, present) if shifts else "nothing to change")
        for j, why in skipped.items():
            print(f"skipped: {j} ({why})")
        note = _rest_note(cfg, shifts)
        if shifts and note:
            print(note)
        print("zero-adjust: DRY-RUN, nothing written (add --apply --attended to write)")
        return EXIT_OK

    # ------------------------------------------------------------------------------------- apply
    result: dict = {"ok": False, "stamp": stamp, "port": port, "calibration_path": str(cal_path), **source}
    try:
        lock = SerialPortLock(port)
    except ServoAlreadyRunning as e:
        print(f"zero-adjust: refused: the servo process owns {port}; stop it first ({e})", file=sys.stderr)
        return EXIT_REFUSED
    log = ApplyLog(written=[])
    bus = None
    try:
        if bus_factory is None:
            from so101_rl.servo.feetech_bus import FeetechBus

            bus = FeetechBus(replace(cfg, port=port), calibration)
        else:
            bus = bus_factory(port, calibration)
        bus.connect()
        if shifts is None:
            shifts = shifts_to_restore(bus.read_calibration_regs(), backup)
        if not shifts:
            print("nothing to change")
            for j, why in skipped.items():
                print(f"skipped: {j} ({why})")
            return EXIT_OK
        print(format_table(shifts, None))
        for j, why in skipped.items():
            print(f"skipped: {j} ({why})")
        result["shifts"] = [
            {"joint": s.joint, "offset_deg": s.offset_deg, "delta_ticks": s.delta_ticks, "old": asdict(s.old), "new": asdict(s.new)}
            for s in shifts
        ]
        result["skipped"] = skipped
        result.update(
            apply_shifts(bus, shifts, calibration, cal_path, log, restore_json=args.restore, stamp=stamp)
        )
    except ZeroAdjustRefused as e:
        print(f"zero-adjust: refused: {e}", file=sys.stderr)
        result.update(error=str(e), exit_code=EXIT_REFUSED, backup=str(log.backup) if log.backup else None)
        print(f"zero-adjust: wrote {_write_report(result, args.out_dir, False, stamp)}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # noqa: BLE001 - anything after the first write must print the restore path
        if not log.written and not log.pending:
            print(f"zero-adjust: refused (nothing written): {type(e).__name__}: {e}", file=sys.stderr)
            result.update(error=f"{type(e).__name__}: {e}", exit_code=EXIT_REFUSED)
            _write_report(result, args.out_dir, False, stamp)
            return EXIT_REFUSED
        print(_failure_text(log, cal_path, shifts or [], e), file=sys.stderr)
        result.update(
            error=f"{type(e).__name__}: {e}",
            exit_code=EXIT_WRITTEN_FAILED,
            written=log.written,
            maybe_written=log.pending,
            backup=str(log.backup),
            json_replaced=log.json_replaced,
        )
        print(f"zero-adjust: wrote {_write_report(result, args.out_dir, False, stamp)}", file=sys.stderr)
        return EXIT_WRITTEN_FAILED
    finally:
        if bus is not None:
            try:
                bus.disconnect()  # never changes torque
            except Exception:  # noqa: BLE001, S110 - the port is closed with the process anyway
                pass
        lock.release()

    result.update(ok=True, exit_code=EXIT_OK, written=log.written, backup=str(log.backup))
    for s in shifts:
        j = JOINT_NAMES.index(s.joint)
        print(f"{s.joint}: present {result['present_before'][j]} -> {result['present_after'][j]} ({s.delta_ticks:+d} predicted)")
    print(f"backup: {log.backup}")
    print("calib-check: PASS (servo registers match the new JSON)")
    note = _rest_note(cfg, shifts)
    if note:
        print(note)
        result["note"] = note
    print(f"zero-adjust: APPLIED; wrote {_write_report(result, args.out_dir, True, stamp)}")
    print("next: pixi run calib-check; then pixi run camera-calibrate --method arm-shape --poses 2 (|offsets| < 1 deg)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

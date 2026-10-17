"""H8 camera-moved check (``pixi run camera-moved-check``): the perception process's two camera-moved checks (the
ArUco fiducial of ``calib/fiducial.yaml`` and the table plane of ``calib/camera.yaml``) on the live D435 for a few
seconds, as a verdict and a JSON record. No servo, no arm motion.

``python -m so101_rl.perception.camera_moved [--label L] [--duration-s 5] [--out PATH]`` measures once and writes
``runs/hw/camera-moved-<ts>-<label>.json``; exit 0 = the gate passes, 1 = it fails (camera moved).

``--procedure`` runs the H8 shown-to-fail test in one go (``docs/HARDWARE_CHECKLIST.md`` H8): ``baseline`` (must
pass) -> "nudge the camera by 2 mm" (Enter) -> ``nudged`` (must FAIL) -> "put it back" (Enter) -> ``restored`` (must
pass), and writes ``runs/hw/camera-moved-<ts>.json`` with the three measurements and ``baseline_err_px``,
``nudged_err_px``, ``restored_err_px``, ``pass``; exit 0 only when all three are as required.

One measurement: every frame the fiducial check (mean corner distance to ``expected_corners_px``; frames without the
marker counted as ``not_found``, no evidence), every ``--plane-every`` frames the table-plane check. Fiducial verdict:
found in >= half the frames and median error <= ``tol_px``; plane verdict: >= half the plane checks pass;
``ok`` = both. ``err_px`` is the median error (``null`` if never found). Exit 2 calibration refused (no
``calib/fiducial.yaml``, no table plane, another camera's fiducial), 3 camera refused.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from so101_rl.perception.camera_calib import CameraCalib
from so101_rl.perception.fiducial import FIDUCIAL_YAML, FiducialCheck
from so101_rl.perception.process import PLANE_MAX_ANGLE_DEG, PLANE_MAX_DH_M, TablePlaneCheck

RUNS_HW = Path("runs/hw")
DURATION_S = 5.0
PLANE_EVERY = 10


def _num(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def measure(camera: Any, calib: CameraCalib, fiducial: FiducialCheck, plane: TablePlaneCheck, *,
            duration_s: float = DURATION_S, plane_every: int = PLANE_EVERY, max_frames: int | None = None) -> dict[str, Any]:  # fmt: skip
    """One measurement on the started ``camera`` (module docstring)."""
    c = calib.color
    rgb = np.zeros((c.height, c.width, 3), np.uint8)
    depth = np.zeros((c.height, c.width), np.float32)
    errs: list[float] = []
    frames = failed = not_found = 0
    planes: list[Any] = []
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end and (max_frames is None or frames < max_frames):
        if not camera.read(rgb, depth).ok:
            failed += 1
            continue
        if frames % plane_every == 0:
            planes.append(plane.check(depth))
        frames += 1
        r = fiducial.check(rgb)
        if r.found:
            errs.append(r.reproj_err_px)
        else:
            not_found += 1
    found = len(errs)
    err = float(np.median(errs)) if errs else None
    fid_ok = frames > 0 and found >= frames / 2 and err is not None and err <= fiducial.tol_px
    passed = sum(p.ok for p in planes)
    plane_ok = bool(planes) and passed >= len(planes) / 2
    return {
        "frames": frames,
        "failed_reads": failed,
        "err_px": err,
        "err_px_max": max(errs) if errs else None,
        "fiducial": {"found": found, "not_found": not_found, "tol_px": fiducial.tol_px, "ok": bool(fid_ok)},
        "plane": {
            "checks": len(planes),
            "passed": passed,
            "angle_deg": _num(float(np.median([p.angle_deg for p in planes]))) if planes else None,
            "dh_mm": _num(float(np.median([p.dh_m for p in planes])) * 1e3) if planes else None,
            "limits": {"angle_deg": PLANE_MAX_ANGLE_DEG, "dh_mm": PLANE_MAX_DH_M * 1e3},
            "ok": bool(plane_ok),
        },
        "ok": bool(fid_ok and plane_ok),
    }


def _line(label: str, m: dict[str, Any]) -> str:
    err = "not found" if m["err_px"] is None else f"{m['err_px']:.2f} px"
    p = m["plane"]
    ang = "-" if p["angle_deg"] is None else f"{p['angle_deg']:.2f} deg"
    dh = "-" if p["dh_mm"] is None else f"{p['dh_mm']:.1f} mm"
    return (f"camera-moved-check {label}: fiducial {err} (found {m['fiducial']['found']}/{m['frames']}), plane {ang} / "
            f"{dh} -> {'PASS' if m['ok'] else 'FAIL (camera moved)'}")  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="camera-moved-check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", default="check")
    ap.add_argument("--procedure", action="store_true", help="baseline -> nudged -> restored, prompted (H8)")
    ap.add_argument("--duration-s", type=float, default=DURATION_S)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--plane-every", type=int, default=PLANE_EVERY)
    ap.add_argument("--camera-yaml", type=Path, default=None)
    ap.add_argument("--fiducial-yaml", type=Path, default=FIDUCIAL_YAML)
    ap.add_argument("--out", type=Path, default=None, help="default runs/hw/camera-moved-<ts>[-<label>].json")
    return ap


def main(argv: list[str] | None = None, *, camera: Any | None = None,
         prompt: Callable[[str], Any] | None = None) -> int:  # fmt: skip
    from so101_rl.perception.camera_calib import CameraNotCalibrated, load_camera_calib
    from so101_rl.perception.process import EXIT_CALIB, EXIT_CAMERA, CameraRefused, open_checked

    a = build_parser().parse_args(argv)
    prompt = prompt if prompt is not None else input
    try:
        calib = load_camera_calib(a.camera_yaml)
        plane = TablePlaneCheck(calib)
        fiducial = FiducialCheck.from_yaml(a.fiducial_yaml)
        if fiducial.camera_serial is not None and fiducial.camera_serial != calib.serial:
            raise ValueError(f"{a.fiducial_yaml} was recorded with camera {fiducial.camera_serial}, not {calib.serial}")
    except (CameraNotCalibrated, FileNotFoundError, ValueError, TypeError, KeyError, AttributeError) as e:
        print(f"camera-moved-check: calibration refused: {e}", file=sys.stderr)
        return EXIT_CALIB
    c = calib.color
    if camera is None:
        from so101_rl.perception.camera import D435Camera

        camera = D435Camera(serial=calib.serial, width=c.width, height=c.height, fps=c.fps)
    try:
        open_checked(camera, calib)
    except (CameraRefused, RuntimeError, ImportError) as e:
        print(f"camera-moved-check: camera refused: {e}", file=sys.stderr)
        return EXIT_CAMERA
    started = _dt.datetime.now().astimezone()
    doc: dict[str, Any] = {"started": started.isoformat(timespec="seconds"), "camera_serial": calib.serial,
                           "fiducial_yaml": str(a.fiducial_yaml), **calib.pose_meta()}  # fmt: skip

    def run(label: str) -> dict[str, Any]:
        m = measure(camera, calib, fiducial, plane, duration_s=a.duration_s, plane_every=a.plane_every,
                    max_frames=a.max_frames)  # fmt: skip
        print(_line(label, m), file=sys.stderr, flush=True)
        return m

    try:
        if a.procedure:
            doc["baseline"] = run("baseline")
            prompt("camera-moved-check: nudge the camera by about 2 mm (a light push on the mount), then press Enter ")
            doc["nudged"] = run("nudged")
            prompt("camera-moved-check: put the camera back exactly, then press Enter ")
            doc["restored"] = run("restored")
            for k in ("baseline", "nudged", "restored"):
                doc[f"{k}_err_px"] = doc[k]["err_px"]
            doc["pass"] = bool(doc["baseline"]["ok"] and not doc["nudged"]["ok"] and doc["restored"]["ok"])
            out = a.out or RUNS_HW / f"camera-moved-{started:%Y%m%d-%H%M%S}.json"
        else:
            doc["label"] = a.label
            doc.update(run(a.label))
            doc["pass"] = doc["ok"]
            out = a.out or RUNS_HW / f"camera-moved-{started:%Y%m%d-%H%M%S}-{a.label}.json"
    finally:
        camera.stop()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    verdict = "PASS" if doc["pass"] else "FAIL"
    print(f"camera-moved-check: {verdict}; wrote {out}", file=sys.stderr)
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

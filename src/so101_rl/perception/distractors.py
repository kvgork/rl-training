"""``camera-distractors``: measure what the red-die HSV tracker would mistake for the die (camera parity C.5).

``pixi run camera-distractors [--frames 5] [--out calib/distractors.yaml]``

Captures the D435 colour (and, when the depth stream delivers, aligned depth), with NO die in the workspace, and
runs the real tracker's colour mask (``HsvDepthDieTracker._mask``, the thresholds of ``hsv_red_die.yaml``). Every
connected component whose area lies in the tracker's band is a distractor: the tracker reports the LARGEST such
blob, so each of them is a false positive whenever it is larger than the die's blob. Written to
``calib/distractors.yaml`` (schema: ``calib/README.md``), read by ``sim.sim_tracker.DistractorModel.from_yaml``.

Positions: the blob's median depth deprojected through ``T_base_color`` (``xyz_source: depth``), or, without
depth, the camera ray through the blob centre intersected with the table plane (``xyz_source: table-ray``; the
height of an object on the arm is then wrong, its image position is right). ``camera_pose_method`` records which
pose was used: a provisional pose makes the base-frame positions provisional too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from so101_rl.perception.camera_calib import CAMERA_YAML, CameraCalib, load_camera_calib, today

OUT = CAMERA_YAML.parent / "distractors.yaml"


def measure(rgb: np.ndarray, depth_m: np.ndarray | None, calib: CameraCalib, table_z_m: float) -> list[dict]:
    import cv2

    from so101_rl.perception.die_tracker import HsvDepthDieTracker

    tr = HsvDepthDieTracker.from_camera_calib(calib)
    mask = tr._mask(rgb)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    T = calib.T_base_color
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not tr.min_area_px <= area <= tr.max_area_px:
            continue
        u, v = float(cents[i][0]), float(cents[i][1])
        src = "table-ray"
        p = None
        if depth_m is not None:
            vals = depth_m[labels == i]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size >= 8:
                p = T[:3, :3] @ calib.color.deproject(u, v, float(np.median(vals))) + T[:3, 3]
                src = "depth"
        if p is None:
            ray = T[:3, :3] @ np.array([(u - calib.color.cx) / calib.color.fx, (v - calib.color.cy) / calib.color.fy, 1.0])
            if ray[2] >= -1e-6:
                continue  # above the horizon: never on the table
            p = T[:3, 3] + ray * ((table_z_m - T[2, 3]) / ray[2])
        out.append({"label": f"blob{len(out)}", "u": round(u, 1), "v": round(v, 1), "area_px": area,
                    "xyz_base": [round(float(x), 4) for x in p], "xyz_source": src})  # fmt: skip
    return sorted(out, key=lambda d: -d["area_px"])


def write(path: Path, items: list[dict], calib: CameraCalib, source: str) -> None:
    from so101_rl.perception.die_tracker import DEFAULT_HSV_YAML

    hsv = yaml.safe_load(DEFAULT_HSV_YAML.read_text())
    doc = {
        "version": 1,
        "date": today(),
        "source": source,
        "camera_pose_method": calib.require_pose().method,
        "hsv": {k: hsv[k] for k in ("hsv_lo", "hsv_hi", "min_area_px", "max_area_px")},
        "distractors": items,
    }
    Path(path).write_text("# Red/orange false-positive candidates of the HSV die tracker (camera-distractors, C.5). Generated.\n"
                          + yaml.safe_dump(doc, sort_keys=False))  # fmt: skip


def main(argv: list[str] | None = None, *, camera=None) -> int:
    from so101_rl.sim.model import TABLE_Z_M

    ap = argparse.ArgumentParser(prog="python -m so101_rl.perception.distractors", description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--camera-yaml", type=Path, default=CAMERA_YAML)
    ap.add_argument("--png", type=Path, default=None, help="also save the colour frame with the blobs marked")
    ap.add_argument("--from-npz", type=Path, default=None, help="a saved capture (rgb [K,H,W,3] or [H,W,3], optional depth)")
    ap.add_argument("--note", default="", help="appended to the source line (e.g. what else is in the frame)")
    a = ap.parse_args(argv)
    calib = load_camera_calib(a.camera_yaml)
    depth = None
    if a.from_npz is not None:
        z = np.load(a.from_npz)
        rgb = z["rgb"][-1] if z["rgb"].ndim == 4 else z["rgb"]
        if "depth" in z.files:
            from so101_rl.perception.table_plane import median_depth

            depth = median_depth(z["depth"] if z["depth"].ndim == 3 else z["depth"][None])
        how = f"saved capture {a.from_npz.name}" + (" with depth" if depth is not None else ", colour only")
    else:
        rgb, depth, how = _capture(calib, a.frames, camera)
    items = measure(rgb, depth, calib, TABLE_Z_M)
    write(a.out, items, calib, f"camera-distractors {today()}, D435 {calib.serial}, {how}" + (f"; {a.note}" if a.note else ""))
    if a.png is not None:
        import cv2

        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for d in items:
            cv2.circle(img, (int(d["u"]), int(d["v"])), 12, (0, 255, 255), 2)
            cv2.putText(img, f"{d['label']} {d['area_px']}px", (int(d["u"]) + 14, int(d["v"])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        a.png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.png), img)
    print(f"{a.out}: {len(items)} distractor blob(s) in the tracker's band: "
          + ", ".join(f"{d['label']} {d['area_px']} px at ({d['u']:.0f}, {d['v']:.0f})" for d in items))  # fmt: skip
    return 0


def _capture(calib: CameraCalib, n: int, camera) -> tuple[np.ndarray, np.ndarray | None, str]:
    from so101_rl.perception.camera_calibrate import capture_d435
    from so101_rl.perception.table_plane import median_depth

    try:
        rgb, frames = (camera or capture_d435)(calib, n)
        return rgb, median_depth(frames), "colour + aligned depth"
    except RuntimeError as e:
        print(f"camera-distractors: no depth ({e}); colour only, positions by table ray", file=sys.stderr)
        return _colour_only(calib), None, "colour only (depth stream gave no frames)"


def _colour_only(calib: CameraCalib) -> np.ndarray:
    from so101_rl.perception.camera import D435Camera

    c = calib.color
    cam = D435Camera(serial=calib.serial, width=c.width, height=c.height, fps=c.fps, timeout_ms=5000, depth=False)
    cam.start()
    try:
        rgb = np.zeros((c.height, c.width, 3), np.uint8)
        d = np.zeros((c.height, c.width), np.float32)
        if not cam.read(rgb, d).ok:
            raise RuntimeError("D435 colour frame timeout")
        return rgb
    finally:
        cam.stop()


if __name__ == "__main__":
    sys.exit(main())

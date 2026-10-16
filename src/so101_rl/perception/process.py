"""The perception process (plan F9; ``plans/die-to-zone-plan.md`` D3): D435 -> die tracker -> object table.

``pixi run perception [--backend d435|sim] [--shm-name so101_objects] [--rate-hz 30] [--max-frames N]
[--duration-s S] [--summary PATH] [--allow-provisional] [--no-plane-check]``

One OS process, the single writer of the object table (``perception/object_table.py``); the policy process
(``So101GoalEnv(object_table=ObjectTableShm.attach(name))``) reads it. It never touches the serial bus or the servo
shm, and it imports nothing from ``rl`` / ``wm`` / ``env`` / ``safety``.

**Start-up (refuses, nothing is created):** ``calib/camera.yaml`` must have a calibrated ``T_base_color``
(``arm-shape`` / ``hand-eye``; a provisional pose only with ``--allow-provisional``) and a ``table_plane`` (unless
``--no-plane-check``) -> exit 2. A single-instance ``flock`` on the table's name (``servo.shm.InstanceLock``) -> exit 4
when another perception process holds it. The camera is opened by the calibration's serial; its device serial must
equal ``camera.yaml``'s and its colour intrinsics (resolution, fx, fy, cx, cy within 0.05 px) must be the calibrated
ones -> exit 3 otherwise. Only then is the shm segment created.

**Per frame** (colour + depth aligned to colour; the D435 paces the loop at its 30 fps, ``--rate-hz`` below the
camera rate skips frames): ``HsvDepthDieTracker.detect`` -> ``centroid_base(det, T_base_color)``, the die CENTROID
in the base frame (ARCHITECTURE §5 Perception) -> a workspace sanity check (the centroid inside the tip box grown by
5 cm, not below the table; outside = not visible, counted) -> one object-table write, slot 0, label ``die``:

- ``visible`` = the tracker's verdict for THIS frame (``detect`` not None and inside the workspace);
- ``xyz_base`` / ``t_mono_ns`` / ``confidence`` always describe the same measurement: the last visible detection.
  While the die is invisible they keep the last-seen centroid and its frame time (zeros before the first sighting)
  and the confidence is 0. The reader (``So101GoalEnv._update_die``) uses ``xyz_base`` only when ``visible`` is set
  and fresh and holds its own last-seen position otherwise (plan §10 2026-10-16 "Perception process (D3)");
- record ``t_mono_ns`` = this frame's time (``time.monotonic_ns`` when the frame arrived, the servo's clock), so
  the record shows the process is alive even while the die is hidden; the frame count is ``seq // 2`` (one
  seqlock write per processed frame);
- record ``fiducial_ok`` = the camera-moved verdict (below). The reader ANDs it into visibility.

A failed camera read writes nothing: the reader's freshness rule (detection age <= one policy period + 2 frames)
turns a stalled table into "not visible". No good frame for ``--camera-timeout-s`` (3 s) ends the process (exit 1).

**Camera-moved check (what can be checked before H8).** :class:`TablePlaneCheck` every ``PLANE_CHECK_EVERY`` frames
(1 s): the depth pixels (every 4th row / column) within 30 mm of ``camera.yaml``'s ``table_plane`` are refitted
(SVD), then those within 15, 8 and 8 mm of each refit; the last refit must agree with the calibrated plane to
<= 1 deg and <= 5 mm, from >= 500 points (sim: a 1.5 deg pitch or a 5 mm lift fails, the calibrated scene reads
0.23 deg / 0.0 mm). A failed check clears ``fiducial_ok`` at once; ``PLANE_TRIP_AFTER`` (3) consecutive failures latch it off until restart (a
moved camera invalidates the calibration). This sees the camera's height, roll and pitch only (3 of 6 DOF): a yaw
or an in-plane shift leaves the table plane unchanged. **TODO(H8): the ArUco fiducial check** (plan §5 camera-moved
gate: marker at (0.10, -0.16), reprojection error <= 2 px, ``perception/fiducial.py`` ``FiducialCheck``) needs
``calib/fiducial.yaml`` with the expected corners recorded at the H8 hand-eye calibration. It does not exist yet, so
the process does NOT run a fiducial check, and ``fiducial_ok`` is the table-plane verdict alone; the summary says
``"fiducial": "not implemented (needs H8 calib/fiducial.yaml)"``.

**Shutdown:** SIGINT / SIGTERM / SIGHUP set a flag; the loop ends after the current frame, the camera stops, the
owner closes and unlinks the shm segment, the lock is released and the summary JSON is written (default
``runs/perception/<ts>.json``): frames, detection rate, latency (frame arrival -> table write) and frame-interval
percentiles, the camera-check counts, the exit reason.

Exit codes: 0 normal stop, 1 camera died or an exception, 2 calibration refused, 3 camera refused, 4 already running.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from so101_rl.perception.camera import Camera
from so101_rl.perception.camera_calib import CAMERA_YAML, CameraCalib, CameraNotCalibrated, load_camera_calib
from so101_rl.perception.die_tracker import DEFAULT_HSV_YAML, HsvDepthDieTracker
from so101_rl.perception.object_table import ObjectTableShm, new_object_table, set_object

DEFAULT_SHM_NAME = "so101_objects"
DIE_LABEL = "die"  # = env.goal_env.DIE_LABEL (test-enforced; perception must not import env)
RUNS = Path("runs/perception")

INTRINSICS_TOL_PX = 0.05
CAMERA_TIMEOUT_S = 3.0
WORKSPACE_MARGIN_M = 0.05
BELOW_TABLE_M = 0.005

PLANE_STRIDE_PX = 4
PLANE_BANDS_M = (0.030, 0.015, 0.008, 0.008)  # refit schedule: first around the calibrated plane, then the refit
PLANE_MAX_ANGLE_DEG = 1.0
PLANE_MAX_DH_M = 0.005
PLANE_MIN_POINTS = 500
PLANE_CHECK_EVERY = 30
PLANE_TRIP_AFTER = 3

FIDUCIAL_STATUS = "not implemented (needs H8 calib/fiducial.yaml)"

EXIT_OK, EXIT_FAILED, EXIT_CALIB, EXIT_CAMERA, EXIT_RUNNING = 0, 1, 2, 3, 4


class CameraRefused(RuntimeError):
    """The opened camera is not the calibrated one (serial or intrinsics)."""


# ------------------------------------------------------------------------------------------------ camera check
@dataclass(frozen=True)
class PlaneReport:
    ok: bool
    angle_deg: float  # refit normal vs the calibrated normal
    dh_m: float  # refit camera height minus the calibrated height
    n_points: int


class TablePlaneCheck:
    """Refit the table near ``calib.table_plane`` on an aligned depth frame and compare (module docstring)."""

    def __init__(
        self,
        calib: CameraCalib,
        *,
        stride_px: int = PLANE_STRIDE_PX,
        bands_m: tuple[float, ...] = PLANE_BANDS_M,
        max_angle_deg: float = PLANE_MAX_ANGLE_DEG,
        max_dh_m: float = PLANE_MAX_DH_M,
        min_points: int = PLANE_MIN_POINTS,
    ) -> None:
        if calib.table_plane is None:
            raise CameraNotCalibrated("calib/camera.yaml has no table_plane: the camera-moved plane check needs one")
        self.n0 = np.asarray(calib.table_plane.normal_color, dtype=np.float64)
        self.h0 = float(calib.table_plane.height_m)
        self.bands_m, self.max_angle_deg, self.max_dh_m = tuple(map(float, bands_m)), float(max_angle_deg), float(max_dh_m)
        self.min_points = int(min_points)
        c = calib.color
        v, u = np.mgrid[0 : c.height : stride_px, 0 : c.width : stride_px]
        self._sl = (slice(0, c.height, stride_px), slice(0, c.width, stride_px))
        self._rays = np.stack([(u - c.cx) / c.fx, (v - c.cy) / c.fy, np.ones_like(u, dtype=np.float64)], axis=-1)

    def check(self, depth_m: np.ndarray) -> PlaneReport:
        z = np.asarray(depth_m[self._sl], dtype=np.float64)
        ok = np.isfinite(z) & (z > 0.1) & (z < 2.0)
        pts = self._rays[ok] * z[ok, None]
        n, h = self.n0, self.h0
        sel = np.zeros(0, bool)
        for band in self.bands_m:
            sel = np.abs(pts @ n + h) < band
            if int(sel.sum()) < self.min_points:
                return PlaneReport(False, math.inf, math.inf, int(sel.sum()))
            q = pts[sel]
            c = q.mean(axis=0)
            nn = np.linalg.svd(q - c, full_matrices=False)[2][2]
            hh = -float(nn @ c)
            if hh < 0:  # orient towards the camera (origin on the positive side), as camera.yaml's normal
                nn, hh = -nn, -hh
            n, h = nn, hh
        angle = math.degrees(math.acos(float(np.clip(n @ self.n0, -1.0, 1.0))))
        dh = h - self.h0
        good = angle <= self.max_angle_deg and abs(dh) <= self.max_dh_m
        return PlaneReport(good, angle, dh, int(sel.sum()))


@dataclass
class CameraMovedGate:
    """``ok`` = the last plane check passed and the gate has not latched; ``trip_after`` consecutive failures latch
    it (until the process restarts). ``plane is None`` (``--no-plane-check``) = always ok, recorded as disabled."""

    plane: TablePlaneCheck | None
    trip_after: int = PLANE_TRIP_AFTER
    checks: int = 0
    fails: int = 0
    consecutive: int = 0
    tripped: bool = False
    last: PlaneReport | None = None
    worst_angle_deg: float = 0.0
    worst_dh_m: float = 0.0

    @property
    def ok(self) -> bool:
        if self.plane is None:
            return True
        return not self.tripped and self.last is not None and self.last.ok

    def update(self, depth_m: np.ndarray) -> bool:
        if self.plane is None:
            return True
        r = self.plane.check(depth_m)
        self.last = r
        self.checks += 1
        if r.ok:
            self.consecutive = 0
            self.worst_angle_deg = max(self.worst_angle_deg, r.angle_deg)
            self.worst_dh_m = max(self.worst_dh_m, abs(r.dh_m))
        else:
            self.fails += 1
            self.consecutive += 1
            if self.consecutive >= self.trip_after and not self.tripped:
                self.tripped = True
                print(
                    f"perception: CAMERA MOVED (table plane off by {r.angle_deg:.2f} deg / {r.dh_m * 1e3:.1f} mm, "
                    f"{r.n_points} points, {self.consecutive} checks in a row): fiducial_ok latched off; "
                    "recalibrate (camera-calibrate) and restart",
                    file=sys.stderr,
                )
        return self.ok

    def report(self) -> dict[str, Any]:
        last = self.last
        return {
            "table_plane": {
                "enabled": self.plane is not None,
                "checks": self.checks,
                "fails": self.fails,
                "tripped": self.tripped,
                "last_ok": None if last is None else last.ok,
                "last_angle_deg": None if last is None or not math.isfinite(last.angle_deg) else last.angle_deg,
                "last_dh_mm": None if last is None or not math.isfinite(last.dh_m) else last.dh_m * 1e3,
                "last_points": None if last is None else last.n_points,
                "worst_passing_angle_deg": self.worst_angle_deg,
                "worst_passing_dh_mm": self.worst_dh_m * 1e3,
                "limits": {"angle_deg": PLANE_MAX_ANGLE_DEG, "dh_mm": PLANE_MAX_DH_M * 1e3, "trip_after": self.trip_after},
            },
            "fiducial": FIDUCIAL_STATUS,
        }


# ------------------------------------------------------------------------------------------------ camera open
def open_checked(camera: Camera, calib: CameraCalib) -> None:
    """Start ``camera`` and refuse (:class:`CameraRefused`, camera stopped) unless it is the calibrated D435."""
    intr = camera.start()
    try:
        got = getattr(camera, "device_serial", None) or getattr(camera, "serial", None)
        if got is None or str(got) != calib.serial:
            raise CameraRefused(f"camera serial {got!r} != calib/camera.yaml serial {calib.serial!r}")
        c = calib.color
        if (intr.width, intr.height) != (c.width, c.height):
            raise CameraRefused(f"camera stream {intr.width}x{intr.height} != calibrated {c.width}x{c.height}")
        diff = max(abs(intr.fx - c.fx), abs(intr.fy - c.fy), abs(intr.cx - c.cx), abs(intr.cy - c.cy))
        if diff > INTRINSICS_TOL_PX:
            raise CameraRefused(
                f"camera colour intrinsics differ from calib/camera.yaml by {diff:.3f} px "
                f"(got fx {intr.fx:.3f} fy {intr.fy:.3f} cx {intr.cx:.3f} cy {intr.cy:.3f}): rerun camera-intrinsics"
            )
    except BaseException:
        camera.stop()
        raise


# ------------------------------------------------------------------------------------------------ the loop
def _pct(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    a = np.asarray(xs, dtype=np.float64)
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {"n": len(xs), "mean": float(a.mean()), "p50": float(p50), "p95": float(p95), "p99": float(p99), "max": float(a.max())}


@dataclass
class PerceptionStats:
    read: int = 0
    ok: int = 0
    failed: int = 0
    skipped_rate: int = 0
    processed: int = 0
    detections: int = 0
    out_of_workspace: int = 0
    latency_ms: list[float] = field(default_factory=list)
    interval_ms: list[float] = field(default_factory=list)


class PerceptionLoop:
    """Camera (already opened by :func:`open_checked`) -> tracker -> ``table`` (the owner side). One :meth:`step` =
    one camera read; :meth:`run` until ``stop()`` / ``max_frames`` processed / ``duration_s``."""

    def __init__(
        self,
        camera: Camera,
        calib: CameraCalib,
        table: ObjectTableShm,
        *,
        tracker: HsvDepthDieTracker | None = None,
        gate: CameraMovedGate | None = None,
        rate_hz: float | None = None,
        workspace: tuple[np.ndarray, np.ndarray] | None = None,
        plane_every: int = PLANE_CHECK_EVERY,
        camera_timeout_s: float = CAMERA_TIMEOUT_S,
    ) -> None:
        self.camera, self.calib, self.table = camera, calib, table
        self.T = np.asarray(calib.T_base_color, dtype=np.float64)
        self.tracker = tracker if tracker is not None else HsvDepthDieTracker.from_camera_calib(calib)
        self.gate = gate if gate is not None else CameraMovedGate(None)
        fps = float(calib.color.fps)
        self.rate_hz = fps if rate_hz is None else float(rate_hz)
        if not 0 < self.rate_hz:
            raise ValueError(f"rate_hz must be > 0, got {self.rate_hz}")
        # accept a frame when at least the period minus half a camera frame has passed (no drift to 15 Hz at 30)
        self._min_dt_ns = max(0, round(1e9 / self.rate_hz - 0.5e9 / fps)) if self.rate_hz < fps else 0
        self.workspace = workspace if workspace is not None else default_workspace()
        self.plane_every = int(plane_every)
        self.camera_timeout_ns = round(camera_timeout_s * 1e9)
        c = calib.color
        self._rgb = np.zeros((c.height, c.width, 3), np.uint8)
        self._depth = np.zeros((c.height, c.width), np.float32)
        self._rec = new_object_table()
        self._xyz = np.zeros(3)
        self._t_seen = 0
        self._conf = 0.0
        self._last_t: int | None = None
        self._last_ok_ns = time.monotonic_ns()
        self.stats = PerceptionStats()
        self.last_visible = False

    def _in_workspace(self, p: np.ndarray) -> bool:
        lo, hi = self.workspace
        return bool(np.all(np.isfinite(p)) and np.all(p >= lo) and np.all(p <= hi))

    def step(self) -> bool:
        """One camera read (and, unless rate-skipped or failed, one table write). False = the camera timed out."""
        s = self.stats
        fr = self.camera.read(self._rgb, self._depth)
        s.read += 1
        if not fr.ok:
            s.failed += 1
            return time.monotonic_ns() - self._last_ok_ns <= self.camera_timeout_ns
        s.ok += 1
        self._last_ok_ns = time.monotonic_ns()
        if self._last_t is not None and fr.t_mono_ns - self._last_t < self._min_dt_ns:
            s.skipped_rate += 1
            return True
        if self._last_t is not None:
            s.interval_ms.append((fr.t_mono_ns - self._last_t) / 1e6)
        self._last_t = fr.t_mono_ns
        if s.processed % self.plane_every == 0:
            self.gate.update(self._depth)
        s.processed += 1

        visible = False
        det = self.tracker.detect(self._rgb, self._depth)
        if det is not None:
            p = self.tracker.centroid_base(det, self.T)
            if self._in_workspace(p):
                visible = True
                s.detections += 1
                self._xyz, self._t_seen, self._conf = p, fr.t_mono_ns, float(det.confidence)
            else:
                s.out_of_workspace += 1
        rec = self._rec
        set_object(rec, 0, DIE_LABEL, self._xyz, self._conf if visible else 0.0, visible, self._t_seen)
        rec["t_mono_ns"] = fr.t_mono_ns
        rec["fiducial_ok"] = 1 if self.gate.ok else 0
        self.table.write(rec)
        s.latency_ms.append((time.monotonic_ns() - fr.t_mono_ns) / 1e6)
        self.last_visible = visible
        return True

    def run(
        self, stop: Callable[[], bool] = lambda: False, *, max_frames: int | None = None, duration_s: float | None = None
    ) -> str:
        """The exit reason: ``stop`` / ``max_frames`` / ``duration`` / ``camera_timeout``."""
        t_end = None if duration_s is None else time.monotonic() + duration_s
        while True:
            if stop():
                return "stop"
            if max_frames is not None and self.stats.processed >= max_frames:
                return "max_frames"
            if t_end is not None and time.monotonic() >= t_end:
                return "duration"
            if not self.step():
                return "camera_timeout"

    def summary(self) -> dict[str, Any]:
        s = self.stats
        return {
            "frames": {"read": s.read, "ok": s.ok, "failed": s.failed, "skipped_rate": s.skipped_rate, "processed": s.processed},
            "detections": s.detections,
            "detection_rate": s.detections / s.processed if s.processed else None,
            "rejected_out_of_workspace": s.out_of_workspace,
            "latency_ms": _pct(s.latency_ms),
            "frame_interval_ms": _pct(s.interval_ms),
            "rate_hz": self.rate_hz,
            "table_seq": self.table.last_seq,
            "camera_check": self.gate.report(),
        }


def default_workspace() -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) base-frame bounds of a plausible die centroid: the tip box grown by 5 cm, not below the table."""
    from so101_rl.config.loader import load_geometry

    table_z, box = load_geometry()
    m = WORKSPACE_MARGIN_M
    lo = np.array([box.x[0] - m, box.y[0] - m, table_z - BELOW_TABLE_M])
    hi = np.array([box.x[1] + m, box.y[1] + m, box.z[1] + m])
    return lo, hi


# ------------------------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m so101_rl.perception.process", description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=("d435", "sim"), default="d435", help="sim: MuJoCo renders (no hardware)")
    ap.add_argument("--shm-name", default=DEFAULT_SHM_NAME)
    ap.add_argument("--camera-yaml", type=Path, default=None, help=f"default {CAMERA_YAML}")
    ap.add_argument("--hsv-yaml", type=Path, default=DEFAULT_HSV_YAML)
    ap.add_argument("--rate-hz", type=float, default=None, help="default: the calibrated camera fps (30)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--summary", type=Path, default=None, help="default runs/perception/<ts>.json")
    ap.add_argument("--allow-provisional", action="store_true", help="accept a provisional camera pose")
    ap.add_argument("--no-plane-check", action="store_true", help="disable the table-plane camera-moved check")
    ap.add_argument("--camera-timeout-s", type=float, default=CAMERA_TIMEOUT_S)
    ap.add_argument(
        "--sim-die", type=float, nargs="+", default=None, metavar="X Y [YAW]",
        help="--backend sim: die resting on the table at (x, y) with yaw (rad); default hidden",
    )  # fmt: skip
    return ap


def _write_summary(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=float) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None, *, camera: Camera | None = None) -> int:
    a = build_parser().parse_args(argv)
    started = _dt.datetime.now().astimezone()
    summary_path = a.summary if a.summary is not None else RUNS / f"{started:%Y%m%d-%H%M%S}.json"
    doc: dict[str, Any] = {"started": started.isoformat(timespec="seconds"), "shm_name": a.shm_name, "backend": a.backend}

    def finish(code: int, reason: str) -> int:
        doc["exit_code"], doc["exit_reason"] = code, reason
        doc["duration_s"] = (_dt.datetime.now().astimezone() - started).total_seconds()
        _write_summary(summary_path, doc)
        print(f"perception: {reason} (exit {code}); summary {summary_path}", file=sys.stderr)
        return code

    try:
        calib = load_camera_calib(a.camera_yaml)
        pose = calib.require_pose(calibrated=not a.allow_provisional)
        gate = CameraMovedGate(None if a.no_plane_check else TablePlaneCheck(calib))
        tracker = HsvDepthDieTracker.from_camera_calib(calib, a.hsv_yaml)
    except (CameraNotCalibrated, FileNotFoundError, ValueError, TypeError) as e:
        return finish(EXIT_CALIB, f"calibration refused: {e}")
    doc["calib"] = {"serial": calib.serial, **calib.pose_meta(), "residual_mm": pose.residual_mm}
    if not pose.calibrated:
        print(f"perception: WARNING provisional camera pose ({pose.method}) accepted by --allow-provisional", file=sys.stderr)

    from so101_rl.servo.shm import InstanceLock, ServoAlreadyRunning

    try:
        lock = InstanceLock(a.shm_name, what=f"object table {a.shm_name!r}")
    except ServoAlreadyRunning as e:
        return finish(EXIT_RUNNING, f"already running: {e}")

    stop_signal: list[str] = []

    def on_signal(signum: int, _frame: object) -> None:
        stop_signal.append(signal.Signals(signum).name)

    old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    table: ObjectTableShm | None = None
    cam: Camera | None = None
    code, reason = EXIT_FAILED, "not started"
    try:
        code, reason, cam = _open_camera(a, calib, camera)
        if cam is not None:
            doc["camera"] = {"type": type(cam).__name__, "device_serial": getattr(cam, "device_serial", None)}
            table = ObjectTableShm.create(a.shm_name)
            loop = PerceptionLoop(cam, calib, table, tracker=tracker, gate=gate, rate_hz=a.rate_hz,
                                  camera_timeout_s=a.camera_timeout_s)  # fmt: skip
            print(f"perception: writing object table {a.shm_name!r} at {loop.rate_hz:g} Hz", file=sys.stderr)
            reason = loop.run(lambda: bool(stop_signal), max_frames=a.max_frames, duration_s=a.duration_s)
            doc.update(loop.summary())
            code = EXIT_FAILED if reason == "camera_timeout" else EXIT_OK
            if reason == "stop":
                reason = f"signal {stop_signal[0]}"
    except Exception as e:  # noqa: BLE001 - recorded in the summary, the exit code says it failed
        code, reason = EXIT_FAILED, f"exception: {type(e).__name__}: {e}"
    finally:
        if cam is not None:
            try:
                cam.stop()
            except Exception as e:  # noqa: BLE001 - shutdown continues
                print(f"perception: camera stop failed: {e}", file=sys.stderr)
        if table is not None:
            table.close()  # owner: unlinks the segment
        lock.release()
        for s, h in old.items():
            signal.signal(s, h)
    doc["table_created_and_unlinked"] = table is not None
    return finish(code, reason)


def _open_camera(a: argparse.Namespace, calib: CameraCalib, camera: Camera | None) -> tuple[int, str, Camera | None]:
    """(code, reason, the opened camera or None when refused)."""
    if camera is not None:
        cam = camera
    elif a.backend == "sim":
        from so101_rl.perception.sim_camera import SimRenderCamera
        from so101_rl.sim.model import DIE_EDGE_M, TABLE_Z_M

        die = None
        if a.sim_die is not None:
            if len(a.sim_die) not in (2, 3):
                return EXIT_CALIB, "--sim-die takes X Y [YAW]", None
            x, y, *yaw = a.sim_die
            die = (x, y, TABLE_Z_M + DIE_EDGE_M / 2, yaw[0] if yaw else 0.0)
        cam = SimRenderCamera(calib, die_xyz_yaw=die, realtime=True)
    else:
        from so101_rl.perception.camera import D435Camera

        c = calib.color
        cam = D435Camera(serial=calib.serial, width=c.width, height=c.height, fps=c.fps)
    try:
        open_checked(cam, calib)
    except (CameraRefused, RuntimeError, ImportError) as e:
        return EXIT_CAMERA, f"camera refused: {e}", None  # open_checked stopped it
    return EXIT_OK, "", cam

if __name__ == "__main__":
    sys.exit(main())

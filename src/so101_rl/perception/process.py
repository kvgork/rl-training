"""The perception process (plan F9; ``plans/die-to-zone-plan.md`` D3): D435 -> die tracker -> object table.

``pixi run perception [--backend d435|sim] [--shm-name so101_objects] [--rate-hz 30] [--max-frames N]
[--duration-s S] [--summary PATH] [--allow-provisional] [--no-plane-check] [--fiducial-yaml calib/fiducial.yaml]
[--no-fiducial-check] [--plane-defers-to-fiducial] [--servo-shm so101_servo] [--no-arm-mask]``

One OS process, the single writer of the object table (``perception/object_table.py``); the policy process
(``So101GoalEnv(object_table=ObjectTableShm.attach(name))``) reads it. It never touches the serial bus; the servo shm
it only READS (the arm mask's joint ticks, ``perception/arm_mask.py`` ``ServoStateReader``: attach + ``read_state``,
never a write). It imports nothing from ``rl`` / ``wm`` / ``env`` / ``safety``.

**Start-up (refuses, nothing is created):** ``calib/camera.yaml`` must have a calibrated ``T_base_color``
(``arm-shape`` / ``hand-eye``; a provisional pose only with ``--allow-provisional``) and a ``table_plane`` (unless
``--no-plane-check``) -> exit 2. A single-instance ``flock`` on the table's name (``servo.shm.InstanceLock``) -> exit 4
when another perception process holds it. The camera is opened by the calibration's serial; its device serial must
equal ``camera.yaml``'s and its colour intrinsics (resolution, fx, fy, cx, cy within 0.05 px) must be the calibrated
ones -> exit 3 otherwise. Only then is the shm segment created.

**Per frame** (colour + depth aligned to colour; the D435 paces the loop at its 30 fps, ``--rate-hz`` below the
camera rate skips frames): the arm mask (below) -> ``HsvDepthDieTracker.detect(rgb, depth, accept=on the table,
exclude=arm mask)`` -> ``centroid_base(det, T_base_color)``, the die CENTROID in the base frame (ARCHITECTURE §5
Perception); candidate blobs are tried largest first, a blob more than half on the arm or whose centroid lies off the
table workspace (the tip box grown by 5 cm, not below the table) is skipped and counted -> one object-table write,
slot 0, label ``die``:

- ``visible`` = the tracker's verdict for THIS frame (``detect`` not None and inside the workspace);
- ``xyz_base`` / ``t_mono_ns`` / ``confidence`` always describe the same measurement: the last visible detection.
  While the die is invisible they keep the last-seen centroid and its frame time (zeros before the first sighting)
  and the confidence is 0. The reader (``So101GoalEnv._update_die``) uses ``xyz_base`` only when ``visible`` is set
  and fresh and holds its own last-seen position otherwise (plan §10 2026-10-16 "Perception process (D3)");
- record ``t_mono_ns`` = this frame's time (``time.monotonic_ns`` when the frame arrived, the servo's clock), so
  the record shows the process is alive even while the die is hidden; the frame count is ``seq // 2`` (one
  seqlock write per processed frame);
- record ``fiducial_ok`` = the camera-moved verdict (below). The reader ANDs it into visibility.

**Arm mask** (plan §10 2026-10-17 "Perception: arm mask"; ``perception/arm_mask.py``): the servo's present ticks
(``--servo-shm``, default ``so101.yaml`` ``shm_name``) -> FK -> the arm's surface points projected into the colour
frame, dilated 6 px, depth-gated (a surface 10 mm or more in front of the arm is not the arm) and open between the
fingertips (a die held in the jaws stays visible). With no servo segment, a state older than 100 ms or a failed bus
read the frame runs without the mask (as before) and is counted; the summary's ``arm_mask`` gives the active fraction,
the reasons, the blobs rejected on the arm and the mask's cost. ``--no-arm-mask`` disables it (so does a motor
calibration that cannot be loaded; the summary records why).

A failed camera read writes nothing: the reader's freshness rule (detection age <= one policy period + 2 frames)
turns a stalled table into "not visible". No good frame for ``--camera-timeout-s`` (3 s) ends the process (exit 1).

**Camera-moved check** (:class:`CameraMovedGate`, once a second = every ``PLANE_CHECK_EVERY`` frames), two halves:

- :class:`TablePlaneCheck`: the depth pixels (every 4th row / column) within 30 mm of ``camera.yaml``'s
  ``table_plane`` are refitted (SVD), then those within 15, 8 and 8 mm of each refit; the last refit must agree with
  the calibrated plane to <= 1 deg and <= 5 mm, from >= 500 points (sim: a 1.5 deg pitch or a 5 mm lift fails, the
  calibrated scene reads 0.23 deg / 0.0 mm). It sees the camera's height, roll and pitch only (3 of 6 DOF). When
  the arm mask is active for the frame, the depth pixels inside it are left out of the fit (plan §10 2026-10-23: the
  arm low over the table corrupted the refit on the real rig, a false 2.25 deg / 12.6 mm from 1741 points latched
  ``fiducial_ok`` off); the summary counts the masked checks. ``--plane-defers-to-fiducial`` (off by default: a 5 mm
  lift moves the marker's corners < 1 px, which only the plane sees) skips the plane check whenever the fiducial
  check of the same update found the marker and passed (counted as ``deferred``; a latched plane stays latched).
- the ArUco fiducial (H8; ``perception/fiducial.py`` ``FiducialCheck``, plan §5 camera-moved gate, plan §10
  2026-10-17 "H8 software"), when ``--fiducial-yaml`` (default ``calib/fiducial.yaml``, written by
  ``pixi run fiducial-capture``) exists: marker found and mean corner error <= ``tol_px`` (2) = pass; found and off =
  failure; NOT found (occluded by the arm or a hand) = no evidence, neither a pass nor a failure (counted as
  ``not_found``). It sees the remaining yaw and in-plane shift. A fiducial recorded with another camera serial is
  refused (exit 2); ``--no-fiducial-check`` disables it.

``fiducial_ok`` = plane check AND fiducial check: a failed check of either clears it at once; ``PLANE_TRIP_AFTER`` (3)
consecutive failures of either latch it off until restart (a moved camera invalidates the calibration; fiducial
checks without evidence neither reset nor extend the run). Without ``calib/fiducial.yaml`` the plane verdict alone
is ``fiducial_ok`` and the summary's ``camera_check.fiducial`` says why (``enabled: false``, ``reason``); with it, the
counts ``checks`` / ``passes`` / ``fails`` / ``not_found``, ``tripped``, the last and worst passing error.

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

from so101_rl.perception.arm_mask import ARM_MAJORITY, ArmMaskSource
from so101_rl.perception.camera import Camera
from so101_rl.perception.camera_calib import CAMERA_YAML, CameraCalib, CameraNotCalibrated, load_camera_calib
from so101_rl.perception.die_tracker import (
    DEFAULT_HSV_YAML,
    TABLE_WORKSPACE_MARGIN_M,
    Detection,
    HsvDepthDieTracker,
)
from so101_rl.perception.fiducial import FIDUCIAL_YAML, FiducialCheck, FiducialReport
from so101_rl.perception.object_table import ObjectTableShm, new_object_table, set_object

DEFAULT_SHM_NAME = "so101_objects"
DIE_LABEL = "die"  # = env.goal_env.DIE_LABEL (test-enforced; perception must not import env)
RUNS = Path("runs/perception")

INTRINSICS_TOL_PX = 0.05
CAMERA_TIMEOUT_S = 3.0
WORKSPACE_MARGIN_M = TABLE_WORKSPACE_MARGIN_M  # the table workspace (die_tracker.table_workspace)

PLANE_STRIDE_PX = 4
PLANE_BANDS_M = (0.030, 0.015, 0.008, 0.008)  # refit schedule: first around the calibrated plane, then the refit
PLANE_MAX_ANGLE_DEG = 1.0
PLANE_MAX_DH_M = 0.005
PLANE_MIN_POINTS = 500
PLANE_CHECK_EVERY = 30
PLANE_TRIP_AFTER = 3

FIDUCIAL_STATUS = "no calib/fiducial.yaml (record it with `pixi run fiducial-capture`, H8)"

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

    def check(self, depth_m: np.ndarray, exclude: np.ndarray | None = None) -> PlaneReport:
        """``exclude`` bool [h, w] (the arm mask): those pixels are left out of the fit."""
        z = np.asarray(depth_m[self._sl], dtype=np.float64)
        ok = np.isfinite(z) & (z > 0.1) & (z < 2.0)
        if exclude is not None:
            ok &= ~np.asarray(exclude[self._sl], dtype=bool)
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
class FiducialGateState:
    """The fiducial half of :class:`CameraMovedGate`: found and within ``tol_px`` = pass, found and off = failure,
    not found (occluded by the arm or a hand) = no evidence (neither; counted)."""

    check: FiducialCheck
    checks: int = 0
    passes: int = 0
    fails: int = 0
    not_found: int = 0
    consecutive: int = 0
    tripped: bool = False
    last_err_px: float | None = None
    last_found: bool | None = None
    last_evidence_ok: bool | None = None  # the verdict of the last check that found the marker
    worst_passing_err_px: float = 0.0

    def update(self, rgb: np.ndarray) -> FiducialReport:
        r = self.check.check(rgb)
        self.checks += 1
        self.last_found = r.found
        if not r.found:
            self.not_found += 1
            return r
        self.last_err_px = r.reproj_err_px
        self.last_evidence_ok = r.ok
        if r.ok:
            self.passes += 1
            self.consecutive = 0
            self.worst_passing_err_px = max(self.worst_passing_err_px, r.reproj_err_px)
        else:
            self.fails += 1
            self.consecutive += 1
        return r

    @property
    def ok(self) -> bool:
        return not self.tripped and self.last_evidence_ok is not False

    def report(self, trip_after: int) -> dict[str, Any]:
        return {
            "enabled": True,
            "marker_id": self.check.marker_id,
            "tol_px": self.check.tol_px,
            "checks": self.checks,
            "passes": self.passes,
            "fails": self.fails,
            "not_found": self.not_found,
            "tripped": self.tripped,
            "last_found": self.last_found,
            "last_err_px": self.last_err_px,
            "worst_passing_err_px": self.worst_passing_err_px,
            "trip_after": trip_after,
        }


@dataclass
class CameraMovedGate:
    """``ok`` = the plane check AND the fiducial check, neither latched. Plane: the last check passed. Fiducial
    (when ``calib/fiducial.yaml`` exists): the last check that FOUND the marker passed (a check that did not find it
    is no evidence: it changes nothing; before any evidence the fiducial does not block). ``trip_after`` consecutive
    failures of either check latch the gate (until the process restarts); for the fiducial, checks without evidence
    neither reset nor extend the run of failures. ``plane is None`` (``--no-plane-check``) and ``fiducial is None``
    (no ``calib/fiducial.yaml`` / ``--no-fiducial-check``) = that half always ok, recorded as disabled. ``exclude``
    (the arm mask of the frame) is left out of the plane fit. ``plane_defers_to_fiducial``: a fiducial check of the
    same update that found the marker and passed stands in for the plane check (module docstring)."""

    plane: TablePlaneCheck | None
    fiducial: FiducialCheck | None = None
    trip_after: int = PLANE_TRIP_AFTER
    fiducial_disabled_reason: str = FIDUCIAL_STATUS
    plane_defers_to_fiducial: bool = False
    checks: int = 0
    masked_checks: int = 0
    deferred: int = 0
    fails: int = 0
    consecutive: int = 0
    tripped: bool = False
    last: PlaneReport | None = None
    last_deferred: bool = False
    worst_angle_deg: float = 0.0
    worst_dh_m: float = 0.0
    fid: FiducialGateState | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.fid = None if self.fiducial is None else FiducialGateState(self.fiducial)

    @property
    def plane_ok(self) -> bool:
        if self.plane is None:
            return True
        return not self.tripped and (self.last_deferred or (self.last is not None and self.last.ok))

    @property
    def ok(self) -> bool:
        return self.plane_ok and (self.fid is None or self.fid.ok)

    def update(self, depth_m: np.ndarray, rgb: np.ndarray | None = None, exclude: np.ndarray | None = None) -> bool:
        fid_pass = False
        if self.fid is not None and rgb is not None:
            f = self.fid
            r = f.update(rgb)
            fid_pass = bool(r.found and r.ok and not f.tripped)
            if r.found and not r.ok and f.consecutive >= self.trip_after and not f.tripped:
                f.tripped = True
                print(
                    f"perception: CAMERA MOVED (fiducial corners off by {r.reproj_err_px:.2f} px > {f.check.tol_px} px, "
                    f"{f.consecutive} checks in a row): fiducial_ok latched off; recalibrate (H8) and restart",
                    file=sys.stderr,
                )
        if self.plane is not None:
            if self.plane_defers_to_fiducial and fid_pass:
                self.deferred += 1
                self.last_deferred = True
                self.consecutive = 0
            else:
                self.last_deferred = False
                self._update_plane(depth_m, exclude)
        return self.ok

    def _update_plane(self, depth_m: np.ndarray, exclude: np.ndarray | None = None) -> None:
        assert self.plane is not None
        r = self.plane.check(depth_m, exclude)
        self.last = r
        self.checks += 1
        self.masked_checks += int(exclude is not None)
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

    def report(self) -> dict[str, Any]:
        last = self.last
        return {
            "table_plane": {
                "enabled": self.plane is not None,
                "checks": self.checks,
                "fails": self.fails,
                "tripped": self.tripped,
                "arm_masked_checks": self.masked_checks,
                "defers_to_fiducial": self.plane_defers_to_fiducial,
                "deferred": self.deferred,
                "last_ok": None if last is None else last.ok,
                "last_angle_deg": None if last is None or not math.isfinite(last.angle_deg) else last.angle_deg,
                "last_dh_mm": None if last is None or not math.isfinite(last.dh_m) else last.dh_m * 1e3,
                "last_points": None if last is None else last.n_points,
                "worst_passing_angle_deg": self.worst_angle_deg,
                "worst_passing_dh_mm": self.worst_dh_m * 1e3,
                "limits": {"angle_deg": PLANE_MAX_ANGLE_DEG, "dh_mm": PLANE_MAX_DH_M * 1e3, "trip_after": self.trip_after},
            },
            "fiducial": (
                {"enabled": False, "reason": self.fiducial_disabled_reason}
                if self.fid is None
                else self.fid.report(self.trip_after)
            ),
            "ok": self.ok,
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
    on_arm: int = 0
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
        arm: ArmMaskSource | None = None,
        arm_disabled_reason: str = "not configured",
    ) -> None:
        self.camera, self.calib, self.table = camera, calib, table
        self.T = np.asarray(calib.T_base_color, dtype=np.float64)
        self.tracker = tracker if tracker is not None else HsvDepthDieTracker.from_camera_calib(calib)
        self.gate = gate if gate is not None else CameraMovedGate(None)
        self.arm = arm
        self.arm_disabled_reason = arm_disabled_reason
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

    def _on_table(self, det: Detection) -> bool:
        """``detect``'s candidate filter: the blob's die centroid lies on the table workspace."""
        return self._in_workspace(self.tracker.centroid_base(det, self.T))

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
        exclude = self.arm.mask(self._depth) if self.arm is not None else None
        if s.processed % self.plane_every == 0:
            self.gate.update(self._depth, self._rgb, exclude)  # the arm's pixels stay out of the plane fit
        s.processed += 1

        visible = False
        tr = self.tracker
        before, before_arm = tr.rejected, tr.rejected_excluded
        det = tr.detect(self._rgb, self._depth, accept=self._on_table, exclude=exclude)
        s.out_of_workspace += tr.rejected - before  # red / orange blobs off the table, skipped
        s.on_arm += tr.rejected_excluded - before_arm  # red / orange blobs on the arm itself, skipped
        if det is not None:
            visible = True
            s.detections += 1
            self._xyz, self._t_seen, self._conf = self.tracker.centroid_base(det, self.T), fr.t_mono_ns, float(det.confidence)
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
            "rejected_on_arm": s.on_arm,
            "arm_mask": self.arm_report(),
            "latency_ms": _pct(s.latency_ms),
            "frame_interval_ms": _pct(s.interval_ms),
            "rate_hz": self.rate_hz,
            "table_seq": self.table.last_seq,
            "camera_check": self.gate.report(),
        }

    def arm_report(self) -> dict[str, Any]:
        """The arm mask's share of the processed frames, why it was off, and its cost."""
        a = self.arm
        if a is None:
            return {"enabled": False, "reason": self.arm_disabled_reason, "active_fraction": 0.0}
        m = a.masker
        n = a.active + sum(a.inactive.values())
        age = getattr(a.joints, "max_age_ns", None)
        return {
            "enabled": True,
            "servo_shm": getattr(a.joints, "name", None),
            "active_frames": a.active,
            "inactive_frames": dict(a.inactive),
            "active_fraction": a.active / n if n else None,
            "mask_ms": _pct(a.ms),
            "dilate_px": m.dilate_px,
            "depth_tol_mm": m.depth_tol_m * 1e3,
            "grasp_margin_mm": m.grasp_margin_m * 1e3,
            "majority": ARM_MAJORITY,
            "max_state_age_ms": None if age is None else age / 1e6,
        }


def default_workspace() -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) base-frame bounds of a plausible die centroid: ``die_tracker.table_workspace`` (the tip box grown
    by 5 cm, not below the table)."""
    from so101_rl.perception.die_tracker import table_workspace

    return table_workspace()


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
    ap.add_argument("--fiducial-yaml", type=Path, default=FIDUCIAL_YAML,
                    help="the ArUco camera-moved fiducial (checked when the file exists; H8 fiducial-capture)")
    ap.add_argument("--no-fiducial-check", action="store_true", help="disable the fiducial camera-moved check")
    ap.add_argument("--plane-defers-to-fiducial", action="store_true", help="skip the table-plane check while the "
                    "fiducial check of the same update finds the marker and passes")  # fmt: skip
    ap.add_argument("--camera-timeout-s", type=float, default=CAMERA_TIMEOUT_S)
    ap.add_argument("--servo-shm", default=None, help="servo state shm to read the arm mask's joint ticks from "
                    "(read-only; default so101.yaml shm_name)")  # fmt: skip
    ap.add_argument("--no-arm-mask", action="store_true", help="do not mask the arm's own pixels")
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
    # one OpenCV thread: its default pool spun ~14 of 16 cores and starved the servo loop (no SCHED_FIFO on the
    # host; the servo state went stale on the real arm, 2026-10-23). The pixi task also caps the BLAS / OpenMP pools.
    import cv2

    cv2.setNumThreads(1)
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
        fiducial, fid_reason = load_fiducial(a, calib)
        gate = CameraMovedGate(None if a.no_plane_check else TablePlaneCheck(calib), fiducial,
                               fiducial_disabled_reason=fid_reason,
                               plane_defers_to_fiducial=a.plane_defers_to_fiducial)  # fmt: skip
        tracker = HsvDepthDieTracker.from_camera_calib(calib, a.hsv_yaml)
    except (CameraNotCalibrated, FileNotFoundError, ValueError, TypeError, KeyError, AttributeError) as e:
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
    arm: ArmMaskSource | None = None
    code, reason = EXIT_FAILED, "not started"
    try:
        code, reason, cam = _open_camera(a, calib, camera)
        if cam is not None:
            doc["camera"] = {"type": type(cam).__name__, "device_serial": getattr(cam, "device_serial", None)}
            table = ObjectTableShm.create(a.shm_name)
            arm, why = _arm_source(a, calib)
            doc["arm_mask_source"] = why
            loop = PerceptionLoop(cam, calib, table, tracker=tracker, gate=gate, rate_hz=a.rate_hz,
                                  camera_timeout_s=a.camera_timeout_s, arm=arm, arm_disabled_reason=why)  # fmt: skip
            print(f"perception: writing object table {a.shm_name!r} at {loop.rate_hz:g} Hz", file=sys.stderr)
            reason = loop.run(lambda: bool(stop_signal), max_frames=a.max_frames, duration_s=a.duration_s)
            doc.update(loop.summary())
            code = EXIT_FAILED if reason == "camera_timeout" else EXIT_OK
            if reason == "stop":
                reason = f"signal {stop_signal[0]}"
    except Exception as e:  # noqa: BLE001 - recorded in the summary, the exit code says it failed
        code, reason = EXIT_FAILED, f"exception: {type(e).__name__}: {e}"
    finally:
        if arm is not None:
            arm.close()
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


def load_fiducial(a: argparse.Namespace, calib: CameraCalib) -> tuple[FiducialCheck | None, str]:
    """(the fiducial check, "") when ``--fiducial-yaml`` exists and is enabled, else (None, why not). A fiducial
    recorded with another camera is refused (ValueError: exit 2)."""
    if a.no_fiducial_check:
        return None, "disabled (--no-fiducial-check)"
    path = Path(a.fiducial_yaml)
    if not path.is_file():
        return None, f"no {path} (record it with `pixi run fiducial-capture`, H8)"
    check = FiducialCheck.from_yaml(path)
    if check.camera_serial is not None and check.camera_serial != calib.serial:
        raise ValueError(f"{path} was recorded with camera {check.camera_serial}, calib/camera.yaml is {calib.serial}")
    return check, ""


def _arm_source(a: argparse.Namespace, calib: CameraCalib) -> tuple[ArmMaskSource | None, str]:
    """(the arm mask on the servo shm, a description) or (None, why not). Never refuses the process: without the
    mask the tracker works as before."""
    if a.no_arm_mask:
        return None, "disabled (--no-arm-mask)"
    from so101_rl.config import load_calibration, load_config
    from so101_rl.perception.arm_mask import ArmMasker, ServoStateReader

    try:
        cfg = load_config()
        calibration = load_calibration(cfg.calibration_path)
        masker = ArmMasker(calib)
    except (OSError, ValueError, KeyError) as e:
        print(f"perception: WARNING no arm mask: {e}", file=sys.stderr)
        return None, f"disabled: {type(e).__name__}: {e}"
    name = a.servo_shm if a.servo_shm is not None else cfg.shm_name
    return ArmMaskSource(masker, ServoStateReader(name), calibration), f"servo shm {name!r} (read-only)"


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

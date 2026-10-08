"""Hand-eye calibration driver (port of LIT/scripts/calibrate_hand_eye.py to the F9/F7 interfaces).

Procedure: visit >= 10 poses spread over the workspace box (greedy farthest-point selection), at each one record
the gripper-tip position from forward kinematics (base frame) and from the die tracker looking at a red marker on
the tip (camera frame), then fit ``T_base_cam`` with :func:`fit_hand_eye` and write ``hand_eye.yaml``.

* ``--dry-run`` uses a synthetic rig (ground truth: the camera of ``calib/camera.yaml``, rendered marker, real
  tracker) and never touches the arm or a camera. It proves the pipeline and the residual gate end to end.
* ``--execute`` is required to move the arm. The real rig adapter needs the env ``ShmBackend`` (unit U8) and
  ``So101FK``; until those exist it exits with code 3 and a clear message. H8 finishes and runs it on the real arm.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

from so101_rl.config.schema import WorkspaceBox
from so101_rl.perception.camera import DEFAULT_SERIAL, FakeCamera, Intrinsics
from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.perception.die_tracker import HsvDepthDieTracker
from so101_rl.perception.frames import MIN_POSES, HandEye, fit_hand_eye, load_hand_eye, save_hand_eye

DEFAULT_DRY_OUT = Path("runs/hand-eye-dry/hand_eye.yaml")
DEFAULT_REAL_OUT = Path("calib/hand_eye.yaml")


class CalibrationRig(Protocol):
    def measure(self, target_xyz_base: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Move the marker to ``target_xyz_base`` and return ``(fk_xyz_base, cam_xyz)``; None = not detected."""
        ...


def farthest_point_subset(points: np.ndarray, k: int, start: int = 0) -> np.ndarray:
    """Indices of ``k`` points chosen greedily to maximise the minimum pairwise distance."""
    n = len(points)
    if not 1 <= k <= n:
        raise ValueError(f"cannot pick {k} of {n} points")
    chosen = [start]
    d = np.linalg.norm(points - points[start], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen)


def sample_box_points(box: WorkspaceBox, n: int, rng: np.random.Generator, margin_m: float = 0.02) -> np.ndarray:
    lo = np.array([box.x[0], box.y[0], box.z[0]]) + margin_m
    hi = np.array([box.x[1], box.y[1], box.z[1]]) - margin_m
    return rng.uniform(lo, hi, (n, 3))


class SyntheticRig:
    """Ground-truth camera model: renders a red square marker and runs the real tracker on it."""

    def __init__(self, T_true_base_cam: np.ndarray, intr: Intrinsics, tracker: HsvDepthDieTracker, marker_px: int = 15):
        self.T = T_true_base_cam
        self.intr = intr
        self.tracker = tracker
        self.half = marker_px // 2
        self.size = 2 * self.half + 1  # odd: the centroid is an exact pixel
        self._rgb = np.full((intr.height, intr.width, 3), 120, np.uint8)
        self._depth = np.full((intr.height, intr.width), 0.75, np.float32)
        self.cam = FakeCamera(self._rgb, self._depth, intr)
        self.cam.start()

    def measure(self, target_xyz_base: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        R, t = self.T[:3, :3], self.T[:3, 3]
        p_cam = R.T @ (np.asarray(target_xyz_base, dtype=np.float64) - t)
        if p_cam[2] <= 0.05:
            return None
        u = round(self.intr.fx * p_cam[0] / p_cam[2] + self.intr.cx)
        v = round(self.intr.fy * p_cam[1] / p_cam[2] + self.intr.cy)
        h = self.half
        if not (h + 3 <= u < self.intr.width - h - 3 and h + 3 <= v < self.intr.height - h - 3):
            return None
        rgb = self._rgb.copy()
        depth = self._depth.copy()
        rgb[v - h : v + h + 1, u - h : u + h + 1] = (200, 20, 20)
        depth[v - h : v + h + 1, u - h : u + h + 1] = p_cam[2]
        self.cam.set_frames(rgb, depth)
        rgb_buf = np.empty_like(rgb)
        depth_buf = np.empty_like(depth)
        if not self.cam.read(rgb_buf, depth_buf).ok:
            return None
        det = self.tracker.detect(rgb_buf, depth_buf)
        if det is None:
            return None
        return np.asarray(target_xyz_base, dtype=np.float64), det.xyz_cam


def run_calibration(
    rig: CalibrationRig,
    candidates_base: np.ndarray,
    *,
    n_poses: int,
    camera_serial: str,
    date: str | None = None,
) -> HandEye:
    """Visit farthest-point poses, collect correspondences, fit. Raises if fewer than MIN_POSES are usable."""
    if n_poses < MIN_POSES:
        raise ValueError(f"n_poses must be >= {MIN_POSES}")
    order = farthest_point_subset(candidates_base, min(len(candidates_base), n_poses + 4))
    p_base: list[np.ndarray] = []
    p_cam: list[np.ndarray] = []
    for idx in order:
        m = rig.measure(candidates_base[idx])
        if m is None:
            continue
        p_base.append(m[0])
        p_cam.append(m[1])
        if len(p_base) == n_poses:
            break
    if len(p_base) < MIN_POSES:
        raise RuntimeError(f"only {len(p_base)} usable poses, need >= {MIN_POSES}")
    T, residual = fit_hand_eye(np.array(p_base), np.array(p_cam))
    return HandEye(T, residual, len(p_base), date or _dt.datetime.now(_dt.UTC).date().isoformat(), camera_serial)


def _dry_run(out: Path, n_poses: int, seed: int) -> int:
    # the synthetic ground truth is the camera of calib/camera.yaml (intrinsics and pose; plan C.1 single source)
    calib = load_camera_calib()
    intr = calib.color_intrinsics()
    tracker = HsvDepthDieTracker.from_yaml(intr)
    T_true = calib.T_base_color
    rig = SyntheticRig(T_true, intr, tracker)
    rng = np.random.default_rng(seed)
    cands = sample_box_points(WorkspaceBox(), 200, rng)
    he = run_calibration(rig, cands, n_poses=n_poses, camera_serial="dry-run")
    save_hand_eye(he, out)
    load_hand_eye(out)  # the artifact must pass its own gate
    err_mm = float(np.max(np.abs(he.T_base_cam - T_true)) * 1000)
    print(f"dry-run: {he.n_poses} poses, residual {he.residual_mm:.3f} mm, max |T - T_true| {err_mm:.3f} mm -> {out}")
    return 0


def _execute(out: Path, n_poses: int) -> int:
    try:
        from so101_rl.env.backend import ShmBackend  # noqa: F401  (U8)
    except ImportError:
        print(
            "calibrate_hand_eye --execute: needs so101_rl.env.backend.ShmBackend and So101FK (unit U8), which are "
            "not available yet. No motion was commanded.",
            file=sys.stderr,
        )
        return 3
    print(
        "calibrate_hand_eye --execute: the real-rig adapter is completed and run in H8, with the operator present.",
        file=sys.stderr,
    )
    return 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="synthetic rig; no arm, no camera")
    mode.add_argument("--execute", action="store_true", help="move the real arm (required to move)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-poses", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serial", default=DEFAULT_SERIAL)
    args = ap.parse_args(argv)
    if args.dry_run:
        return _dry_run(args.out or DEFAULT_DRY_OUT, args.n_poses, args.seed)
    return _execute(args.out or DEFAULT_REAL_OUT, args.n_poses)


if __name__ == "__main__":
    raise SystemExit(main())

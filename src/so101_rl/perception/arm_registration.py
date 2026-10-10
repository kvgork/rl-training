"""Camera-only extrinsics by registering the arm's FK shape against the depth cloud (camera parity C.2.3, C.2.4).

Inputs: one or more observations ``(depth cloud of the aligned depth, colour frame; probed joint angles)`` and a table
plane fit (``perception/table_plane.py``). The plane fixes the camera's height, roll and pitch; the arm fixes the
rest.

1. **Coarse search (3 DOF, plane-constrained).** For yaw on a 10 deg grid and the base position on a 1.5 cm grid of
   the visible table, score = the fraction of the arm's camera-facing surface samples (``sim/arm_cloud.py``, FK at
   the probed angles) that land within 8 mm of an occupied scene voxel. Several distinct best hypotheses are kept
   (the seeds).
2. **Planar ICP** (x, y, yaw; point-to-plane on the model normals, trimmed) from every seed.
3. **6-DOF ICP** from the best planar result, point-to-plane, trimmed at 15 -> 8 mm, model visibility (back-face
   culling + a z-buffer) recomputed every outer iteration. Each trim stage runs until the step is below
   ``ICP_CONV_ROT_RAD`` / ``ICP_CONV_TRANS_M`` (or ``icp6_stage_iters``); ``icp6_converged`` means the final (8 mm)
   stage converged. Two things make it converge (the 2026-10-09 rig runs: 30/30 iterations "not converged" while
   the pose had settled after ~10, the step jittering at ~0.05 deg / 0.1 mm): the scene subsample is FIXED (one random
   key per scene point, drawn once) instead of redrawn every iteration, and the correspondences are weighted by a
   Tukey biweight of their distance (``(1 - (d / trim)^2)^2``, times the 5 mm Huber weight of the point-to-plane
   residual) instead of a hard 0/1 trim, so a point crossing the trim radius no longer flips the problem (a hard trim
   left a 2-cycle of ~0.05 deg).
4. **Metrics / gate** (:class:`ArmShapeResult`): the RMS point-to-point distance from every scene point within
   2 cm of the visible model to the nearest model sample (<= 5 mm; NOT the trimmed ICP residual, which is small by
   construction: with the shoulder and elbow 0.25 rad wrong the 8 mm-trimmed point-to-plane RMS was 2.6 mm, this
   one 6-7 mm), the coverage of the visible model (fraction of visible model samples with a scene point within
   10 mm; >= 0.8, an added criterion: right angles 0.95, those wrong angles 0.79 on synthetic frames), the
   plane-vs-6-DOF agreement (camera height within 5 mm, tilt within 2 deg), and the FK gripper tip projected into the
   colour image within 8 px of an arm pixel at the tip's depth (+-20 mm). Bootstrap over the correspondences gives
   the uncertainty (2 sigma).
5. **Diagnostics** (``ArmShapeResult.diagnostics``, report only, no gate): where the table plane lies in the base
   frame under the 6-DOF pose (the axis and angle of the plane-vs-6-DOF tilt, the table height at the base origin),
   the information matrix of the final point-to-plane problem (rotation axes in the base frame with their 1-sigma, and
   the angle between the tilt axis and the weakest one: a disagreement about a well-constrained axis is not ICP
   noise), and the **level-base alternative**: the plane-constrained pose with a free height plus zero offsets of
   shoulder_lift, elbow_flex, wrist_flex and wrist_roll. When that fits as well as the 6-DOF pose, a single pose cannot
   tell a tilted base from joint-zero (calibration) errors: shoulder_lift and a tilt about base y correlate at ~0.9.
   **Tilt vs offsets** (``tilt_vs_offsets``, 2026-10-10): the 6-DOF pose and the joint offsets fitted TOGETHER, with
   the separation VIF (how much the free offsets inflate the tilt's sigma; pose geometry only) and a verdict
   (``tilted_base`` / ``joint_offsets`` / ``both`` / ``neither``, or ``ambiguous`` when the VIF is above
   :data:`MAX_SEPARATION_VIF`). A second pose with the base TURNED (the shoulder_lift axis then lies at an angle to
   the tilt axis) is what separates them: one rest pose VIF ~3, + the base turned 45 deg ~2.0, 60 deg ~1.5-1.7;
   raising the upper arm adds little.

**Degeneracy.** A shoulder_pan error is indistinguishable from a camera yaw about the base's vertical axis (only
the static base mesh breaks the symmetry, weakly): 60 ticks of pan error moved the result 39 mm with identical
metrics. The registered camera is then consistent with the arm's FK frame (what the policy uses), not with the
physical base; H8 hand-eye shares the same property.

Nearest neighbours use chunked ``torch.cdist`` on the CPU (no scipy in the environments).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from so101_rl.perception.camera_calib import StreamIntrinsics, plane_constrained_pose, rot_z
from so101_rl.perception.table_plane import PlaneFit

MAX_RMS_MM = 5.0
MIN_COVERAGE = 0.8
GATE_TRIM_M = 0.020  # the RMS gate counts every scene point within 2 cm of the visible model (clutter beyond)
MAX_PLANE_HEIGHT_MM = 5.0
MAX_PLANE_TILT_DEG = 2.0
MAX_TIP_PX = 8.0
ABOVE_TABLE_M = 0.008
MIN_INLIERS = 1000  # scene arm points within GATE_TRIM_M of the visible model: an RMS over fewer points proves little
MIN_VISIBLE_SAMPLES = 500  # visible model surface samples per observation (a sliver of overlap cannot pin 6 DOF)
MIN_ICP_ITERS = 3  # 6-DOF ICP Gauss-Newton steps taken (or converged after >= 1 step)
# 6-DOF ICP convergence: the Gauss-Newton step is below 1e-4 rad (0.006 deg) AND 1e-5 m, two orders of magnitude under
# the bootstrap uncertainty (~0.15 deg / 0.4 mm on the rig).
ICP_CONV_ROT_RAD = 1e-4
ICP_CONV_TRANS_M = 1e-5
ICP_STAGE_ITERS = 40  # per trim stage (15 mm, then 8 mm)
# Diagnostic joint-offset fit (level-base alternative): the joints whose zero error mimics a camera tilt.
OFFSET_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TICKS_PER_DEG = 4096 / 360.0
# Tilt-vs-offset verdict (diagnostics ``tilt_vs_offsets``, report only). The separation VIF is pose geometry only
# (2026-10-10 synthetic study, rest pose + a second pose): one pose ~2.2-3.1 (rig: 2.5-4.7), base turned 30 deg ~2.4,
# 45 deg ~2.0, 60 deg ~1.5-1.7. The significance levels sit above the bias an unmodelled cable + a 1.5 % depth scale
# put on the fit (two poses: tilt <= 0.3 deg, offsets <= 1.35 deg), not at the statistical sigma (~0.03 deg).
MAX_SEPARATION_VIF = 2.2
TILT_SIGNIFICANT_DEG = 1.5
OFFSET_SIGNIFICANT_DEG = 2.0


@dataclass
class Observation:
    pts_cam: np.ndarray  # [P, 3] colour-frame points of the aligned depth (median over frames)
    pix: np.ndarray  # [P, 2] (v, u) pixel of each point
    model_pts: np.ndarray  # [M, 3] base-frame arm surface samples at the probed angles
    model_nrm: np.ndarray  # [M, 3]
    tip_base: np.ndarray  # [3] FK gripper tip
    depth_m: np.ndarray  # [H, W] aligned depth (median), for the tip check
    # (pts, nrm) of the model with zero offsets [rad] added to OFFSET_JOINTS; diagnostics only (None: skipped)
    model_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None


@dataclass
class ArmShapeResult:
    T_base_color: np.ndarray
    rms_mm: float
    coverage: float
    n_inliers: int
    plane_height_err_mm: float
    plane_tilt_err_deg: float
    tip_px_err: float
    uncertainty_trans_mm: float
    uncertainty_rot_deg: float
    seeds: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    visible_samples: list[int] = field(default_factory=list)  # visible model samples per observation at the final pose
    icp6_iterations: int = 0
    icp6_converged: bool = False
    min_inliers: int = MIN_INLIERS
    min_visible_samples: int = MIN_VISIBLE_SAMPLES
    min_icp_iters: int = MIN_ICP_ITERS
    diagnostics: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_json(self) -> dict:
        return {
            "T_base_color": self.T_base_color.tolist(),
            "rms_mm": self.rms_mm,
            "coverage": self.coverage,
            "n_inliers": self.n_inliers,
            "plane_height_err_mm": self.plane_height_err_mm,
            "plane_tilt_err_deg": self.plane_tilt_err_deg,
            "tip_px_err": self.tip_px_err,
            "visible_samples": self.visible_samples,
            "icp6_iterations": self.icp6_iterations,
            "icp6_converged": self.icp6_converged,
            "uncertainty": {"trans_mm": self.uncertainty_trans_mm, "rot_deg": self.uncertainty_rot_deg},
            "gate": {
                "rms_mm<=": MAX_RMS_MM,
                "coverage>=": MIN_COVERAGE,
                "plane_height_mm<=": MAX_PLANE_HEIGHT_MM,
                "plane_tilt_deg<=": MAX_PLANE_TILT_DEG,
                "tip_px<=": MAX_TIP_PX,
                "n_inliers>=": self.min_inliers,
                "visible_samples_per_obs>=": self.min_visible_samples,
                "icp6_iterations>=": self.min_icp_iters,
            },
            "passed": self.passed,
            "failures": self.failures,
            "seeds": self.seeds,
            "diagnostics": self.diagnostics,
        }


# ------------------------------------------------------------------------------------------------ helpers
def _nn(a: np.ndarray, b: np.ndarray, max_elems: int = 1 << 24) -> tuple[np.ndarray, np.ndarray]:
    """For each row of ``a``: (distance, index) of the nearest row of ``b``.

    Each distance block holds at most ``max_elems`` floats (64 MB), whatever the size of ``b``.
    """
    ta = torch.as_tensor(a, dtype=torch.float32)
    tb = torch.as_tensor(b, dtype=torch.float32)
    chunk = max(1, max_elems // max(1, len(tb)))
    ds, ix = [], []
    for i in range(0, len(ta), chunk):
        d = torch.cdist(ta[i : i + chunk], tb)
        m, j = d.min(dim=1)
        ds.append(m)
        ix.append(j)
    return torch.cat(ds).numpy().astype(np.float64), torch.cat(ix).numpy()


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def visible_model(T_base_cam: np.ndarray, pts: np.ndarray, nrm: np.ndarray, intr: StreamIntrinsics, *, scale: int = 4, tol_m: float = 0.006) -> np.ndarray:
    """bool [M]: model samples facing the camera and not hidden by nearer samples (z-buffer at 1/``scale``)."""
    c = T_base_cam[:3, 3]
    facing = ((c - pts) * nrm).sum(1) > 0
    pc = (pts - c) @ T_base_cam[:3, :3]
    z = pc[:, 2]
    ok = facing & (z > 0.05)
    u = np.floor((intr.fx * pc[:, 0] / np.maximum(z, 1e-6) + intr.cx) / scale).astype(np.int64)
    v = np.floor((intr.fy * pc[:, 1] / np.maximum(z, 1e-6) + intr.cy) / scale).astype(np.int64)
    W, H = -(-intr.width // scale), -(-intr.height // scale)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zb = np.full(W * H, np.inf)
    key = v * W + u
    np.minimum.at(zb, key[ok], z[ok])
    vis = np.zeros(len(pts), bool)
    vis[ok] = z[ok] <= zb[key[ok]] + tol_m
    return vis


def _rotvec(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))))


# ------------------------------------------------------------------------------------------------ registration
class ArmShapeRegistration:
    def __init__(self, obs: list[Observation], plane: PlaneFit, intr: StreamIntrinsics, table_z_m: float, *, seed: int = 0) -> None:
        if not obs:
            raise ValueError("need at least one observation")
        self.obs, self.plane, self.intr, self.table_z = obs, plane, intr, float(table_z_m)
        self.rng = np.random.default_rng(seed)
        self.icp6_iterations, self.icp6_converged = 0, False
        # levelled scene points (plane frame with x = y = yaw = 0): above the table only
        self.T0 = plane_constrained_pose(plane.normal_up, plane.height_m, self.table_z, 0.0, 0.0, 0.0)
        self.scene: list[np.ndarray] = []
        self._keys: list[np.ndarray] = []  # fixed subsampling priority per scene point (module docstring, step 3)
        for o in obs:
            h = o.pts_cam @ plane.normal_up + plane.height_m
            keep = (h > ABOVE_TABLE_M) & (h < 0.6)
            q = _apply(self.T0, o.pts_cam[keep])
            if len(q) > 40_000:
                q = q[self.rng.choice(len(q), 40_000, replace=False)]
            self.scene.append(q)
            self._keys.append(self.rng.random(len(q)))

    def pose(self, x: float, y: float, yaw: float) -> np.ndarray:
        return plane_constrained_pose(self.plane.normal_up, self.plane.height_m, self.table_z, x, y, yaw)

    # ---------------------------------------------------------------- coarse
    def coarse(self, *, yaw_step_deg: float = 10.0, xy_step_m: float = 0.015, keep: int = 6, n_model: int = 1200) -> list[tuple[float, float, float, float]]:
        """(score, x, y, yaw) of the best distinct plane-constrained hypotheses (vectorised over the base grid)."""
        vox = 0.008
        lev = np.concatenate(self.scene)
        lo3 = lev.min(0) - 2 * vox
        shape = np.ceil((lev.max(0) + 2 * vox - lo3) / vox).astype(np.int64) + 1
        cam = np.array([0.0, 0.0, self.table_z + self.plane.height_m])
        grids = []
        for q in self.scene:
            g = np.zeros(shape, bool)
            k = np.floor((q - lo3) / vox).astype(np.int64)
            g[k[:, 0], k[:, 1], k[:, 2]] = True
            grids.append(g)
        bx, by = np.meshgrid(np.arange(lev[:, 0].min(), lev[:, 0].max(), xy_step_m), np.arange(lev[:, 1].min(), lev[:, 1].max(), xy_step_m), indexing="ij")
        B = np.stack([bx.ravel(), by.ravel(), np.zeros(bx.size)], 1)
        results = []
        for yaw in np.radians(np.arange(0.0, 360.0, yaw_step_deg)):
            Rz = rot_z(yaw)
            score = np.zeros(len(B))
            for oi, o in enumerate(self.obs):
                sub = self.rng.choice(len(o.model_pts), size=min(len(o.model_pts), n_model), replace=False)
                m_l = o.model_pts[sub] @ Rz  # Rz^T m: the model for a base at the levelled origin
                n_l = o.model_nrm[sub] @ Rz
                for c0 in range(0, len(B), 256):
                    p = m_l[None] + B[c0 : c0 + 256, None]  # [b, M, 3]
                    facing = (n_l[None] * (cam - p)).sum(-1) > 0
                    k = np.floor((p - lo3) / vox).astype(np.int64)
                    inside = ((k >= 0) & (k < shape)).all(-1)
                    kk = np.where(inside[..., None], k, 0)
                    hit = grids[oi][kk[..., 0], kk[..., 1], kk[..., 2]] & inside & facing
                    nf = facing.sum(-1)
                    score[c0 : c0 + 256] += np.where(nf >= 50, hit.sum(-1) / np.maximum(nf, 1), 0.0)
            for i in np.argsort(-score)[: 4 * keep]:
                results.append((float(score[i] / len(self.obs)), B[i, 0], B[i, 1], float(yaw)))
        results.sort(key=lambda r: -r[0])
        out: list[tuple[float, float, float, float]] = []
        for sc, bxx, byy, yaw in results:
            # base at levelled b: base->levelled is q = Rz^T (p - t), so t = -Rz b
            t = -rot_z(yaw) @ np.array([bxx, byy, 0.0])
            cand = (sc, float(t[0]), float(t[1]), float(yaw))
            if all(np.hypot(cand[1] - c[1], cand[2] - c[2]) > 0.04 or abs(math.remainder(cand[3] - c[3], 2 * math.pi)) > math.radians(25) for c in out):
                out.append(cand)
            if len(out) >= keep:
                break
        return out

    def _scene_base(self, T: np.ndarray, oi: int) -> np.ndarray:
        """Scene points of observation ``oi`` in the base frame under the full pose ``T`` (T_base_color)."""
        return _apply(T @ np.linalg.inv(self.T0), self.scene[oi])

    def _gather(self, T: np.ndarray, trim_m: float, *, max_pts: int = 6000, with_dist: bool = False):
        """(scene, model, normal) correspondences within ``trim_m`` (plus their distances with ``with_dist``)."""
        S, M, N, D = [], [], [], []
        for oi, o in enumerate(self.obs):
            vis = visible_model(T, o.model_pts, o.model_nrm, self.intr)
            mp, mn = o.model_pts[vis], o.model_nrm[vis]
            if len(mp) < 20:
                continue
            s = self._scene_base(T, oi)
            # keep the scene points near the model's bounding box; subsample by the fixed keys (deterministic in T)
            lo, hi = mp.min(0) - 0.03, mp.max(0) + 0.03
            inb = ((s >= lo) & (s <= hi)).all(1)
            s = s[inb]
            if len(s) > max_pts:
                s = s[np.argpartition(self._keys[oi][inb], max_pts)[:max_pts]]
            if len(s) == 0:
                continue
            d, j = _nn(s, mp)
            keep = d < trim_m
            S.append(s[keep])
            M.append(mp[j[keep]])
            N.append(mn[j[keep]])
            D.append(d[keep])
        if not S:
            out = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        else:
            out = (np.concatenate(S), np.concatenate(M), np.concatenate(N), np.concatenate(D))
        return out if with_dist else out[:3]

    def icp_planar(self, x: float, y: float, yaw: float, *, iters: int = 25) -> tuple[float, float, float]:
        for it in range(iters):
            trim = 0.03 if it < 8 else 0.015
            T = self.pose(x, y, yaw)
            s, m, n = self._gather(T, trim)
            if len(s) < 30:
                break
            # r = n . (s - m); d s/d x = (1,0,0); d s / d yaw = z x (s - c_rot) with rotation about base z through origin
            J = np.stack([n[:, 0], n[:, 1], n[:, 1] * s[:, 0] - n[:, 0] * s[:, 1]], 1)
            r = ((s - m) * n).sum(1)
            w = 1.0 / np.maximum(1.0, np.abs(r) / 0.005)
            A = J.T @ (J * w[:, None]) + 1e-6 * np.eye(3)
            dx = -np.linalg.solve(A, J.T @ (w * r))
            # update: s' = Rz(dyaw) s + (dx, dy); in pose params: T' = [Rz(dyaw) | (dx,dy,0)] T
            dR = rot_z(dx[2])
            t = dR @ np.array([x, y, 0.0]) + np.array([dx[0], dx[1], 0.0])
            x, y, yaw = float(t[0]), float(t[1]), yaw + float(dx[2])
            if np.abs(dx[:2]).max() < 1e-5 and abs(dx[2]) < 1e-5:
                break
        return x, y, yaw

    def icp_6dof(self, T: np.ndarray, *, stage_iters: int = ICP_STAGE_ITERS, trims: tuple[float, ...] = (0.015, 0.008)) -> np.ndarray:
        """Point-to-plane ICP in 6 DOF, one stage per trim radius, each until converged (module docstring, step 3) or
        ``stage_iters``. ``self.icp6_iterations`` / ``icp6_converged`` record what ran: 0 iterations means the
        correspondence set was empty at the start (the pose returned is the input, unrefined); converged means the
        LAST stage converged."""
        T = T.copy()
        self.icp6_iterations, self.icp6_converged = 0, False
        for trim in trims:
            self.icp6_converged = False
            for _ in range(stage_iters):
                s, m, n, d = self._gather(T, trim, with_dist=True)
                if len(s) < 30:
                    return T
                T = self._step6(T, s, m, n, (1.0 - (d / trim) ** 2) ** 2) @ T
                self.icp6_iterations += 1
                if self._last_rot < ICP_CONV_ROT_RAD and self._last_trans < ICP_CONV_TRANS_M:
                    self.icp6_converged = True
                    break
        return T

    def visible_counts(self, T: np.ndarray) -> list[int]:
        """Visible model surface samples per observation under ``T``."""
        return [int(visible_model(T, o.model_pts, o.model_nrm, self.intr).sum()) for o in self.obs]

    def _step6(self, T: np.ndarray, s: np.ndarray, m: np.ndarray, n: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        """One Gauss-Newton point-to-plane step: returns dT (applied on the left, base frame)."""
        J = np.concatenate([np.cross(s, n), n], 1)  # d r / d (omega, t)
        r = ((s - m) * n).sum(1)
        ww = 1.0 / np.maximum(1.0, np.abs(r) / 0.005)
        if w is not None:
            ww = ww * w
        A = J.T @ (J * ww[:, None]) + 1e-8 * np.eye(6)
        dx = -np.linalg.solve(A, J.T @ (ww * r))
        dT = np.eye(4)
        dT[:3, :3] = _rotvec(dx[:3])
        dT[:3, 3] = dx[3:]
        self._last_rot, self._last_trans = float(np.linalg.norm(dx[:3])), float(np.linalg.norm(dx[3:]))
        return dT

    # ---------------------------------------------------------------- metrics
    def metrics(self, T: np.ndarray, trim_m: float = GATE_TRIM_M) -> tuple[float, float, int]:
        """(RMS point-to-point distance [mm] of the scene arm points within ``trim_m`` of the visible model, the
        model coverage, the number of those points)."""
        s, m, _n = self._gather(T, trim_m, max_pts=20_000)
        if len(s) == 0:
            return math.inf, 0.0, 0
        rms = float(np.sqrt(np.mean(np.sum((s - m) ** 2, axis=1))) * 1000)
        covs = []
        for oi, o in enumerate(self.obs):
            vis = visible_model(T, o.model_pts, o.model_nrm, self.intr)
            if vis.sum() == 0:
                covs.append(0.0)
                continue
            sc = self._scene_base(T, oi)
            d, _ = _nn(o.model_pts[vis], sc)
            covs.append(float((d < 0.010).mean()))
        return rms, float(np.mean(covs)), len(s)

    def bootstrap(self, T: np.ndarray, *, n: int = 20) -> tuple[float, float]:
        trim = 0.008
        s, m, nr, d = self._gather(T, trim, max_pts=20_000, with_dist=True)
        if len(s) < 30:
            return math.inf, math.inf
        w = (1.0 - (d / trim) ** 2) ** 2  # the weights of the final ICP stage, so T is (nearly) a fixed point
        ts, rs = [], []
        for _ in range(n):
            idx = self.rng.integers(0, len(s), len(s))
            Tb = np.eye(4)
            for _k in range(5):
                ss = _apply(Tb, s[idx])
                Tb = self._step6(Tb, ss, m[idx], nr[idx], w[idx]) @ Tb
            Tf = Tb @ T
            ts.append(Tf[:3, 3])
            rs.append(rotation_angle_deg(Tf[:3, :3] @ T[:3, :3].T))
        ts = np.array(ts)
        return float(2 * np.linalg.norm(ts.std(0)) * 1000), float(2 * np.sqrt(np.mean(np.square(rs))))

    def tip_error_px(self, T: np.ndarray) -> float:
        """Pixel distance from the projected FK tip to the nearest arm pixel (scene point within 10 mm of the model)
        whose depth is within 20 mm of the tip's (worst over observations)."""
        worst = 0.0
        for oi, o in enumerate(self.obs):
            Ti = np.linalg.inv(T)
            tc = _apply(Ti, o.tip_base[None])[0]
            uv = np.array([self.intr.fx * tc[0] / tc[2] + self.intr.cx, self.intr.fy * tc[1] / tc[2] + self.intr.cy])
            pb = _apply(T, o.pts_cam)
            near = np.abs(o.pts_cam[:, 2] - tc[2]) < 0.020
            if near.sum() == 0:
                return math.inf
            d, _ = _nn(pb[near], o.model_pts)
            arm = d < 0.010
            if arm.sum() == 0:
                return math.inf
            px = o.pix[near][arm][:, ::-1].astype(np.float64)  # (u, v)
            worst = max(worst, float(np.min(np.linalg.norm(px - uv, axis=1))))
        return worst

    # ---------------------------------------------------------------- diagnostics (report only, no gate)
    def plane_in_base(self, T: np.ndarray) -> dict:
        """The table plane in the base frame under ``T``: its up normal, the tilt against base +z (angle, and the
        rotation axis taking base +z onto the normal), the direction the table descends along, its slope and its
        height at the base origin."""
        n_b = T[:3, :3] @ self.plane.normal_up
        p0 = _apply(T, (-self.plane.height_m * self.plane.normal_up)[None])[0]  # foot of the camera on the plane
        z0 = p0[2] + (n_b[0] * p0[0] + n_b[1] * p0[1]) / n_b[2]
        ax = np.cross([0.0, 0.0, 1.0], n_b)
        sn = float(np.linalg.norm(ax))
        down = n_b[:2] / max(float(np.hypot(n_b[0], n_b[1])), 1e-12)
        return {
            "normal_base": n_b.tolist(),
            "tilt_deg": math.degrees(math.atan2(sn, float(n_b[2]))),
            "tilt_axis_base": (ax / max(sn, 1e-12)).tolist(),
            "descends_toward_base_xy": down.tolist(),
            "slope_mm_per_100mm": 100.0 * float(np.hypot(n_b[0], n_b[1]) / n_b[2]),
            "table_z_at_base_origin_mm": 1000.0 * float(z0),
            "table_z_assumed_mm": 1000.0 * self.table_z,
        }

    def information(self, T: np.ndarray) -> dict:
        """The point-to-plane information matrix at ``T`` (8 mm trim): rotation axes (base frame, translation
        marginalised) and translation axes (rotation marginalised), weakest first, each with its 1-sigma for the
        observed residual spread. A LOWER BOUND on the real uncertainty (residuals are spatially correlated)."""
        s, m, n = self._gather(T, 0.008, max_pts=20_000)
        if len(s) < 30:
            return {}
        J = np.concatenate([np.cross(s, n), n], 1)
        r = ((s - m) * n).sum(1)
        w = 1.0 / np.maximum(1.0, np.abs(r) / 0.005)
        A = J.T @ (J * w[:, None])
        sig = float(np.sqrt(np.sum(w * r**2) / np.sum(w)))
        Arr = A[:3, :3] - A[:3, 3:] @ np.linalg.solve(A[3:, 3:], A[3:, :3])
        Att = A[3:, 3:] - A[3:, :3] @ np.linalg.solve(A[:3, :3], A[:3, 3:])
        er, Vr = np.linalg.eigh(Arr)
        et, Vt = np.linalg.eigh(Att)
        return {
            "residual_sigma_mm": 1000.0 * sig,
            "rotation_axes_base": [{"axis": Vr[:, k].tolist(), "sigma_deg": math.degrees(sig / math.sqrt(er[k]))} for k in range(3)],
            "translation_axes_base": [{"axis": Vt[:, k].tolist(), "sigma_mm": 1000.0 * sig / math.sqrt(et[k])} for k in range(3)],
        }

    def _offset_problem(self, T: np.ndarray, delta: np.ndarray, trim: float, *, eps: float = 1e-3, max_pts: int = 6000):
        """Point-to-plane linearisation of the pose-plus-joint-offset model at (``T``, ``delta``): (J [N, 6 + k],
        r [N], distances [N]). Pose columns: (omega, t) of a left perturbation in the base frame (as :meth:`_step6`);
        offset columns: the model motion per rad of each :data:`OFFSET_JOINTS` offset (finite differences)."""
        k = len(OFFSET_JOINTS)
        Js, rs, ds = [], [], []
        for oi, o in enumerate(self.obs):
            assert o.model_fn is not None
            mp, mn = o.model_fn(delta)
            dms = [(o.model_fn(delta + eps * np.eye(k)[j])[0] - mp) / eps for j in range(k)]
            idx = np.nonzero(visible_model(T, mp, mn, self.intr))[0]
            if len(idx) < 20:
                continue
            sc = self._scene_base(T, oi)
            lo, hi = mp[idx].min(0) - 0.03, mp[idx].max(0) + 0.03
            inb = ((sc >= lo) & (sc <= hi)).all(1)
            sc = sc[inb]
            if len(sc) > max_pts:
                sc = sc[np.argpartition(self._keys[oi][inb], max_pts)[:max_pts]]
            d, j = _nn(sc, mp[idx])
            keep = d < trim
            sc, mi = sc[keep], idx[j[keep]]
            nn = mn[mi]
            rs.append(((sc - mp[mi]) * nn).sum(1))
            ds.append(d[keep])
            Js.append(np.concatenate([np.cross(sc, nn), nn, np.stack([-(nn * dm[mi]).sum(1) for dm in dms], 1)], 1))
        if not Js:
            return np.zeros((0, 6 + k)), np.zeros(0), np.zeros(0)
        return np.concatenate(Js), np.concatenate(rs), np.concatenate(ds)

    def offset_fit(self, T0: np.ndarray, *, level: bool, iters: int = 40) -> dict | None:
        """Fit the camera pose TOGETHER with zero offsets of :data:`OFFSET_JOINTS` (shared by all observations),
        from ``T0``. ``level``: the pose keeps the table plane's roll / pitch (free x, y, yaw and height: the level-base
        explanation); otherwise all 6 DOF are free (tilt and offsets at once: separable only from several poses at
        different pan angles, :meth:`tilt_vs_offsets`). None when an observation has no ``model_fn``.

        ``cov`` is the Gauss-Newton covariance of the parameters for the observed (Huber-weighted) residual spread,
        independent points assumed: a LOWER BOUND (depth noise is spatially correlated); its correlations are exact."""
        if any(o.model_fn is None for o in self.obs):
            return None
        k = len(OFFSET_JOINTS)
        free = np.array([2, 3, 4, 5] if level else [0, 1, 2, 3, 4, 5]).tolist() + list(range(6, 6 + k))
        delta = np.zeros(k)
        T = T0.copy()
        A = np.eye(len(free))
        sig = math.nan
        for it in range(iters):
            trim = 0.015 if it < iters // 2 else 0.008
            J, r, d = self._offset_problem(T, delta, trim)
            if len(r) < 30:
                return None
            J = J[:, free]
            w = 1.0 / np.maximum(1.0, np.abs(r) / 0.005) * (1.0 - (d / trim) ** 2) ** 2
            A = J.T @ (J * w[:, None])
            sig = float(np.sqrt(np.sum(w * r**2) / np.sum(w)))
            dx = np.zeros(6 + k)
            dx[free] = -np.linalg.solve(A + 1e-8 * np.eye(len(free)), J.T @ (w * r))
            dT = np.eye(4)
            dT[:3, :3] = _rotvec(dx[:3])
            dT[:3, 3] = dx[3:6]
            T = dT @ T
            delta += dx[6:]
            if np.abs(dx).max() < 1e-6:
                break
        cov = np.full((6 + k, 6 + k), 0.0)
        cov[np.ix_(free, free)] = sig**2 * np.linalg.inv(A + 1e-12 * np.eye(len(free)))
        info = np.zeros((6 + k, 6 + k))
        info[np.ix_(free, free)] = A / sig**2
        saved = [(o.model_pts, o.model_nrm) for o in self.obs]
        try:
            for o in self.obs:
                o.model_pts, o.model_nrm = o.model_fn(delta)
            rms, coverage, nin = self.metrics(T)
        finally:
            for o, (mp, mn) in zip(self.obs, saved, strict=True):
                o.model_pts, o.model_nrm = mp, mn
        table_z = float(T[2, 3] - self.plane.height_m)
        sd = np.sqrt(np.diag(cov))
        return {
            "rms_mm": rms,
            "coverage": coverage,
            "n_inliers": nin,
            "table_z_mm": 1000.0 * table_z,
            "table_z_minus_assumed_mm": 1000.0 * (table_z - self.table_z),
            "offsets_deg": {n: math.degrees(float(v)) for n, v in zip(OFFSET_JOINTS, delta, strict=True)},
            "offsets_sigma_deg": {n: math.degrees(float(v)) for n, v in zip(OFFSET_JOINTS, sd[6:], strict=True)},
            "offsets_ticks_approx": {n: math.degrees(float(v)) * TICKS_PER_DEG for n, v in zip(OFFSET_JOINTS, delta, strict=True)},
            "T_base_color": T.tolist(),
            "_T": T,
            "_cov": cov,
            "_info": info,
        }

    def level_base_alternative(self, T_plane: np.ndarray, *, iters: int = 40) -> dict | None:
        """The plane-constrained pose with a FREE height plus zero offsets of :data:`OFFSET_JOINTS`, from ``T_plane``:
        the competing explanation of a plane-vs-6-DOF tilt, a level base whose FK is off (:meth:`offset_fit`)."""
        fit = self.offset_fit(T_plane, level=True, iters=iters)
        return None if fit is None else {k: v for k, v in fit.items() if not k.startswith("_")}

    def tilt_vs_offsets(self, T: np.ndarray) -> dict | None:
        """Base tilt AND joint-zero offsets fitted together (:meth:`offset_fit` with all 6 pose DOF free, from the
        6-DOF pose ``T``), with a verdict (module docstring, step 5). None when an observation has no ``model_fn``.

        ``separation_vif``: how much freeing the joint offsets inflates the 1-sigma of the tilt about base y (the
        axis a shoulder_lift error mimics at pan 0): sqrt(var(tilt | offsets free) / var(tilt | offsets fixed)). It
        depends only on the arm poses and the camera, not on the noise. Above :data:`MAX_SEPARATION_VIF` the poses
        cannot tell the two apart (one pose: ~3; a second pose with the base turned 45 / 60 deg: ~1.8-2.0 / ~1.6), and
        the verdict is ``ambiguous`` whatever the fit says. Otherwise the verdict is ``tilted_base``, ``joint_offsets``,
        ``both`` or ``neither`` by :data:`TILT_SIGNIFICANT_DEG` / :data:`OFFSET_SIGNIFICANT_DEG`."""
        fit = self.offset_fit(T, level=False)
        if fit is None:
            return None
        cov, info = fit["_cov"], fit["_info"]
        pb = self.plane_in_base(fit["_T"])
        n = np.asarray(pb["normal_base"])
        var_fixed = float(np.linalg.inv(info[:6, :6] + 1e-12 * np.eye(6))[1, 1])
        vif = math.sqrt(float(cov[1, 1]) / max(var_fixed, 1e-30))
        offs = {j: v for j, v in fit["offsets_deg"].items() if j != "wrist_roll"}  # wrist_roll: barely visible, report only
        tilt_sig = pb["tilt_deg"] >= TILT_SIGNIFICANT_DEG
        off_sig = max(abs(v) for v in offs.values()) >= OFFSET_SIGNIFICANT_DEG
        if vif > MAX_SEPARATION_VIF:
            verdict = "ambiguous"
        else:
            verdict = {(True, False): "tilted_base", (False, True): "joint_offsets", (True, True): "both"}.get((tilt_sig, off_sig), "neither")
        c = cov[[0, 1, 6, 8]][:, [0, 1, 6, 8]]
        sd = np.sqrt(np.diag(c))
        return {
            "verdict": verdict,
            "separation_vif": vif,
            "max_separation_vif": MAX_SEPARATION_VIF,
            "rms_mm": fit["rms_mm"],
            "coverage": fit["coverage"],
            "tilt_deg": pb["tilt_deg"],
            "tilt_about_base_x_deg": math.degrees(math.atan2(-n[1], n[2])),
            "tilt_about_base_y_deg": math.degrees(math.atan2(n[0], n[2])),
            "tilt_sigma_deg": {"x": math.degrees(float(sd[0])), "y": math.degrees(float(sd[1]))},
            "table_z_mm": pb["table_z_at_base_origin_mm"],
            "table_z_minus_assumed_mm": pb["table_z_at_base_origin_mm"] - 1000.0 * self.table_z,
            "offsets_deg": fit["offsets_deg"],
            "offsets_sigma_deg": fit["offsets_sigma_deg"],
            "offsets_ticks_approx": fit["offsets_ticks_approx"],
            "corr_tilt_y_shoulder_lift": float(c[1, 2] / max(sd[1] * sd[2], 1e-30)),
            "sigma_note": "Gauss-Newton 1-sigma for independent points: a lower bound (depth errors are spatially correlated)",
            "T_base_color": fit["T_base_color"],
        }

    def diagnostics(self, T: np.ndarray, T_plane: np.ndarray) -> dict:
        """Report-only diagnostics of the final pose ``T`` (module docstring, step 5)."""
        pb = self.plane_in_base(T)
        info = self.information(T)
        out = {"plane_in_base": pb, "icp_information": info}
        if info and pb["tilt_deg"] > 1e-6:
            weakest = np.array(info["rotation_axes_base"][0]["axis"])
            c = abs(float(weakest @ np.array(pb["tilt_axis_base"])))
            out["tilt_axis_vs_weakest_rotation_deg"] = math.degrees(math.acos(min(1.0, c)))
        alt = self.level_base_alternative(T_plane)
        if alt is not None:
            out["level_base_alternative"] = alt
        tvo = self.tilt_vs_offsets(T)
        if tvo is not None:
            out["tilt_vs_offsets"] = tvo
        return out

    # ---------------------------------------------------------------- driver
    def register(self, *, keep_seeds: int = 6, seeds: list[tuple[float, float, float, float]] | None = None) -> tuple[np.ndarray, list[dict]]:
        """Steps 1-3: coarse seeds (or the given ``(score, x, y, yaw)`` seeds), planar ICP from each, 6-DOF ICP from
        the best. Returns the 6-DOF pose and the refined seeds, best first."""
        seeds = self.coarse(keep=keep_seeds) if seeds is None else seeds
        if not seeds:
            raise ValueError("no seed: no scene points above the table near any arm hypothesis")
        refined = []
        for sc, x, y, yaw in seeds:
            x2, y2, yaw2 = self.icp_planar(x, y, yaw)
            T = self.pose(x2, y2, yaw2)
            rms, cov, _nin = self.metrics(T)
            refined.append({"coarse_score": sc, "x": x2, "y": y2, "yaw_deg": math.degrees(yaw2), "rms_mm": rms, "coverage": cov, "T": T})
        refined.sort(key=lambda r: (-(r["coverage"] >= 0.3), r["rms_mm"] / max(r["coverage"], 1e-3)))
        return self.icp_6dof(refined[0]["T"]), refined

    def run(
        self,
        *,
        bootstrap: int = 20,
        keep_seeds: int = 6,
        seeds: list[tuple[float, float, float, float]] | None = None,
        min_inliers: int = MIN_INLIERS,
        min_visible_samples: int = MIN_VISIBLE_SAMPLES,
        min_icp_iters: int = MIN_ICP_ITERS,
        diagnostics: bool = True,
    ) -> ArmShapeResult:
        T, refined = self.register(keep_seeds=keep_seeds, seeds=seeds)
        best = refined[0]
        rms, cov, nin = self.metrics(T)
        # plane agreement
        h6 = T[2, 3] - self.table_z
        dh = abs(h6 - self.plane.height_m) * 1000
        up6 = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        dt = math.degrees(math.acos(float(np.clip(up6 @ self.plane.normal_up, -1, 1))))
        tip = self.tip_error_px(T)
        ut, ur = self.bootstrap(T, n=bootstrap) if bootstrap else (math.nan, math.nan)
        vis_counts = self.visible_counts(T)
        iters6, conv6 = self.icp6_iterations, self.icp6_converged
        fails = []
        if not nin >= min_inliers:
            fails.append(f"only {nin} scene points match the arm model (< {min_inliers}): too little overlap to trust the RMS")
        low = [i for i, c in enumerate(vis_counts) if c < min_visible_samples]
        if low:
            fails.append(f"visible model samples {vis_counts} (observations {low} < {min_visible_samples}): arm out of view / occluded")
        if not (iters6 >= min_icp_iters or (conv6 and iters6 >= 1)):
            # without the 6-DOF refinement T is still the plane-constrained pose, which satisfies the plane check by
            # construction: the plane-vs-6DOF comparison below would pass trivially
            fails.append(f"6-DOF ICP ran {iters6} iteration(s) (< {min_icp_iters}, not converged): the plane check is vacuous")
        if not rms <= MAX_RMS_MM:
            fails.append(f"ICP RMS {rms:.2f} mm > {MAX_RMS_MM} mm")
        if not cov >= MIN_COVERAGE:
            fails.append(f"model coverage {cov:.2f} < {MIN_COVERAGE}")
        if not dh <= MAX_PLANE_HEIGHT_MM:
            fails.append(f"plane height disagrees by {dh:.1f} mm > {MAX_PLANE_HEIGHT_MM} mm")
        if not dt <= MAX_PLANE_TILT_DEG:
            fails.append(f"plane tilt disagrees by {dt:.2f} deg > {MAX_PLANE_TILT_DEG} deg")
        if bootstrap and not (math.isfinite(ut) and math.isfinite(ur)):
            fails.append("bootstrap uncertainty not finite (too few correspondences)")
        if not tip <= MAX_TIP_PX:
            fails.append(f"FK tip reprojects {tip:.1f} px from the arm > {MAX_TIP_PX} px")
        seeds_json = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in refined]
        diag = self.diagnostics(T, best["T"]) if diagnostics else {}
        return ArmShapeResult(
            T, rms, cov, nin, dh, dt, tip, ut, ur, seeds_json, fails, vis_counts, iters6, conv6,
            min_inliers, min_visible_samples, min_icp_iters, diag,
        )  # fmt: skip

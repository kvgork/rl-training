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
   culling + a z-buffer) recomputed every outer iteration.
4. **Metrics / gate** (:class:`ArmShapeResult`): the RMS point-to-point distance from every scene point within
   2 cm of the visible model to the nearest model sample (<= 5 mm; NOT the trimmed ICP residual, which is small by
   construction: with the shoulder and elbow 0.25 rad wrong the 8 mm-trimmed point-to-plane RMS was 2.6 mm, this
   one 6-7 mm), the coverage of the visible model (fraction of visible model samples with a scene point within
   10 mm; >= 0.8, an added criterion: right angles 0.95, those wrong angles 0.79 on synthetic frames), the
   plane-vs-6-DOF agreement (camera height within 5 mm, tilt within 2 deg), and the FK gripper tip projected into the
   colour image within 8 px of an arm pixel at the tip's depth (+-20 mm). Bootstrap over the correspondences gives
   the uncertainty (2 sigma).

**Degeneracy.** A shoulder_pan error is indistinguishable from a camera yaw about the base's vertical axis (only
the static base mesh breaks the symmetry, weakly): 60 ticks of pan error moved the result 39 mm with identical
metrics. The registered camera is then consistent with the arm's FK frame (what the policy uses), not with the
physical base; H8 hand-eye shares the same property.

Nearest neighbours use chunked ``torch.cdist`` on the CPU (no scipy in the environments).
"""

from __future__ import annotations

import math
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


@dataclass
class Observation:
    pts_cam: np.ndarray  # [P, 3] colour-frame points of the aligned depth (median over frames)
    pix: np.ndarray  # [P, 2] (v, u) pixel of each point
    model_pts: np.ndarray  # [M, 3] base-frame arm surface samples at the probed angles
    model_nrm: np.ndarray  # [M, 3]
    tip_base: np.ndarray  # [3] FK gripper tip
    depth_m: np.ndarray  # [H, W] aligned depth (median), for the tip check


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
            "uncertainty": {"trans_mm": self.uncertainty_trans_mm, "rot_deg": self.uncertainty_rot_deg},
            "gate": {
                "rms_mm<=": MAX_RMS_MM,
                "coverage>=": MIN_COVERAGE,
                "plane_height_mm<=": MAX_PLANE_HEIGHT_MM,
                "plane_tilt_deg<=": MAX_PLANE_TILT_DEG,
                "tip_px<=": MAX_TIP_PX,
            },
            "passed": self.passed,
            "failures": self.failures,
            "seeds": self.seeds,
        }


# ------------------------------------------------------------------------------------------------ helpers
def _nn(a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """For each row of ``a``: (distance, index) of the nearest row of ``b``."""
    ta = torch.as_tensor(a, dtype=torch.float32)
    tb = torch.as_tensor(b, dtype=torch.float32)
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
        # levelled scene points (plane frame with x = y = yaw = 0): above the table only
        self.T0 = plane_constrained_pose(plane.normal_up, plane.height_m, self.table_z, 0.0, 0.0, 0.0)
        self.scene: list[np.ndarray] = []
        for o in obs:
            h = o.pts_cam @ plane.normal_up + plane.height_m
            keep = (h > ABOVE_TABLE_M) & (h < 0.6)
            q = _apply(self.T0, o.pts_cam[keep])
            if len(q) > 40_000:
                q = q[self.rng.choice(len(q), 40_000, replace=False)]
            self.scene.append(q)

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

    def _gather(self, T: np.ndarray, trim_m: float, *, max_pts: int = 6000):
        S, M, N = [], [], []
        for oi, o in enumerate(self.obs):
            vis = visible_model(T, o.model_pts, o.model_nrm, self.intr)
            mp, mn = o.model_pts[vis], o.model_nrm[vis]
            if len(mp) < 20:
                continue
            s = self._scene_base(T, oi)
            # keep the scene points near the model's bounding box
            lo, hi = mp.min(0) - 0.03, mp.max(0) + 0.03
            s = s[((s >= lo) & (s <= hi)).all(1)]
            if len(s) > max_pts:
                s = s[self.rng.choice(len(s), max_pts, replace=False)]
            if len(s) == 0:
                continue
            d, j = _nn(s, mp)
            keep = d < trim_m
            S.append(s[keep])
            M.append(mp[j[keep]])
            N.append(mn[j[keep]])
        if not S:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
        return np.concatenate(S), np.concatenate(M), np.concatenate(N)

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

    def icp_6dof(self, T: np.ndarray, *, iters: int = 30, trims: tuple[float, float] = (0.015, 0.008)) -> np.ndarray:
        T = T.copy()
        for it in range(iters):
            trim = trims[0] if it < iters // 2 else trims[1]
            s, m, n = self._gather(T, trim)
            if len(s) < 30:
                break
            T = self._step6(T, s, m, n) @ T
            if self._last_step < 1e-6:
                break
        return T

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
        self._last_step = float(np.abs(dx).max())
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
        s, m, nr = self._gather(T, 0.008, max_pts=20_000)
        if len(s) < 30:
            return math.inf, math.inf
        ts, rs = [], []
        for _ in range(n):
            idx = self.rng.integers(0, len(s), len(s))
            Tb = np.eye(4)
            for _k in range(5):
                ss = _apply(Tb, s[idx])
                Tb = self._step6(Tb, ss, m[idx], nr[idx]) @ Tb
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

    # ---------------------------------------------------------------- driver
    def run(self, *, bootstrap: int = 20, keep_seeds: int = 6) -> ArmShapeResult:
        seeds = self.coarse(keep=keep_seeds)
        if not seeds:
            raise ValueError("no seed: no scene points above the table near any arm hypothesis")
        refined = []
        for sc, x, y, yaw in seeds:
            x2, y2, yaw2 = self.icp_planar(x, y, yaw)
            T = self.pose(x2, y2, yaw2)
            rms, cov, nin = self.metrics(T)
            refined.append({"coarse_score": sc, "x": x2, "y": y2, "yaw_deg": math.degrees(yaw2), "rms_mm": rms, "coverage": cov, "T": T})
        refined.sort(key=lambda r: (-(r["coverage"] >= 0.3), r["rms_mm"] / max(r["coverage"], 1e-3)))
        best = refined[0]
        T = self.icp_6dof(best["T"])
        rms, cov, nin = self.metrics(T)
        # plane agreement
        h6 = T[2, 3] - self.table_z
        dh = abs(h6 - self.plane.height_m) * 1000
        up6 = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        dt = math.degrees(math.acos(float(np.clip(up6 @ self.plane.normal_up, -1, 1))))
        tip = self.tip_error_px(T)
        ut, ur = self.bootstrap(T, n=bootstrap) if bootstrap else (math.nan, math.nan)
        fails = []
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
        return ArmShapeResult(T, rms, cov, nin, dh, dt, tip, ut, ur, seeds_json, fails)

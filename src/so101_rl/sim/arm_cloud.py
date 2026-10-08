"""The arm's surface as a point cloud at given joint angles (camera parity C.2, arm-shape registration).

:class:`ArmSurface` samples the arm's VISUAL meshes (group 2, every body from ``base`` down, the shape the camera
sees) once, area-weighted, in each geom's own frame; :meth:`ArmSurface.points` places them with MuJoCo FK
(``mj_kinematics`` on the given ``qpos``) and returns base-frame points and outward normals.
:func:`qpos_from_ticks` turns probed servo ticks into the scene's qpos with the same conversion as ``MujocoBus``
(``ticks_to_rad``, the gripper through ``gripper_map``).
"""

from __future__ import annotations

import mujoco
import numpy as np

from so101_rl.config import JOINT_NAMES, MotorCalibration, ticks_to_rad
from so101_rl.sim.gripper_map import gripper_ticks_to_rad

__all__ = ["ArmSurface", "qpos_from_ticks"]


def qpos_from_ticks(model: mujoco.MjModel, ticks: np.ndarray, calibration: dict[str, MotorCalibration], base_qpos: np.ndarray | None = None) -> np.ndarray:
    """Scene qpos with the six joints set from Present_Position ``ticks`` (JOINT_NAMES order)."""
    q = ticks_to_rad(np.asarray(ticks), calibration)
    q[len(JOINT_NAMES) - 1] = gripper_ticks_to_rad(float(ticks[-1]), calibration)
    qpos = np.array(model.qpos0 if base_qpos is None else base_qpos, dtype=np.float64)
    for j, n in enumerate(JOINT_NAMES):
        qpos[model.jnt_qposadr[model.joint(n).id]] = q[j]
    return qpos


def _descends_from(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body > 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


class ArmSurface:
    """Area-weighted surface samples of the arm's visual meshes (module docstring)."""

    def __init__(self, model: mujoco.MjModel, *, spacing_m: float = 0.003, seed: int = 0) -> None:
        self.model = model
        base = model.body("base").id
        rng = np.random.default_rng(seed)
        gids, pts, nrm = [], [], []
        for g in range(model.ngeom):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 2:
                continue
            if not _descends_from(model, int(model.geom_bodyid[g]), base):
                continue
            m = int(model.geom_dataid[g])
            v0, nv = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
            f0, nf = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
            V = model.mesh_vert[v0 : v0 + nv].astype(np.float64)
            F = model.mesh_face[f0 : f0 + nf]
            a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            cr = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(cr, axis=1)
            total = float(area.sum())
            k = max(8, round(total / spacing_m**2))
            idx = rng.choice(len(F), size=k, p=area / total)
            r1, r2 = rng.uniform(size=(2, k))
            s = np.sqrt(r1)
            p = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
            n = cr[idx] / np.maximum(np.linalg.norm(cr[idx], axis=1, keepdims=True), 1e-12)
            gids.append(np.full(k, g))
            pts.append(p)
            nrm.append(n)
        if not pts:
            raise ValueError("no visual arm meshes found")
        self.geom = np.concatenate(gids)
        self._p = np.concatenate(pts)
        self._n = np.concatenate(nrm)
        self._data = mujoco.MjData(model)

    def __len__(self) -> int:
        return len(self._p)

    def points(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(points [P, 3], unit outward normals [P, 3]) in the base frame at ``qpos``."""
        d = self._data
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, d)
        R = d.geom_xmat[self.geom].reshape(-1, 3, 3)
        t = d.geom_xpos[self.geom]
        return np.einsum("pij,pj->pi", R, self._p) + t, np.einsum("pij,pj->pi", R, self._n)

    def site(self, qpos: np.ndarray, name: str) -> np.ndarray:
        d = self._data
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.model.site(name).id].copy()

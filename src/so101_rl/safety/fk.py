"""Pure-numpy forward kinematics from the extracted chain (plan F5; no placo, L7).

The chain comes from ``config/so101_kinematics.yaml`` (generated from the MJCF by ``scripts/extract_kinematics.py``).
Each arm joint is a body frame placed at ``parent_body_pos/quat`` (wxyz) relative to its parent and then rotated by
``q`` about the joint axis in the child frame (MuJoCo convention). All SO-101 axes are +z, which the loader
enforces, so a joint rotation is a plain ``Rz(q)``. The gripper joint does not move the fixed-jaw ``gripperframe``
tip, so only the 5 arm joints are used.

Joint angle zero is the LeRobot homing pose, tick ``CENTER_TICK``; ``drive_mode`` of the calibration gives the
sign. ``homing_offset`` is never applied: the servo firmware already applies it to ``Present_Position``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from so101_rl.config import CENTER_TICK, JOINT_NAMES, TICKS_PER_REV, MotorCalibration, WorkspaceBox

N_ARM_JOINTS = 5
_RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_REV


def _quat_wxyz_to_matrix(q: list[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero quaternion in kinematic chain")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform(pos: list[float], quat: list[float]) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _quat_wxyz_to_matrix(quat)
    t[:3, 3] = np.asarray(pos, dtype=np.float64)
    return t


@dataclass(frozen=True, eq=False)
class KinematicChain:
    """Constant transforms of the chain. ``parent_T[i]`` places joint i's body in its parent's frame."""

    base_T: np.ndarray  # [4,4]
    parent_T: np.ndarray  # [5,4,4] shoulder_pan .. wrist_roll
    tip_T: np.ndarray  # [4,4] gripperframe in the wrist_roll (gripper) body frame
    joint_names: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> KinematicChain:
        data = yaml.safe_load(Path(path).read_text())
        joints = data["joints"]
        names = tuple(j["name"] for j in joints)
        if names != JOINT_NAMES:
            raise ValueError(f"kinematics yaml joints {names} != {JOINT_NAMES}")
        for j in joints:
            if [float(a) for a in j["axis"]] != [0.0, 0.0, 1.0]:
                raise ValueError(f"joint {j['name']}: only +z joint axes are supported, got {j['axis']}")
        base = _transform(data["base"]["pos"], data["base"]["quat"])
        parent = np.stack([_transform(j["parent_body_pos"], j["parent_body_quat"]) for j in joints[:N_ARM_JOINTS]])
        tip = _transform(data["gripper_tip"]["pos"], data["gripper_tip"]["quat"])
        return cls(base_T=base, parent_T=parent, tip_T=tip, joint_names=names)


class So101FK:
    """Allocation-free FK with preallocated 4x4 buffers. Not thread-safe (owns scratch buffers)."""

    def __init__(self, chain: KinematicChain, calibration: dict[str, MotorCalibration] | None = None) -> None:
        self._chain = chain
        self._signs = np.array(
            [(-1.0 if calibration[n].drive_mode else 1.0) if calibration else 1.0 for n in JOINT_NAMES]
        )
        self._a = np.empty((4, 4))
        self._b = np.empty((4, 4))
        self._rz = np.eye(4)
        self._q = np.empty(len(JOINT_NAMES))
        self._tip_pos = chain.tip_T[:3, 3].copy()

    def ticks_to_rad(self, ticks: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Present_Position ticks -> joint angles [rad] in ``out`` (same convention as ``config.ticks_to_rad``)."""
        np.subtract(ticks, CENTER_TICK, out=out)
        np.multiply(out, self._signs, out=out)
        np.multiply(out, _RAD_PER_TICK, out=out)
        return out

    def gripper_xyz(self, q_rad: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Fixed-jaw tip position in the base frame, metres, written into ``out`` (float64[3]); uses q_rad[:5]."""
        a, b, rz = self._a, self._b, self._rz
        a[...] = self._chain.base_T
        for i in range(N_ARM_JOINTS):
            np.matmul(a, self._chain.parent_T[i], out=b)
            c = math.cos(q_rad[i])
            s = math.sin(q_rad[i])
            rz[0, 0] = c
            rz[0, 1] = -s
            rz[1, 0] = s
            rz[1, 1] = c
            np.matmul(b, rz, out=a)
        np.matmul(a[:3, :3], self._tip_pos, out=out)
        np.add(out, a[:3, 3], out=out)
        return out

    def gripper_xyz_from_ticks(self, ticks: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Convenience for the safety pipeline: ticks -> rad -> tip xyz, no allocation."""
        return self.gripper_xyz(self.ticks_to_rad(ticks, self._q), out)

    @staticmethod
    def in_box(xyz: np.ndarray, box: WorkspaceBox) -> bool:
        """Inclusive box test. NaN is never inside."""
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        return bool(box.x[0] <= x <= box.x[1] and box.y[0] <= y <= box.y[1] and box.z[0] <= z <= box.z[1])

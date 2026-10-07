"""Pure-numpy forward kinematics from the extracted chain (plan F5; no placo, L7).

The chain comes from ``config/so101_kinematics.yaml`` (generated from the MJCF by ``scripts/extract_kinematics.py``).
Each arm joint is a body frame placed at ``parent_body_pos/quat`` (wxyz) relative to its parent and then rotated by
``q`` about the joint axis in the child frame (MuJoCo convention). All SO-101 axes are +z, which the loader
enforces, so a joint rotation is a plain ``Rz(q)``. The gripper joint does not move the fixed-jaw ``gripperframe``
tip, so only the 5 arm joints are used.

Joint angle zero is the LeRobot homing pose, tick ``CENTER_TICK``; ``drive_mode`` of the calibration gives the
sign. ``homing_offset`` is never applied: the servo firmware already applies it to ``Present_Position``.

**Gripper floor** (plan §10 2026-10-06 "Gripper floor check", 2026-10-07 "Gripper floor clearance"): besides the
tip box, the LOWEST point of the wrist and gripper must stay at or above ``box.gripper_floor_z_m`` (table + 1 mm).
``floor_points`` of the yaml are fixed points in the ``wrist`` body frame (after wrist_flex) and the ``gripper`` body
frame (after wrist_roll), derived from the MJCF collision meshes by ``scripts/extract_kinematics.py`` (the moving jaw
swept over its whole range, so the check never depends on the gripper reading; in any orientation the lowest point
is at most ``tolerance_m`` = 0.5 mm above the lowest mesh vertex, tests/test_gripper_floor.py).
:meth:`So101FK.tip_and_floor_z` returns the tip AND ``floor_z`` = the lowest point's z; :meth:`So101FK.in_box_gripper`
applies the unchanged tip box (x, y, z incl. the tip floor z[0] = -0.015) to the tip and ``gripper_floor_z_m`` to
``floor_z``. ``sim/batched_servo.py`` implements the same arithmetic (point z = ((x * r20 + y * r21) + z * r22) + t2,
elementwise in float64).
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
    floor_wrist: np.ndarray  # [Kw,3] gripper-floor points in the wrist body frame (after joint 4, wrist_flex)
    floor_gripper: np.ndarray  # [Kg,3] gripper-floor points in the gripper body frame (after joint 5, wrist_roll)

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
        fp = data.get("floor_points")
        if not isinstance(fp, dict):
            raise ValueError(  # noqa: TRY004 - a missing yaml section is a value error by contract
                "kinematics yaml: 'floor_points' missing (regenerate with scripts/extract_kinematics.py)"
            )
        pts = {}
        for body in ("wrist", "gripper"):
            a = np.asarray(fp.get(body), dtype=np.float64)
            if a.ndim != 2 or a.shape[1] != 3 or len(a) == 0 or not np.isfinite(a).all():
                raise ValueError(f"kinematics yaml: floor_points.{body} must be a non-empty list of finite [x, y, z]")
            pts[body] = a
        return cls(
            base_T=base,
            parent_T=parent,
            tip_T=tip,
            joint_names=names,
            floor_wrist=pts["wrist"],
            floor_gripper=pts["gripper"],
        )


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
        self._aw = np.empty((4, 4))  # wrist body frame of the last FK
        self._pw = np.ascontiguousarray(chain.floor_wrist.T)  # [3, Kw]
        self._pg = np.ascontiguousarray(chain.floor_gripper.T)  # [3, Kg]
        self._zw = np.empty(self._pw.shape[1])
        self._zg = np.empty(self._pg.shape[1])
        self._zt = np.empty(max(self._pw.shape[1], self._pg.shape[1]))

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
            if i == N_ARM_JOINTS - 2:
                self._aw[...] = a  # the wrist body frame (gripper-floor points)
        np.matmul(a[:3, :3], self._tip_pos, out=out)
        np.add(out, a[:3, 3], out=out)
        return out

    def _points_min_z(self, frame: np.ndarray, pts: np.ndarray, z: np.ndarray) -> float:
        """min over points of ((x * r20 + y * r21) + z * r22) + t2 (elementwise float64, the batched chain's order)."""
        tmp = self._zt[: z.shape[0]]
        np.multiply(pts[0], frame[2, 0], out=z)
        np.multiply(pts[1], frame[2, 1], out=tmp)
        np.add(z, tmp, out=z)
        np.multiply(pts[2], frame[2, 2], out=tmp)
        np.add(z, tmp, out=z)
        np.add(z, frame[2, 3], out=z)
        return float(z.min())

    def tip_and_floor_z(self, q_rad: np.ndarray, out: np.ndarray) -> float:
        """Tip xyz into ``out`` (as :meth:`gripper_xyz`) and return ``floor_z`` = the lowest z of every gripper-floor
        point (wrist and gripper bodies), metres, base frame. NaN propagates (never inside)."""
        self.gripper_xyz(q_rad, out)
        zw = self._points_min_z(self._aw, self._pw, self._zw)
        zg = self._points_min_z(self._a, self._pg, self._zg)
        return float("nan") if np.isnan(zw) or np.isnan(zg) else min(zw, zg)

    def tip_and_floor_z_from_ticks(self, ticks: np.ndarray, out: np.ndarray) -> float:
        """ticks -> rad -> (tip xyz in ``out``, returned floor_z); no allocation."""
        return self.tip_and_floor_z(self.ticks_to_rad(ticks, self._q), out)

    def gripper_xyz_from_ticks(self, ticks: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Convenience for the safety pipeline: ticks -> rad -> tip xyz, no allocation."""
        return self.gripper_xyz(self.ticks_to_rad(ticks, self._q), out)

    @staticmethod
    def in_box(xyz: np.ndarray, box: WorkspaceBox, *, margin: float = 0.0) -> bool:
        """Inclusive box test. NaN is never inside. ``margin`` > 0 grows the box on every face (the present-position
        tolerance of review S3), ``margin`` < 0 shrinks it (the loader's "home well inside the box" check)."""
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        m = float(margin)
        return bool(
            box.x[0] - m <= x <= box.x[1] + m and box.y[0] - m <= y <= box.y[1] + m and box.z[0] - m <= z <= box.z[1] + m
        )

    @staticmethod
    def in_box_gripper(
        xyz: np.ndarray, floor_z: float, box: WorkspaceBox, *, margin: float = 0.0, point_margin: float = 0.0
    ) -> bool:
        """The safety box test (pipeline BOX / BOX_PRESENT, batched chain): the TIP inside the box (every face grown
        by ``margin``, as :meth:`in_box`) AND ``floor_z`` (lowest gripper point, :meth:`tip_and_floor_z`) at or above
        ``box.gripper_floor_z_m - point_margin``. NaN is never inside. Targets: both margins 0; present:
        ``box.present_tolerance_m`` / ``box.gripper_present_tolerance_m``."""
        x, y, z, zf = float(xyz[0]), float(xyz[1]), float(xyz[2]), float(floor_z)
        m = float(margin)
        return bool(
            box.x[0] - m <= x <= box.x[1] + m
            and box.y[0] - m <= y <= box.y[1] + m
            and box.z[0] - m <= z <= box.z[1] + m
            and box.gripper_floor_z_m - float(point_margin) <= zf
        )

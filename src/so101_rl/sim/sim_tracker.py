"""Simulated D435 die tracker for N worlds (FS6, sim track S1.2): the sim twin of ``HsvDepthDieTracker``.

The camera is the real one: :class:`CameraModel` comes from ``calib/camera.yaml`` (camera parity C.3; the oblique
mount, intrinsics from the device), so FOV and occlusion are computed from the real viewpoint, from the colour AND
the depth sensor (14.8 mm apart). Per episode the true camera pose is drawn around the calibrated one (DR on the
camera pose, :class:`TrackerNoise`), and red/orange distractors can replace the die (:class:`DistractorModel`, C.5).

Semantics pinned to the real tracker (``perception/die_tracker.py``, which returns ``None`` = UNKNOWN for no
blob, bad depth or an occluding nearer object):

* A camera frame is either a MEASUREMENT or nothing. Nothing when the die is outside the camera FOV, off the
  table region or below the table, occluded by the gripper, or dropped (per-frame dropout bursts: depth holes,
  missed blobs).
* The policy sees the frame ``latency`` frames old (a per-world FIFO; latency drawn per episode).
* ``visible`` = the emitted frame was a measurement; while invisible the output HOLDS the last-known position and
  ``age_frames`` grows (0 when visible; a large value until the first sighting after a reset, zeros position).
  ``tasks.success_*`` never fires while invisible (UNKNOWN is never SUCCESS, L11) - that is the env's job.
* Occlusion: the die's projected disc (radius ``DIE_DISC_RADIUS_M``, 32 fixed sample points) is covered by the
  occluder boxes that are NEARER to the camera than the die (ray cast from the true colour and depth camera
  centres); invisible when the covered fraction exceeds ``occlusion_fraction``, OR when the gripper tip lies within
  ``occlusion_radius_m`` of the camera's ray to the die, in front of it (the cheap fallback; overhead it was "the
  tip above the die"). ALSO invisible when more than ``RING_OCCLUDED_FRACTION`` (15 %) of a ring around the disc
  (``RING_PX`` pixels outside its edge at the die's depth) sees an occluder box at least ``RING_DEPTH_M`` (2 cm)
  nearer than the die (the ring's middle: the blob's equivalent radius plus half the band): the real tracker's ring test (``HsvDepthDieTracker._occluded``, a 7x7 dilation ring and a
  2 cm depth window), which rejects a die sitting in a recess such as the start-well tray from the oblique camera
  even when the die itself is unobstructed (camera parity, plan section 10 2026-10-16).
* Noise per frame: Gaussian in the CAMERA frame (``sigma_xy_m`` across, ``sigma_z_m`` along the optical axis), then
  a per-episode camera-pose error: the true camera is the calibrated one rotated about its centre (up to
  ``rot_bias_max_rad``) and shifted (up to ``bias_max_m``); visibility uses the true pose, the reported position
  the calibrated one, which is the same as a rotation + offset of the measurement.
* False positives (C.5): the real tracker reports the largest blob in its area band; a distractor larger than the
  die's projected blob is reported instead (visible, at the distractor's position).

The reported position is the die CENTRE (FS3 ``die_xyz``: "last-known die centroid"). The real tracker
deprojects the median depth of the TOP face (~8 mm above the centre) and converts it to the centroid with
``HsvDepthDieTracker.centroid_base`` before it is published in the object table (integration 2026-10-06;
``tests/test_die_convention.py`` compares both trackers on rendered scenes).

Occluder boxes are passed as ``jaw_centers`` / ``jaw_half`` ``[N, M, 3]`` (FS6 names) plus the optional
``jaw_rot [N, M, 3, 3]`` (an additive keyword: without it the boxes are base-frame axis-aligned).
:class:`Occluders` computes them from the bodies' poses (``WarpSim.body_xpos`` / ``body_xquat``): the four jaw
boxes AND the local bounding boxes of the arm links' visual meshes. Deviation from the plan text (jaws only): on
rendered frames the forearm / wrist hid the die from the real tracker in 3 of 20 configurations the jaw-only rule
called visible (agreement 17/20 < 18/20); with the links 19/20 (the miss: a link bounding box larger than
the mesh, conservative), tests/test_sim_tracker.py.

All per-frame work is batched torch on the tracker's device; nothing synchronises with the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import torch

from so101_rl.perception.camera_calib import CameraCalib, load_camera_calib
from so101_rl.sim.model import _JAW_BOXES, DIE_EDGE_M, TABLE_Z_M

__all__ = [
    "DIE_DISC_RADIUS_M",
    "NEVER_SEEN_AGE",
    "CameraModel",
    "DistractorModel",
    "Occluders",
    "SimDieTracker",
    "TrackerNoise",
    "TrackerOut",
]

DIE_DISC_RADIUS_M: Final[float] = DIE_EDGE_M / 2  # 8 mm
# the real tracker's ring test (perception/die_tracker.py: _RING_KERNEL 7x7 -> 3 px, _OCCLUDED_RING_FRACTION, the
# default depth_window_m); tests/test_die_convention.py asserts they match
RING_PX: Final[float] = 3.0
RING_OCCLUDED_FRACTION: Final[float] = 0.15
RING_DEPTH_M: Final[float] = 0.02
NEVER_SEEN_AGE: Final[int] = 10_000  # age_frames until the first sighting after a reset
TABLE_HALF_M: Final[float] = 0.6  # scene table plane half size (so101_rl_scene.xml)
_N_DISC: Final[int] = 32


@dataclass(frozen=True, eq=False)
class CameraModel:
    """One D435 stream as mounted. ``T_base_cam`` maps camera-frame points (RealSense/OpenCV optical frame: x
    right, y down, z forward) to base. Built from ``calib/camera.yaml`` only (:meth:`from_calib`, :meth:`default`;
    plan C.1: no hard-coded intrinsics or pose)."""

    T_base_cam: np.ndarray  # 4x4
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    frame_hz: float = 30.0

    @classmethod
    def from_calib(cls, calib: CameraCalib, stream: str = "color") -> CameraModel:
        if stream == "color":
            T, s = calib.T_base_color, calib.color
        elif stream == "depth":
            T, s = calib.T_base_depth, calib.depth
        else:
            raise ValueError(f"stream must be 'color' or 'depth', got {stream!r}")
        return cls(T_base_cam=np.array(T), fx=s.fx, fy=s.fy, cx=s.cx, cy=s.cy, width=s.width, height=s.height,
                   frame_hz=float(s.fps))  # fmt: skip

    @classmethod
    def default(cls, path: Path | None = None, stream: str = "color") -> CameraModel:
        """The stream of ``calib/camera.yaml`` (or ``path``)."""
        return cls.from_calib(load_camera_calib(path), stream)

    def project(self, p_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """numpy: base ``[..., 3]`` -> (pixels ``[..., 2]``, depth along the optical axis ``[...]``)."""
        R, t = self.T_base_cam[:3, :3], self.T_base_cam[:3, 3]
        pc = (np.asarray(p_base, dtype=np.float64) - t) @ R
        z = pc[..., 2]
        return np.stack([self.fx * pc[..., 0] / z + self.cx, self.fy * pc[..., 1] / z + self.cy], axis=-1), z


@dataclass(frozen=True)
class TrackerNoise:
    sigma_xy_m: float = 0.0015
    sigma_z_m: float = 0.003
    # Per-episode camera-pose error (DR on the camera pose, plan C.3): the TRUE camera is the calibrated one moved
    # by a rotation about its centre of up to rot_bias_max_rad and a translation of up to bias_max_m; visibility
    # and occlusion use the true pose, the reported position is converted with the calibrated one. Floor +-5 mm /
    # +-1.5 deg until H8; ``with_camera_dr(calib)`` widens it to the calibration uncertainty.
    bias_max_m: float = 0.005
    rot_bias_max_rad: float = math.radians(1.5)
    latency_frames: tuple[int, int] = (1, 3)  # per-episode integer draw (inclusive)
    dropout_p: float = 0.02
    dropout_burst_frames: tuple[int, int] = (1, 5)
    # tip-to-camera-ray distance below which a tip in front of the die covers it. 0 = off (default since camera
    # parity C.4: with the oblique camera the tip site, between the fingers, is often near the ray without hiding
    # the die; the ray-cast jaw / link boxes model the gripper)
    occlusion_radius_m: float = 0.0
    occlusion_fraction: float = 0.5  # covered fraction of the projected die disc above which the die is invisible

    def __post_init__(self) -> None:
        lo, hi = self.latency_frames
        if not 0 <= lo <= hi:
            raise ValueError(f"latency_frames must be 0 <= lo <= hi, got {self.latency_frames}")
        b0, b1 = self.dropout_burst_frames
        if not 1 <= b0 <= b1:
            raise ValueError(f"dropout_burst_frames must be 1 <= lo <= hi, got {self.dropout_burst_frames}")
        if not 0.0 <= self.dropout_p <= 1.0 or not 0.0 <= self.occlusion_fraction <= 1.0:
            raise ValueError("dropout_p and occlusion_fraction must be in [0, 1]")
        for k in ("sigma_xy_m", "sigma_z_m", "bias_max_m", "rot_bias_max_rad", "occlusion_radius_m"):
            if not getattr(self, k) >= 0.0:
                raise ValueError(f"{k} must be >= 0")

    def with_camera_dr(self, calib: CameraCalib) -> TrackerNoise:
        """This noise with the camera-pose DR range of ``calib`` (``CameraCalib.dr_ranges``), never narrower than
        the current one."""
        t, r = calib.dr_ranges()
        return replace(self, bias_max_m=max(self.bias_max_m, t), rot_bias_max_rad=max(self.rot_bias_max_rad, r))


NO_NOISE: Final[TrackerNoise] = TrackerNoise(
    sigma_xy_m=0.0, sigma_z_m=0.0, bias_max_m=0.0, rot_bias_max_rad=0.0, latency_frames=(0, 0), dropout_p=0.0
)
DEFAULT_MIN_AREA_PX: Final[float] = 40.0  # the real tracker's area band (perception/hsv_red_die.yaml, test-pinned)
DEFAULT_MAX_AREA_PX: Final[float] = 4000.0
DISTRACTORS_YAML: Final[Path] = Path(__file__).resolve().parents[3] / "calib" / "distractors.yaml"
DISTRACTORS_VERSION: Final[int] = 1


@dataclass(frozen=True, eq=False)
class DistractorModel:
    """Red / orange things the real HSV tracker can take for the die (plan C.5).

    The real tracker reports the LARGEST blob whose area lies in ``[min_area_px, max_area_px]``
    (``HsvDepthDieTracker.detect``). So a distractor in view whose blob is larger than the die's replaces the die:
    a false positive at the distractor's position, reported as visible, also while the die itself is hidden.

    * ``fixed_*``: the distractors measured in the real colour frame (``camera-distractors`` ->
      ``calib/distractors.yaml``: the servo cables, the orange tip marker). Static in the base frame (cables that
      move with the arm are modelled where they were measured).
    * ``random_*``: train-time DR. Per episode, with probability ``random_p``, ``random_count`` red blobs of
      ``random_area_px`` on the table inside ``random_x`` x ``random_y``.
    """

    fixed_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # [K, 3] base frame
    fixed_area_px: np.ndarray = field(default_factory=lambda: np.zeros(0))  # [K]
    random_p: float = 0.0
    random_count: tuple[int, int] = (1, 2)
    random_area_px: tuple[float, float] = (40.0, 900.0)
    random_x: tuple[float, float] = (0.10, 0.32)
    random_y: tuple[float, float] = (-0.20, 0.20)
    min_area_px: float = DEFAULT_MIN_AREA_PX  # the real tracker's band (perception/hsv_red_die.yaml)
    max_area_px: float = DEFAULT_MAX_AREA_PX

    def __post_init__(self) -> None:
        xyz = np.asarray(self.fixed_xyz, dtype=np.float64).reshape(-1, 3)
        area = np.asarray(self.fixed_area_px, dtype=np.float64).reshape(-1)
        if xyz.shape[0] != area.shape[0]:
            raise ValueError("fixed_xyz and fixed_area_px must have the same length")
        object.__setattr__(self, "fixed_xyz", xyz)
        object.__setattr__(self, "fixed_area_px", area)
        if not 0.0 <= self.random_p <= 1.0:
            raise ValueError("random_p must be in [0, 1]")
        if not 0 <= self.random_count[0] <= self.random_count[1]:
            raise ValueError("random_count must be 0 <= lo <= hi")
        if not 0 < self.min_area_px < self.max_area_px:
            raise ValueError("need 0 < min_area_px < max_area_px")

    @property
    def max_random(self) -> int:
        return int(self.random_count[1]) if self.random_p > 0 else 0

    @classmethod
    def from_yaml(cls, path: Path | None = None, **dr: float | tuple) -> DistractorModel:
        """The measured distractors of ``calib/distractors.yaml`` (schema: ``calib/README.md``) plus the DR
        settings ``dr`` (keyword arguments of this class). Band from ``perception/hsv_red_die.yaml``."""
        import yaml

        p = DISTRACTORS_YAML if path is None else Path(path)
        doc = yaml.safe_load(p.read_text())
        if not isinstance(doc, dict) or doc.get("version") != DISTRACTORS_VERSION:
            raise ValueError(f"{p}: expected version {DISTRACTORS_VERSION}")
        want = {"version", "date", "source", "camera_pose_method", "hsv", "distractors"}
        if set(doc) != want:
            raise ValueError(f"{p}: keys must be exactly {sorted(want)}, got {sorted(doc)}")
        items = doc["distractors"] or []
        for i, d in enumerate(items):
            keys = {"label", "u", "v", "area_px", "xyz_base", "xyz_source"}
            if not isinstance(d, dict) or set(d) != keys:
                raise ValueError(f"{p}: distractors[{i}] keys must be exactly {sorted(keys)}")
        hsv = doc["hsv"]
        return cls(
            fixed_xyz=np.array([d["xyz_base"] for d in items], dtype=np.float64).reshape(-1, 3),
            fixed_area_px=np.array([d["area_px"] for d in items], dtype=np.float64),
            min_area_px=float(hsv["min_area_px"]),
            max_area_px=float(hsv["max_area_px"]),
            **dr,
        )


@dataclass
class TrackerOut:
    xyz: torch.Tensor  # float32[N,3] last-known base-frame position (held while invisible)
    visible: torch.Tensor  # bool[N]  seen in THIS policy step's latest frame
    age_frames: torch.Tensor  # int64[N] frames since last seen (0 when visible)
    confidence: torch.Tensor  # float32[N]

    @classmethod
    def empty(cls, nworld: int, device: torch.device | str) -> TrackerOut:
        return cls(
            xyz=torch.zeros((nworld, 3), dtype=torch.float32, device=device),
            visible=torch.zeros(nworld, dtype=torch.bool, device=device),
            age_frames=torch.full((nworld,), NEVER_SEEN_AGE, dtype=torch.int64, device=device),
            confidence=torch.zeros(nworld, dtype=torch.float32, device=device),
        )


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """wxyz [..., 4] -> rotation matrices [..., 3, 3]."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)  # fmt: skip


class Occluders:
    """Boxes that can hide the die from the D435, as oriented boxes from their bodies' poses (batched).

    The four ``jaw_col`` boxes (exact), plus (``include_links``) the local bounding box of every visual mesh of the
    arm bodies (base .. moving jaw), plus (``include_static``) the static scene boxes above the table (the start-well
    tray). The links matter: the forearm and wrist hide the die as often as the jaws do (seen on rendered frames,
    S1.2); the base and the tray rim matter because the real camera is oblique (camera parity C.3)."""

    def __init__(
                self,
        model: mujoco.MjModel,
        device: torch.device | str,
        *,
        include_links: bool = True,
        include_static: bool = True,
        slabs: int = 3,
    ) -> None:
        dev = torch.device(device)
        jaw = [model.geom(n).id for n, *_ in _JAW_BOXES]
        gids = list(jaw)
        if include_links:
            base = model.body("base").id
            arm_bodies = {b for b in range(model.nbody) if _descends_from(model, b, base)}
            gids += [
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] in arm_bodies
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and model.geom_contype[g] == 0
                and model.geom_conaffinity[g] == 0
                and model.geom_group[g] == 2
            ]
        if include_static:
            tray = model.body("tray").id
            gids += [g for g in range(model.ngeom) if model.geom_bodyid[g] == tray and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        # boxes in each geom's frame: primitives exactly; a mesh as ``slabs`` boxes, the AABBs of its vertices cut
        # into equal slabs along the mesh's longest axis (one AABB per mesh is too loose for an oblique camera: the
        # L-shaped gripper body's box hid dice the real tracker saw, camera parity C.4)
        box_g, box_c, box_h = [], [], []
        for g in gids:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and slabs > 1:
                m = int(model.geom_dataid[g])
                v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]].astype(np.float64)
                lo, hi = v.min(0), v.max(0)
                ax = int(np.argmax(hi - lo))
                edges = np.linspace(lo[ax], hi[ax], slabs + 1)
                for k in range(slabs):
                    sel = (v[:, ax] >= edges[k]) & (v[:, ax] <= edges[k + 1])
                    if sel.sum() < 3:
                        continue
                    a, b = v[sel].min(0), v[sel].max(0)
                    box_g.append(g)
                    box_c.append((a + b) / 2)
                    box_h.append(np.maximum((b - a) / 2, 1e-4))
            else:
                box_g.append(g)
                box_c.append(model.geom_aabb[g][:3])
                box_h.append(model.geom_aabb[g][3:])
        self.geom_ids = tuple(int(g) for g in box_g)  # one entry per box (a mesh may own several)
        bodies = [int(model.geom_bodyid[g]) for g in box_g]
        self.bodies: tuple[str, ...] = tuple(dict.fromkeys(model.body(b).name for b in bodies))
        self._body_idx = torch.tensor([self.bodies.index(model.body(b).name) for b in bodies], device=dev)
        gpos = np.stack([model.geom_pos[g] for g in box_g])
        gquat = np.stack([model.geom_quat[g] for g in box_g])
        self._grot = _quat_to_mat(torch.as_tensor(gquat, dtype=torch.float32, device=dev))  # [M,3,3]
        self._lpos = torch.as_tensor(gpos, dtype=torch.float32, device=dev) + torch.einsum(
            "mij,mj->mi", self._grot, torch.as_tensor(np.stack(box_c), dtype=torch.float32, device=dev)
        )
        self._half = torch.as_tensor(np.stack(box_h), dtype=torch.float32, device=dev)

    @property
    def n(self) -> int:
        return len(self.geom_ids)

    def compute(
        self, body_xpos: torch.Tensor, body_xquat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``body_xpos [N, B, 3]`` / ``body_xquat [N, B, 4]`` in ``self.bodies`` order -> (centres [N,M,3], half
        sizes in the box frame [N,M,3], box rotations [N,M,3,3])."""
        p = body_xpos[:, self._body_idx].float()
        R = _quat_to_mat(body_xquat[:, self._body_idx].float())  # [N,M,3,3]
        centers = p + torch.einsum("nkij,kj->nki", R, self._lpos)
        rot = torch.matmul(R, self._grot)
        return centers, self._half.expand(p.shape[0], -1, -1), rot


def _descends_from(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body > 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


def _disc_points() -> torch.Tensor:
    """32 sunflower points filling the unit disc (deterministic)."""
    k = torch.arange(_N_DISC, dtype=torch.float64) + 0.5
    r = torch.sqrt(k / _N_DISC)
    th = k * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=-1).float()


def _small_rotations(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues for [N,3] axis-angle vectors -> [N,3,3]."""
    theta = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = axis_angle / theta
    K = torch.zeros((*axis_angle.shape[:-1], 3, 3), dtype=axis_angle.dtype, device=axis_angle.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    t = theta[..., None]
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(K)
    return eye + torch.sin(t) * K + (1 - torch.cos(t)) * (K @ K)


class SimDieTracker:
    """See the module docstring. One instance per batched env; per-world state, seeded torch generator.

    ``cam`` is the colour stream (blobs are found in colour; the real pipeline aligns depth to colour), ``depth_cam``
    the depth stream (occlusion and field of view are checked from both sensors when given), ``distractors`` the
    false-positive model (C.5)."""

    def __init__(
        self,
        cam: CameraModel,
        noise: TrackerNoise,
        nworld: int,
        *,
        device: torch.device | str,
        seed: int,
        depth_cam: CameraModel | None = None,
        distractors: DistractorModel | None = None,
    ) -> None:
        if nworld < 1:
            raise ValueError("nworld must be >= 1")
        self.cam, self.noise, self.nworld = cam, noise, int(nworld)
        self.depth_cam = depth_cam
        self.distractors = distractors
        self.device = dev = torch.device(device)
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(int(seed))
        T = torch.as_tensor(cam.T_base_cam, dtype=torch.float32, device=dev)
        self._R = T[:3, :3].contiguous()  # calibrated cam -> base
        self._t = T[:3, 3].contiguous()
        self._Rcd: torch.Tensor | None = None
        if depth_cam is not None:
            T_cd = np.linalg.inv(cam.T_base_cam) @ depth_cam.T_base_cam  # depth -> colour, rigid
            self._Rcd = torch.as_tensor(T_cd[:3, :3], dtype=torch.float32, device=dev)
            self._tcd = torch.as_tensor(T_cd[:3, 3], dtype=torch.float32, device=dev)
        self._disc = _disc_points().to(dev)
        self.max_latency = int(noise.latency_frames[1])
        K = self.max_latency + 1
        n = self.nworld
        self._ring_xyz = torch.zeros((n, K, 3), dtype=torch.float32, device=dev)
        self._ring_ok = torch.zeros((n, K), dtype=torch.bool, device=dev)
        self._ring_conf = torch.zeros((n, K), dtype=torch.float32, device=dev)
        self._head = 0  # index of the newest frame (shared: all worlds advance together)
        self.latency = torch.zeros(n, dtype=torch.int64, device=dev)
        self.bias = torch.zeros((n, 3), dtype=torch.float32, device=dev)
        self.rot = torch.eye(3, dtype=torch.float32, device=dev).repeat(n, 1, 1)
        # the per-world TRUE camera poses (colour, depth) implied by rot / bias (TrackerNoise docstring)
        self.cam_R = self._R.repeat(n, 1, 1)
        self.cam_t = self._t.repeat(n, 1)
        self._burst = torch.zeros(n, dtype=torch.int64, device=dev)  # dropped frames still to come
        self._held = TrackerOut.empty(n, dev)
        self._arange = torch.arange(n, device=dev)
        self._sig_key: TrackerNoise | None = None
        self._sig = torch.zeros(3, device=dev)
        # distractors: fixed (measured) + per-episode random, [N, Kd]
        dm = distractors
        kf = 0 if dm is None else dm.fixed_xyz.shape[0]
        kr = 0 if dm is None else dm.max_random
        self._dist_xyz = torch.zeros((n, kf + kr, 3), dtype=torch.float32, device=dev)
        self._dist_area = torch.zeros((n, kf + kr), dtype=torch.float32, device=dev)
        self._dist_on = torch.zeros((n, kf + kr), dtype=torch.bool, device=dev)
        if kf:
            self._dist_xyz[:, :kf] = torch.as_tensor(dm.fixed_xyz, dtype=torch.float32, device=dev)
            self._dist_area[:, :kf] = torch.as_tensor(dm.fixed_area_px, dtype=torch.float32, device=dev)
            self._dist_on[:, :kf] = True
        self._kf = kf
        self.false_positive = torch.zeros(n, dtype=torch.bool, device=dev)  # the last emitted frame was a distractor

    # ------------------------------------------------------------------ per episode
    def reset(self, mask: torch.Tensor, die_xyz_true: torch.Tensor) -> None:
        """Redraw the per-episode camera-pose error, latency and random distractors for the worlds in ``mask``;
        their FIFO starts at "never seen" (the env then runs ``max_latency`` warm-up frames from the reset state).
        ``die_xyz_true`` is accepted for the interface (FS6) and validated; the warm-up frames do the measuring."""
        m = mask.to(device=self.device, dtype=torch.bool)
        if die_xyz_true.shape != (self.nworld, 3):
            raise ValueError(f"die_xyz_true must be [{self.nworld}, 3]")
        nz, g = self.noise, self._gen
        n = self.nworld
        lat = torch.randint(nz.latency_frames[0], nz.latency_frames[1] + 1, (n,), generator=g, device=self.device)
        self.latency.copy_(torch.where(m, lat, self.latency))
        b = torch.randn((n, 3), generator=g, device=self.device)
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        b = b * (nz.bias_max_m * torch.rand((n, 1), generator=g, device=self.device) ** (1 / 3))  # uniform in the ball
        self.bias.copy_(torch.where(m[:, None], b, self.bias))
        a = torch.randn((n, 3), generator=g, device=self.device)
        a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a = a * (nz.rot_bias_max_rad * torch.rand((n, 1), generator=g, device=self.device))
        self.rot.copy_(torch.where(m[:, None, None], _small_rotations(a), self.rot))
        rT = self.rot.transpose(-1, -2)
        self.cam_R.copy_(torch.where(m[:, None, None], rT @ self._R, self.cam_R))
        self.cam_t.copy_(torch.where(m[:, None], self._t - torch.einsum("nij,nj->ni", rT, self.bias), self.cam_t))
        dm = self.distractors
        if dm is not None and dm.max_random:
            kf, kr = self._kf, dm.max_random
            on = torch.rand(n, generator=g, device=self.device) < dm.random_p
            cnt = torch.randint(dm.random_count[0], dm.random_count[1] + 1, (n,), generator=g, device=self.device)
            slot = torch.arange(kr, device=self.device)[None, :] < cnt[:, None]
            xy = torch.rand((n, kr, 2), generator=g, device=self.device)
            x = dm.random_x[0] + xy[..., 0] * (dm.random_x[1] - dm.random_x[0])
            y = dm.random_y[0] + xy[..., 1] * (dm.random_y[1] - dm.random_y[0])
            xyz = torch.stack([x, y, torch.full_like(x, TABLE_Z_M + 0.002)], dim=-1)
            lo, hi = dm.random_area_px
            area = lo + torch.rand((n, kr), generator=g, device=self.device) * (hi - lo)
            self._dist_xyz[:, kf:] = torch.where(m[:, None, None], xyz, self._dist_xyz[:, kf:])
            self._dist_area[:, kf:] = torch.where(m[:, None], area, self._dist_area[:, kf:])
            self._dist_on[:, kf:] = torch.where(m[:, None], on[:, None] & slot, self._dist_on[:, kf:])
        self._ring_ok.masked_fill_(m[:, None], False)
        self._ring_xyz.masked_fill_(m[:, None, None], 0.0)
        self._ring_conf.masked_fill_(m[:, None], 0.0)
        self._burst.masked_fill_(m, 0)
        self.false_positive.masked_fill_(m, False)
        h = self._held
        h.xyz.masked_fill_(m[:, None], 0.0)
        h.visible.masked_fill_(m, False)
        h.age_frames.masked_fill_(m, NEVER_SEEN_AGE)
        h.confidence.masked_fill_(m, 0.0)

    def prime(self, mask: torch.Tensor, die_xyz_true: torch.Tensor, out: TrackerOut | None = None) -> None:
        """After :meth:`reset`: fill the masked worlds' whole FIFO with noisy measurements of the (resting) die,
        visible when in view, as if the camera had been watching it during the reset (the gripper is at ``home``,
        never over the start well). Other worlds are untouched (the shared FIFO head does not move)."""
        m = mask.to(device=self.device, dtype=torch.bool)
        die = die_xyz_true.float()
        ok = self.in_view(die) & m
        K = self.max_latency + 1
        for k in range(K):
            meas = self._measure(die)
            self._ring_xyz[:, k] = torch.where(m[:, None], meas, self._ring_xyz[:, k])
            self._ring_ok[:, k] = torch.where(m, ok, self._ring_ok[:, k])
            self._ring_conf[:, k] = torch.where(m, ok.float(), self._ring_conf[:, k])
        h = self._held
        idx = (self._head - self.latency) % K
        e_xyz = self._ring_xyz[self._arange, idx]
        h.xyz.copy_(torch.where(ok[:, None], e_xyz, h.xyz))
        h.visible.copy_(torch.where(m, ok, h.visible))
        h.age_frames.copy_(torch.where(ok, torch.zeros_like(h.age_frames), h.age_frames))
        h.confidence.copy_(torch.where(m, ok.float(), h.confidence))
        if out is not None:
            for a in ("xyz", "visible", "age_frames", "confidence"):
                getattr(out, a).copy_(getattr(h, a))

    def _measure(self, p_true: torch.Tensor) -> torch.Tensor:
        """One noisy measurement per world of base points ``[N, 3]``: camera-frame noise, then the per-episode
        camera-pose error (rotation about the camera centre, offset)."""
        nz = self.noise
        if self._sig_key is not nz:  # built once per noise spec: no host->device copy per frame
            self._sig = torch.tensor([nz.sigma_xy_m, nz.sigma_xy_m, nz.sigma_z_m], dtype=torch.float32, device=self.device)
            self._sig_key = nz
        sig = self._sig
        noise_cam = torch.randn((self.nworld, 3), generator=self._gen, device=self.device) * sig
        meas = p_true + noise_cam @ self._R.T
        return torch.einsum("nij,nj->ni", self.rot, meas - self._t) + self._t + self.bias

    # ------------------------------------------------------------------ geometry
    def _poses(self) -> list[tuple[torch.Tensor, torch.Tensor, CameraModel]]:
        """The per-world TRUE (R [N,3,3] cam->base, t [N,3], intrinsics) of the colour and the depth sensor."""
        out = [(self.cam_R, self.cam_t, self.cam)]
        if self.depth_cam is not None and self._Rcd is not None:
            Rd = self.cam_R @ self._Rcd
            td = self.cam_t + torch.einsum("nij,j->ni", self.cam_R, self._tcd)
            out.append((Rd, td, self.depth_cam))
        return out

    @staticmethod
    def _to_cam(p: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """base points ``[N, ..., 3]`` -> camera frame with per-world ``R [N,3,3]`` (cam->base) and ``t [N,3]``."""
        shape = (t.shape[0],) + (1,) * (p.dim() - 2) + (3,)
        d = p - t.reshape(shape)
        return torch.einsum("nij,n...i->n...j", R, d)

    def _project(self, p: torch.Tensor, pose: tuple[torch.Tensor, torch.Tensor, CameraModel] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """base [N, ..., 3] -> (uv [N, ..., 2] pixels, depth [N, ...] along the optical axis) of the true colour camera
        (or ``pose``)."""
        R, t, cm = pose if pose is not None else self._poses()[0]
        pc = self._to_cam(p, R, t)
        z = pc[..., 2]
        zs = z.clamp_min(1e-6)
        u = cm.fx * pc[..., 0] / zs + cm.cx
        v = cm.fy * pc[..., 1] / zs + cm.cy
        return torch.stack([u, v], dim=-1), z

    @staticmethod
    def _box_hits(pts, o, jaw_centers, jaw_half, jaw_rot, s_max) -> torch.Tensor:
        """bool[N,P]: the ray ``o + s (pts - o)`` enters one of the boxes at some ``s < s_max`` (slab test)."""
        n, m = jaw_centers.shape[:2]
        d = pts - o[:, None]  # ray o + s d reaches the point at s = 1
        c = jaw_centers.float()
        h = jaw_half.float()
        if jaw_rot is None:
            o_l = (o[:, None] - c)[:, None]  # [N,1,M,3]
            d_l = d[:, :, None, :].expand(n, d.shape[1], m, 3)
        else:
            Rt = jaw_rot.float().transpose(-1, -2)  # [N,M,3,3]
            o_l = torch.einsum("nmij,nmj->nmi", Rt, o[:, None] - c)[:, None]
            d_l = torch.einsum("nmij,npj->npmi", Rt, d)
        d_l = torch.where(d_l.abs() < 1e-12, torch.full_like(d_l, 1e-12), d_l)
        t1 = (-h[:, None] - o_l) / d_l
        t2 = (h[:, None] - o_l) / d_l
        t_in = torch.minimum(t1, t2).amax(dim=-1)
        t_out = torch.maximum(t1, t2).amin(dim=-1)
        if not torch.is_tensor(s_max):
            s_max = torch.tensor(float(s_max), device=pts.device)
        hit = (t_out >= t_in.clamp_min(0.0)) & (t_in < s_max[..., None])  # [N,P,M]
        return hit.any(dim=-1)

    def _disc_pts(self, die, R, radius) -> torch.Tensor:
        ax, ay = R[:, :, 0], R[:, :, 1]  # [N,3] camera x / y axes in base
        r = radius if torch.is_tensor(radius) else torch.full((die.shape[0],), float(radius), device=die.device)
        return die[:, None, :] + r[:, None, None] * (self._disc[None, :, 0:1] * ax[:, None] + self._disc[None, :, 1:2] * ay[:, None])

    def _covered_from(self, die, R, t, jaw_centers, jaw_half, jaw_rot) -> torch.Tensor:
        pts = self._disc_pts(die, R, DIE_DISC_RADIUS_M)
        return self._box_hits(pts, t, jaw_centers, jaw_half, jaw_rot, 1.0).float().mean(dim=1)

    def _ring_from(self, die, R, t, cm, jaw_centers, jaw_half, jaw_rot) -> torch.Tensor:
        z = self._to_cam(die, R, t)[:, 2].clamp_min(1e-6)  # the die's depth along this sensor's optical axis
        # the ring's middle: the blob's equivalent radius (from its projected area) plus half the 3 px band
        blob_px = (self.die_area_px(die).clamp_min(0.0) / math.pi).sqrt()
        radius = (blob_px + RING_PX / 2) * z / float(cm.fx)
        pts = self._disc_pts(die, R, radius)
        rng = (pts - t[:, None]).norm(dim=-1).clamp_min(1e-6)  # [N,P]
        return self._box_hits(pts, t, jaw_centers, jaw_half, jaw_rot, 1.0 - RING_DEPTH_M / rng).float().mean(dim=1)

    def ring_occluded_fraction(
        self,
        die_xyz_true: torch.Tensor,
        jaw_centers: torch.Tensor,
        jaw_half: torch.Tensor,
        jaw_rot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """float[N]: fraction of the ring around the die's disc (``RING_PX`` outside its edge) whose ray meets an
        occluder box at least ``RING_DEPTH_M`` before the ring point (the real ring test), the larger of the colour
        and depth sensors."""
        die = die_xyz_true.float()
        fr = [self._ring_from(die, R, t, cm, jaw_centers, jaw_half, jaw_rot) for R, t, cm in self._poses()]
        return torch.stack(fr).amax(dim=0)

    def covered_fraction(
        self,
        die_xyz_true: torch.Tensor,
        jaw_centers: torch.Tensor,
        jaw_half: torch.Tensor,
        jaw_rot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """float[N]: fraction of the die's disc (32 points, perpendicular to the optical axis) whose camera ray hits
        one of the occluder boxes BEFORE reaching the die (slab test), from the true colour camera and, with a depth
        camera, the larger of the two (the blob needs colour, its depth needs the depth sensor). Boxes: centres
        ``[N,M,3]``, half sizes ``[N,M,3]`` in the box frame, rotations ``jaw_rot [N,M,3,3]`` (None: axis-aligned in
        the base frame)."""
        die = die_xyz_true.float()
        cov = [self._covered_from(die, R, t, jaw_centers, jaw_half, jaw_rot) for R, t, _ in self._poses()]
        return torch.stack(cov).amax(dim=0)

    def _tip_in_front(self, die: torch.Tensor, tip: torch.Tensor) -> torch.Tensor:
        """bool[N]: the gripper tip within ``occlusion_radius_m`` of the colour camera's ray to the die, between the
        camera and the die (the cheap fallback; for an overhead camera: the tip above the die)."""
        o = self.cam_t
        d = die - o
        s = ((tip - o) * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-12)
        dist = (tip - (o + s[:, None] * d)).norm(dim=-1)
        return (dist < self.noise.occlusion_radius_m) & (s > 0.0) & (s < 1.0)

    def occluded(
        self,
        die_xyz_true: torch.Tensor,
        tip_xyz: torch.Tensor,
        jaw_centers: torch.Tensor,
        jaw_half: torch.Tensor,
        jaw_rot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """bool[N]; the rule the real tracker's ring test approximates (module docstring)."""
        die, tip = die_xyz_true.float(), tip_xyz.float()
        cov = self.covered_fraction(die, jaw_centers, jaw_half, jaw_rot)
        ring = self.ring_occluded_fraction(die, jaw_centers, jaw_half, jaw_rot)
        return self._tip_in_front(die, tip) | (cov > self.noise.occlusion_fraction) | (ring > RING_OCCLUDED_FRACTION)

    def _in_image(self, p: torch.Tensor, margin_px: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Projected point inside the image(s). ``margin_px`` (scalar or ``[N, ...]``) shrinks the COLOUR image on all
        four sides: the real tracker rejects a blob that touches the image border (``HsvDepthDieTracker.detect``), so
        a die is visible only while its centre is at least the projected half-blob width from the border."""
        ok = torch.ones(p.shape[:-1], dtype=torch.bool, device=self.device)
        for k, pose in enumerate(self._poses()):
            uv, z = self._project(p, pose)
            cm = pose[2]
            m = margin_px if k == 0 else 0.0
            ok &= (z > 0) & (uv[..., 0] >= m) & (uv[..., 0] <= cm.width - 1 - m) & (uv[..., 1] >= m)
            ok &= uv[..., 1] <= cm.height - 1 - m
        return ok

    def in_view(self, die_xyz_true: torch.Tensor) -> torch.Tensor:
        """bool[N]: projected centre inside the colour (and depth) image by at least the projected half-blob width
        (the real tracker drops a blob cut by the border), in front of the camera, die on the table region."""
        die = die_xyz_true.float()
        table_ok = (die[:, :2].abs() <= TABLE_HALF_M).all(dim=-1) & (die[:, 2] >= TABLE_Z_M)
        half_blob = 0.5 * self.die_area_px(die).clamp_min(0.0).sqrt()
        return self._in_image(die, half_blob) & table_ok

    def die_area_px(self, die_xyz_true: torch.Tensor) -> torch.Tensor:
        """float[N]: the die's blob area in the colour image, face-up cube axis-aligned with the base frame:
        (f / z)^2 e^2 (|d_x| + |d_y| + |d_z|) with d the unit viewing ray (exact for an orthographic view)."""
        die = die_xyz_true.float()
        d = die - self.cam_t
        dist = d.norm(dim=-1).clamp_min(1e-6)
        _, z = self._project(die)
        f2 = self.cam.fx * self.cam.fy
        return f2 / z.clamp_min(1e-6) ** 2 * DIE_EDGE_M**2 * (d.abs().sum(-1) / dist)

    # ------------------------------------------------------------------ per frame
    def update(
        self,
        die_xyz_true: torch.Tensor,
        tip_xyz: torch.Tensor,
        jaw_centers: torch.Tensor,
        jaw_half: torch.Tensor,
        frames: int,
        out: TrackerOut,
        *,
        jaw_rot: torch.Tensor | None = None,
    ) -> TrackerOut:
        """Advance ``frames`` camera frames with the given (current) state and fill ``out`` with what the policy
        sees after the last one. Per frame: visibility (FOV/table, occlusion, dropout burst), the largest blob in the
        area band (die or a distractor, C.5), a noisy measurement, push into the FIFO; the emitted frame is
        ``latency`` frames old."""
        nz, g, dev = self.noise, self._gen, self.device
        n = self.nworld
        die = die_xyz_true.float()
        covered = self.covered_fraction(die, jaw_centers, jaw_half, jaw_rot)
        ring = self.ring_occluded_fraction(die, jaw_centers, jaw_half, jaw_rot)
        base_ok = (
            self.in_view(die)
            & ~self._tip_in_front(die, tip_xyz.float())
            & ~(covered > nz.occlusion_fraction)
            & ~(ring > RING_OCCLUDED_FRACTION)
        )
        conf = (1.0 - covered).clamp(0.0, 1.0)
        target = die
        fp = torch.zeros(n, dtype=torch.bool, device=dev)
        dm = self.distractors
        lo_a, hi_a = (dm.min_area_px, dm.max_area_px) if dm is not None else (DEFAULT_MIN_AREA_PX, DEFAULT_MAX_AREA_PX)
        area_die = torch.where(base_ok, self.die_area_px(die) * conf, torch.zeros_like(conf))
        band_die = (area_die >= lo_a) & (area_die <= hi_a)  # the real tracker's area band, always
        base_ok = base_ok & band_die
        area_die = torch.where(band_die, area_die, torch.zeros_like(area_die))
        if dm is not None and self._dist_xyz.shape[1]:
            a_d = self._dist_area
            cand = self._dist_on & self._in_image(self._dist_xyz) & (a_d >= dm.min_area_px) & (a_d <= dm.max_area_px)
            a_d = torch.where(cand, a_d, torch.zeros_like(a_d))
            best, k = a_d.max(dim=1)
            fp = best > area_die
            target = torch.where(fp[:, None], self._dist_xyz[self._arange, k], die)
            base_ok = base_ok | fp
            conf = torch.where(fp, torch.ones_like(conf), conf)
        K = self.max_latency + 1
        h = self._held
        for _ in range(int(frames)):
            # dropout bursts
            start = (self._burst == 0) & (torch.rand(n, generator=g, device=dev) < nz.dropout_p)
            blen = torch.randint(
                nz.dropout_burst_frames[0], nz.dropout_burst_frames[1] + 1, (n,), generator=g, device=dev
            )
            self._burst = torch.where(start, blen, self._burst)
            dropped = self._burst > 0
            self._burst = (self._burst - 1).clamp_min(0)
            ok = base_ok & ~dropped
            meas = self._measure(target)
            self._head = (self._head + 1) % K
            self._ring_xyz[:, self._head] = meas
            self._ring_ok[:, self._head] = ok
            self._ring_conf[:, self._head] = conf
            idx = (self._head - self.latency) % K
            e_ok = self._ring_ok[self._arange, idx]
            e_xyz = self._ring_xyz[self._arange, idx]
            h.xyz.copy_(torch.where(e_ok[:, None], e_xyz, h.xyz))
            h.visible.copy_(e_ok)
            h.age_frames.copy_(torch.where(e_ok, torch.zeros_like(h.age_frames), h.age_frames + 1))
            h.confidence.copy_(torch.where(e_ok, self._ring_conf[self._arange, idx], torch.zeros_like(h.confidence)))
        self.false_positive.copy_(fp & base_ok)
        out.xyz.copy_(h.xyz)
        out.visible.copy_(h.visible)
        out.age_frames.copy_(h.age_frames)
        out.confidence.copy_(h.confidence)
        return out

"""Simulated overhead die tracker for N worlds (FS6, sim track S1.2): the sim twin of ``HsvDepthDieTracker``.

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
  projected jaw boxes that are NEARER to the camera than the die; invisible when the covered fraction exceeds
  ``occlusion_fraction``, OR when the gripper tip is above the die within ``occlusion_radius_m`` horizontally
  (the cheap fallback). The real tracker's ring test ("a nearer object touches the blob") approximates the same.
* Noise per frame: Gaussian in the CAMERA frame (``sigma_xy_m`` across, ``sigma_z_m`` along the optical axis), then
  a per-episode rigid error in the base frame: a small rotation of the camera pose (``rot_bias_max_rad``, about
  the camera centre) and a constant offset (``bias_max_m``, the hand-eye residual).

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
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import torch

from so101_rl.sim.model import _JAW_BOXES, DIE_EDGE_M, TABLE_Z_M

__all__ = [
    "DIE_DISC_RADIUS_M",
    "NEVER_SEEN_AGE",
    "CameraModel",
    "Occluders",
    "SimDieTracker",
    "TrackerNoise",
    "TrackerOut",
]

DIE_DISC_RADIUS_M: Final[float] = DIE_EDGE_M / 2  # 8 mm
NEVER_SEEN_AGE: Final[int] = 10_000  # age_frames until the first sighting after a reset
TABLE_HALF_M: Final[float] = 0.6  # scene table plane half size (so101_rl_scene.xml)
_N_DISC: Final[int] = 32
DEFAULT_HAND_EYE: Final[Path] = Path("calib/hand_eye.yaml")

# Nominal mount (no calib/hand_eye.yaml yet): 0.614 m above the table at (0.20, 0), looking straight down, image
# x = base +x, image down = base -y. Same pose as the scene's ``overhead`` camera (test-enforced).
NOMINAL_CAM_POS: Final[tuple[float, float, float]] = (0.20, 0.0, TABLE_Z_M + 0.614)
NOMINAL_F_PX: Final[float] = 615.0  # D435 colour at 640 x 480: 1 px ~ 1 mm on the table


@dataclass(frozen=True, eq=False)
class CameraModel:
    """The D435 as mounted. ``T_base_cam`` maps camera-frame points (OpenCV: x right, y down, z forward) to base."""

    T_base_cam: np.ndarray  # 4x4
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
    frame_hz: float = 30.0

    @classmethod
    def nominal(cls) -> CameraModel:
        T = np.eye(4)
        T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # columns: cam x, y, z in base
        T[:3, 3] = NOMINAL_CAM_POS
        # principal point at the image centre in pixel-centre coordinates (MuJoCo's pinhole render, OpenCV indices)
        return cls(T_base_cam=T, fx=NOMINAL_F_PX, fy=NOMINAL_F_PX, cx=319.5, cy=239.5)

    @classmethod
    def default(cls, hand_eye: Path = DEFAULT_HAND_EYE) -> CameraModel:
        """``calib/hand_eye.yaml`` (``perception.frames.load_hand_eye``) when present, else :meth:`nominal`."""
        if Path(hand_eye).exists():
            from so101_rl.perception.frames import load_hand_eye

            nom = cls.nominal()
            return cls(T_base_cam=load_hand_eye(Path(hand_eye)).T_base_cam, fx=nom.fx, fy=nom.fy, cx=nom.cx, cy=nom.cy)
        return cls.nominal()

    def mujoco_fovy_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / 2.0 / self.fy))


@dataclass(frozen=True)
class TrackerNoise:
    sigma_xy_m: float = 0.0015
    sigma_z_m: float = 0.003
    bias_max_m: float = 0.003  # per-episode constant offset (hand-eye residual <= 3 mm)
    rot_bias_max_rad: float = 0.0175  # per-episode small rigid error of the camera pose (~1 deg)
    latency_frames: tuple[int, int] = (1, 3)  # per-episode integer draw (inclusive)
    dropout_p: float = 0.02
    dropout_burst_frames: tuple[int, int] = (1, 5)
    occlusion_radius_m: float = 0.020  # tip-to-die horizontal distance below which the tip above the die covers it
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


NO_NOISE: Final[TrackerNoise] = TrackerNoise(
    sigma_xy_m=0.0, sigma_z_m=0.0, bias_max_m=0.0, rot_bias_max_rad=0.0, latency_frames=(0, 0), dropout_p=0.0
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
    """Boxes that can hide the die from the overhead camera, as oriented boxes from their bodies' poses (batched).

    The four ``jaw_col`` boxes (exact), plus (``include_links``) the local bounding box of every visual mesh of the
    moving arm bodies (shoulder .. moving jaw; the base never passes over the die). The links matter: the forearm
    and wrist hide the die from above as often as the jaws do (seen on rendered frames, S1.2)."""

    def __init__(self, model: mujoco.MjModel, device: torch.device | str, *, include_links: bool = True) -> None:
        dev = torch.device(device)
        jaw = [model.geom(n).id for n, *_ in _JAW_BOXES]
        gids = list(jaw)
        if include_links:
            base = model.body("base").id
            arm_bodies = {b for b in range(model.nbody) if b != base and _descends_from(model, b, base)}
            gids += [
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] in arm_bodies
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and model.geom_contype[g] == 0
                and model.geom_conaffinity[g] == 0
                and model.geom_group[g] == 2
            ]
        self.geom_ids = tuple(int(g) for g in gids)
        bodies = [int(model.geom_bodyid[g]) for g in gids]
        self.bodies: tuple[str, ...] = tuple(dict.fromkeys(model.body(b).name for b in bodies))
        self._body_idx = torch.tensor([self.bodies.index(model.body(b).name) for b in bodies], device=dev)
        # geom frame in body frame, then the geom-local AABB centre (mesh geoms are not centred on their frame)
        gpos = np.stack([model.geom_pos[g] for g in gids])
        gquat = np.stack([model.geom_quat[g] for g in gids])
        aabb = np.stack([model.geom_aabb[g] for g in gids])
        self._grot = _quat_to_mat(torch.as_tensor(gquat, dtype=torch.float32, device=dev))  # [M,3,3]
        self._lpos = torch.as_tensor(gpos, dtype=torch.float32, device=dev) + torch.einsum(
            "mij,mj->mi", self._grot, torch.as_tensor(aabb[:, :3], dtype=torch.float32, device=dev)
        )
        self._half = torch.as_tensor(aabb[:, 3:], dtype=torch.float32, device=dev)

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
    """See the module docstring. One instance per batched env; per-world state, seeded torch generator."""

    def __init__(self, cam: CameraModel, noise: TrackerNoise, nworld: int, *, device: torch.device | str, seed: int) -> None:
        if nworld < 1:
            raise ValueError("nworld must be >= 1")
        self.cam, self.noise, self.nworld = cam, noise, int(nworld)
        self.device = dev = torch.device(device)
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(int(seed))
        T = torch.as_tensor(cam.T_base_cam, dtype=torch.float32, device=dev)
        self._R = T[:3, :3].contiguous()  # cam -> base
        self._t = T[:3, 3].contiguous()
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
        self._burst = torch.zeros(n, dtype=torch.int64, device=dev)  # dropped frames still to come
        self._held = TrackerOut.empty(n, dev)
        self._arange = torch.arange(n, device=dev)
        self._sig_key: TrackerNoise | None = None
        self._sig = torch.zeros(3, device=dev)

    # ------------------------------------------------------------------ per episode
    def reset(self, mask: torch.Tensor, die_xyz_true: torch.Tensor) -> None:
        """Redraw the per-episode bias, rotation and latency for the worlds in ``mask``; their FIFO starts at "never
        seen" (the env then runs ``max_latency`` warm-up frames from the reset state). ``die_xyz_true`` is accepted
        for the interface (FS6) and validated; the warm-up frames do the measuring."""
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
        self._ring_ok.masked_fill_(m[:, None], False)
        self._ring_xyz.masked_fill_(m[:, None, None], 0.0)
        self._ring_conf.masked_fill_(m[:, None], 0.0)
        self._burst.masked_fill_(m, 0)
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

    def _measure(self, die: torch.Tensor) -> torch.Tensor:
        """One noisy measurement per world: camera-frame noise, then the per-episode rotation about the camera
        centre and the bias."""
        nz = self.noise
        if self._sig_key is not nz:  # built once per noise spec: no host->device copy per frame
            self._sig = torch.tensor([nz.sigma_xy_m, nz.sigma_xy_m, nz.sigma_z_m], dtype=torch.float32, device=self.device)
            self._sig_key = nz
        sig = self._sig
        noise_cam = torch.randn((self.nworld, 3), generator=self._gen, device=self.device) * sig
        meas = die + noise_cam @ self._R.T
        return torch.einsum("nij,nj->ni", self.rot, meas - self._t) + self._t + self.bias

    # ------------------------------------------------------------------ geometry
    def _project(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """base [..., 3] -> (uv [..., 2] pixels, depth [...] metres along the optical axis)."""
        pc = (p - self._t) @ self._R  # R^T (p - t), row vectors
        z = pc[..., 2]
        zs = z.clamp_min(1e-6)
        u = self.cam.fx * pc[..., 0] / zs + self.cam.cx
        v = self.cam.fy * pc[..., 1] / zs + self.cam.cy
        return torch.stack([u, v], dim=-1), z

    def covered_fraction(
        self,
        die_xyz_true: torch.Tensor,
        jaw_centers: torch.Tensor,
        jaw_half: torch.Tensor,
        jaw_rot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """float[N]: fraction of the die's disc (32 points, perpendicular to the optical axis) whose camera ray hits
        one of the occluder boxes BEFORE reaching the die (slab test). Boxes: centres ``[N,M,3]``, half sizes
        ``[N,M,3]`` in the box frame, rotations ``jaw_rot [N,M,3,3]`` (None: axis-aligned in the base frame)."""
        die = die_xyz_true.float()
        n, m = jaw_centers.shape[:2]
        ax, ay = self._R[:, 0], self._R[:, 1]
        r = DIE_DISC_RADIUS_M
        pts = die[:, None, :] + r * (self._disc[None, :, 0:1] * ax + self._disc[None, :, 1:2] * ay)  # [N,32,3]
        o = self._t
        d = pts - o  # ray o + t d reaches the disc point at t = 1
        c = jaw_centers.float()
        h = jaw_half.float()
        if jaw_rot is None:
            o_l = (o - c)[:, None]  # [N,1,M,3]
            d_l = d[:, :, None, :].expand(n, d.shape[1], m, 3)
        else:
            Rt = jaw_rot.float().transpose(-1, -2)  # [N,M,3,3]
            o_l = torch.einsum("nmij,nmj->nmi", Rt, o - c)[:, None]
            d_l = torch.einsum("nmij,npj->npmi", Rt, d)
        d_l = torch.where(d_l.abs() < 1e-12, torch.full_like(d_l, 1e-12), d_l)
        t1 = (-h[:, None] - o_l) / d_l
        t2 = (h[:, None] - o_l) / d_l
        t_in = torch.minimum(t1, t2).amax(dim=-1)
        t_out = torch.maximum(t1, t2).amin(dim=-1)
        hit = (t_out >= t_in.clamp_min(0.0)) & (t_in < 1.0)  # [N,32,M]
        return hit.any(dim=-1).float().mean(dim=1)

    def occluded(
        self,
        die_xyz_true: torch.Tensor,
        tip_xyz: torch.Tensor,
        jaw_centers: torch.Tensor,
        jaw_half: torch.Tensor,
        jaw_rot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """bool[N]; the rule the real tracker's ring test approximates (module docstring)."""
        nz = self.noise
        die, tip = die_xyz_true.float(), tip_xyz.float()
        horiz = (tip[:, :2] - die[:, :2]).norm(dim=-1)
        over = (horiz < nz.occlusion_radius_m) & (tip[:, 2] > die[:, 2])
        return over | (self.covered_fraction(die, jaw_centers, jaw_half, jaw_rot) > nz.occlusion_fraction)

    def in_view(self, die_xyz_true: torch.Tensor) -> torch.Tensor:
        """bool[N]: projected centre inside the image, in front of the camera, die on the table region."""
        die = die_xyz_true.float()
        uv, z = self._project(die)
        cam_ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= self.cam.width - 1) & (uv[:, 1] >= 0)
        cam_ok &= uv[:, 1] <= self.cam.height - 1
        table_ok = (die[:, :2].abs() <= TABLE_HALF_M).all(dim=-1) & (die[:, 2] >= TABLE_Z_M)
        return cam_ok & table_ok

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
        sees after the last one. Per frame: visibility (FOV/table, occlusion, dropout burst), a noisy measurement,
        push into the FIFO; the emitted frame is ``latency`` frames old."""
        nz, g, dev = self.noise, self._gen, self.device
        n = self.nworld
        die = die_xyz_true.float()
        covered = self.covered_fraction(die, jaw_centers, jaw_half, jaw_rot)
        horiz = (tip_xyz.float()[:, :2] - die[:, :2]).norm(dim=-1)
        over = (horiz < nz.occlusion_radius_m) & (tip_xyz.float()[:, 2] > die[:, 2])
        base_ok = self.in_view(die) & ~over & ~(covered > nz.occlusion_fraction)
        conf = (1.0 - covered).clamp(0.0, 1.0)
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
            meas = self._measure(die)
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
        out.xyz.copy_(h.xyz)
        out.visible.copy_(h.visible)
        out.age_frames.copy_(h.age_frames)
        out.confidence.copy_(h.confidence)
        return out


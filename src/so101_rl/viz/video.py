"""Rollout videos per checkpoint (plan Task V.2). Supersedes S4.3's ``rl/render.py``.

``record_rollout`` runs the current deterministic policy on a :class:`RolloutEnv` for ``seconds`` of env time and
writes one mp4 per recorded world, ``<out_dir>/<env_step>_<world>.mp4``. Each frame shows two cameras side by side:

1. **overhead**: the D435 pose of the real arm (so a sim and a real frame line up), a fixed MuJoCo camera at
   ``T_base_cam``. Source of the pose: ``calib/hand_eye.yaml`` (the measured hand-eye result) when that file
   exists; otherwise ``perception.calibrate_hand_eye.default_synthetic_camera_pose()`` (0.6 m above the workspace
   box centre, looking straight down), because ``so101.yaml`` does not carry the camera mount (it says so in
   docs/ARCHITECTURE.md). :func:`overhead_camera_pose` returns the matrix and a string naming the source.
   The camera frame is the OpenCV one (x right, y down, z forward); MuJoCo cameras look along -z with y up, hence
   ``R_mj = R_cv @ diag(1, -1, -1)``. The field of view is the D435 colour stream's (``fy`` ~615 px at 480 rows).
2. a free 3/4 view.

Rendering is CPU ``mujoco.Renderer`` on copied states, which is always available. MuJoCo Warp's batch renderer is
NOT used: it needs a different model/scene setup and is only an optimisation for many frames per step; at four
worlds and 10 fps the CPU path is a small fraction of a video interval. (This is a recorded deviation, see the
report.)

GL backend: ``MUJOCO_GL`` must be chosen before ``import mujoco`` (it is read once). This module sets it to ``egl``
when it is the first to import mujoco and the variable is unset, so headless rendering works; if mujoco was
imported earlier the existing backend is kept (GLFW needs a display). ``scripts`` that want to be sure export
``MUJOCO_GL=egl`` themselves.

Encoding is ffmpeg through a pipe (``libx264``, falling back to ``mpeg4``); overlays are drawn with OpenCV.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

if "mujoco" not in sys.modules:
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

if TYPE_CHECKING:
    import mujoco
    import torch

__all__ = [
    "FREE_CAM",
    "OVERHEAD_CAM",
    "ReplayRolloutEnv",
    "RolloutEnv",
    "RolloutResult",
    "RolloutStatus",
    "VideoWriter",
    "build_render_model",
    "draw_overlay",
    "find_ffmpeg",
    "overhead_camera_pose",
    "overlay_lines",
    "record_rollout",
]

OVERHEAD_CAM: Final[str] = "d435_overhead"
FREE_CAM: Final[str] = "free_3q"
D435_FY_PX: Final[float] = 615.0  # nominal D435 colour intrinsics at 640x480
HAND_EYE_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "calib" / "hand_eye.yaml"


@dataclass(frozen=True)
class RolloutStatus:
    """Per-recorded-world status after a step (host arrays, length = number of recorded worlds)."""

    success: np.ndarray  # bool
    ep_return: np.ndarray  # float
    dr_stage: int = 0


class RolloutEnv(Protocol):
    """What :func:`record_rollout` needs from an env. ``obs``/``action`` are batched over ALL of the env's worlds.

    A video env is normally a SEPARATE small env (``video_worlds`` worlds) so that recording never disturbs the
    training worlds' episodes. ``qpos`` rows have the same layout as ``WarpSim.qpos`` (arm 6 + die freejoint 7).
    """

    def reset(self) -> torch.Tensor: ...

    def step(self, action: torch.Tensor) -> torch.Tensor: ...

    def qpos(self, worlds: Sequence[int]) -> np.ndarray: ...

    def status(self, worlds: Sequence[int]) -> RolloutStatus: ...


@dataclass
class RolloutResult:
    paths: list[Path]
    n_frames: int
    tb_frames: np.ndarray | None  # uint8 [T, H, W, 3] of the first world, half resolution, for TensorBoard add_video


# --------------------------------------------------------------------------------------------- cameras and model
def overhead_camera_pose(path: Path | None = None) -> tuple[np.ndarray, str]:
    """``(T_base_cam 4x4 in the OpenCV camera convention, source description)``."""
    p = HAND_EYE_PATH if path is None else Path(path)
    if p.exists():
        from so101_rl.perception.frames import load_hand_eye

        return load_hand_eye(p).T_base_cam, f"hand-eye calibration {p}"
    from so101_rl.perception.calibrate_hand_eye import default_synthetic_camera_pose

    return default_synthetic_camera_pose(), "default_synthetic_camera_pose (no calib/hand_eye.yaml yet)"


def cv_to_mujoco_rotation(r_cv: np.ndarray) -> np.ndarray:
    return np.asarray(r_cv, dtype=np.float64) @ np.diag([1.0, -1.0, -1.0])


def build_render_model(
    height: int = 360, width: int = 480, *, hand_eye_path: Path | None = None
) -> tuple[mujoco.MjModel, str]:
    """The RL scene with the D435 overhead camera and offscreen buffers sized for ``height x width``.

    Adding a camera changes neither the dof layout nor the body/geom order, so ``qpos`` rows of ``WarpSim`` apply
    unchanged. Returns ``(model, camera pose source)``.
    """
    import mujoco

    from so101_rl.sim.model import build_spec

    T, source = overhead_camera_pose(hand_eye_path)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, cv_to_mujoco_rotation(T[:3, :3]).ravel())
    spec = build_spec()
    fovy = float(np.degrees(2.0 * np.arctan(0.5 * 480.0 / D435_FY_PX)))
    spec.worldbody.add_camera(name=OVERHEAD_CAM, pos=[float(x) for x in T[:3, 3]], quat=[float(x) for x in quat], fovy=fovy)
    spec.visual.global_.offheight = int(height)
    spec.visual.global_.offwidth = int(width)
    return spec.compile(), source


# --------------------------------------------------------------------------------------------- overlay
def overlay_lines(env_step: int, world: int, t_s: float, dr_stage: int, ep_return: float, success: bool) -> list[str]:
    return [
        f"step {int(env_step)}  world {int(world)}  t {t_s:.1f}s",
        f"DR stage {int(dr_stage)}  return {float(ep_return):+.2f}  success {'YES' if success else 'no'}",
    ]


def draw_overlay(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Draw ``lines`` on a dark band at the top of ``frame`` (uint8 HxWx3, edited in place and returned)."""
    import cv2

    scale, thick, pad = 0.5, 1, 6
    band = pad + len(lines) * 18
    frame[:band] = (frame[:band] * 0.35).astype(np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + 12 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------------------------- encoding
def find_ffmpeg() -> str:
    """ffmpeg of the pixi env (next to the interpreter) if present, else PATH."""
    env_bin = Path(sys.executable).parent / "ffmpeg"
    found = str(env_bin) if env_bin.exists() else shutil.which("ffmpeg")
    if found is None:
        raise RuntimeError("ffmpeg not found (it ships with the pixi env: pixi run -e default ...)")
    return found


_CODEC: list[str] | None = None


def _codec_args() -> list[str]:
    """libx264 when this ffmpeg has it, else mpeg4 (probed once)."""
    global _CODEC
    if _CODEC is None:
        out = subprocess.run([find_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False).stdout
        _CODEC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"] if "libx264 " in out else ["-c:v", "mpeg4", "-q:v", "4"]
    return _CODEC


class VideoWriter:
    """Pipe raw RGB frames into ffmpeg -> mp4 (yuv420p so every player opens it)."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = width - width % 2, height - height % 2  # yuv420p needs even sizes
        self.frames = 0
        codec = _codec_args()
        self._proc: subprocess.Popen[bytes] | None = subprocess.Popen(
            [
                find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-vf", f"crop={self.w}:{self.h}:0:0",
                *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._size = (width, height)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self._size[1], self._size[0]) or frame.dtype != np.uint8:
            raise ValueError(f"frame {frame.shape} {frame.dtype} does not match {self._size[::-1]} uint8")
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        assert proc.stdin is not None and proc.stderr is not None
        with contextlib.suppress(BrokenPipeError):
            proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} (exit {rc}): {err.strip()[:400]}")


# --------------------------------------------------------------------------------------------- recording
def _free_camera() -> Any:
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.15, 0.0, 0.03]
    cam.distance, cam.azimuth, cam.elevation = 0.75, 135.0, -30.0
    return cam


def record_rollout(
    env: RolloutEnv,
    policy: Callable[[torch.Tensor], torch.Tensor],
    *,
    env_step: int,
    out_dir: Path,
    worlds: Sequence[int] = (0, 1, 2, 3),
    seconds: float = 10.0,
    fps: float = 10.0,
    height: int = 360,
    width: int = 480,
    hand_eye_path: Path | None = None,
    keep_tb_frames: bool = True,
) -> RolloutResult:
    """Roll ``policy`` out on ``env`` for ``seconds * fps`` env steps, writing ``<env_step>_<world>.mp4`` per world.

    One frame per env step (the env's policy tick), played at ``fps``. Raises on any renderer / ffmpeg failure; the
    :class:`~so101_rl.viz.monitor.TrainingMonitor` is what swallows it (visualisation-only errors).
    """
    import mujoco
    import torch

    n_steps = max(1, round(seconds * fps))
    model, _src = build_render_model(height, width, hand_eye_path=hand_eye_path)
    data = mujoco.MjData(model)
    free = _free_camera()
    out_dir = Path(out_dir)
    writers: dict[int, VideoWriter] = {}
    tb: list[np.ndarray] = []
    renderer = mujoco.Renderer(model, height, width)
    try:
        for w in worlds:
            writers[w] = VideoWriter(out_dir / f"{int(env_step)}_{int(w)}.mp4", 2 * width, height, fps)
        with torch.no_grad():
            obs = env.reset()
            for t in range(n_steps + 1):
                qpos = env.qpos(worlds)
                st = env.status(worlds)
                for i, w in enumerate(worlds):
                    data.qpos[:] = qpos[i]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=OVERHEAD_CAM)
                    top = renderer.render().copy()
                    renderer.update_scene(data, camera=free)
                    side = renderer.render().copy()
                    frame = np.concatenate([top, side], axis=1)
                    draw_overlay(frame, overlay_lines(env_step, w, t / fps, st.dr_stage, float(st.ep_return[i]), bool(st.success[i])))
                    writers[w].write(frame)
                    if keep_tb_frames and i == 0:
                        tb.append(frame[::2, ::2].copy())
                if t < n_steps:
                    obs = env.step(policy(obs))
    finally:
        with contextlib.suppress(Exception):
            renderer.close()
        errs: list[Exception] = []
        for wr in writers.values():
            try:
                wr.close()
            except Exception as e:  # noqa: BLE001 - every writer is closed before the first error is raised
                errs.append(e)
        if errs and sys.exc_info()[0] is None:
            raise errs[0]
    paths = [wr.path for wr in writers.values()]
    return RolloutResult(paths, n_steps + 1, np.stack(tb) if tb else None)


class ReplayRolloutEnv:
    """A stub env that replays given qpos trajectories (tests, demos, and the CPU renderer check).

    ``trajectory``: ``[T, nq]`` or ``[T, W, nq]``; ``step`` advances one row. Obs/action are dummy tensors.
    """

    def __init__(self, trajectory: np.ndarray, nworld: int = 1) -> None:
        import torch

        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim == 2:
            traj = np.repeat(traj[:, None, :], nworld, axis=1)
        self.traj, self.nworld, self.t = traj, traj.shape[1], 0
        self._torch = torch

    def reset(self) -> torch.Tensor:
        self.t = 0
        return self._torch.zeros(self.nworld, 1)

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self.t = min(self.t + 1, len(self.traj) - 1)
        return self._torch.zeros(self.nworld, 1)

    def qpos(self, worlds: Sequence[int]) -> np.ndarray:
        return self.traj[self.t][list(worlds)]

    def status(self, worlds: Sequence[int]) -> RolloutStatus:
        n = len(worlds)
        return RolloutStatus(np.zeros(n, bool), np.full(n, -float(self.t)), dr_stage=0)

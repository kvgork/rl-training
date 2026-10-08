"""Rollout videos per checkpoint (plan Task V.2). Supersedes S4.3's ``rl/render.py``.

``record_rollout`` runs the current deterministic policy on a :class:`RolloutEnv` for ``seconds`` of env time and
writes one mp4 per recorded world, ``<out_dir>/<env_step>_<world>.mp4``. Each frame shows two cameras side by side:

1. **d435**: the real D435's colour view (so a sim and a real frame line up): the scene camera ``d435_color``,
   built from ``calib/camera.yaml`` by ``sim/model.py`` (camera parity C.3: intrinsics from the device, pose from
   the camera calibration; ``sim/d435.py`` holds the RealSense <-> MuJoCo frame conversion). The real camera is
   oblique (64 deg from vertical), not overhead. :func:`d435_camera_pose` returns the pose and its source.
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
    "D435_CAM",
    "FREE_CAM",
    "ReplayRolloutEnv",
    "RolloutEnv",
    "RolloutResult",
    "RolloutStatus",
    "VideoWriter",
    "add_scene_markers",
    "build_render_model",
    "d435_camera_pose",
    "draw_banner",
    "draw_marker_overlay",
    "draw_overlay",
    "find_ffmpeg",
    "overlay_lines",
    "project_points",
    "record_rollout",
]

D435_CAM: Final[str] = "d435_color"  # sim/d435.py COLOR_CAM (kept literal: importing sim/d435 loads mujoco)
FREE_CAM: Final[str] = "free_3q"


@dataclass(frozen=True)
class RolloutStatus:
    """Per-recorded-world status after a step (host arrays, length = number of recorded worlds)."""

    success: np.ndarray  # bool
    ep_return: np.ndarray  # float (must stop accumulating once the episode has ended)
    dr_stage: int = 0
    # Optional episode semantics + goal, all per recorded world. ``terminated``/``truncated`` None = the env does not
    # report episode ends and the clip simply runs for ``seconds`` (legacy behaviour).
    terminated: np.ndarray | None = None  # bool, sticky: the world's episode ended with success
    truncated: np.ndarray | None = None  # bool, sticky: the world's episode ended without success (TIMEOUT)
    goal: np.ndarray | None = None  # [n, 3] world-frame desired goal (drawn in both views)
    success_radius: float = 0.0  # m, ball around the goal (reach); 0 = none
    zone_half: float | None = None  # m, die tasks: half edge of the square target zone (drawn instead of a ball)


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
def d435_camera_pose(path: Path | None = None) -> tuple[np.ndarray, str]:
    """``(T_base_color 4x4, RealSense optical convention; source description)`` of ``calib/camera.yaml`` (or
    ``path``)."""
    from so101_rl.perception.camera_calib import CAMERA_YAML, load_camera_calib

    p = CAMERA_YAML if path is None else Path(path)
    pose = load_camera_calib(p).require_pose()
    return pose.matrix, f"{p} ({pose.method}, {pose.date})"


def build_render_model(
    height: int = 360, width: int = 480, *, camera_path: Path | None = None, backlit: bool = False
) -> tuple[mujoco.MjModel, str]:
    """The RL scene with the D435 cameras of ``calib/camera.yaml`` (or ``camera_path``) and offscreen buffers sized
    for ``height x width`` (use the D435's 4:3 aspect so its intrinsics hold). ``backlit``: add the window light
    behind the workspace (pixel world-model renders, C.5).

    Adding cameras and a light changes neither the dof layout nor the body/geom order, so ``qpos`` rows of
    ``WarpSim`` apply unchanged. Returns ``(model, camera pose source)``.
    """
    from so101_rl.perception.camera_calib import load_camera_calib
    from so101_rl.sim.model import build_spec

    calib = load_camera_calib(camera_path)
    _T, source = d435_camera_pose(camera_path)
    spec = build_spec(camera=calib)
    if backlit:
        from so101_rl.sim.d435 import add_window_backlight

        add_window_backlight(spec, calib)
    spec.visual.global_.offheight = max(int(height), calib.color.height, calib.depth.height)
    spec.visual.global_.offwidth = max(int(width), calib.color.width, calib.depth.width)
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


# --------------------------------------------------------------------------------------------- goal markers
GOAL_RGB: Final[tuple[int, int, int]] = (0, 255, 0)
TIP_RGB: Final[tuple[int, int, int]] = (255, 0, 255)
HOLD_SECONDS: Final[float] = 1.0


def _half_width(cam: Any, width: int, height: int) -> float:
    """Half frustum width at the near plane; free cameras leave it 0 (the renderer derives it from the aspect)."""
    return float(cam.frustum_width) or 0.5 * (cam.frustum_top - cam.frustum_bottom) * width / height


def project_points(scene: Any, pts: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel ``(x, y, depth_m)`` of world points ``pts [n, 3]`` for the camera of the LAST ``update_scene`` (read from
    the scene's GL camera: position, forward, up and frustum, so it holds for fixed and free cameras alike)."""
    cam = scene.camera[0]
    # camera[0]/[1] are the left/right stereo eyes (offset by +-ipd/2 even when stereo is off): the image is the mean
    pos = 0.5 * (np.asarray(cam.pos, float) + np.asarray(scene.camera[1].pos, float))
    fwd, up = np.asarray(cam.forward, float), np.asarray(cam.up, float)
    right = np.cross(fwd, up)
    d = np.atleast_2d(pts).astype(float) - pos
    z = d @ fwd
    z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    xn = (d @ right) / z_safe * cam.frustum_near
    yn = (d @ up) / z_safe * cam.frustum_near
    half_w = _half_width(cam, width, height)
    left, right = cam.frustum_center - half_w, cam.frustum_center + half_w
    px = (xn - left) / (right - left) * width
    py = (cam.frustum_top - yn) / (cam.frustum_top - cam.frustum_bottom) * height
    return np.stack([px, py, z], axis=1)


def add_scene_markers(
    scene: Any, goal: np.ndarray | None, tip: np.ndarray | None, success_radius: float = 0.0, zone_half: float | None = None
) -> None:
    """Append 3D marker geoms to ``scene``: a solid goal core + translucent success ball (or flat zone), a tip dot."""
    import mujoco

    def add(gtype: Any, size: Sequence[float], pos: np.ndarray, rgba: Sequence[float]) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1

    if goal is not None:
        g = np.asarray(goal, float)
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], g, [0, 1, 0, 1])
        if zone_half:
            add(mujoco.mjtGeom.mjGEOM_BOX, [zone_half, zone_half, 0.001], g, [0, 1, 0, 0.35])
        elif success_radius > 0:
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [success_radius, 0, 0], g, [0, 1, 0, 0.3])
    if tip is not None:
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], np.asarray(tip, float), [1, 0, 1, 1])


def draw_marker_overlay(
    frame: np.ndarray, scene: Any, goal: np.ndarray | None, tip: np.ndarray | None, success_radius: float = 0.0
) -> np.ndarray:
    """2D crosshair + success ring at the projected goal and a ring at the tip, drawn ON TOP (never occluded by the
    arm). ``frame`` is one camera view (HxWx3 uint8), edited in place."""
    import cv2

    h, w = frame.shape[:2]
    if goal is not None:
        x, y, z = project_points(scene, np.asarray(goal), w, h)[0]
        if z > 0:
            c = (round(x), round(y))
            if success_radius > 0:
                cam = scene.camera[0]
                rpx = success_radius / z * cam.frustum_near / (2 * _half_width(cam, w, h)) * w
                cv2.circle(frame, c, max(3, round(rpx)), GOAL_RGB, 1, cv2.LINE_AA)
            cv2.line(frame, (c[0] - 12, c[1]), (c[0] + 12, c[1]), GOAL_RGB, 1, cv2.LINE_AA)
            cv2.line(frame, (c[0], c[1] - 12), (c[0], c[1] + 12), GOAL_RGB, 1, cv2.LINE_AA)
            cv2.putText(frame, "GOAL", (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GOAL_RGB, 1, cv2.LINE_AA)
    if tip is not None:
        x, y, z = project_points(scene, np.asarray(tip), w, h)[0]
        if z > 0:
            cv2.circle(frame, (round(x), round(y)), 4, TIP_RGB, 1, cv2.LINE_AA)
    return frame


def draw_banner(frame: np.ndarray, text: str, rgb: tuple[int, int, int]) -> np.ndarray:
    """A solid coloured band with ``text`` across the bottom of ``frame`` (edited in place)."""
    import cv2

    h = frame.shape[0]
    frame[h - 36 :] = (frame[h - 36 :] * 0.3 + np.asarray(rgb) * 0.7).astype(np.uint8)
    cv2.putText(frame, text, (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
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


def _has_site(model: Any, name: str) -> bool:
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0


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
    camera_path: Path | None = None,
    keep_tb_frames: bool = True,
) -> RolloutResult:
    """Roll ``policy`` out on ``env`` for ``seconds * fps`` env steps, writing ``<env_step>_<world>.mp4`` per world.

    One frame per env step (the env's policy tick), played at ``fps``. Both views draw the goal (crosshair, success ring,
    3D marker) and the gripper tip. When the env reports episode ends (``RolloutStatus.terminated``/``truncated``), a
    world's clip stops at its end: last frame held ~1 s with a "SUCCESS at t=..." / "TIMEOUT at t=..." banner. Raises on any renderer / ffmpeg failure; the
    :class:`~so101_rl.viz.monitor.TrainingMonitor` is what swallows it (visualisation-only errors).
    """
    import mujoco
    import torch

    n_steps = max(1, round(seconds * fps))
    model, _src = build_render_model(height, width, camera_path=camera_path)
    data = mujoco.MjData(model)
    free = _free_camera()
    out_dir = Path(out_dir)
    writers: dict[int, VideoWriter] = {}
    tb: list[np.ndarray] = []
    renderer = mujoco.Renderer(model, height, width)
    try:
        for w in worlds:
            writers[w] = VideoWriter(out_dir / f"{int(env_step)}_{int(w)}.mp4", 2 * width, height, fps)
        hold = max(1, round(HOLD_SECONDS * fps))
        site = model.site("gripper_tip").id if _has_site(model, "gripper_tip") else -1
        finished: set[int] = set()

        def render_world(i: int, w: int, t: int, qpos: np.ndarray, st: RolloutStatus) -> np.ndarray:
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            goal = None if st.goal is None else np.asarray(st.goal[i], float)
            tip = data.site_xpos[site].copy() if site >= 0 else None
            views = []
            for cam in (D435_CAM, free):
                renderer.update_scene(data, camera=cam)
                add_scene_markers(renderer.scene, goal, tip, st.success_radius, st.zone_half)
                img = renderer.render().copy()
                draw_marker_overlay(img, renderer.scene, goal, tip, st.success_radius)
                views.append(img)
            frame = np.concatenate(views, axis=1)
            draw_overlay(frame, overlay_lines(env_step, w, t / fps, st.dr_stage, float(st.ep_return[i]), bool(st.success[i])))
            return frame

        with torch.no_grad():
            obs = env.reset()
            for t in range(n_steps + 1):
                qpos = env.qpos(worlds)
                st = env.status(worlds)
                for i, w in enumerate(worlds):
                    if w in finished:
                        continue
                    frame = render_world(i, w, t, qpos, st)
                    ended = st.terminated is not None and st.truncated is not None and bool(st.terminated[i] or st.truncated[i])
                    if ended:  # freeze here: banner, hold ~1 s, stop this world's clip (no rolling on after the end)
                        ok = bool(st.terminated[i])
                        draw_banner(frame, f"{'SUCCESS' if ok else 'TIMEOUT'} at t={t / fps:.1f}s", (0, 150, 0) if ok else (200, 110, 0))
                        for _ in range(hold):
                            writers[w].write(frame)
                        finished.add(w)
                    else:
                        writers[w].write(frame)
                    if keep_tb_frames and i == 0:
                        tb.append(frame[::2, ::2].copy())
                if len(finished) == len(worlds):
                    break
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
    return RolloutResult(paths, max((wr.frames for wr in writers.values()), default=0), np.stack(tb) if tb else None)


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

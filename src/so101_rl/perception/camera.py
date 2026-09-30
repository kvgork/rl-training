"""Camera protocol, Intel RealSense D435 wrapper and a fake camera (plan F9).

The D435 is selected by serial number, never by ``/dev/videoN`` (device nodes are not stable across reboots).
``pyrealsense2`` is imported lazily so this module imports, and the fake camera works, without the SDK.

Buffers: ``rgb_out`` is ``uint8 [H, W, 3]`` in **RGB** order, ``depth_m_out`` is ``float32 [H, W]`` in metres
(0.0 = no depth), aligned to the colour image.
"""

from __future__ import annotations

import time
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np

DEFAULT_SERIAL = "819312071240"


class Intrinsics(NamedTuple):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float  # metres per raw depth unit


class Frame(NamedTuple):
    t_mono_ns: int
    ok: bool


@runtime_checkable
class Camera(Protocol):
    def start(self) -> Intrinsics: ...

    def read(self, rgb_out: np.ndarray, depth_m_out: np.ndarray) -> Frame: ...

    def stop(self) -> None: ...


class D435Camera:
    """RealSense D435: colour + depth 640x480 @30, depth aligned to colour."""

    def __init__(
        self,
        serial: str = DEFAULT_SERIAL,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 45,
        timeout_ms: int = 2000,
    ) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.timeout_ms = timeout_ms
        self._rs = None
        self._pipeline = None
        self._align = None
        self._intr: Intrinsics | None = None

    def start(self) -> Intrinsics:
        try:
            import pyrealsense2 as rs
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                "D435Camera needs pyrealsense2 (pixi features 'perception'; environments cpu/robot/default). "
                f"Original error: {e}"
            ) from e
        if self._pipeline is not None:
            raise RuntimeError("camera already started")
        self._rs = rs
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        pipeline = rs.pipeline()
        profile = pipeline.start(cfg)  # raises RuntimeError if the serial is not connected
        try:
            self._align = rs.align(rs.stream.color)
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self._intr = Intrinsics(ci.fx, ci.fy, ci.ppx, ci.ppy, ci.width, ci.height, depth_scale)
            self._pipeline = pipeline
            for _ in range(self.warmup_frames):  # auto-exposure settles
                pipeline.wait_for_frames(self.timeout_ms)
        except BaseException:
            self._pipeline = None
            pipeline.stop()
            raise
        return self._intr

    def read(self, rgb_out: np.ndarray, depth_m_out: np.ndarray) -> Frame:
        if self._pipeline is None or self._intr is None:
            raise RuntimeError("camera not started")
        try:
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
            frames = self._align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
        except RuntimeError:  # wait_for_frames timeout
            return Frame(time.monotonic_ns(), False)
        t = time.monotonic_ns()
        if not color or not depth:
            return Frame(t, False)
        rgb_out[:] = np.asanyarray(color.get_data())
        np.multiply(
            np.asanyarray(depth.get_data()), np.float32(self._intr.depth_scale), out=depth_m_out, casting="unsafe"
        )
        return Frame(t, True)

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None


class FakeCamera:
    """Returns fixed (or replaced via :meth:`set_frames`) frames; for tests and the calibration dry run."""

    def __init__(self, rgb: np.ndarray, depth_m: np.ndarray, intr: Intrinsics) -> None:
        self.intr = intr
        self.started = False
        self.fail_next = 0  # number of upcoming reads that report ok=False (timeout simulation)
        self.set_frames(rgb, depth_m)

    def set_frames(self, rgb: np.ndarray, depth_m: np.ndarray) -> None:
        if rgb.shape[:2] != depth_m.shape:
            raise ValueError(f"rgb {rgb.shape} and depth {depth_m.shape} differ")
        self._rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self._depth = np.ascontiguousarray(depth_m, dtype=np.float32)

    def start(self) -> Intrinsics:
        self.started = True
        return self.intr

    def read(self, rgb_out: np.ndarray, depth_m_out: np.ndarray) -> Frame:
        if not self.started:
            raise RuntimeError("camera not started")
        t = time.monotonic_ns()
        if self.fail_next > 0:
            self.fail_next -= 1
            return Frame(t, False)
        rgb_out[:] = self._rgb
        depth_m_out[:] = self._depth
        return Frame(t, True)

    def stop(self) -> None:
        self.started = False

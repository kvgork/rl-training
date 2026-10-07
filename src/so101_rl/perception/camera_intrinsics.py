"""``camera-intrinsics``: read the D435's factory intrinsics into ``calib/camera.yaml`` (camera parity C.1).

``pixi run camera-intrinsics --serial 819312071240 --out calib/camera.yaml``

Streaming only: the pipeline is started with the colour (rgb8) and depth (z16) streams the rest of the stack uses,
the stream profiles' intrinsics, the depth-to-colour extrinsics and the depth scale are read, and the pipeline is
stopped. No option, preset or calibration table is written to the camera. If ``--out`` already exists for the same
serial, its pose (``T_base_color``) and table plane are kept; only the device block is replaced.

pyrealsense2 is imported lazily (``rs_module`` is injectable: ``tests/test_camera_calib.py`` runs this on a fake
device).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from so101_rl.perception.camera import DEFAULT_SERIAL
from so101_rl.perception.camera_calib import (
    CAMERA_YAML,
    CameraCalib,
    StreamIntrinsics,
    load_camera_calib,
    save_camera_calib,
)

_MODEL_NAMES = {"none": "none", "brown_conrady": "brown_conrady", "inverse_brown_conrady": "inverse_brown_conrady"}


def _model_name(model: Any) -> str:
    name = str(model).split(".")[-1]
    if name not in _MODEL_NAMES:
        raise ValueError(f"unsupported distortion model {model!r}")
    return _MODEL_NAMES[name]


def _stream(vsp: Any) -> StreamIntrinsics:
    i = vsp.get_intrinsics()
    return StreamIntrinsics(
        width=int(i.width),
        height=int(i.height),
        fps=int(vsp.fps()),
        fx=float(i.fx),
        fy=float(i.fy),
        cx=float(i.ppx),
        cy=float(i.ppy),
        distortion_model=_model_name(i.model),
        distortion_coeffs=tuple(float(c) for c in i.coeffs),
    )


def read_device_calib(serial: str, *, width: int = 640, height: int = 480, fps: int = 30, rs_module: Any = None) -> CameraCalib:
    """Factory intrinsics and depth-to-colour extrinsics of device ``serial`` (pose and table plane: None)."""
    rs = rs_module
    if rs is None:
        import pyrealsense2 as rs
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    try:
        dev = profile.get_device()
        cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ds = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ext = ds.get_extrinsics_to(cs)
        R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3).T  # librealsense: column-major
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array(ext.translation, dtype=np.float64)
        # re-orthonormalise the float32 rotation so the strict loader's rigidity check is exact
        u, _, vt = np.linalg.svd(T[:3, :3])
        T[:3, :3] = u @ vt
        calib = CameraCalib(
            serial=str(dev.get_info(rs.camera_info.serial_number)),
            firmware=str(dev.get_info(rs.camera_info.firmware_version)),
            color=_stream(cs),
            depth=_stream(ds),
            depth_scale=float(dev.first_depth_sensor().get_depth_scale()),
            T_color_depth=T,
            pose=None,
            table_plane=None,
        )
    finally:
        pipe.stop()
    if calib.serial != serial:
        raise RuntimeError(f"asked for serial {serial}, the device reports {calib.serial}")
    return calib


def main(argv: list[str] | None = None, *, rs_module: Any = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m so101_rl.perception.camera_intrinsics", description=__doc__.split("\n")[0])
    ap.add_argument("--serial", default=DEFAULT_SERIAL)
    ap.add_argument("--out", type=Path, default=CAMERA_YAML)
    a = ap.parse_args(argv)
    dev = read_device_calib(a.serial, rs_module=rs_module)
    out = dev
    if a.out.exists():
        old = load_camera_calib(a.out)
        if old.serial != dev.serial:
            print(f"{a.out} belongs to camera {old.serial}, not {dev.serial}: pose and table plane dropped", file=sys.stderr)
        else:
            out = CameraCalib(
                serial=dev.serial,
                firmware=dev.firmware,
                color=dev.color,
                depth=dev.depth,
                depth_scale=dev.depth_scale,
                T_color_depth=dev.T_color_depth,
                pose=old.pose,
                table_plane=old.table_plane,
            )
    save_camera_calib(out, a.out)
    c, d = out.color, out.depth
    print(
        f"{a.out}: D435 {out.serial} fw {out.firmware}; colour {c.width}x{c.height}@{c.fps} fx {c.fx:.2f} fy {c.fy:.2f} "
        f"cx {c.cx:.2f} cy {c.cy:.2f}; depth fx {d.fx:.2f} cx {d.cx:.2f} cy {d.cy:.2f} scale {out.depth_scale:g}; "
        f"depth->colour t {np.round(out.T_color_depth[:3, 3] * 1000, 2).tolist()} mm; pose "
        f"{'kept (' + out.pose.method + ')' if out.pose else 'none'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

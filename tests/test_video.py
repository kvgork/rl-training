"""Task V.2: rollout videos (plan Unit SUV)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import so101_rl
from so101_rl.viz import video

SRC = str(Path(so101_rl.__file__).resolve().parents[1])


def _ffprobe(path: Path) -> dict:
    exe = Path(video.find_ffmpeg()).with_name("ffprobe")
    exe = str(exe) if exe.exists() else "ffprobe"
    out = subprocess.run(
        [exe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    import json

    return json.loads(out)["streams"][0]


def test_overlay_text_present():
    lines = video.overlay_lines(1500, 3, 4.5, 2, -7.25, True)
    assert lines[0] == "step 1500  world 3  t 4.5s"
    assert "DR stage 2" in lines[1] and "return -7.25" in lines[1] and "success YES" in lines[1]
    assert "success no" in video.overlay_lines(0, 0, 0.0, 0, 0.0, False)[1]
    frame = np.full((120, 400, 3), 90, np.uint8)
    out = video.draw_overlay(frame.copy(), lines)
    band = out[: 6 + 18 * 2]
    assert (band > 200).any(axis=-1).sum() > 100  # bright text pixels on the darkened band
    assert (out[-10:] == 90).all()  # nothing drawn outside the band
    other = video.draw_overlay(frame.copy(), video.overlay_lines(1501, 3, 4.5, 2, -7.25, True))
    assert not np.array_equal(out, other)  # the text really encodes the step


def test_d435_view_is_the_calibrated_camera():
    import mujoco

    from so101_rl.perception.camera_calib import load_camera_calib

    calib = load_camera_calib()
    T, src = video.d435_camera_pose()
    assert "camera.yaml" in src and calib.require_pose().method in src
    model, _ = video.build_render_model(120, 160)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    cid = model.camera(video.D435_CAM).id
    np.testing.assert_allclose(d.cam_xpos[cid], T[:3, 3], atol=1e-9)
    r_mj = d.cam_xmat[cid].reshape(3, 3)
    np.testing.assert_allclose(r_mj, T[:3, :3] @ np.diag([1.0, -1.0, -1.0]), atol=1e-9)
    np.testing.assert_allclose(-r_mj[:, 2], T[:3, 2], atol=1e-9)  # optical axis (z forward in the RealSense frame)
    assert tuple(model.cam_resolution[cid]) == (calib.color.width, calib.color.height)  # intrinsics, not fovy


def test_d435_view_follows_camera_yaml(tmp_path):
    import mujoco

    from so101_rl.perception.camera_calib import CameraPose, load_camera_calib, save_camera_calib

    calib = load_camera_calib()
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [0.30, -0.05, 0.55]
    p = tmp_path / "camera.yaml"
    save_camera_calib(calib.with_pose(CameraPose(T, "hand-eye", "2026-10-07", 1.2, 3.0, 0.5)), p)
    pose, source = video.d435_camera_pose(p)
    assert "hand-eye" in source
    model, _ = video.build_render_model(120, 160, camera_path=p)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    np.testing.assert_allclose(d.cam_xpos[model.camera(video.D435_CAM).id], [0.30, -0.05, 0.55], atol=1e-9)
    np.testing.assert_allclose(pose, T)


def test_render_model_keeps_the_dof_layout():
    from so101_rl.sim.model import load_model

    base, _ = load_model()
    model, _ = video.build_render_model(120, 160)
    assert (model.nq, model.nv, model.nbody, model.ngeom) == (base.nq, base.nv, base.nbody, base.ngeom)  # WarpSim rows apply as-is


def test_video_writer_roundtrip_odd_size(tmp_path):
    p = tmp_path / "a" / "x.mp4"
    w = video.VideoWriter(p, 101, 63, 10)  # odd sizes are cropped to even for yuv420p
    rng = np.random.default_rng(0)
    for _ in range(12):
        w.write(rng.integers(0, 255, (63, 101, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        w.write(np.zeros((10, 10, 3), np.uint8))
    w.close()
    info = _ffprobe(p)
    assert (info["width"], info["height"], int(info["nb_read_frames"])) == (100, 62, 12)
    assert info["codec_name"] in ("h264", "mpeg4")


_RENDER = textwrap.dedent(
    """
    import sys
    import numpy as np
    try:
        import mujoco
        from so101_rl.sim.model import load_model
        from so101_rl.viz import video
        m, _ = load_model()
        mujoco.Renderer(m, 32, 32).close()
    except Exception as e:  # no usable GL backend here
        print("NO-GL", e)
        sys.exit(77)
    q = np.asarray(m.key("rest").qpos)
    traj = np.stack([q + 0.02 * t * np.r_[np.ones(6), np.zeros(7)] for t in range(40)])
    res = video.record_rollout(video.ReplayRolloutEnv(traj, nworld=2), lambda o: o, env_step=777, out_dir=sys.argv[1],
                               worlds=[0, 1], seconds=2.0, fps=10, height=96, width=128)
    print("OK", [p.name for p in res.paths], res.n_frames, res.tb_frames.shape)
    """
)


def test_video_written_cpu_renderer(tmp_path):
    """Stub env (replayed qpos), CPU ``mujoco.Renderer`` (EGL, so it also works headless), real ffmpeg."""
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p = subprocess.run([sys.executable, "-c", _RENDER, str(tmp_path)], env=env, capture_output=True, text=True, timeout=300, check=False)
    if p.returncode == 77:
        pytest.skip(f"no usable GL backend for mujoco.Renderer: {p.stdout.strip()}")
    assert p.returncode == 0, p.stderr[-2000:] + p.stdout
    assert "OK ['777_0.mp4', '777_1.mp4'] 21 (21, 48, 128, 3)" in p.stdout
    for w in (0, 1):
        info = _ffprobe(tmp_path / f"777_{w}.mp4")
        assert (info["width"], info["height"], int(info["nb_read_frames"])) == (256, 96, 21)  # two cameras side by side

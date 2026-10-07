"""Task V.3: TrainingMonitor (FS11) - never raises into training, viz_disabled, attach, TensorBoard (plan Unit SUV)."""

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from so101_rl.viz import live_viewer as lv
from so101_rl.viz import monitor as mon_mod
from so101_rl.viz.monitor import TrainingMonitor, VizConfig
from so101_rl.viz.video import RolloutResult, RolloutStatus


def _sim(n=8):
    return SimpleNamespace(qpos=torch.zeros(n, 13), qvel=torch.zeros(n, 12))


def _mon(tmp_path, **kw):
    cfg = VizConfig(**{"tensorboard": False, "video_every_min": 0.0, **kw})
    return TrainingMonitor(cfg, _sim(), tmp_path / "run", task="reach", render_cfg=None)


def _step(m, n=8, i=1):
    m.on_step(i, success=torch.zeros(n, dtype=torch.bool), ep_return=torch.zeros(n))


class _Boom:
    def __getattr__(self, name):
        def f(*a, **k):
            raise RuntimeError("boom")

        return f


def test_failure_never_raises_and_flags_viz_disabled(tmp_path, caplog):
    m = _mon(tmp_path)
    assert m.csv_fields() == {"viz_disabled": 0}
    m._publisher = _Boom()  # a crash inside the viewer path
    with caplog.at_level("ERROR", logger="so101_rl.viz"):
        _step(m)  # must not raise
        _step(m, i=2)
    assert m.csv_fields() == {"viz_disabled": 1}
    assert sum("visualisation disabled" in r.message for r in caplog.records) == 1  # logged once
    status = json.loads((tmp_path / "run" / "viz" / "status.json").read_text())
    assert status["viz_disabled"] == 1 and "boom" in status["error"]
    # everything is a no-op from now on, and still never raises
    m.on_metrics(5, {"alpha": 0.1})
    m.add_text(5, "review", "x")
    assert m.maybe_record(5, lambda o: o, force=True) is None
    m.close()
    m.close()


def test_stale_status_from_a_previous_run_is_cleared(tmp_path):
    (tmp_path / "run" / "viz").mkdir(parents=True)
    (tmp_path / "run" / "viz" / "status.json").write_text('{"viz_disabled": 1}')
    m = _mon(tmp_path)
    assert not (tmp_path / "run" / "viz" / "status.json").exists() and m.viz_disabled == 0


def test_headless_viewer_request_is_not_a_failure(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with caplog.at_level("WARNING", logger="so101_rl.viz"):
        m = _mon(tmp_path, live_viewer=True)
    assert m.viz_disabled == 0 and m._publisher is None
    assert any("needs a display" in r.message for r in caplog.records)
    _step(m)
    m.close()


def test_bad_viewer_selector_disables_viz_but_not_training(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    m = _mon(tmp_path, live_viewer=True, viewer_worlds=("0", "median"))
    assert m.viz_disabled == 1 and "median" in m.error
    _step(m)  # still fine


def test_attach_request_turns_on_the_publisher(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    m = _mon(tmp_path)  # started WITHOUT --viewer
    assert m._publisher is None
    req = tmp_path / "run" / "viz" / lv.ATTACH_REQUEST
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text("1")
    _step(m)
    assert m._publisher is not None and not req.exists() and m.viz_disabled == 0
    assert (tmp_path / "run" / "viz" / lv.DESCRIPTOR).exists()
    for i in range(3):
        _step(m, i=i + 2)
        time.sleep(0.05)
    assert m._publisher.published >= 1  # the grace period lets it publish before the viewer's pid file exists
    m.close()
    assert not (tmp_path / "run" / "viz" / lv.DESCRIPTOR).exists()


def test_viewer_process_crash_disables_viz_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    m = _mon(tmp_path)
    pub = m._make_publisher(spawn=False)
    pub._proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    pub._proc.wait()
    _step(m)
    assert m.viz_disabled == 1 and "exit code 3" in m.error
    _step(m, i=2)  # training goes on


def test_closing_the_window_is_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    m = _mon(tmp_path)
    pub = m._make_publisher(spawn=False)
    pub._proc = subprocess.Popen([sys.executable, "-c", "pass"])  # exit code 0 = the user closed the window
    pub._proc.wait()
    _step(m)
    assert m.viz_disabled == 0
    m.close()


class _FakeTb:
    def __init__(self):
        self.scalars, self.texts = [], []

    def add_scalar(self, tag, val, step):
        self.scalars.append((tag, val, step))

    def add_text(self, tag, text, step):
        self.texts.append((tag, text, step))

    def add_video(self, *a, **k):
        self.scalars.append(("video", None, a[2]))

    def flush(self):
        pass

    def close(self):
        pass


def test_on_metrics_maps_names_and_adds_system_scalars(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_mod, "gpu_health", lambda: {"system/gpu_temp_c": 61.0, "system/gpu_mem_used_mb": 100.0, "system/gpu_util_pct": 50.0})
    m = _mon(tmp_path)
    m._tb = _FakeTb()
    m.on_step(10, success=torch.tensor([True, False]), ep_return=torch.tensor([-1.0, -3.0]))
    m.on_metrics(10, {"success_rate": 0.25, "alpha": 0.2, "critic_loss": 1.5, "custom_thing": 7})
    time.sleep(0.05)
    m.on_metrics(110, {"entropy": -1.0})
    tags = {t: v for t, v, s in m._tb.scalars if s == 10}
    assert tags["train/success_rate"] == 0.25 and tags["sac/alpha"] == 0.2 and tags["sac/critic_loss"] == 1.5
    assert tags["custom_thing"] == 7 and tags["train/current_mean_return"] == -2.0 and tags["train/current_success_fraction"] == 0.5
    assert tags["system/gpu_temp_c"] == 61.0 and tags["system/viz_disabled"] == 0.0
    second = {t: v for t, v, s in m._tb.scalars if s == 110}
    assert second["sac/entropy"] == -1.0 and second["system/env_steps_per_s"] > 0
    m.add_text(110, "review/trigger", "return plateau")
    assert m._tb.texts == [("review/trigger", "return plateau", 110)]


def test_gpu_health_falls_back_to_nvidia_smi(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # force ImportError
    monkeypatch.setattr(mon_mod.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(mon_mod.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="58, 2048, 17\n"))
    assert mon_mod.gpu_health() == {"system/gpu_temp_c": 58.0, "system/gpu_mem_used_mb": 2048.0, "system/gpu_util_pct": 17.0}
    monkeypatch.setattr(mon_mod.shutil, "which", lambda n: None)
    assert mon_mod.gpu_health() == {}


def test_missing_tensorboard_is_not_a_failure(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)  # import raises ImportError
    with caplog.at_level("WARNING", logger="so101_rl.viz"):
        m = TrainingMonitor(VizConfig(tensorboard=True, video_every_min=0), _sim(), tmp_path / "run", task="reach", render_cfg=None)
    assert m.viz_disabled == 0 and m._tb is None
    assert any("TensorBoard writer off" in r.message for r in caplog.records)
    m.on_metrics(1, {"alpha": 0.1})  # no-op, no raise
    assert m.viz_disabled == 0


def test_tensorboard_events_written(tmp_path):
    # tensorboard is in the `train` pixi feature (integration 2026-10-06): no skip, a missing package fails here
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    m = TrainingMonitor(VizConfig(tensorboard=True, video_every_min=0), _sim(), tmp_path / "run", task="reach", render_cfg=None)
    assert m._tb is not None
    m.on_step(100, success=torch.zeros(8, dtype=torch.bool), ep_return=torch.zeros(8))
    m.on_metrics(100, {"success_rate": 0.5, "mean_return": -4.0, "alpha": 0.2})
    m.add_text(100, "review/trigger", "plateau")
    m.close()
    acc = EventAccumulator(str(tmp_path / "run" / "tensorboard"), size_guidance={"scalars": 0, "tensors": 0})
    acc.Reload()
    scalars = set(acc.Tags()["scalars"])
    assert {"train/success_rate", "train/mean_return", "sac/alpha", "system/viz_disabled"} <= scalars
    assert acc.Scalars("train/success_rate")[0].value == pytest.approx(0.5)


def test_tensorboard_rollout_video_is_embedded(tmp_path, monkeypatch):
    """Plan §10 2026-10-06 "TensorBoard videos": the rollout shows inline in the dashboard. torch's add_video encodes a
    GIF through ``moviepy.editor`` (moviepy < 2, `train` feature) and silently writes NOTHING without it, so this test
    reads the event file back. No skip: a missing or incompatible moviepy fails here."""
    import moviepy.editor  # noqa: F401  (the import torch's make_video needs; moviepy 2.x removed it)
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from so101_rl.viz import video

    t, h, w = 6, 24, 32
    frames = np.zeros((t, h, w, 3), np.uint8)
    frames[:, :, :, 0] = np.arange(t, dtype=np.uint8)[:, None, None] * 40  # a changing red channel

    def rec(env, policy, *, env_step, out_dir, worlds, seconds, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{env_step}_0.mp4"
        p.write_bytes(b"x")
        return RolloutResult([p], t, frames)

    monkeypatch.setattr(video, "record_rollout", rec)
    m = TrainingMonitor(VizConfig(tensorboard=True, video_every_min=0), _sim(), tmp_path / "run", task="reach", render_cfg=None)
    m.attach_rollout_env(_StubEnv())
    assert m.maybe_record(700, lambda o: o, force=True) is not None
    assert m.viz_disabled == 0
    m.close()
    acc = EventAccumulator(str(tmp_path / "run" / "tensorboard"), size_guidance={"images": 0})
    acc.Reload()
    assert "rollout/world0" in acc.Tags()["images"], acc.Tags()
    ev = acc.Images("rollout/world0")[0]
    assert ev.step == 700 and (ev.width, ev.height) == (w, h)
    assert ev.encoded_image_string[:6] in (b"GIF87a", b"GIF89a") and len(ev.encoded_image_string) > 100


class _StubEnv:
    nworld = 4


def _fake_record(calls):
    def rec(env, policy, *, env_step, out_dir, worlds, seconds, **kw):
        calls.append((env_step, list(worlds), seconds))
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for w in worlds:
            p = out_dir / f"{env_step}_{w}.mp4"
            p.write_bytes(b"x")
            paths.append(p)
        return RolloutResult(paths, 11, None)

    return rec


def test_maybe_record_due_logic(tmp_path, monkeypatch):
    from so101_rl.viz import video

    calls = []
    monkeypatch.setattr(video, "record_rollout", _fake_record(calls))
    m = _mon(tmp_path, video_every_min=10.0, video_worlds=3, video_seconds=2.0)
    m.attach_rollout_env(_StubEnv())
    assert m.maybe_record(1, lambda o: o) is None and not calls  # not due
    p = m.maybe_record(500, lambda o: o, force=True)  # a checkpoint
    assert p == tmp_path / "run" / "videos" / "500_0.mp4" and calls == [(500, [0, 1, 2], 2.0)]
    assert m.maybe_record(501, lambda o: o) is None  # the timer restarted
    m._last_video_t -= 601.0
    assert m.maybe_record(900, lambda o: o) is not None and len(calls) == 2
    m2 = _mon(tmp_path / "b", video_every_min=0.0)
    m2.attach_rollout_env(_StubEnv())
    m2._last_video_t -= 1e6
    assert m2.maybe_record(1, lambda o: o) is None  # 0 disables the timer


def test_maybe_record_without_env_is_quiet(tmp_path):
    m = _mon(tmp_path, video_every_min=10.0)
    assert m.maybe_record(1, lambda o: o, force=True) is None and m.viz_disabled == 0


def test_video_writer_failure_disables_viz_and_training_continues(tmp_path, monkeypatch):
    from so101_rl.viz import video

    def boom(*a, **k):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(video, "record_rollout", boom)
    m = _mon(tmp_path)
    m.attach_rollout_env(_StubEnv())
    assert m.maybe_record(1, lambda o: o, force=True) is None
    assert m.csv_fields() == {"viz_disabled": 1} and "ffmpeg exploded" in m.error
    _step(m)
    m.on_metrics(2, {"alpha": 1.0})


def test_rollout_status_is_plain_data():
    s = RolloutStatus(success=__import__("numpy").zeros(2, bool), ep_return=__import__("numpy").zeros(2))
    assert s.dr_stage == 0


def test_interface_surface_matches_fs11():
    import inspect

    sig = inspect.signature
    assert list(sig(TrainingMonitor.__init__).parameters) == ["self", "cfg", "sim", "run_dir", "task", "render_cfg"]
    assert list(sig(TrainingMonitor.on_step).parameters) == ["self", "env_step", "success", "ep_return"]
    assert list(sig(TrainingMonitor.on_metrics).parameters) == ["self", "env_step", "scalars"]
    assert list(sig(TrainingMonitor.maybe_record).parameters)[:3] == ["self", "env_step", "policy"]
    d = VizConfig()
    assert (d.live_viewer, d.viewer_worlds, d.viewer_hz, d.video_every_min, d.video_worlds, d.video_seconds, d.tensorboard) == (
        False, ("0", "best", "worst"), 30.0, 10.0, 4, 10.0, True,
    )


def test_viz_never_imports_safety_or_servo_hardware():
    """viz imports sim/env read-only (and servo.shm for the seqlock); never safety, never a serial backend."""
    import ast

    root = Path(mon_mod.__file__).parent
    for f in root.glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for n in names:
                assert not n.startswith(("so101_rl.safety", "so101_rl.rl", "so101_rl.wm", "so101_rl.servo.feetech_bus", "so101_rl.servo.bus", "serial")), (f.name, n)


def test_dashboard_wrapper_builds_the_tensorboard_command(tmp_path, capsys):
    from so101_rl.viz.dashboard import build_parser, main, tensorboard_argv

    ns = build_parser().parse_args(["--logdir", str(tmp_path)])
    assert tensorboard_argv(ns) == ["tensorboard", "--logdir", str(tmp_path), "--host", "127.0.0.1", "--port", "6006"]
    assert build_parser().parse_args([]).logdir.as_posix() == "runs"
    assert main(["--logdir", str(tmp_path / "missing")]) == 2  # refuses a missing logdir, never starts a server
    assert "does not exist" in capsys.readouterr().err

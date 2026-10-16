"""The perception process (plans/die-to-zone-plan.md D3): camera -> HsvDepthDieTracker -> object table.

No hardware: :class:`SimRenderCamera` renders the sim scene from the committed ``calib/camera.yaml`` camera (colour +
depth aligned to colour, as ``tests/test_die_convention.py``), so the published centroid can be compared with the true
die pose.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from so101_rl.perception import process as pp
from so101_rl.perception.camera import FakeCamera, Frame, Intrinsics
from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.perception.object_table import ObjectTableShm, new_object_table
from so101_rl.perception.sim_camera import SimRenderCamera

REPO = Path(__file__).resolve().parents[1]
WELL = (0.20, -0.05)  # env/tasks.py DIE_TO_ZONE.die_start.well_center_xy_m: hidden by the tray rim at home
# resting die positions in the calibrated view, clear of the arm at home: the zone centre and the start stages
SCENES = [(0.15, -0.13, 0.3), (0.14, -0.06, 0.7), (0.25, -0.12, 0.2), (0.12, -0.10, 0.0)]


def _name() -> str:
    return f"so101_perc_{uuid.uuid4().hex[:8]}"


def _die(x: float, y: float, yaw: float = 0.0) -> tuple[float, float, float, float]:
    from so101_rl.sim.model import DIE_EDGE_M, TABLE_Z_M

    return (x, y, TABLE_Z_M + DIE_EDGE_M / 2, yaw)


@pytest.fixture(scope="module")
def calib():
    return load_camera_calib()


@pytest.fixture(scope="module")
def cam(calib):
    c = SimRenderCamera(calib)
    try:
        pp.open_checked(c, calib)
    except Exception as exc:  # noqa: BLE001 - no OpenGL context on this host
        pytest.skip(f"no MuJoCo offscreen rendering: {exc}")
    yield c
    c.stop()


@pytest.fixture()
def table():
    t = ObjectTableShm.create(_name())
    yield t
    t.close()


def _read(name: str) -> np.ndarray:
    r = ObjectTableShm.attach(name)
    try:
        out = new_object_table()
        assert r.read(out)
        return out[0]
    finally:
        r.close()


def _loop(cam, calib, table, **kw) -> pp.PerceptionLoop:
    return pp.PerceptionLoop(cam, calib, table, gate=pp.CameraMovedGate(pp.TablePlaneCheck(calib)), **kw)


def test_die_label_is_the_env_label():
    from so101_rl.env.goal_env import DIE_LABEL

    assert pp.DIE_LABEL.encode() == DIE_LABEL


def test_publishes_the_die_centroid_of_rendered_scenes(cam, calib, table):
    loop = _loop(cam, calib, table)
    name = table._shm.name.lstrip("/")
    for k, (x, y, yaw) in enumerate(SCENES, start=1):
        cam.set_die(_die(x, y, yaw))
        assert loop.step()
        rec = _read(name)
        o = rec["objects"][0]
        assert int(rec["n"]) == 1 and bytes(o["label"]).rstrip(b"\0") == b"die"
        assert bool(o["visible"]), (x, y)
        err = np.linalg.norm(np.asarray(o["xyz_base"], np.float64) - cam.die_true_xyz())
        assert err < 0.003, ((x, y), err)  # the convention's tolerance (tests/test_die_convention.py)
        assert 0.0 < float(o["confidence"]) <= 1.0
        assert int(o["t_mono_ns"]) == int(rec["t_mono_ns"]) > 0  # a fresh detection: its own frame time
        assert int(rec["fiducial_ok"]) == 1  # the committed calibration's table plane matches the scene
        assert int(rec["seq"]) // 2 == k  # one seqlock write per processed frame
    assert loop.stats.detections == len(SCENES)
    assert loop.gate.checks == 1 and not loop.gate.tripped


def test_hidden_die_is_not_visible_and_the_last_position_is_held(cam, calib, table):
    loop = _loop(cam, calib, table)
    name = table._shm.name.lstrip("/")
    cam.set_die(_die(*SCENES[0]))
    loop.step()
    seen = _read(name)["objects"][0].copy()
    assert bool(seen["visible"])
    for die in (_die(*WELL), None):  # in the start well from the home pose (tray rim nearer), then out of view
        cam.set_die(die)
        loop.step()
        rec = _read(name)
        o = rec["objects"][0]
        assert not bool(o["visible"]), die
        assert float(o["confidence"]) == 0.0
        np.testing.assert_array_equal(o["xyz_base"], seen["xyz_base"])  # last-seen centroid and its time
        assert int(o["t_mono_ns"]) == int(seen["t_mono_ns"]) < int(rec["t_mono_ns"])
    assert loop.stats.detections == 1 and loop.stats.processed == 3


def test_never_seen_die_publishes_zeros(cam, calib, table):
    cam.set_die(None)
    _loop(cam, calib, table).step()
    o = _read(table._shm.name.lstrip("/"))["objects"][0]
    assert not bool(o["visible"]) and int(o["t_mono_ns"]) == 0
    np.testing.assert_array_equal(o["xyz_base"], 0.0)


def test_concurrent_reads_are_consistent(cam, calib, table):
    """A reader hammering the table while the writer alternates two scenes only ever sees whole records."""
    rgb = np.zeros((calib.color.height, calib.color.width, 3), np.uint8)
    depth = np.zeros(rgb.shape[:2], np.float32)
    frames = []
    for die in (_die(*SCENES[0]), None):
        cam.set_die(die)
        cam.read(rgb, depth)
        frames.append((rgb.copy(), depth.copy()))
    fake = FakeCamera(*frames[0], calib.color_intrinsics())
    fake.start()
    loop = pp.PerceptionLoop(fake, calib, table)
    loop.step()
    xyz_seen = loop._xyz.astype(np.float32)
    stop = threading.Event()
    bad: list[str] = []
    n_ok = [0]

    def reader() -> None:
        r = ObjectTableShm.attach(table._shm.name.lstrip("/"))
        out = new_object_table()
        try:
            while not stop.is_set():
                if not r.read(out):
                    continue
                n_ok[0] += 1
                o = out[0]["objects"][0]
                if not np.array_equal(o["xyz_base"], xyz_seen):
                    bad.append(f"xyz {o['xyz_base']}")
                if bool(o["visible"]) != (int(o["t_mono_ns"]) == int(out[0]["t_mono_ns"])):
                    bad.append("visible / time mismatch")
                if bool(o["visible"]) != (float(o["confidence"]) > 0):
                    bad.append("visible / confidence mismatch")
        finally:
            r.close()

    th = threading.Thread(target=reader)
    th.start()
    try:
        for i in range(120):
            fake.set_frames(*frames[i % 2])
            loop.step()
    finally:
        stop.set()
        th.join()
    assert not bad, bad[:5]
    assert n_ok[0] > 0
    assert loop.stats.detections == 61


def test_camera_moved_plane_check(calib):
    """A camera pitched by 3 deg fails the table-plane check and latches after PLANE_TRIP_AFTER checks; a pure yaw
    (or an in-plane shift) leaves the table plane unchanged and is NOT detected: that needs the H8 fiducial."""
    from so101_rl.sim.d435 import _axis_angle

    def moved(axis) -> SimRenderCamera:
        dT = np.eye(4)
        dT[:3, :3] = _axis_angle(np.asarray(axis, float))
        return SimRenderCamera(calib, dT_color=dT)

    pitched = moved([np.radians(3.0), 0.0, 0.0])  # about the optical x axis
    try:
        pp.open_checked(pitched, calib)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no MuJoCo offscreen rendering: {exc}")
    t = ObjectTableShm.create(_name())
    try:
        loop = _loop(pitched, calib, t, plane_every=1)
        loop.step()
        assert not loop.gate.ok and not loop.gate.tripped and loop.gate.last.angle_deg > 2.0
        assert int(_read(t._shm.name.lstrip("/"))["fiducial_ok"]) == 0
        for _ in range(pp.PLANE_TRIP_AFTER - 1):
            loop.step()
        assert loop.gate.tripped
        rep = loop.summary()["camera_check"]
        assert rep["table_plane"]["tripped"] and rep["fiducial"] == pp.FIDUCIAL_STATUS
    finally:
        t.close()
        pitched.stop()

    # yaw about the plane normal through the camera centre: same plane, passes (the documented blind spot)
    n_base = np.array([0.0, 0.0, 1.0])
    yawed = moved(calib.T_base_color[:3, :3].T @ n_base * np.radians(3.0))
    pp.open_checked(yawed, calib)
    try:
        g = pp.CameraMovedGate(pp.TablePlaneCheck(calib))
        rgb = np.zeros((calib.color.height, calib.color.width, 3), np.uint8)
        depth = np.zeros(rgb.shape[:2], np.float32)
        yawed.read(rgb, depth)
        assert g.update(depth), g.last
    finally:
        yawed.stop()


def test_plane_check_needs_table_points(calib):
    g = pp.TablePlaneCheck(calib)
    r = g.check(np.zeros((calib.color.height, calib.color.width), np.float32))
    assert not r.ok and r.n_points == 0


def test_serial_mismatch_is_refused(calib, tmp_path):
    wrong = SimRenderCamera(calib, serial="000000000000")
    with pytest.raises(pp.CameraRefused, match="serial"):
        pp.open_checked(wrong, calib)
    assert wrong._renderer is None  # stopped on refusal
    name = _name()
    code = pp.main(["--shm-name", name, "--summary", str(tmp_path / "s.json")], camera=SimRenderCamera(calib, serial="1"))
    assert code == pp.EXIT_CAMERA
    with pytest.raises(FileNotFoundError):
        ObjectTableShm.attach(name)  # nothing created
    assert "serial" in json.loads((tmp_path / "s.json").read_text())["exit_reason"]


def test_intrinsics_mismatch_is_refused(calib):
    c = calib.color
    intr = Intrinsics(c.fx + 1.0, c.fy, c.cx, c.cy, c.width, c.height, calib.depth_scale)
    fake = FakeCamera(np.zeros((c.height, c.width, 3), np.uint8), np.zeros((c.height, c.width), np.float32), intr)
    fake.serial = calib.serial
    with pytest.raises(pp.CameraRefused, match="intrinsics"):
        pp.open_checked(fake, calib)
    assert not fake.started


def test_provisional_pose_is_refused_without_the_flag(tmp_path):
    provisional = REPO / "tests" / "data" / "camera_provisional_2026-10-07.yaml"  # conftest.PROVISIONAL_CAMERA
    assert load_camera_calib(provisional).pose.method == "provisional"
    code = pp.main(["--camera-yaml", str(provisional), "--shm-name", _name(), "--summary", str(tmp_path / "s.json")])
    assert code == pp.EXIT_CALIB
    assert "provisional" in json.loads((tmp_path / "s.json").read_text())["exit_reason"]


def test_second_instance_is_refused(tmp_path):
    from so101_rl.servo.shm import InstanceLock

    name = _name()
    lock = InstanceLock(name, what="test holder")
    try:
        code = pp.main(["--backend", "sim", "--shm-name", name, "--summary", str(tmp_path / "s.json")])
    finally:
        lock.release()
    assert code == pp.EXIT_RUNNING


class _ClockCamera(FakeCamera):
    """FakeCamera with synthetic 30 fps frame times."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.t = 0

    def read(self, rgb_out, depth_m_out) -> Frame:
        fr = super().read(rgb_out, depth_m_out)
        self.t += round(1e9 / 30)
        return Frame(self.t, fr.ok)


def test_rate_below_the_camera_fps_skips_frames(calib, table):
    c = calib.color
    cam = _ClockCamera(np.zeros((c.height, c.width, 3), np.uint8), np.zeros((c.height, c.width), np.float32),
                       calib.color_intrinsics())  # fmt: skip
    cam.start()
    loop = pp.PerceptionLoop(cam, calib, table, rate_hz=10.0)
    for _ in range(30):
        loop.step()
    assert loop.stats.processed == 10 and loop.stats.skipped_rate == 20
    assert loop.summary()["frame_interval_ms"]["p50"] == pytest.approx(100.0, abs=0.1)


def test_camera_timeout_ends_the_loop(calib, table):
    c = calib.color
    cam = FakeCamera(np.zeros((c.height, c.width, 3), np.uint8), np.zeros((c.height, c.width), np.float32),
                     calib.color_intrinsics())  # fmt: skip
    cam.start()
    cam.fail_next = 10**6
    loop = pp.PerceptionLoop(cam, calib, table, camera_timeout_s=0.05)
    assert loop.run(max_frames=5) == "camera_timeout"
    assert loop.stats.processed == 0 and loop.stats.failed > 0


def test_out_of_workspace_detection_is_not_visible(cam, calib, table):
    cam.set_die(_die(*SCENES[0]))
    lo, hi = pp.default_workspace()
    loop = pp.PerceptionLoop(cam, calib, table, workspace=(lo + 10.0, hi + 10.0))
    loop.step()
    o = _read(table._shm.name.lstrip("/"))["objects"][0]
    assert not bool(o["visible"]) and loop.stats.out_of_workspace == 1


def test_process_sigterm_unlinks_and_writes_the_summary(tmp_path):
    """The real entry point as a child process (sim backend): writes the table, stops cleanly on SIGTERM, unlinks."""
    name = _name()
    summary = tmp_path / "summary.json"
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "2"}  # fmt: skip
    x, y, yaw = SCENES[0]
    p = subprocess.Popen(
        [sys.executable, "-m", "so101_rl.perception.process", "--backend", "sim", "--sim-die", str(x), str(y), str(yaw),
         "--shm-name", name, "--summary", str(summary)],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        rec = None
        t_end = time.monotonic() + 60
        while time.monotonic() < t_end and p.poll() is None:
            try:
                r = ObjectTableShm.attach(name)
            except FileNotFoundError:
                time.sleep(0.1)
                continue
            out = new_object_table()
            try:
                for _ in range(100):
                    if r.read(out) and int(out[0]["seq"]) >= 10:
                        rec = out[0].copy()
                        break
                    time.sleep(0.05)
            finally:
                r.close()
            break
        assert rec is not None, p.stderr.read() if p.poll() is not None else "no table"
        assert bool(rec["objects"][0]["visible"]) and int(rec["fiducial_ok"]) == 1
        p.send_signal(15)
        _, err = p.communicate(timeout=30)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    assert p.returncode == 0, err
    with pytest.raises(FileNotFoundError):
        ObjectTableShm.attach(name)
    doc = json.loads(summary.read_text())
    assert doc["exit_reason"] == "signal SIGTERM" and doc["table_created_and_unlinked"]
    assert doc["frames"]["processed"] >= 5 and doc["detection_rate"] == pytest.approx(1.0)
    assert doc["latency_ms"]["p50"] is not None and doc["calib"]["camera_pose_method"] == "arm-shape"
    assert doc["camera_check"]["fiducial"] == pp.FIDUCIAL_STATUS

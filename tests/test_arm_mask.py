"""The arm mask (plan section 10 2026-10-17 "Perception: arm mask"; ``perception/arm_mask.py``).

Scenes are MuJoCo renders from the calibrated colour camera (``sim/d435.py`` ``COLOR_CAM``, colour + depth aligned to
colour, as ``tests/test_die_convention.py`` and ``perception/sim_camera.py``). The sim arm is yellow, outside the red
HSV band, so a red arm part is made by recolouring one of its materials (the servos, the wrist motor holder): without
the mask the tracker takes it for the die, with the mask it is skipped. A die on the table near the arm and a die
held between open and closed jaws must stay detectable. No hardware: the servo state comes from a test-owned shm
segment or a fake joint source.
"""

from __future__ import annotations

import importlib.util
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from so101_rl.config import JOINT_NAMES, rad_to_ticks
from so101_rl.perception import process as pp
from so101_rl.perception.arm_mask import (
    ARM_BODIES,
    ARM_POINTS_NPZ,
    ArmGeometry,
    ArmMasker,
    ArmMaskSource,
    ServoStateReader,
    joint_angles,
)
from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.perception.die_tracker import HsvDepthDieTracker, table_workspace
from so101_rl.perception.object_table import ObjectTableShm
from so101_rl.sim.gripper_map import gripper_rad_to_ticks

REPO = Path(__file__).resolve().parents[1]
RED = (0.85, 0.08, 0.05, 1.0)
RED_PARTS = ("sts3215_03a_v1_material", "motor_holder_so101_wrist_v1_material")  # servos (horns), wrist holder
# die resting positions in the calibrated view, clear of the arm at home (tests/test_perception_process.py)
TABLE_SCENES = [(0.15, -0.13, 0.3), (0.14, -0.06, 0.7), (0.25, -0.12, 0.2), (0.12, -0.10, 0.0)]
GRASP_SITES = [(0.15, -0.13), (0.25, -0.12), (0.20, 0.0)]
ROLLS = [0.0, 0.5, 1.0, 1.5, 2.5, 3.0]
GRIPPER_CLOSED_ON_DIE = 0.0  # rad: jaw-tip box centres 27.8 mm apart = their 12 mm half widths + the 16 mm die
GRIPPER_OPEN = 0.6  # rad: 70 mm apart


class Scene:
    """The RL scene from the calibrated colour camera, posed and rendered on demand."""

    def __init__(self) -> None:
        from so101_rl.sim.d435 import COLOR_CAM
        from so101_rl.sim.model import build_spec

        self.calib = load_camera_calib()
        c = self.calib.color
        spec = build_spec(camera=self.calib)
        spec.visual.global_.offwidth, spec.visual.global_.offheight = c.width, c.height
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, c.height, c.width)
        self.cam = COLOR_CAM
        self.jq = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINT_NAMES])
        self.jv = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in JOINT_NAMES])
        self.dq = self.m.jnt_qposadr[self.m.joint("die_free").id]
        self.rgba0 = self.m.mat_rgba.copy()
        self.tips = (self.m.geom("jaw_fixed_tip").id, self.m.geom("jaw_moving_tip").id)
        self.gripper = self.m.body("gripper").id

    def pose(self, q: np.ndarray | None = None, die: tuple | None = None, red: str | None = None) -> None:
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        if q is not None:
            d.qpos[self.jq] = q
        d.qpos[self.dq : self.dq + 7] = (2.0, 2.0, 0.5, 1, 0, 0, 0) if die is None else die
        m.mat_rgba[:] = self.rgba0
        if red is not None:
            m.mat_rgba[m.material(red).id] = RED
        mujoco.mj_forward(m, d)

    def q(self) -> np.ndarray:
        return self.d.qpos[self.jq].copy()

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.r
        r.update_scene(self.d, camera=self.cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=self.cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        return rgb, depth

    def grip_centre(self) -> np.ndarray:
        return 0.5 * (self.d.geom_xpos[self.tips[0]] + self.d.geom_xpos[self.tips[1]])

    def ik_down(self, target: np.ndarray, roll: float) -> np.ndarray:
        """Arm angles with the jaw-tip midpoint at ``target`` and the fingers pointing down (damped least squares
        on shoulder_pan .. wrist_flex; wrist_roll = ``roll``)."""
        m, d = self.m, self.d
        self.pose()
        d.qpos[self.jq[4]] = roll
        jp = [np.zeros((3, m.nv)) for _ in range(2)]
        jr = np.zeros((3, m.nv))
        for _ in range(300):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = self.grip_centre()
            ax = (p - d.xpos[self.gripper]) / np.linalg.norm(p - d.xpos[self.gripper])
            for k, g in enumerate(self.tips):
                mujoco.mj_jac(m, d, jp[k], None, d.geom_xpos[g], m.geom_bodyid[g])
            mujoco.mj_jacBody(m, d, None, jr, self.gripper)
            J = np.vstack([0.5 * (jp[0] + jp[1])[:, self.jv[:4]], 0.3 * jr[:, self.jv[:4]]])
            e = np.concatenate([target - p, 0.3 * np.cross(ax, [0.0, 0.0, -1.0])])
            d.qpos[self.jq[:4]] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        mujoco.mj_kinematics(m, d)
        assert np.linalg.norm(target - self.grip_centre()) < 1e-3
        return self.q()

    def die_in_jaws(self, q: np.ndarray) -> tuple:
        """Die pose resting on the table between the fingertips of pose ``q``, square to the gripper."""
        from so101_rl.sim.model import DIE_EDGE_M, TABLE_Z_M

        self.pose(q)
        c = self.grip_centre()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.xmat[self.gripper])
        return (c[0], c[1], TABLE_Z_M + DIE_EDGE_M / 2, *quat)


@pytest.fixture(scope="module")
def scene():
    try:
        s = Scene()
    except Exception as exc:  # noqa: BLE001 - no OpenGL context on this host
        pytest.skip(f"no MuJoCo offscreen rendering: {exc}")
    yield s
    s.r.close()


@pytest.fixture(scope="module")
def masker(scene):
    return ArmMasker(scene.calib)


@pytest.fixture(scope="module")
def tracker(scene):
    return HsvDepthDieTracker.from_camera_calib(scene.calib)


@pytest.fixture(scope="module")
def on_table(scene, tracker):
    lo, hi = table_workspace()
    T = scene.calib.T_base_color

    def accept(det) -> bool:
        p = tracker.centroid_base(det, T)
        return bool(np.all(p >= lo) and np.all(p <= hi))

    return accept


def _die(x: float, y: float, yaw: float = 0.0) -> tuple:
    from so101_rl.sim.model import DIE_EDGE_M, TABLE_Z_M

    return (x, y, TABLE_Z_M + DIE_EDGE_M / 2, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))


def _ticks(q: np.ndarray, calibration) -> np.ndarray:
    t = rad_to_ticks(q, calibration).astype(np.float64)
    t[-1] = gripper_rad_to_ticks(float(q[-1]), calibration)
    return t


# ------------------------------------------------------------------------------------------------ geometry
def test_points_file_is_the_mjcf_surface():
    """``arm_points.npz`` is what ``scripts/extract_arm_points.py`` makes from the current MJCF."""
    spec = importlib.util.spec_from_file_location("extract_arm_points", REPO / "scripts" / "extract_arm_points.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh = mod.arm_points()
    saved = np.load(ARM_POINTS_NPZ)
    assert [str(b) for b in saved["bodies"]] == list(ARM_BODIES)
    for k in ("body", "grasp_body"):
        np.testing.assert_array_equal(saved[k], fresh[k])
    for k in ("points", "grasp_corners"):
        np.testing.assert_allclose(saved[k], fresh[k], atol=1e-6)


def test_fk_matches_mujoco(scene):
    g = ArmGeometry()
    rng = np.random.default_rng(1)
    for _ in range(25):
        q = rng.uniform(-1.6, 1.6, 6)
        scene.pose(q)
        T = g.body_frames(q)
        for i, b in enumerate(ARM_BODIES):
            bid = scene.m.body(b).id
            np.testing.assert_allclose(T[i, :3, 3], scene.d.xpos[bid], atol=1e-9)
            np.testing.assert_allclose(T[i, :3, :3], scene.d.xmat[bid].reshape(3, 3), atol=1e-9)


def test_mask_covers_the_rendered_arm(scene, masker):
    """The dilated point render covers every arm pixel MuJoCo draws (segmentation), at home and in a grasp pose,
    and adds no more than the dilation band."""
    base = scene.m.body("base").id

    def arm_body(b: int) -> bool:
        while b > 0:
            if b == base:
                return True
            b = int(scene.m.body_parentid[b])
        return False

    arm_geom = np.array([arm_body(int(scene.m.geom_bodyid[g])) for g in range(scene.m.ngeom)])
    for q in (None, scene.ik_down(np.array([0.20, 0.0, 0.03]), 0.8)):
        scene.pose(q)
        r = scene.r
        r.enable_segmentation_rendering()
        r.update_scene(scene.d, camera=scene.cam)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        truth = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & arm_geom[np.clip(seg[..., 0], 0, None)] & (seg[..., 0] >= 0)
        cover = masker.render(scene.q()) < 1e3
        assert truth.sum() > 10_000
        assert (cover & truth).sum() / truth.sum() > 0.995
        assert (cover & ~truth).sum() < 0.4 * truth.sum()


# ------------------------------------------------------------------------------------------------ the rule
@pytest.mark.parametrize("part", RED_PARTS)
def test_red_arm_part_is_taken_for_the_die_without_the_mask_and_rejected_with_it(scene, masker, tracker, on_table, part):
    scene.pose(red=part)
    rgb, depth = scene.render()
    det = tracker.detect(rgb, depth, accept=on_table)
    assert det is not None, "the red arm part should pass the table-only rule (the gap the mask closes)"
    before = tracker.rejected_excluded
    assert tracker.detect(rgb, depth, accept=on_table, exclude=masker.mask(scene.q(), depth)) is None
    assert tracker.rejected_excluded > before


@pytest.mark.parametrize("x,y,yaw", TABLE_SCENES)
def test_red_arm_part_no_longer_hides_the_die(scene, masker, tracker, on_table, x, y, yaw):
    scene.pose(die=_die(x, y, yaw), red=RED_PARTS[0])
    rgb, depth = scene.render()
    truth = scene.d.xpos[scene.m.body("die").id]
    det = tracker.detect(rgb, depth, accept=on_table, exclude=masker.mask(scene.q(), depth))
    assert det is not None
    got = tracker.centroid_base(det, scene.calib.T_base_color)
    assert np.linalg.norm(got - truth) < 0.006


def test_die_on_the_table_near_the_arm_stays_visible(scene, masker, tracker, on_table):
    """Dice on the table around the fingertips (3-5 cm from the grasp point, 8 directions, 2 poses): every die the
    tracker sees without the mask it still sees with it, at the same place."""
    seen = 0
    for (x, y), roll in (((0.20, 0.0), 0.0), ((0.22, -0.08), 1.0)):
        from so101_rl.sim.model import TABLE_Z_M

        q = scene.ik_down(np.array([x, y, TABLE_Z_M + 0.04]), roll)
        q[5] = GRIPPER_OPEN
        for rad in (0.03, 0.05):
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                scene.pose(q, die=_die(x + rad * np.cos(a), y + rad * np.sin(a), a))
                rgb, depth = scene.render()
                plain = tracker.detect(rgb, depth, accept=on_table)
                if plain is None:
                    continue
                seen += 1
                det = tracker.detect(rgb, depth, accept=on_table, exclude=masker.mask(q, depth))
                assert det is not None, (x, y, roll, rad, a)
                assert abs(det.u - plain.u) < 1e-6 and abs(det.v - plain.v) < 1e-6
    assert seen >= 16


@pytest.mark.parametrize("gripper", [GRIPPER_CLOSED_ON_DIE, GRIPPER_OPEN], ids=["closed", "open"])
def test_die_between_the_jaws_stays_visible(scene, masker, tracker, on_table, gripper):
    """A die on the table between the fingertips (3 places x 6 wrist rolls): every one the tracker sees without
    the mask it sees with it; the arm covers less than half of every such blob (the grasp region)."""
    from so101_rl.sim.model import TABLE_Z_M

    seen = 0
    for x, y in GRASP_SITES:
        for roll in ROLLS:
            q = scene.ik_down(np.array([x, y, TABLE_Z_M + 0.012]), roll)
            q[5] = gripper
            die = scene.die_in_jaws(q)
            scene.pose(q, die=die)
            rgb, depth = scene.render()
            plain = tracker.detect(rgb, depth, accept=on_table)
            if plain is None:  # hidden by a jaw (the ring test): the mask cannot make it worse
                continue
            seen += 1
            det = tracker.detect(rgb, depth, accept=on_table, exclude=masker.mask(q, depth))
            assert det is not None, (x, y, roll)
            assert abs(det.u - plain.u) < 1e-6 and abs(det.v - plain.v) < 1e-6
    assert seen >= 10


def test_without_the_grasp_region_closed_jaws_hide_the_die(scene, tracker, on_table):
    """Without the grasp region (silhouette + depth gate only) the closed jaws eat the die: the region is needed."""
    from so101_rl.sim.model import TABLE_Z_M

    bare = ArmMasker(scene.calib, grasp_margin_m=0.0)
    bare._grasp_region = lambda *a: None
    q = scene.ik_down(np.array([0.20, 0.0, TABLE_Z_M + 0.012]), 0.0)
    q[5] = GRIPPER_CLOSED_ON_DIE
    scene.pose(q, die=scene.die_in_jaws(q))
    rgb, depth = scene.render()
    assert tracker.detect(rgb, depth, accept=on_table) is not None
    assert tracker.detect(rgb, depth, accept=on_table, exclude=bare.mask(q, depth)) is None


def test_mask_cost_per_frame(scene, masker, capsys):
    """Budget: < 5 ms per 640 x 480 frame on the CPU (FK + projection + z-buffer + dilation + gates)."""
    q = scene.ik_down(np.array([0.20, 0.0, 0.03]), 0.8)
    scene.pose(q)
    _, depth = scene.render()
    masker.mask(q, depth)
    ts = []
    for _ in range(60):
        t0 = time.perf_counter()
        masker.mask(q, depth)
        ts.append((time.perf_counter() - t0) * 1e3)
    p50, p95 = np.percentile(ts, [50, 95])
    with capsys.disabled():
        print(f"\narm mask {masker.w}x{masker.h}: p50 {p50:.2f} ms, p95 {p95:.2f} ms, max {max(ts):.2f} ms")
    assert p50 < 5.0


# ------------------------------------------------------------------------------------------------ servo state
@pytest.fixture()
def servo_shm():
    from so101_rl.servo.shm import ServoShm

    s = ServoShm.create(f"so101_armtest_{uuid.uuid4().hex[:8]}")
    yield s
    s.close()


def _publish(shm, ticks, *, age_ns: int = 0, read_ok: int = 1) -> None:
    from so101_rl.servo.shm import new_state

    st = new_state()
    st["t_mono_ns"] = time.monotonic_ns() - age_ns
    st["tick"] = 1
    st["present_ticks"][0] = ticks
    st["read_ok"] = read_ok
    shm.write_state(st)


def _segment_name(shm) -> str:
    return shm._shm.name.lstrip("/")


def test_servo_state_reader_fresh_stale_failed_and_read_only(servo_shm):
    name = _segment_name(servo_shm)
    r = ServoStateReader(name)
    try:
        assert r.read() is None and r.status == "stale"  # attached, nothing published yet
        ticks = np.array([2048, 1500, 2600, 2100, 2048, 1900])
        _publish(servo_shm, ticks)
        raw = bytes(servo_shm._shm.buf)
        np.testing.assert_array_equal(r.read(), ticks)
        assert r.status == "ok"
        assert bytes(servo_shm._shm.buf) == raw, "the reader wrote into the servo segment"
        _publish(servo_shm, ticks, age_ns=150_000_000)
        assert r.read() is None and r.status == "stale"
        _publish(servo_shm, ticks, read_ok=0)
        assert r.read() is None and r.status == "read_failed"
    finally:
        r.close()
    assert servo_shm.last_cmd_seq == 0  # nothing ever wrote a command


def test_servo_state_reader_without_a_servo():
    r = ServoStateReader(f"so101_noservo_{uuid.uuid4().hex[:8]}")
    assert r.read() is None and r.status == "no_servo"
    r.close()


# ------------------------------------------------------------------------------------------------ the process
class _Joints:
    def __init__(self, ticks: np.ndarray | None, status: str = "ok") -> None:
        self.ticks, self.status, self.name = ticks, status, "fake"

    def read(self) -> np.ndarray | None:
        return self.ticks

    def close(self) -> None:
        pass


@pytest.fixture()
def table():
    t = ObjectTableShm.create(f"so101_perc_{uuid.uuid4().hex[:8]}")
    yield t
    t.close()


@pytest.fixture(scope="module")
def red_cam(scene):
    from so101_rl.perception.sim_camera import SimRenderCamera

    cam = SimRenderCamera(scene.calib)
    pp.open_checked(cam, scene.calib)
    m = cam._model
    m.mat_rgba[m.material(RED_PARTS[0]).id] = RED
    yield cam
    cam.stop()


def test_loop_masks_the_arm_with_fresh_joint_state(scene, red_cam, table, calibration):
    q = scene.ik_down(np.array([0.22, -0.02, 0.03]), 0.5)
    red_cam.set_arm(q)
    red_cam.set_die(None)
    arm = ArmMaskSource(ArmMasker(scene.calib), _Joints(_ticks(q, calibration)), calibration)
    np.testing.assert_allclose(joint_angles(_ticks(q, calibration), calibration), q, atol=2e-3)
    loop = pp.PerceptionLoop(red_cam, scene.calib, table, arm=arm)
    loop.step()
    assert not loop.last_visible and loop.stats.on_arm >= 1
    s = loop.summary()["arm_mask"]
    assert s["enabled"] and s["active_fraction"] == 1.0 and s["mask_ms"]["n"] == 1
    # without the mask the same frame reports the red servo as the die
    plain = pp.PerceptionLoop(red_cam, scene.calib, table)
    plain.step()
    assert plain.last_visible
    assert plain.summary()["arm_mask"] == {"enabled": False, "reason": "not configured", "active_fraction": 0.0}
    red_cam.set_arm(None)


def test_loop_without_fresh_joint_state_runs_as_before_and_counts(scene, red_cam, table, calibration):
    red_cam.set_arm(None)
    red_cam.set_die(None)
    arm = ArmMaskSource(ArmMasker(scene.calib), _Joints(None, "stale"), calibration)
    loop = pp.PerceptionLoop(red_cam, scene.calib, table, arm=arm)
    for _ in range(3):
        loop.step()
    s = loop.summary()["arm_mask"]
    assert s["active_fraction"] == 0.0 and s["inactive_frames"] == {"stale": 3}
    assert loop.last_visible  # no mask: the red servo wins, as before the mask existed
    assert loop.stats.on_arm == 0


def test_main_without_a_servo_counts_no_servo(scene, tmp_path, monkeypatch, calibration):
    """``--servo-shm`` naming no segment: the process runs, the summary says the mask was never active and why."""
    from so101_rl import config
    from so101_rl.perception.sim_camera import SimRenderCamera

    real_load_config = config.load_config
    monkeypatch.setattr(config, "load_config", lambda: real_load_config(calibration=calibration))
    monkeypatch.setattr(config, "load_calibration", lambda _p: calibration)  # tests/data copy, not ~/.cache
    out = tmp_path / "s.json"
    rc = pp.main(["--backend", "sim", "--shm-name", f"so101_perc_{uuid.uuid4().hex[:8]}", "--max-frames", "3",
                  "--summary", str(out), "--servo-shm", f"so101_noservo_{uuid.uuid4().hex[:8]}"],
                 camera=SimRenderCamera(scene.calib))  # fmt: skip
    import json

    doc = json.loads(out.read_text())
    assert rc == 0, doc
    a = doc["arm_mask"]
    assert a["enabled"] and a["active_fraction"] == 0.0 and a["inactive_frames"] == {"no_servo": 3}
    assert "read-only" in doc["arm_mask_source"]


def test_main_no_arm_mask_flag(scene, tmp_path):
    from so101_rl.perception.sim_camera import SimRenderCamera

    out = tmp_path / "s.json"
    rc = pp.main(["--backend", "sim", "--shm-name", f"so101_perc_{uuid.uuid4().hex[:8]}", "--max-frames", "2",
                  "--summary", str(out), "--no-arm-mask"], camera=SimRenderCamera(scene.calib))  # fmt: skip
    import json

    doc = json.loads(out.read_text())
    assert rc == 0 and doc["arm_mask"]["enabled"] is False and "--no-arm-mask" in doc["arm_mask"]["reason"]


# ------------------------------------------------------------------------------------------------ the plane check
# (plan §10 2026-10-23): the camera-moved table-plane refit leaves the arm mask's pixels out. On the real rig the arm
# low over the table corrupted the refit (a false 2.25 deg / 12.6 mm from 1741 points latched fiducial_ok off). A
# clean MuJoCo depth render does not reproduce the D435's mixed pixels, so they are put in: inside the arm's
# silhouette the depth reads as a surface tilted about the table, within the refit's bands.
def _mixed_pixels(calib, depth: np.ndarray, mask: np.ndarray, k: float = 0.03) -> np.ndarray:
    c = calib.color
    v, u = np.mgrid[0 : c.height, 0 : c.width]
    rays = np.stack([(u - c.cx) / c.fx, (v - c.cy) / c.fy, np.ones_like(u, dtype=np.float64)], axis=-1)
    z_table = -calib.table_plane.height_m / (rays @ np.asarray(calib.table_plane.normal_color))
    out = depth.copy()
    vc = v[mask].mean()
    out[mask] = (z_table * (1.0 - k * (v - vc) / 100.0))[mask]
    return out


@pytest.fixture(scope="module")
def arm_over_table(scene, masker):
    q = scene.ik_down(np.array([0.20, 0.0, 0.03]), 0.8)  # the jaws 3 cm over the table, the forearm across it
    scene.pose(q)
    rgb, depth = scene.render()
    m = masker.mask(q, depth)
    return q, rgb, depth, _mixed_pixels(scene.calib, depth, m), m


def test_plane_check_leaves_the_arm_out(scene, arm_over_table):
    _q, _rgb, depth, bad, m = arm_over_table
    chk = pp.TablePlaneCheck(scene.calib)
    assert m.sum() > 20_000 and chk.check(depth).ok  # the clean render passes with or without the mask
    plain = chk.check(bad)
    assert not plain.ok  # the arm's mixed pixels drag the refit off: a false "camera moved"
    masked = chk.check(bad, m)
    assert masked.ok and masked.angle_deg < 0.5 and abs(masked.dh_m) < 0.001 and masked.n_points >= pp.PLANE_MIN_POINTS


def test_loop_feeds_the_arm_mask_to_the_plane_check(scene, arm_over_table, table, calibration):
    """Through ``PerceptionLoop``: with fresh joint state the arm mask goes into the plane check and fiducial_ok stays
    set over PLANE_TRIP_AFTER checks; without it the same frames latch the gate off (the session's false trip)."""
    from fiducial_synth import SequenceCamera

    q, rgb, _depth, bad, _m = arm_over_table
    for masked in (True, False):
        cam = SequenceCamera(scene.calib, lambda: (rgb, bad))
        cam.start()
        arm = ArmMaskSource(ArmMasker(scene.calib), _Joints(_ticks(q, calibration) if masked else None, "stale"),
                            calibration)  # fmt: skip
        gate = pp.CameraMovedGate(pp.TablePlaneCheck(scene.calib))
        loop = pp.PerceptionLoop(cam, scene.calib, table, gate=gate, plane_every=1, arm=arm)
        for _ in range(pp.PLANE_TRIP_AFTER):
            loop.step()
        rep = loop.summary()["camera_check"]["table_plane"]
        if masked:
            assert gate.ok and rep["arm_masked_checks"] == pp.PLANE_TRIP_AFTER and rep["fails"] == 0
        else:
            assert not gate.ok and rep["tripped"] and rep["arm_masked_checks"] == 0

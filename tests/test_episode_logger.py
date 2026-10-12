"""Task 6.1: episode logger stores the APPLIED action, writes atomically, honours the full-frame policy."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from so101_rl.logging.episode_logger import EpisodeLogger, StepRecord

N = 5


def make_rec(i: int, *, small: bool = True, full: bool = False, cmd=None, applied=None) -> StepRecord:
    rng = np.random.default_rng(i)
    return StepRecord(
        t_mono_ns=1_000 + i,
        obs={
            "observation": rng.standard_normal(15).astype(np.float32),
            "achieved_goal": rng.standard_normal(3).astype(np.float32),
            "desired_goal": rng.standard_normal(3).astype(np.float32),
        },
        action_cmd=np.full(6, 0.5, np.float32) if cmd is None else cmd,
        applied_delta_ticks=np.full(6, 7.0, np.float32) if applied is None else applied,
        target_ticks=np.full(6, 2000 + i, np.int32),
        present_ticks=np.full(6, 1990 + i, np.int32),
        reward=-0.1 * i,
        terminated=False,
        truncated=i == N - 1,
        trips=0,
        verifier=0,
        frame_small=np.full((96, 96, 3), i, np.uint8) if small else None,
        frame_full=np.full((480, 640, 3), i, np.uint8) if full else None,
    )


def run_episode(lg: EpisodeLogger, outcome="failure", *, full=True, **kw):
    lg.start_episode({"goal": [0.1, 0.0, 0.2]})
    for i in range(N):
        lg.log_step(make_rec(i, full=full, **kw))
    return lg.end_episode(outcome)


def test_round_trip(tmp_path):
    lg = EpisodeLogger(tmp_path, "run1", config_dump={"a": np.int64(3), "arr": np.arange(2)})
    p = run_episode(lg, "success")
    assert p == tmp_path / "run1" / "episodes" / "ep_000000.h5"
    assert not list(p.parent.glob("*.tmp"))
    with h5py.File(p, "r") as f:
        assert f["t_mono_ns"][:].tolist() == [1000 + i for i in range(N)]
        assert f["obs/observation"].shape == (N, 15) and f["obs/observation"].dtype == np.float32
        assert f["target_ticks"].dtype == np.int32 and f["present_ticks"].shape == (N, 6)
        np.testing.assert_allclose(f["reward"][:], [-0.1 * i for i in range(N)], rtol=1e-6)
        assert f["truncated"][:].tolist() == [0, 0, 0, 0, 1] and f["truncated"].dtype == np.uint8
        assert f["trips"].dtype == np.uint32 and f["verifier"].dtype == np.uint8
        assert f["frames_small"].shape == (N, 96, 96, 3)
        assert f["frames_small"].compression == "gzip"
        assert int(f["frames_small"][3].mean()) == 3
        a = dict(f.attrs)
        assert a["run_id"] == "run1" and a["episode"] == 0 and a["outcome"] == "success"
        assert json.loads(a["config_dump"]) == {"a": 3, "arr": [0, 1]}
        assert a["git_sha"] and a["created"]


def test_episode_index_increments(tmp_path):
    lg = EpisodeLogger(tmp_path, "r")
    p0 = run_episode(lg)
    p1 = run_episode(lg)
    assert (p0.name, p1.name) == ("ep_000000.h5", "ep_000001.h5")
    assert EpisodeLogger(tmp_path, "r").start_episode({}) == 2  # new logger continues numbering


def test_logs_applied_action_not_commanded(tmp_path):
    cmd = np.full(6, 1.0, np.float32)  # policy asked for +1.0 (full scale)
    applied = np.array([3, -2, 0, 45, -45, 9], np.float32)  # what the clamp let through
    lg = EpisodeLogger(tmp_path, "r")
    p = run_episode(lg, cmd=cmd, applied=applied)
    with h5py.File(p, "r") as f:
        np.testing.assert_array_equal(f["applied_delta_ticks"][0], applied)
        np.testing.assert_array_equal(f["action_cmd"][0], cmd)
        assert not np.array_equal(f["applied_delta_ticks"][0], f["action_cmd"][0])


@pytest.mark.parametrize(
    ("policy", "outcome", "expect_full"),
    [
        ("failures", "success", False),
        ("failures", "failure", True),
        ("failures", "truncated", True),
        ("failures", "trip", True),
        ("never", "failure", False),
        ("always", "success", True),
    ],
)
def test_full_frame_policy(tmp_path, policy, outcome, expect_full):
    lg = EpisodeLogger(tmp_path, "r", keep_full_frames=policy)
    p = run_episode(lg, outcome)
    with h5py.File(p, "r") as f:
        assert ("frames_full" in f) is expect_full
        if expect_full:
            assert f["frames_full"].shape == (N, 480, 640, 3)
        assert f["frames_small"].shape[0] == N  # small frames are always kept


def test_crash_before_end_leaves_only_tmp(tmp_path):
    lg = EpisodeLogger(tmp_path, "r")
    lg.start_episode({})
    for i in range(3):
        lg.log_step(make_rec(i))
    lg.close()  # process dies: handle released, end_episode never called
    files = sorted(p.name for p in (tmp_path / "r" / "episodes").iterdir())
    assert files == ["ep_000000.h5.tmp"]
    assert not list((tmp_path / "r" / "episodes").glob("*.h5"))


def test_no_frames_episode(tmp_path):
    lg = EpisodeLogger(tmp_path, "r")
    p = run_episode(lg, small=False, full=False)
    with h5py.File(p, "r") as f:
        assert "frames_small" not in f and "frames_full" not in f


def test_misuse_rejected(tmp_path):
    lg = EpisodeLogger(tmp_path, "r")
    with pytest.raises(RuntimeError):
        lg.log_step(make_rec(0))
    with pytest.raises(RuntimeError):
        lg.end_episode("success")
    lg.start_episode({})
    with pytest.raises(RuntimeError):
        lg.start_episode({})
    bad = make_rec(0)
    bad.applied_delta_ticks = np.zeros(5, np.float32)
    with pytest.raises(ValueError, match="applied_delta_ticks"):
        lg.log_step(bad)
    bad = make_rec(0)
    bad.obs = {"observation": bad.obs["observation"]}
    with pytest.raises(ValueError, match="obs"):
        lg.log_step(bad)
    lg.log_step(make_rec(0))
    with pytest.raises(ValueError, match="frame_small"):
        lg.log_step(make_rec(1, small=False))
    with pytest.raises(ValueError, match="outcome"):
        lg.end_episode(SimpleNamespace())  # type: ignore[arg-type]
    lg.close()
    with pytest.raises(ValueError, match="policy"):
        EpisodeLogger(tmp_path, "r", keep_full_frames="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="run_id"):
        EpisodeLogger(tmp_path, "../x")


def test_observation_width_is_fixed_per_file_by_its_first_step(tmp_path):
    """15 values (excitation) or 21 (the reach policy obs): each file takes its first step's width, then keeps it."""
    from so101_rl.logging.checks import check_episode

    lg = EpisodeLogger(tmp_path, "r", keep_full_frames="never")
    lg.start_episode({})
    for i in range(N):
        rec = make_rec(i, small=False)
        rec.obs["observation"] = np.full(21, i, np.float32)
        lg.log_step(rec)
    wide = rec
    path = lg.end_episode("truncated")
    with h5py.File(path, "r") as f:
        assert f["obs/observation"].shape == (N, 21)
    assert check_episode(path).ok
    lg.start_episode({})
    lg.log_step(make_rec(0, small=False))  # a new file: 15 wide
    with pytest.raises(ValueError, match="obs.observation"):
        lg.log_step(wide)
    with pytest.raises(ValueError, match="1-D"):
        lg2 = EpisodeLogger(tmp_path, "r2")
        lg2.start_episode({})
        bad = make_rec(0)
        bad.obs["observation"] = np.zeros((3, 5), np.float32)
        lg2.log_step(bad)

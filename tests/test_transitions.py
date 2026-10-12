"""``logging.transitions``: training arrays from F8 episode files written by ``deploy-policy``."""

import numpy as np
import pytest

from so101_rl.logging.episode_logger import EpisodeLogger, StepRecord
from so101_rl.logging.transitions import load_episode, load_transitions
from so101_rl.rl.deploy_policy import EpisodeTrace, episode_outcome, write_transitions

MAX_DELTA = np.array([40, 40, 40, 40, 40, 40], np.float32)


def _obs(t: float) -> dict[str, np.ndarray]:
    return {"observation": np.full(15, t, np.float32), "achieved_goal": np.full(3, t, np.float32),
            "desired_goal": np.full(3, -1.0, np.float32)}  # fmt: skip


def _trace(n: int) -> EpisodeTrace:
    tr = EpisodeTrace()
    for t in range(n):
        tr.steps.append(StepRecord(
            t_mono_ns=t, obs=_obs(t), action_cmd=np.full(6, 0.5, np.float32),
            applied_delta_ticks=np.array([20, -20, 60, 0, 0, 0], np.float32), target_ticks=np.full(6, 2048, np.int32),
            present_ticks=np.full(6, 2048, np.int32), reward=-1.0, terminated=t == n - 1, truncated=False, trips=0,
            verifier=1 if t == n - 1 else 0,
        ))  # fmt: skip
    tr.final_obs = _obs(n)
    return tr


def _ep(i: int, **kw) -> dict:
    return {"episode": i, "speed": 1.0, "goal": [0.2, 0.0, 0.1], "success": True, "tripped": False,
            "servo_stalled": False, **kw}  # fmt: skip


def test_rows_pair_with_the_next_observation_and_the_applied_action(tmp_path):
    log = EpisodeLogger(tmp_path, "run", keep_full_frames="never")
    paths = [write_transitions(log, _trace(n), _ep(i), MAX_DELTA) for i, n in enumerate((3, 2))]
    tr = load_transitions(paths)
    assert tr["episode"].tolist() == [0, 0, 0, 1, 1]
    assert tr["observation"][:, 0].tolist() == [0, 1, 2, 0, 1]
    assert tr["next_observation"][:, 0].tolist() == [1, 2, 3, 1, 2]  # the last row's next obs is meta.final_obs
    np.testing.assert_allclose(tr["action"][0], [0.5, -0.5, 1.0, 0, 0, 0])  # applied / max_delta, clipped
    np.testing.assert_allclose(tr["action_cmd"], 0.5)
    assert tr["terminated"].tolist() == [False, False, True, False, True]


def test_outcome_and_refusals(tmp_path):
    assert episode_outcome(_ep(0)) == "success"
    assert episode_outcome(_ep(0, success=False, tripped=True)) == "trip"
    assert episode_outcome(_ep(0, success=False, servo_stalled=True)) == "failure"
    assert episode_outcome(_ep(0, success=False)) == "truncated"
    log = EpisodeLogger(tmp_path, "run", keep_full_frames="never")
    with pytest.raises(ValueError, match="no steps"):
        write_transitions(log, EpisodeTrace(), _ep(0), MAX_DELTA)
    log.start_episode({"segment": 0})  # an excitation-style episode: no final_obs
    for r in _trace(2).steps:
        log.log_step(r)
    with pytest.raises(ValueError, match="final_obs"):
        load_episode(log.end_episode("truncated"))
    with pytest.raises(ValueError, match="no episode files"):
        load_transitions([])

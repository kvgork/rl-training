"""On-disk layout of one episode file (F8). Shared by the writer and the artifact checks."""

from __future__ import annotations

from typing import Final

import numpy as np

from so101_rl.config.schema import N_JOINTS, TICKS_PER_REV

ObsDict = dict[str, np.ndarray]  # keys exactly: "observation", "achieved_goal", "desired_goal"

ANY_WIDTH: Final[int] = -1  # a per-step shape entry fixed per file by its first step (obs/observation: 15 excitation, 21 reach)
GOAL_DIM: Final[int] = 3
FRAME_SMALL_SHAPE: Final[tuple[int, int, int]] = (96, 96, 3)
FRAME_FULL_SHAPE: Final[tuple[int, int, int]] = (480, 640, 3)

FINAL_SUFFIX: Final[str] = ".h5"
TMP_SUFFIX: Final[str] = ".h5.tmp"
OUTCOMES: Final[tuple[str, ...]] = ("success", "failure", "truncated", "trip")
MAX_TICK: Final[int] = TICKS_PER_REV - 1

# name -> (dtype, per-step shape). Every one of these has N rows along axis 0.
REQUIRED_DATASETS: Final[dict[str, tuple[str, tuple[int, ...]]]] = {
    "t_mono_ns": ("<i8", ()),
    "obs/observation": ("<f4", (ANY_WIDTH,)),
    "obs/achieved_goal": ("<f4", (GOAL_DIM,)),
    "obs/desired_goal": ("<f4", (GOAL_DIM,)),
    "action_cmd": ("<f4", (N_JOINTS,)),
    "applied_delta_ticks": ("<f4", (N_JOINTS,)),  # the post-clamp delta actually applied (L4)
    "target_ticks": ("<i4", (N_JOINTS,)),
    "present_ticks": ("<i4", (N_JOINTS,)),
    "reward": ("<f4", ()),
    "terminated": ("u1", ()),
    "truncated": ("u1", ()),
    "trips": ("<u4", ()),
    "verifier": ("u1", ()),
}
OPTIONAL_DATASETS: Final[dict[str, tuple[str, tuple[int, ...]]]] = {
    "frames_small": ("u1", FRAME_SMALL_SHAPE),
    "frames_full": ("u1", FRAME_FULL_SHAPE),
}
FLOAT_DATASETS: Final[tuple[str, ...]] = tuple(
    k for k, (dt, _) in REQUIRED_DATASETS.items() if dt == "<f4"
)
REQUIRED_ATTRS: Final[tuple[str, ...]] = (
    "run_id",
    "episode",
    "outcome",
    "config_dump",
    "git_sha",
    "created",
)


def episode_name(index: int) -> str:
    return f"ep_{index:06d}"

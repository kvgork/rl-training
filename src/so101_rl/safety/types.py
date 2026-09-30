"""Safety enums and the preallocated verdict (plan F5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag
from typing import Final

import numpy as np

from so101_rl.config import N_JOINTS


class TripCause(IntFlag):
    NONE = 0
    NAN = 1
    RANGE = 2
    TRACKING = 4
    JOINT_LIMIT = 8
    BOX = 16
    THERMAL = 32
    PROTECTION_BIT = 64
    STUCK = 128
    STALE_CMD = 256
    BUS_FAIL = 512
    HEARTBEAT_FAIL = 1024
    OOD = 2048
    USER_STOP = 4096
    CALIB_MISMATCH = 8192
    DEADMAN = 16384
    BOX_PRESENT = 32768


class SmState(IntEnum):
    BOOT = 0
    CHECK = 1
    IDLE_TORQUE_OFF = 2
    RUN = 3
    HOLD = 4
    COOLDOWN = 5
    SAFE_STOPPING = 6
    TORQUE_OFF = 7
    LATCHED = 8


class ResumePolicy(Enum):
    AUTO = "auto"
    FRESH_PROCESS = "fresh_process_and_checkpoint"
    HUMAN_PHYSICAL_RESET = "human_physical_reset"


RESUME_BY_CAUSE: Final[dict[TripCause, ResumePolicy]] = {
    TripCause.THERMAL: ResumePolicy.AUTO,  # <=3/h, then HUMAN
    TripCause.STALE_CMD: ResumePolicy.AUTO,  # HOLD->RUN on fresh seq; after SAFE_STOPPING -> FRESH_PROCESS
    TripCause.NAN: ResumePolicy.FRESH_PROCESS,
    TripCause.STUCK: ResumePolicy.FRESH_PROCESS,
    TripCause.OOD: ResumePolicy.FRESH_PROCESS,
    TripCause.RANGE: ResumePolicy.FRESH_PROCESS,
    TripCause.BOX_PRESENT: ResumePolicy.FRESH_PROCESS,
    TripCause.DEADMAN: ResumePolicy.HUMAN_PHYSICAL_RESET,
    TripCause.HEARTBEAT_FAIL: ResumePolicy.HUMAN_PHYSICAL_RESET,
    TripCause.BUS_FAIL: ResumePolicy.HUMAN_PHYSICAL_RESET,
    TripCause.PROTECTION_BIT: ResumePolicy.HUMAN_PHYSICAL_RESET,
    TripCause.CALIB_MISMATCH: ResumePolicy.HUMAN_PHYSICAL_RESET,
    TripCause.USER_STOP: ResumePolicy.HUMAN_PHYSICAL_RESET,
}
# TRACKING, JOINT_LIMIT, BOX are *clamps* (reported in clamped_mask / trips), not state-machine trips.


@dataclass
class Verdict:
    """Preallocated, filled by ``SafetyPipeline.apply``."""

    target_ticks: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS, dtype=np.int32))
    goal_ticks: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS, dtype=np.int32))
    applied_delta_ticks: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS, dtype=np.float32))
    clamped_mask: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS, dtype=bool))
    trips: TripCause = TripCause.NONE

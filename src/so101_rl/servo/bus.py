"""Bus protocol (plan F2). Only the servo process touches the serial bus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from so101_rl.config import N_JOINTS, MotorCalibration

BLOCK_ADDR = 56  # Present_Position
BLOCK_LEN = 11  # 56..66: position, velocity, load, voltage, temperature, status, moving


class BlockRead:
    """Preallocated result of one block read. All arrays are int32[6] unless noted."""

    __slots__ = ("load", "moving", "ok", "position", "status", "t_mono_ns", "temperature_c", "velocity", "voltage_dv")

    def __init__(self) -> None:
        self.position = np.zeros(N_JOINTS, dtype=np.int32)  # ticks 0..4095 (sign bit 15 decoded)
        self.velocity = np.zeros(N_JOINTS, dtype=np.int32)  # raw units, sign bit 15 decoded
        self.load = np.zeros(N_JOINTS, dtype=np.int32)  # sign bit 10 decoded; 1000 = 100 %
        self.voltage_dv = np.zeros(N_JOINTS, dtype=np.int32)  # 0.1 V units
        self.temperature_c = np.zeros(N_JOINTS, dtype=np.int32)
        self.status = np.zeros(N_JOINTS, dtype=np.int32)  # Status (65) protection bitfield, uint8 semantics
        self.moving = np.zeros(N_JOINTS, dtype=np.int32)
        self.t_mono_ns: int = 0
        self.ok: bool = False  # False on ANY comm failure; arrays then hold the PREVIOUS values


@runtime_checkable
class Bus(Protocol):
    def connect(self) -> None:
        """Open the port WITHOUT a motor handshake: before ARM the servo side of the watchdog relay is unpowered
        (U3 review item 1). The servo core pings the motors itself, after ARM and before ``reenergize``."""

    def disconnect(self) -> None:
        """Never changes torque implicitly."""

    def ping_all(self) -> dict[int, bool]:
        """Motor id -> answered."""

    def read_block(self, out: BlockRead) -> bool:
        """ONE sync_read of addr 56, length 11 (56..66); returns ``out.ok``."""

    def write_goal(self, goal_ticks: np.ndarray) -> bool:
        """sync_write Goal_Position, raw; the caller guarantees int32 in 0..4095.

        Un-acked: True means the packet was TRANSMITTED, not that any motor received it (the next block read is
        the evidence). The servo core's "write accepted" heartbeat condition therefore means "transmitted"."""

    def set_torque(self, enabled: bool, ids: Sequence[int] | None = None) -> bool: ...

    def read_reg(self, name: str, motor_id: int) -> int:
        """Sign-decoded register value."""

    def write_reg(self, name: str, motor_id: int, value: int) -> None: ...

    def read_calibration_regs(self) -> dict[str, MotorCalibration]:
        """Homing_Offset (decoded), Min/Max_Position_Limit, id; drive_mode comes from config."""

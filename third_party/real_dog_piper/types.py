from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


@dataclass
class Go2WState:
    """Low-level Go2W feedback in policy joint order."""

    joint_pos: np.ndarray = field(default_factory=lambda: zeros(16))
    joint_vel: np.ndarray = field(default_factory=lambda: zeros(16))
    joint_tau: np.ndarray = field(default_factory=lambda: zeros(16))
    base_quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    base_ang_vel: np.ndarray = field(default_factory=lambda: zeros(3))
    stamp: float = 0.0


@dataclass
class PiperState:
    """Piper feedback in policy arm joint order."""

    joint_pos: np.ndarray = field(default_factory=lambda: zeros(6))
    joint_vel: np.ndarray = field(default_factory=lambda: zeros(6))
    ee_pos_local: np.ndarray = field(default_factory=lambda: zeros(3))
    ee_rpy_local: np.ndarray = field(default_factory=lambda: zeros(3))
    gripper_m: float = 0.0
    stamp: float = 0.0


@dataclass
class Go2WCommand:
    q: np.ndarray = field(default_factory=lambda: zeros(16))
    dq: np.ndarray = field(default_factory=lambda: zeros(16))
    kp: np.ndarray = field(default_factory=lambda: zeros(16))
    kd: np.ndarray = field(default_factory=lambda: zeros(16))
    tau: np.ndarray = field(default_factory=lambda: zeros(16))


@dataclass
class ArmCommand:
    joint_pos: np.ndarray = field(default_factory=lambda: zeros(6))
    gripper_m: float = 0.0

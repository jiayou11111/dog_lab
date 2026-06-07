from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


THIRD_PARTY_ROOT = Path(__file__).resolve().parents[1]
DOG_LAB_ROOT = THIRD_PARTY_ROOT.parent
PROJECT_ROOT = DOG_LAB_ROOT.parent
DEPLOY_MUJOCO_ROOT = THIRD_PARTY_ROOT / "deploy_mujoco"


BASE_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FR_foot_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FL_foot_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RR_foot_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RL_foot_joint",
]
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
WHEEL_JOINT_NAMES = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]
WHEEL_INDICES = [BASE_JOINT_NAMES.index(name) for name in WHEEL_JOINT_NAMES]

# Unitree low-state order follows rl_sar's Go2W order: 12 leg motors, then 4 wheels.
POLICY_TO_UNITREE_MOTOR = [0, 1, 2, 12, 3, 4, 5, 13, 6, 7, 8, 14, 9, 10, 11, 15]


@dataclass
class PolicyConfig:
    actor_path: Path = DEPLOY_MUJOCO_ROOT / "pre_train" / "go2w_piper_cost" / "traced_actor.pt"
    hist_encoder_path: Path | None = DEPLOY_MUJOCO_ROOT / "pre_train" / "go2w_piper_cost" / "traced_hist_encoder.pt"
    num_proprio: int = 71
    history_len: int = 10
    num_actions: int = 22
    num_base_actions: int = 16
    clip_obs: float = 100.0
    clip_actions: float = 100.0


@dataclass
class Go2WConfig:
    network_interface: str = "eth0"
    base_joint_names: list[str] = field(default_factory=lambda: list(BASE_JOINT_NAMES))
    wheel_indices: list[int] = field(default_factory=lambda: list(WHEEL_INDICES))
    policy_to_unitree_motor: list[int] = field(default_factory=lambda: list(POLICY_TO_UNITREE_MOTOR))
    default_joint_pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, 0.67, -1.3, 0.0] * 4,
            dtype=np.float32,
        )
    )
    kp: np.ndarray = field(
        default_factory=lambda: np.array(
            [40.0, 40.0, 40.0, 0.0] * 4,
            dtype=np.float32,
        )
    )
    kd: np.ndarray = field(
        default_factory=lambda: np.array(
            [1.0, 1.0, 1.0, 0.5] * 4,
            dtype=np.float32,
        )
    )
    leg_action_scale: float = 0.25
    wheel_action_scale_vel: float = 10.0
    pos_stop: float = 2.146e9
    vel_stop: float = 16000.0
    lowcmd_topic: str = "rt/lowcmd"
    lowstate_topic: str = "rt/lowstate"


@dataclass
class PiperConfig:
    can_name: str = "can0"
    arm_joint_names: list[str] = field(default_factory=lambda: list(ARM_JOINT_NAMES))
    default_joint_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.57, -0.8, 0.0, -0.7, 0.0], dtype=np.float32)
    )
    joint_lower: np.ndarray = field(
        default_factory=lambda: np.array([-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439], dtype=np.float32)
    )
    joint_upper: np.ndarray = field(
        default_factory=lambda: np.array([2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439], dtype=np.float32)
    )
    move_speed_percent: int = 40
    ik_damping: float = 0.05
    ik_step_scale: float = 0.5
    ik_iterations: int = 6
    ik_position_tolerance_m: float = 0.01
    enable_timeout_s: float = 5.0


@dataclass
class EEGoalConfig:
    step_dt: float = 0.01
    traj_time: float = 2.0
    hold_time: float = 1.0
    init_pos_start: tuple[float, float, float] = (0.5, 0.3, 0.0)
    init_pos_end: tuple[float, float, float] = (0.5, 0.6, 0.0)
    pos_l: tuple[float, float] = (0.5, 0.7)
    pos_p: tuple[float, float] = (-np.pi / 6.0, np.pi / 3.0)
    pos_y: tuple[float, float] = (-1.57, 1.57)


@dataclass
class RealDogPiperConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    go2w: Go2WConfig = field(default_factory=Go2WConfig)
    piper: PiperConfig = field(default_factory=PiperConfig)
    ee_goal: EEGoalConfig = field(default_factory=EEGoalConfig)
    obs_ang_vel_scale: float = 0.25
    obs_dof_vel_scale: float = 0.05
    command_scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    control_dt: float = 0.01

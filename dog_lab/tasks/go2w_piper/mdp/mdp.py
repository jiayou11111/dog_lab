# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2W-Piper specific MDP terms migrated from the Loco-Manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


BASE_JOINTS = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", ".*_foot_joint"]
ARM_JOINTS = ["joint[1-6]"]


def base_action_rate_l2(env: ManagerBasedRLEnv, num_base_actions: int = 16) -> torch.Tensor:
    """Penalize only the chassis action rate.

    The first stage keeps the arm fixed with a zero-scale hold action. This term avoids
    punishing unused arm action channels while preserving the Loco action-rate penalty
    on the 12 leg position actions plus 4 wheel velocity actions.
    """

    action_delta = (
        env.action_manager.action[:, :num_base_actions] - env.action_manager.prev_action[:, :num_base_actions]
    )
    return torch.sum(torch.square(action_delta), dim=1)


def loco_policy_proprio(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS),
) -> torch.Tensor:
    """Loco-Manipulation proprioceptive observation block.

    Layout: roll/pitch(2), base angular velocity(3), chassis joint error(16),
    chassis joint velocity(16), last chassis actions(16), velocity command(3),
    stage-1 EE-goal placeholder(3).
    """

    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch = _base_roll_pitch(asset)
    base_ang_vel = asset.data.root_ang_vel_b * 0.25
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_err = joint_pos - default_joint_pos
    local_wheel_ids = _local_joint_ids(asset, asset_cfg.joint_ids, [".*_foot_joint"])
    if local_wheel_ids:
        joint_err[:, local_wheel_ids] = 0.0
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids] * 0.05
    actions = env.action_manager.action[:, :16]
    commands = env.command_manager.get_command(command_name)[:, :3]
    ee_goal_local = torch.zeros(env.num_envs, 3, device=env.device)
    return torch.cat((roll_pitch, base_ang_vel, joint_err, joint_vel, actions, commands, ee_goal_local), dim=-1)


def loco_privileged_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Loco privileged observation block.

    Stage 1 keeps the shape expected by the Loco ROA modules. Real mass,
    friction, and motor randomization values should be wired here when those
    domain-randomization terms are migrated fully.
    """

    return torch.zeros(env.num_envs, 22, device=env.device)


def loco_policy_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compatibility shim; the Lab config composes policy obs from smaller terms."""

    return torch.cat((loco_policy_proprio(env), loco_privileged_obs(env)), dim=-1)


def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize chassis joint power, matching Loco's joint_power term."""

    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(power), dim=1)


def joint_vel_l2_without_wheels(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS)
) -> torch.Tensor:
    """Loco dof_vel penalty with wheel joints masked out."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids].clone()
    local_wheel_ids = _local_joint_ids(asset, asset_cfg.joint_ids, [".*_foot_joint"])
    if local_wheel_ids:
        joint_vel[:, local_wheel_ids] = 0.0
    return torch.sum(torch.square(joint_vel), dim=1)


def joint_acc_l2_without_wheels(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS)
) -> torch.Tensor:
    """Loco dof_acc penalty with wheel joints masked out."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids].clone()
    local_wheel_ids = _local_joint_ids(asset, asset_cfg.joint_ids, [".*_foot_joint"])
    if local_wheel_ids:
        joint_acc[:, local_wheel_ids] = 0.0
    return torch.sum(torch.square(joint_acc), dim=1)


def stand_still_loco(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS),
) -> torch.Tensor:
    """Penalize chassis posture error when the command is close to zero."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_err = _base_joint_error_without_wheels(asset, asset_cfg.joint_ids)
    command = env.command_manager.get_command(command_name)[:, :3]
    return torch.norm(joint_err, dim=1) * (torch.norm(command, dim=1) < 0.1)


def run_still_loco(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS),
) -> torch.Tensor:
    """Penalize chassis posture error when the command is non-zero, matching Loco."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_err = _base_joint_error_without_wheels(asset, asset_cfg.joint_ids)
    command = env.command_manager.get_command(command_name)[:, :3]
    return torch.norm(joint_err, dim=1) * (torch.norm(command, dim=1) > 0.1)


def joint_mirror_l2(
    env: ManagerBasedRLEnv,
    mirror_joint_pairs: tuple[tuple[str, str], ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize left-right leg asymmetry.

    Hip abduction joints mirror with opposite sign; thigh/calf mirror with the same sign.
    This follows the Loco Go2W-Piper mirror penalty, restricted to chassis joints.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.zeros(env.num_envs, device=env.device)

    for left_key, right_key in mirror_joint_pairs:
        left_ids, _ = asset.find_joints(left_key)
        right_ids, _ = asset.find_joints(right_key)
        for left_id, right_id in zip(left_ids, right_ids):
            left = asset.data.joint_pos[:, left_id]
            right = asset.data.joint_pos[:, right_id]
            if "hip" in asset.joint_names[left_id] and "hip" in asset.joint_names[right_id]:
                diff = -left - right
                coef = 2.0
            else:
                diff = left - right
                coef = 1.0
            penalty += coef * torch.square(diff)

    return penalty / max(1, sum(len(asset.find_joints(left_key)[0]) for left_key, _ in mirror_joint_pairs))


def arm_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Measure arm deviation from its default pose.

    Weight is set to zero in stage 1 by default because the zero-scale hold action already
    keeps the arm at the default target. The term is kept as a migration hook.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(err), dim=1)


def cost_joint_pos_limits_loco(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS)
) -> torch.Tensor:
    """Loco P3O cost for chassis joint-position limit violation."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pos_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    violation = torch.clip(pos_limits[..., 0] - joint_pos, min=0.0)
    violation += torch.clip(joint_pos - pos_limits[..., 1], min=0.0)
    local_wheel_ids = _local_joint_ids(asset, asset_cfg.joint_ids, [".*_foot_joint"])
    if local_wheel_ids:
        violation[:, local_wheel_ids] = 0.0
    return torch.sum(violation, dim=1)


def cost_joint_vel_limits_loco(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS)
) -> torch.Tensor:
    """Loco P3O cost for chassis joint-velocity limit violation."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_limits = asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids]
    violation = torch.clip(torch.abs(joint_vel) - vel_limits, min=0.0, max=1.0)
    local_wheel_ids = _local_joint_ids(asset, asset_cfg.joint_ids, [".*_foot_joint"])
    if local_wheel_ids:
        violation[:, local_wheel_ids] = 0.0
    return torch.sum(violation, dim=1)


def zero_arm_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage-1 placeholder arm reward while the Piper arm is fixed."""

    return torch.zeros(env.num_envs, device=env.device)


def _base_roll_pitch(asset: Articulation) -> torch.Tensor:
    quat = asset.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    return torch.stack((roll, pitch), dim=-1)


def _base_joint_error_without_wheels(asset: Articulation, joint_ids: list[int]) -> torch.Tensor:
    joint_err = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    local_wheel_ids = _local_joint_ids(asset, joint_ids, [".*_foot_joint"])
    if local_wheel_ids:
        joint_err[:, local_wheel_ids] = 0.0
    return joint_err


def _local_joint_ids(asset: Articulation, selected_joint_ids: list[int], joint_name_patterns: list[str]) -> list[int]:
    matched_ids, _ = asset.find_joints(joint_name_patterns, preserve_order=False)
    global_to_local = {joint_id: i for i, joint_id in enumerate(selected_joint_ids)}
    return [global_to_local[joint_id] for joint_id in matched_ids if joint_id in global_to_local]

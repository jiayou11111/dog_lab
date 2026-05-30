# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward and observation terms migrated from Loco-Manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from ._helpers import (
    ARM_JOINTS,
    BASE_JOINTS,
    WHEEL_JOINTS,
    base_joint_error_without_wheels,
    base_roll_pitch,
    local_joint_ids,
    orientation_error,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_action_rate_l2(env: ManagerBasedRLEnv, num_base_actions: int = 16) -> torch.Tensor:
    """Penalize only the chassis action rate."""

    action_delta = (
        env.action_manager.action[:, :num_base_actions] - env.action_manager.prev_action[:, :num_base_actions]
    )
    return torch.sum(torch.square(action_delta), dim=1)


def loco_policy_proprio(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS),
) -> torch.Tensor:
    """Loco-Manipulation proprioceptive observation block."""

    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch = base_roll_pitch(asset)
    base_ang_vel = asset.data.root_ang_vel_b * 0.25
    all_joint_ids, _ = asset.find_joints(BASE_JOINTS + ARM_JOINTS, preserve_order=False)
    joint_pos = asset.data.joint_pos[:, all_joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, all_joint_ids]
    joint_err = joint_pos - default_joint_pos
    local_wheel_ids = local_joint_ids(asset, all_joint_ids, WHEEL_JOINTS)
    if local_wheel_ids:
        joint_err[:, local_wheel_ids] = 0.0
    joint_vel = asset.data.joint_vel[:, all_joint_ids] * 0.05
    actions = env.action_manager.action[:, :16]
    commands = env.command_manager.get_command(command_name)[:, :3]
    arm_action = _get_arm_action(env)
    ee_goal_local = arm_action.ee_goal_local_cart if arm_action is not None else torch.zeros(env.num_envs, 3, device=env.device)
    return torch.cat((roll_pitch, base_ang_vel, joint_err, joint_vel, actions, commands, ee_goal_local), dim=-1)


def loco_privileged_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Loco privileged observation block."""

    mass_params = getattr(env, "loco_mass_params_tensor", None)
    if mass_params is None:
        mass_params = torch.zeros(env.num_envs, 5, device=env.device)
    friction = getattr(env, "loco_friction_coeffs", None)
    if friction is None:
        friction = torch.ones(env.num_envs, 1, device=env.device) * 0.9
    motor_strength = getattr(env, "loco_motor_strength", None)
    if motor_strength is None:
        motor_strength = torch.ones(env.num_envs, 16, device=env.device)
    return torch.cat((mass_params, friction, motor_strength[:, :16] - 1.0), dim=-1)


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
    local_wheel_ids = local_joint_ids(asset, asset_cfg.joint_ids, WHEEL_JOINTS)
    if local_wheel_ids:
        joint_vel[:, local_wheel_ids] = 0.0
    return torch.sum(torch.square(joint_vel), dim=1)


def joint_acc_l2_without_wheels(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS)
) -> torch.Tensor:
    """Loco dof_acc penalty with wheel joints masked out."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids].clone()
    local_wheel_ids = local_joint_ids(asset, asset_cfg.joint_ids, WHEEL_JOINTS)
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
    joint_err = base_joint_error_without_wheels(asset, asset_cfg.joint_ids)
    command = env.command_manager.get_command(command_name)[:, :3]
    return torch.norm(joint_err, dim=1) * (torch.norm(command, dim=1) < 0.1)


def run_still_loco(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS),
) -> torch.Tensor:
    """Penalize chassis posture error when the command is non-zero, matching Loco."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_err = base_joint_error_without_wheels(asset, asset_cfg.joint_ids)
    command = env.command_manager.get_command(command_name)[:, :3]
    return torch.norm(joint_err, dim=1) * (torch.norm(command, dim=1) > 0.1)


def joint_mirror_l2(
    env: ManagerBasedRLEnv,
    mirror_joint_pairs: tuple[tuple[str, str], ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize left-right leg asymmetry."""

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


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_hip_joint"),
) -> torch.Tensor:
    """Penalize selected joints for drifting away from their default pose."""

    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(err), dim=1)


def foot_lateral_distance_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
    ),
    min_lateral_distance: float = 0.22,
) -> torch.Tensor:
    """Penalize left-right wheel/foot pairs that collapse toward the centerline."""

    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_pos_b = math_utils.quat_rotate_inverse(
        asset.data.root_quat_w[:, None, :].expand(-1, foot_pos_w.shape[1], -1),
        foot_pos_w - asset.data.root_pos_w[:, None, :],
    )
    front_width = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
    rear_width = torch.abs(foot_pos_b[:, 2, 1] - foot_pos_b[:, 3, 1])
    front_penalty = torch.square(torch.clamp(min_lateral_distance - front_width, min=0.0))
    rear_penalty = torch.square(torch.clamp(min_lateral_distance - rear_width, min=0.0))
    return front_penalty + rear_penalty


def arm_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Measure arm deviation from its default pose."""

    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(err), dim=1)


def tracking_ee_cart_world(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link7"),
    tracking_ee_sigma: float = 1.0,
) -> torch.Tensor:
    """Reward end-effector Cartesian tracking against the Loco internal EE trajectory."""

    arm_action = _get_arm_action(env)
    if arm_action is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos_error = torch.sum(
        torch.abs(asset.data.body_pos_w[:, asset_cfg.body_ids[0]] - arm_action.curr_ee_goal_cart_world), dim=1
    )
    return torch.exp(-ee_pos_error / tracking_ee_sigma)


def tracking_ee_orn(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link7"),
    tracking_ee_sigma: float = 1.0,
) -> torch.Tensor:
    """Reward end-effector orientation tracking against the Loco internal EE trajectory."""

    arm_action = _get_arm_action(env)
    if arm_action is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    ee_orn_error = orientation_error(arm_action.ee_goal_orn_quat, asset.data.body_quat_w[:, asset_cfg.body_ids[0]])
    ee_orn_error = torch.sum(torch.abs(ee_orn_error), dim=1)
    return torch.exp(-ee_orn_error / tracking_ee_sigma)


def zero_arm_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage-1 placeholder arm reward while the Piper arm is fixed."""

    return torch.zeros(env.num_envs, device=env.device)


def _get_arm_action(env: ManagerBasedRLEnv):
    if not hasattr(env, "action_manager"):
        return None
    for name in ("arm_ik", "arm_hold"):
        try:
            term = env.action_manager.get_term(name)
        except KeyError:
            continue
        if hasattr(term, "curr_ee_goal_cart_world"):
            return term
    return getattr(env, "loco_arm_action", None)


__all__ = [
    "ARM_JOINTS",
    "BASE_JOINTS",
    "WHEEL_JOINTS",
    "arm_deviation_l2",
    "base_action_rate_l2",
    "joint_acc_l2_without_wheels",
    "joint_deviation_l2",
    "joint_mirror_l2",
    "joint_power_l1",
    "foot_lateral_distance_l2",
    "joint_vel_l2_without_wheels",
    "loco_policy_obs",
    "loco_policy_proprio",
    "loco_privileged_obs",
    "run_still_loco",
    "stand_still_loco",
    "tracking_ee_cart_world",
    "tracking_ee_orn",
    "zero_arm_reward",
]

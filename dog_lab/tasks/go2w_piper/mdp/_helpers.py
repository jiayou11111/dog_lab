# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for Go2W-Piper MDP terms."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation

BASE_JOINTS = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", ".*_foot_joint"]
ARM_JOINTS = ["joint[1-6]"]
WHEEL_JOINTS = [".*_foot_joint"]


def base_roll_pitch(asset: Articulation) -> torch.Tensor:
    quat = asset.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    return torch.stack((roll, pitch), dim=-1)


def base_joint_error_without_wheels(asset: Articulation, joint_ids: list[int] | slice) -> torch.Tensor:
    joint_err = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    local_wheel_ids = local_joint_ids(asset, joint_ids, WHEEL_JOINTS)
    if local_wheel_ids:
        joint_err[:, local_wheel_ids] = 0.0
    return joint_err


def local_joint_ids(asset: Articulation, selected_joint_ids: list[int] | slice, joint_name_patterns: list[str]) -> list[int]:
    matched_ids, _ = asset.find_joints(joint_name_patterns, preserve_order=False)
    if isinstance(selected_joint_ids, slice):
        selected_joint_ids = list(range(asset.num_joints))[selected_joint_ids]
    global_to_local = {joint_id: i for i, joint_id in enumerate(selected_joint_ids)}
    return [global_to_local[joint_id] for joint_id in matched_ids if joint_id in global_to_local]

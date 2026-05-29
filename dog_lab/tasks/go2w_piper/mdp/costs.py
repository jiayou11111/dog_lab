# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P3O cost terms migrated from Loco-Manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ._helpers import BASE_JOINTS, WHEEL_JOINTS, local_joint_ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cost_joint_pos_limits_loco(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=BASE_JOINTS)
) -> torch.Tensor:
    """Loco P3O cost for chassis joint-position limit violation."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pos_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    violation = torch.clip(pos_limits[..., 0] - joint_pos, min=0.0)
    violation += torch.clip(joint_pos - pos_limits[..., 1], min=0.0)
    local_wheel_ids = local_joint_ids(asset, asset_cfg.joint_ids, WHEEL_JOINTS)
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
    local_wheel_ids = local_joint_ids(asset, asset_cfg.joint_ids, WHEEL_JOINTS)
    if local_wheel_ids:
        violation[:, local_wheel_ids] = 0.0
    return torch.sum(violation, dim=1)


__all__ = ["cost_joint_pos_limits_loco", "cost_joint_vel_limits_loco"]

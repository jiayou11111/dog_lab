# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for staged Go2W-Piper training."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FixedJointPositionAction(ActionTerm):
    """Hold selected joints at their default positions while optionally consuming policy actions."""

    cfg: "FixedJointPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "FixedJointPositionActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._raw_actions = torch.zeros(self.num_envs, self.cfg.action_dim, device=self.device)
        self._processed_actions = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._asset.data.default_joint_pos[:, self._joint_ids]

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._processed_actions[:] = self._asset.data.default_joint_pos[:, self._joint_ids]
        else:
            self._processed_actions[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]


@configclass
class FixedJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`FixedJointPositionAction`."""

    class_type: type[ActionTerm] = FixedJointPositionAction

    joint_names: list[str] = MISSING
    """Joint names or regular expressions to hold at their default positions."""

    preserve_order: bool = False
    """Whether to preserve the order of the joint name expressions."""

    action_dim: int = 0
    """Number of policy action dimensions to consume while holding the joints fixed."""

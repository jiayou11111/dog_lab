# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration values migrated from Loco-Manipulation Go2W-Piper.

Stage 1 uses the standard Isaac Lab PPO runner and fixes the arm at its default
pose. The P2O/ROA and cost values are kept here as explicit migration anchors so
they can be wired into a custom runner later without hunting through the source.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageOneControlCfg:
    """Base-only control split for the first training stage."""

    num_leg_actions: int = 12
    num_wheel_actions: int = 4
    num_arm_actions: int = 6
    num_base_actions: int = 16
    leg_action_scale: float = 0.25
    wheel_action_scale_vel: float = 10.0


@dataclass(frozen=True)
class LocoConstraintCfg:
    """Dormant P2O/ROA cost values migrated from Loco-Manipulation."""

    enabled: bool = False
    num_costs: int = 2
    dof_pos_limits_scale: float = 0.1
    dof_vel_limits_scale: float = 0.01
    dof_pos_limits_d_value: float = 0.0
    dof_vel_limits_d_value: float = 0.0
    cost_value_loss_coef: float = 1.0
    cost_viol_loss_coef: float = 1.0


STAGE_ONE_CONTROL = StageOneControlCfg()
LOCO_CONSTRAINTS = LocoConstraintCfg()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loco-Manipulation parameters used by the Go2W-Piper environment configs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocoControlCfg:
    """Action split and scaling migrated from Loco-Manipulation."""

    num_leg_actions: int = 16
    num_wheel_actions: int = 4
    num_arm_actions: int = 6
    num_base_actions: int = 16
    leg_action_scale: float = 0.25
    wheel_action_scale_vel: float = 10.0


@dataclass(frozen=True)
class LocoConstraintCfg:
    """P3O cost values migrated from Loco-Manipulation."""

    enabled: bool = True
    num_costs: int = 2
    dof_pos_limits_scale: float = 0.1
    dof_vel_limits_scale: float = 0.01
    dof_pos_limits_d_value: float = 0.0
    dof_vel_limits_d_value: float = 0.0
    cost_value_loss_coef: float = 1.0
    cost_viol_loss_coef: float = 1.0


LOCO_CONTROL = LocoControlCfg()
LOCO_CONSTRAINTS = LocoConstraintCfg()

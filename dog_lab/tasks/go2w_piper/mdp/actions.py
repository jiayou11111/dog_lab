# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for staged Go2W-Piper training."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import math
import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from ._helpers import local_joint_ids, orientation_error, sphere_to_cart

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


class LocoBaseAction(ActionTerm):
    """Loco-Manipulation base action: leg position targets plus wheel velocity targets.

    The original Loco environment consumes one 16-D base action in asset DOF order.
    Non-wheel joints are interpreted as position offsets from the default pose, while
    wheel joints are interpreted as velocity targets.
    """

    cfg: "LocoBaseActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "LocoBaseActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        if len(self._joint_ids) != self.cfg.action_dim:
            raise ValueError(
                f"Expected {self.cfg.action_dim} base joints, found {len(self._joint_ids)}: {self._joint_names}."
            )
        self._wheel_local_ids = local_joint_ids(self._asset, self._joint_ids, self.cfg.wheel_joint_names)
        self._wheel_joint_ids = [self._joint_ids[i] for i in self._wheel_local_ids]
        self._non_wheel_local_ids = [i for i in range(len(self._joint_ids)) if i not in self._wheel_local_ids]
        self._non_wheel_joint_ids = [self._joint_ids[i] for i in self._non_wheel_local_ids]

        self._raw_actions = torch.zeros(self.num_envs, self.cfg.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_pos_target = self._asset.data.default_joint_pos[:, self._non_wheel_joint_ids].clone()
        self._joint_vel_target = torch.zeros(self.num_envs, len(self._wheel_joint_ids), device=self.device)

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
        self._processed_actions[:] = actions
        motor_strength = getattr(self._env, "loco_motor_strength", None)
        if motor_strength is None:
            motor_strength = 1.0
        pos_actions = actions[:, self._non_wheel_local_ids] * self.cfg.position_scale
        wheel_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)
        vel_actions = actions[:, self._wheel_local_ids] * self.cfg.velocity_scale * wheel_sign
        # vel_actions = actions[:, self._wheel_local_ids] * self.cfg.velocity_scale
        if isinstance(motor_strength, torch.Tensor):
            pos_actions = pos_actions * motor_strength[:, self._non_wheel_local_ids]
            vel_actions = vel_actions * motor_strength[:, self._wheel_local_ids]
        self._joint_pos_target[:] = self._asset.data.default_joint_pos[:, self._non_wheel_joint_ids] + pos_actions
        self._joint_vel_target[:] = vel_actions

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_pos_target, joint_ids=self._non_wheel_joint_ids)
        self._asset.set_joint_velocity_target(self._joint_vel_target, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            self._joint_pos_target[:] = self._asset.data.default_joint_pos[:, self._non_wheel_joint_ids]
            self._joint_vel_target[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._joint_pos_target[env_ids] = self._asset.data.default_joint_pos[env_ids][
                :, self._non_wheel_joint_ids
            ]
            self._joint_vel_target[env_ids] = 0.0


@configclass
class LocoBaseActionCfg(ActionTermCfg):
    """Configuration for :class:`LocoBaseAction`."""

    class_type: type[ActionTerm] = LocoBaseAction

    joint_names: list[str] = MISSING
    wheel_joint_names: list[str] = MISSING
    preserve_order: bool = False
    action_dim: int = 16
    position_scale: float = 0.25
    velocity_scale: float = 10.0


class LocoArmIKAction(ActionTerm):
    """Consume arm policy dimensions while tracking internally sampled EE goals with DLS IK.

    This mirrors the original Loco-Manipulation Go2W-Piper control path: the arm action
    channels stay present for the ROA actor heads, but the actual arm target is produced
    from an internally sampled end-effector trajectory and solved with damped least squares.
    """

    cfg: "LocoArmIKActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "LocoArmIKActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        body_ids, body_names = self._asset.find_bodies(self.cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one EE body matching '{self.cfg.ee_body_name}', found {len(body_ids)}: {body_names}."
            )
        self._body_idx = body_ids[0]
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [joint_id + 6 for joint_id in self._joint_ids]

        self._raw_actions = torch.zeros(self.num_envs, self.cfg.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_pos_target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._joint_vel_target = torch.zeros_like(self._joint_pos_target)

        self._goal_center_offset = torch.tensor(self.cfg.goal_center_offset, device=self.device).repeat(
            self.num_envs, 1
        )
        self._arm_base_offset = torch.tensor(self.cfg.arm_base_offset, device=self.device).repeat(self.num_envs, 1)
        self._default_ee_rpy = torch.tensor(self.cfg.default_ee_rpy, device=self.device)
        self._goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_local_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)

        self.traj_timesteps = max(1, int(self.cfg.traj_time / self._env.step_dt))
        self.traj_total_timesteps = max(1, int((self.cfg.traj_time + self.cfg.hold_time) / self._env.step_dt))
        self._collision_lower_limits = torch.tensor(self.cfg.collision_lower_limits, device=self.device)
        self._collision_upper_limits = torch.tensor(self.cfg.collision_upper_limits, device=self.device)
        self._collision_check_t = torch.linspace(
            0.0, 1.0, self.cfg.num_collision_check_samples, device=self.device
        )[None, None, :]

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._initialize_goals(env_ids)
        setattr(self._env, "loco_arm_action", self)

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
        self._processed_actions[:] = 0.0
        self._update_curr_ee_goal()

    def apply_actions(self):
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        pos_error = (self.curr_ee_goal_cart_world - ee_pos_w) * self.cfg.position_error_scale
        orn_error = orientation_error(self.ee_goal_orn_quat, ee_quat_w) * self.cfg.orientation_error_scale
        dpose = torch.cat((pos_error, orn_error), dim=-1)
        delta_joint_pos = self._solve_dls(self._compute_jacobian_w(), dpose)
        delta_joint_pos *= self.cfg.ik_step_scale
        if self.cfg.max_delta_joint_pos is not None:
            delta_joint_pos = torch.clamp(
                delta_joint_pos, -self.cfg.max_delta_joint_pos, self.cfg.max_delta_joint_pos
            )
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_pos_target[:] = joint_pos + delta_joint_pos
        if self.cfg.joint_limit_avoidance_gain > 0.0:
            self._joint_pos_target[:] += self._compute_limit_avoidance_delta(joint_pos)
        if self.cfg.clamp_joint_targets:
            lower, upper = self._joint_limit_bounds()
            self._joint_pos_target[:] = torch.clamp(self._joint_pos_target, lower, upper)
        self._asset.set_joint_position_target(self._joint_pos_target, joint_ids=self._joint_ids)
        if self.cfg.zero_joint_velocity_target:
            self._asset.set_joint_velocity_target(self._joint_vel_target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._joint_pos_target[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]
        self._initialize_goals(env_ids)

    def _initialize_goals(self, env_ids: torch.Tensor) -> None:
        init_start = torch.tensor(self.cfg.init_pos_start, device=self.device)
        init_end = torch.tensor(self.cfg.init_pos_end, device=self.device)
        self.ee_start_sphere[env_ids] = init_start
        self.ee_goal_sphere[env_ids] = init_end
        self.goal_timer[env_ids] = 0.0
        self._resample_ee_goal_orn_once(env_ids)
        if self.cfg.sample_initial_goal:
            active_mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
            for _ in range(10):
                active_env_ids = env_ids[active_mask]
                if active_env_ids.numel() == 0:
                    break
                self._resample_ee_goal_sphere_once(active_env_ids)
                collision_mask = self._collision_check(active_env_ids)
                active_mask_ids = active_mask.nonzero(as_tuple=False).flatten()
                active_mask[active_mask_ids] = collision_mask
        self.curr_ee_goal_sphere[env_ids] = self.ee_start_sphere[env_ids]
        self._update_curr_ee_goal(env_ids=env_ids, advance_timer=False)

    def _resample_ee_goal(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
        self._resample_ee_goal_orn_once(env_ids)
        active_mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        for _ in range(10):
            active_env_ids = env_ids[active_mask]
            if active_env_ids.numel() == 0:
                break
            start_goal_cart = sphere_to_cart(self.ee_start_sphere[active_env_ids])
            self._resample_ee_goal_sphere_once(active_env_ids)
            collision_mask = self._collision_check(active_env_ids)
            if self.cfg.min_resample_goal_distance > 0.0:
                next_goal_cart = sphere_to_cart(self.ee_goal_sphere[active_env_ids])
                too_close_mask = (
                    torch.linalg.norm(next_goal_cart - start_goal_cart, dim=-1)
                    < self.cfg.min_resample_goal_distance
                )
                collision_mask = torch.logical_or(collision_mask, too_close_mask)
            active_mask_ids = active_mask.nonzero(as_tuple=False).flatten()
            active_mask[active_mask_ids] = collision_mask
        self.goal_timer[env_ids] = 0.0


    def _resample_ee_goal_orn_once(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        if n == 0:
            return

        low = torch.tensor(
            [self.cfg.delta_orn_r[0], self.cfg.delta_orn_p[0], self.cfg.delta_orn_y[0]],
            device=self.device,
        )
        high = torch.tensor(
            [self.cfg.delta_orn_r[1], self.cfg.delta_orn_p[1], self.cfg.delta_orn_y[1]],
            device=self.device,
        )

        rand = low + torch.rand(n, 3, device=self.device) * (high - low)
        self._goal_orn_delta_rpy[env_ids] = rand


    def _resample_ee_goal_sphere_once(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        if n == 0:
            return

        low = torch.tensor(
            [self.cfg.pos_l[0], self.cfg.pos_p[0], self.cfg.pos_y[0]],
            device=self.device,
        )
        high = torch.tensor(
            [self.cfg.pos_l[1], self.cfg.pos_p[1], self.cfg.pos_y[1]],
            device=self.device,
        )

        rand = low + torch.rand(n, 3, device=self.device) * (high - low)
        self.ee_goal_sphere[env_ids] = rand


    def _collision_check(self, env_ids: torch.Tensor) -> torch.Tensor:
        ee_target_sphere = torch.lerp(
            self.ee_start_sphere[env_ids, ..., None],
            self.ee_goal_sphere[env_ids, ..., None],
            self._collision_check_t,
        ).squeeze(-1)
        ee_target_cart = sphere_to_cart(
            torch.permute(ee_target_sphere, (2, 0, 1)).reshape(-1, 3)
        ).reshape(self.cfg.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(
            torch.logical_and(
                torch.all(ee_target_cart < self._collision_upper_limits, dim=-1),
                torch.all(ee_target_cart > self._collision_lower_limits, dim=-1),
            ),
            dim=0,
        )
        underground_mask = torch.any(ee_target_cart[..., 2] < self.cfg.underground_limit, dim=0)
        return collision_mask | underground_mask

    def _update_curr_ee_goal(self, env_ids: torch.Tensor | None = None, advance_timer: bool = True) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        t = torch.clamp(self.goal_timer[env_ids] / self.traj_timesteps, 0.0, 1.0)
        self.curr_ee_goal_sphere[env_ids] = torch.lerp(
            self.ee_start_sphere[env_ids], self.ee_goal_sphere[env_ids], t[:, None]
        )
        curr_goal_cart = sphere_to_cart(self.curr_ee_goal_sphere[env_ids])
        root_quat = self._asset.data.root_quat_w[env_ids]
        root_pos = self._asset.data.root_pos_w[env_ids]
        base_yaw_quat = math_utils.yaw_quat(root_quat)
        center = torch.cat((root_pos[:, :2], torch.zeros(len(env_ids), 1, device=self.device)), dim=-1)
        center += math_utils.quat_apply(base_yaw_quat, self._goal_center_offset[env_ids])
        goal_yaw_global = math_utils.quat_apply(base_yaw_quat, curr_goal_cart)
        self.curr_ee_goal_cart_world[env_ids] = center + goal_yaw_global

        default_roll = self._default_ee_rpy[0]
        default_pitch = self._default_ee_rpy[1]
        default_yaw = torch.atan2(goal_yaw_global[:, 1], goal_yaw_global[:, 0]) + self._default_ee_rpy[2]
        self.ee_goal_orn_quat[env_ids] = math_utils.quat_from_euler_xyz(
            self._goal_orn_delta_rpy[env_ids, 0] + default_roll,
            self._goal_orn_delta_rpy[env_ids, 1] + default_pitch,
            self._goal_orn_delta_rpy[env_ids, 2] + default_yaw,
        )
        arm_base_pos = root_pos + math_utils.quat_apply(root_quat, self._arm_base_offset[env_ids])
        self.ee_goal_local_cart[env_ids] = math_utils.quat_rotate_inverse(
            root_quat, self.curr_ee_goal_cart_world[env_ids] - arm_base_pos
        )

        if advance_timer:
            self.goal_timer[env_ids] += 1
            resample_ids = env_ids[self.goal_timer[env_ids] > self.traj_total_timesteps]
            if self.cfg.resample_goals:
                self._resample_ee_goal(resample_ids)
            elif resample_ids.numel() > 0:
                self.goal_timer[resample_ids] = float(self.traj_total_timesteps)

    def _compute_jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :6, self._jacobi_joint_ids]

    def _solve_dls(self, jacobian: torch.Tensor, dpose: torch.Tensor) -> torch.Tensor:
        jacobian_t = torch.transpose(jacobian, 1, 2)
        damping = torch.eye(jacobian.shape[1], device=self.device) * (self.cfg.damping**2)
        delta = jacobian_t @ torch.linalg.solve(jacobian @ jacobian_t + damping[None, ...], dpose.unsqueeze(-1))
        return delta.squeeze(-1)

    def _joint_limit_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        lower = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        upper = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        if self.cfg.joint_limit_safety_margin <= 0.0:
            return lower, upper

        center = 0.5 * (lower + upper)
        safe_lower = torch.minimum(lower + self.cfg.joint_limit_safety_margin, center)
        safe_upper = torch.maximum(upper - self.cfg.joint_limit_safety_margin, center)
        return safe_lower, safe_upper

    def _compute_limit_avoidance_delta(self, joint_pos: torch.Tensor) -> torch.Tensor:
        lower, upper = self._joint_limit_bounds()
        center = 0.5 * (lower + upper)
        half_range = 0.5 * torch.clamp(upper - lower, min=1.0e-6)
        normalized_dist = torch.abs((joint_pos - center) / half_range)
        active = torch.clamp(
            (normalized_dist - self.cfg.joint_limit_avoidance_margin)
            / max(1.0e-6, 1.0 - self.cfg.joint_limit_avoidance_margin),
            min=0.0,
            max=1.0,
        )
        return self.cfg.joint_limit_avoidance_gain * active * (center - joint_pos)


@configclass
class LocoArmIKActionCfg(ActionTermCfg):
    """Configuration for :class:`LocoArmIKAction`."""

    class_type: type[ActionTerm] = LocoArmIKAction

    joint_names: list[str] = MISSING
    ee_body_name: str = "link7"
    preserve_order: bool = False
    action_dim: int = 6
    damping: float = 0.05
    position_error_scale: float = 1.0
    orientation_error_scale: float = 1.0
    ik_step_scale: float = 1.0
    max_delta_joint_pos: float | None = None
    clamp_joint_targets: bool = True
    joint_limit_safety_margin: float = 0.05
    joint_limit_avoidance_gain: float = 0.08
    joint_limit_avoidance_margin: float = 0.65
    zero_joint_velocity_target: bool = True

    arm_base_offset: tuple[float, float, float] = (0.1, 0.0, 0.05)
    goal_center_offset: tuple[float, float, float] = (0.1, 0.0, 0.6)
    traj_time: float = 2.0
    hold_time: float = 1.0
    init_pos_start: tuple[float, float, float] = (0.5, 0.3, 0.0)
    init_pos_end: tuple[float, float, float] = (0.5, 0.6, 0.0)
    sample_initial_goal: bool = True
    resample_goals: bool = True
    pos_l: tuple[float, float] = (0.5, 0.7)
    pos_p: tuple[float, float] = (-math.pi / 6.0, math.pi / 3.0)
    pos_y: tuple[float, float] = (-1.57, 1.57)
    min_resample_goal_distance: float = 0.0
    default_ee_rpy: tuple[float, float, float] = (0.0, math.pi / 2.0, -math.pi / 2.0)
    delta_orn_r: tuple[float, float] = (-0.2, 0.2)
    delta_orn_p: tuple[float, float] = (-0.2, 0.2)
    delta_orn_y: tuple[float, float] = (-0.2, 0.2)
    collision_upper_limits: tuple[float, float, float] = (0.25, 0.25, -0.05)
    collision_lower_limits: tuple[float, float, float] = (-0.45, -0.25, -0.6)
    underground_limit: float = -0.6
    num_collision_check_samples: int = 10

__all__ = [
    "FixedJointPositionAction",
    "FixedJointPositionActionCfg",
    "LocoArmIKAction",
    "LocoArmIKActionCfg",
    "LocoBaseAction",
    "LocoBaseActionCfg",
]

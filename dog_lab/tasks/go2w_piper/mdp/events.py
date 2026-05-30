# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific event helpers for Loco-style privileged observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_loco_physics_and_privileged_params(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float] = (0.8, 1.0),
    restitution_range: tuple[float, float] = (0.0, 0.3),
    added_mass_range: tuple[float, float] = (-1.0, 1.0),
    added_com_range_x: tuple[float, float] = (-0.05, 0.05),
    added_com_range_y: tuple[float, float] = (-0.05, 0.05),
    added_com_range_z: tuple[float, float] = (-0.05, 0.05),
    gripper_added_mass_range: tuple[float, float] = (0.0, 0.1),
    leg_motor_strength_range: tuple[float, float] = (0.9, 1.1),
    num_leg_actions: int = 16,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_name: str = "base",
    gripper_body_name: str = "link7",
) -> None:
    """Apply Loco domain randomization and store the same samples for ROA.

    The original Isaac Gym task samples friction, base mass, base COM, gripper
    mass, and motor strength once per environment, then exposes those exact
    samples through the privileged observation block. This term keeps that
    one-to-one relationship in Isaac Lab instead of sampling physics and
    privileged tensors independently.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if not hasattr(env, "loco_mass_params_tensor"):
        env.loco_mass_params_tensor = torch.zeros(env.num_envs, 5, device=env.device)
    if not hasattr(env, "loco_friction_coeffs"):
        env.loco_friction_coeffs = torch.ones(env.num_envs, 1, device=env.device)
    if not hasattr(env, "loco_motor_strength"):
        env.loco_motor_strength = torch.ones(env.num_envs, num_leg_actions, device=env.device)

    count = len(env_ids)
    base_mass_delta = torch.empty(count, device=env.device).uniform_(*added_mass_range)
    base_com_delta = torch.empty(count, 3, device=env.device)
    base_com_delta[:, 0].uniform_(*added_com_range_x)
    base_com_delta[:, 1].uniform_(*added_com_range_y)
    base_com_delta[:, 2].uniform_(*added_com_range_z)
    gripper_mass_delta = torch.empty(count, device=env.device).uniform_(*gripper_added_mass_range)
    friction = torch.empty(count, 1, device=env.device).uniform_(*friction_range)
    restitution = torch.empty(count, 1, device=env.device).uniform_(*restitution_range)
    motor_strength = torch.empty(count, num_leg_actions, device=env.device).uniform_(*leg_motor_strength_range)

    env.loco_mass_params_tensor[env_ids, 0] = base_mass_delta
    env.loco_mass_params_tensor[env_ids, 1:4] = base_com_delta
    env.loco_mass_params_tensor[env_ids, 4] = gripper_mass_delta
    env.loco_friction_coeffs[env_ids] = friction
    env.loco_motor_strength[env_ids] = motor_strength

    env_ids_cpu = env_ids.cpu()
    base_body_ids, _ = asset.find_bodies(base_body_name)
    gripper_body_ids, _ = asset.find_bodies(gripper_body_name)
    base_body_ids_cpu = torch.tensor(base_body_ids, dtype=torch.long, device="cpu")
    gripper_body_ids_cpu = torch.tensor(gripper_body_ids, dtype=torch.long, device="cpu")
    default_mass = asset.data.default_mass.cpu()
    default_inertia = asset.data.default_inertia.cpu()

    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = friction.cpu()
    materials[env_ids_cpu, :, 1] = friction.cpu()
    materials[env_ids_cpu, :, 2] = restitution.cpu()
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

    masses = asset.root_physx_view.get_masses()
    masses[env_ids_cpu[:, None], base_body_ids_cpu] = (
        default_mass[env_ids_cpu[:, None], base_body_ids_cpu] + base_mass_delta.cpu()[:, None]
    )
    masses[env_ids_cpu[:, None], gripper_body_ids_cpu] = (
        default_mass[env_ids_cpu[:, None], gripper_body_ids_cpu] + gripper_mass_delta.cpu()[:, None]
    )
    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    inertias = asset.root_physx_view.get_inertias()
    base_ratio = (
        masses[env_ids_cpu[:, None], base_body_ids_cpu] / default_mass[env_ids_cpu[:, None], base_body_ids_cpu]
    )
    gripper_ratio = (
        masses[env_ids_cpu[:, None], gripper_body_ids_cpu]
        / default_mass[env_ids_cpu[:, None], gripper_body_ids_cpu]
    )
    inertias[env_ids_cpu[:, None], base_body_ids_cpu] = (
        default_inertia[env_ids_cpu[:, None], base_body_ids_cpu] * base_ratio[..., None]
    )
    inertias[env_ids_cpu[:, None], gripper_body_ids_cpu] = (
        default_inertia[env_ids_cpu[:, None], gripper_body_ids_cpu] * gripper_ratio[..., None]
    )
    asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

    coms = asset.root_physx_view.get_coms()
    cache_name = f"_loco_default_coms_{asset_cfg.name}"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, coms.clone())
    default_coms = getattr(env, cache_name)
    coms[env_ids_cpu[:, None], base_body_ids_cpu, :3] = (
        default_coms[env_ids_cpu[:, None], base_body_ids_cpu, :3] + base_com_delta.cpu()[:, None, :]
    )
    asset.root_physx_view.set_coms(coms, env_ids_cpu)


def randomize_loco_privileged_params(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float] = (0.8, 1.0),
    added_mass_range: tuple[float, float] = (-1.0, 1.0),
    added_com_range_x: tuple[float, float] = (-0.05, 0.05),
    added_com_range_y: tuple[float, float] = (-0.05, 0.05),
    added_com_range_z: tuple[float, float] = (-0.05, 0.05),
    gripper_added_mass_range: tuple[float, float] = (0.0, 0.1),
    leg_motor_strength_range: tuple[float, float] = (0.9, 1.1),
    num_leg_actions: int = 16,
) -> None:
    """Backward-compatible alias for older configs."""

    randomize_loco_physics_and_privileged_params(
        env,
        env_ids,
        friction_range=friction_range,
        added_mass_range=added_mass_range,
        added_com_range_x=added_com_range_x,
        added_com_range_y=added_com_range_y,
        added_com_range_z=added_com_range_z,
        gripper_added_mass_range=gripper_added_mass_range,
        leg_motor_strength_range=leg_motor_strength_range,
        num_leg_actions=num_leg_actions,
    )


def randomize_body_com(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    com_range_x: tuple[float, float] = (-0.05, 0.05),
    com_range_y: tuple[float, float] = (-0.05, 0.05),
    com_range_z: tuple[float, float] = (-0.05, 0.05),
) -> None:
    """Randomize body COM offsets for selected bodies."""

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    body_ids = asset_cfg.body_ids
    if body_ids == slice(None):
        body_ids = list(range(asset.num_bodies))
    body_ids_cpu = torch.as_tensor(body_ids, dtype=torch.long, device="cpu")
    env_ids_cpu = env_ids.cpu()

    coms = asset.root_physx_view.get_coms()
    cache_name = f"_loco_default_coms_{asset_cfg.name}"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, coms.clone())
    default_coms = getattr(env, cache_name)
    default_com = default_coms[env_ids_cpu[:, None], body_ids_cpu, :3].clone()
    offsets = torch.empty_like(default_com)
    offsets[..., 0].uniform_(com_range_x[0], com_range_x[1])
    offsets[..., 1].uniform_(com_range_y[0], com_range_y[1])
    offsets[..., 2].uniform_(com_range_z[0], com_range_z[1])
    coms[env_ids_cpu[:, None], body_ids_cpu, :3] = default_com + offsets
    asset.root_physx_view.set_coms(coms, env_ids_cpu)


__all__ = [
    "randomize_body_com",
    "randomize_loco_physics_and_privileged_params",
    "randomize_loco_privileged_params",
]

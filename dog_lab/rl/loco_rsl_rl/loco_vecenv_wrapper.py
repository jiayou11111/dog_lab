# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab vector-environment wrapper for the Loco-Manipulation RSL-RL fork."""

from __future__ import annotations

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class LocoRslRlVecEnvWrapper(VecEnv):
    """Wrap Isaac Lab envs with Loco reward splitting and P3O cost tensors.

    The original Loco runner expects ``step`` to return locomotion reward, arm
    reward, costs, dones, and extras separately. Isaac Lab computes rewards
    through a single manager, so this wrapper reconstructs the split from
    ``cfg.loco_reward_split`` and computes costs from ``cfg.loco_costs``.
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv):
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must inherit from ManagerBasedRLEnv or DirectRLEnv. "
                f"Received: {type(env)}"
            )
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError("Loco reward splitting currently requires a ManagerBasedRLEnv reward manager.")

        self.env = env
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])

        if hasattr(self.unwrapped, "observation_manager") and "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = None

        runner_cfg = self.unwrapped.cfg.loco_runner
        self.num_proprio = runner_cfg.num_proprio
        self.num_priv = runner_cfg.num_priv
        self.history_len = runner_cfg.history_len
        self.num_leg_actions = runner_cfg.num_leg_actions
        self.num_arm_actions = runner_cfg.num_arm_actions
        self.num_costs = runner_cfg.num_costs

        self.cost_names = self._active_cost_names()
        self.cost_d_values_tensor = torch.tensor(
            [self.unwrapped.cfg.loco_costs.d_values[name] for name in self.cost_names],
            dtype=torch.float,
            device=self.device,
        )
        self.cost_k_values = torch.ones(self.num_costs, dtype=torch.float, device=self.device)

        self.env.reset()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def cfg(self) -> object:
        """Return the Isaac Lab environment configuration."""

        return self.unwrapped.cfg

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict = self._compute_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    def get_privileged_observations(self) -> torch.Tensor | None:
        obs_dict = self._compute_observations()
        return obs_dict.get("critic", None)

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"], {"observations": obs_dict}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, _, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)

        leg_rewards, arm_rewards = self._split_rewards()
        costs = self._compute_costs()

        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return obs_dict["policy"], obs_dict.get("critic", None), leg_rewards, arm_rewards, costs, dones, extras

    def close(self):
        return self.env.close()

    def _compute_observations(self) -> dict[str, torch.Tensor]:
        if hasattr(self.unwrapped, "observation_manager"):
            return self.unwrapped.observation_manager.compute()
        return self.unwrapped._get_observations()

    def _split_rewards(self) -> tuple[torch.Tensor, torch.Tensor]:
        reward_manager = self.unwrapped.reward_manager
        term_names = reward_manager.active_terms
        step_reward = reward_manager._step_reward * self.unwrapped.step_dt
        split_cfg = self.unwrapped.cfg.loco_reward_split

        leg_rewards = self._sum_reward_terms(step_reward, term_names, split_cfg.leg_terms)
        arm_rewards = self._sum_reward_terms(step_reward, term_names, split_cfg.arm_terms)

        if split_cfg.only_positive_rewards:
            leg_rewards = torch.clip(leg_rewards, min=0.0)
            arm_rewards = torch.clip(arm_rewards, min=0.0)
        return leg_rewards, arm_rewards

    def _sum_reward_terms(
        self, step_reward: torch.Tensor, active_terms: list[str], selected_terms: tuple[str, ...]
    ) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name in selected_terms:
            if name in active_terms:
                reward += step_reward[:, active_terms.index(name)]
        return reward

    def _active_cost_names(self) -> list[str]:
        names = []
        for name, value in self.unwrapped.cfg.loco_costs.__dict__.items():
            if name == "d_values" or value is None:
                continue
            if hasattr(value, "func"):
                names.append(name)
        return names

    def _compute_costs(self) -> torch.Tensor:
        cost_values = []
        for name in self.cost_names:
            term_cfg = getattr(self.unwrapped.cfg.loco_costs, name)
            value = term_cfg.func(self.unwrapped, **term_cfg.params)
            cost_values.append(value)
        if not cost_values:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        return torch.stack(cost_values, dim=-1)

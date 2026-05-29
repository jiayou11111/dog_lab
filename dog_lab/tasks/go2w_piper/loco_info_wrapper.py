# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym wrapper that adds Loco-Manipulation training signals to ``infos``."""

from __future__ import annotations

import gymnasium as gym
import torch


class LocoInfoWrapper(gym.Wrapper):
    """Append leg/arm rewards and P3O costs under ``infos["loco"]``.

    The environment remains a normal Isaac Lab Gym environment.  The standard
    ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`` can wrap this class unchanged.
    """

    def step(self, action):
        obs, reward, terminated, truncated, infos = self.env.step(action)
        unwrapped = self.unwrapped
        loco = {
            "leg_rewards": self._compute_weighted_terms(unwrapped.cfg.loco_reward_split.leg_terms),
            "arm_rewards": self._compute_weighted_terms(unwrapped.cfg.loco_reward_split.arm_terms),
            "costs": self._compute_costs(),
        }
        if unwrapped.cfg.loco_reward_split.only_positive_rewards:
            loco["leg_rewards"] = torch.clip(loco["leg_rewards"], min=0.0)
            loco["arm_rewards"] = torch.clip(loco["arm_rewards"], min=0.0)
        infos["loco"] = loco
        return obs, reward, terminated, truncated, infos

    def _compute_weighted_terms(self, term_names: tuple[str, ...]) -> torch.Tensor:
        total = torch.zeros(self.unwrapped.num_envs, device=self.unwrapped.device)
        for name in term_names:
            term_cfg = getattr(self.unwrapped.cfg.rewards, name)
            total += term_cfg.func(self.unwrapped, **term_cfg.params) * term_cfg.weight * self.unwrapped.step_dt
        return total

    def _compute_costs(self) -> torch.Tensor:
        costs = []
        for name, term_cfg in vars(self.unwrapped.cfg.loco_costs).items():
            if hasattr(term_cfg, "func") and hasattr(term_cfg, "weight"):
                costs.append(term_cfg.func(self.unwrapped, **term_cfg.params))
        if not costs:
            return torch.zeros(self.unwrapped.num_envs, 0, device=self.unwrapped.device)
        return torch.stack(costs, dim=-1)

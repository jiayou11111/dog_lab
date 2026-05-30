# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific termination terms.

The current Go2W-Piper task uses Isaac Lab's built-in base-contact and timeout
terminations configured in ``config/rough_env_cfg.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ._helpers import base_roll_pitch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_base_state(
    env: ManagerBasedRLEnv,
    max_roll_pitch: float = 0.8,
    min_base_height: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base roll/pitch is too large or base height is too low."""

    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch = base_roll_pitch(asset)
    tilted = torch.any(torch.abs(roll_pitch) > max_roll_pitch, dim=1)
    too_low = asset.data.root_pos_w[:, 2] < min_base_height
    return tilted | too_low


__all__ = ["illegal_base_state"]

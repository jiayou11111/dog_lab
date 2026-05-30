# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="DogLab-Go2W-Piper-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:Go2wPiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Go2W-Piper-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:Go2wPiperFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Go2W-Piper-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.rough_env_cfg:Go2wPiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperRoughPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Go2W-Piper-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.rough_env_cfg:Go2wPiperRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperRoughPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Go2W-Piper-Flat-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:Go2wPiperFlatGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperFlatGraspPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Go2W-Piper-Flat-Grasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:Go2wPiperFlatGraspEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperFlatGraspPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Go2W-Piper-Rough-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.rough_env_cfg:Go2wPiperRoughGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperRoughGraspPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Go2W-Piper-Rough-Grasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.rough_env_cfg:Go2wPiperRoughGraspEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperRoughGraspPPORunnerCfg",
    },
)

# Backwards-compatible names for older notes/scripts.
gym.register(
    id="DogLab-Velocity-Flat-Go2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:Go2wPiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Velocity-Flat-Go2W-Piper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:Go2wPiperFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Velocity-Rough-Go2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.rough_env_cfg:Go2wPiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperRoughPPORunnerCfg",
    },
)

gym.register(
    id="DogLab-Velocity-Rough-Go2W-Piper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.rough_env_cfg:Go2wPiperRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_rsl_rl_ppo_cfg:Go2wPiperRoughPPORunnerCfg",
    },
)

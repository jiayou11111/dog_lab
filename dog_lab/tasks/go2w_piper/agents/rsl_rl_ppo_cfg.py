# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class Go2wPiperRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 1
    num_steps_per_env = 48
    max_iterations = 15000
    save_interval = 500
    experiment_name = "go2w_piper_cost"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128],
        critic_hidden_dims=[128],
        activation="elu",
    )
    loco_init_std = [[1.0, 1.0, 1.0, 1.0] * 4 + [1.0] * 6]
    loco_min_policy_std = [[0.2, 0.2, 0.2, 0.2] * 4 + [0.2] * 6]
    loco_leg_control_head_hidden_dims = [128, 128]
    loco_arm_control_head_hidden_dims = [128, 128]
    loco_priv_encoder_dims = [64, 20]
    loco_cost_hidden_dims = [128, 128, 128]
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=1.0,
    )
    loco_mixing_schedule = [1.0, 0, 3000]
    loco_dagger_update_freq = 20
    loco_priv_reg_coef_schedual = [0, 0.1, 3000, 7000]
    loco_cost_value_loss_coef = 1.0
    loco_cost_viol_loss_coef = 1.0


@configclass
class Go2wPiperFlatPPORunnerCfg(Go2wPiperRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2w_piper_cost"

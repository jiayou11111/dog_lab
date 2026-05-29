# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration objects for the Loco-Manipulation RSL-RL fork.

The field layout follows :mod:`isaaclab_rl.rsl_rl.rl_cfg`, while exposing the
extra Loco/P3O/ROA parameters consumed by ``third_party/loco_rsl_rl``.
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class LocoRslRlPpoActorCriticCfg:
    """Configuration for the Loco actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name."""

    init_noise_std: list[list[float]] = MISSING
    """Initial per-action standard deviation."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the shared actor backbone."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the shared critic backbone."""

    activation: str = MISSING
    """The activation function for actor, critic, encoders, and heads."""

    leg_control_head_hidden_dims: list[int] = MISSING
    """Hidden dimensions for the locomotion policy/value heads."""

    arm_control_head_hidden_dims: list[int] = MISSING
    """Hidden dimensions for the arm policy/value heads."""

    priv_encoder_dims: list[int] = MISSING
    """Hidden/output dimensions for the privileged-information encoder."""

    cost_hidden_dims: list[int] = MISSING
    """Hidden dimensions for the P3O cost critic."""


@configclass
class LocoRslRlPpoAlgorithmCfg:
    """Configuration for Loco PPO with advantage fusion, P3O costs, and ROA distillation."""

    class_name: str = "PPO"
    """The algorithm class name."""

    value_loss_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    entropy_coef: float = MISSING
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    desired_kl: float | None = MISSING
    max_grad_norm: float = MISSING

    min_policy_std: list[list[float]] = MISSING
    """Minimum per-action policy standard deviation enforced after each PPO update."""

    mixing_schedule: list[float] = MISSING
    """Advantage fusion schedule: max mix, start iteration, ramp duration."""

    dagger_update_freq: int = MISSING
    """Iterations between history-encoder DAgger updates."""

    priv_reg_coef_schedual: list[float] = MISSING
    """ROA privileged-latent regularization schedule used by the original code."""

    cost_value_loss_coef: float = MISSING
    """Coefficient for the cost critic value loss."""

    cost_viol_loss_coef: float = MISSING
    """Coefficient for the P3O constraint violation loss."""


@configclass
class LocoRslRlOnPolicyRunnerCfg:
    """Configuration of the Loco on-policy runner."""

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool = False

    policy: LocoRslRlPpoActorCriticCfg = MISSING
    algorithm: LocoRslRlPpoAlgorithmCfg = MISSING

    policy_class_name: str = "ActorCritic"
    """Name expected by the original Loco runner."""

    algorithm_class_name: str = "PPO"
    """Name expected by the original Loco runner."""

    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"

    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from dog_lab.assets.robots.go2w_piper import GO2W_PIPER_CFG  # isort: skip

from .. import mdp as go2w_mdp
from ..mdp.actions import FixedJointPositionActionCfg
from .loco_params import LOCO_CONSTRAINTS, LOCO_CONTROL


@configclass
class LocoRewardSplitCfg:
    """Reward grouping consumed by the Loco-Manipulation cost-aware runner."""

    leg_terms = (
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "flat_orientation_l2",
        "dof_torques_l2",
        "dof_vel_loco",
        "dof_acc_loco",
        "base_height_l2",
        "undesired_contacts",
        "action_rate_l2",
        "stand_still_loco",
        "dof_pos_limits",
        "run_still_loco",
        "joint_power",
        "joint_mirror",
    )
    arm_terms = ("arm_stage1_zero",)
    only_positive_rewards = True


@configclass
class LocoP3OCostCfg:
    """P3O/constraint costs migrated from Loco-Manipulation."""

    dof_pos_limits = RewTerm(
        func=go2w_mdp.cost_joint_pos_limits_loco,
        weight=LOCO_CONSTRAINTS.dof_pos_limits_scale,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS)},
    )
    dof_vel_limits = RewTerm(
        func=go2w_mdp.cost_joint_vel_limits_loco,
        weight=LOCO_CONSTRAINTS.dof_vel_limits_scale,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS)},
    )
    d_values = {
        "dof_pos_limits": LOCO_CONSTRAINTS.dof_pos_limits_d_value,
        "dof_vel_limits": LOCO_CONSTRAINTS.dof_vel_limits_d_value,
    }


@configclass
class LocoRunnerEnvCfg:
    """Shape metadata required by the Loco-Manipulation RSL-RL fork."""

    num_proprio = 71
    num_priv = 22
    history_len = 10
    num_leg_actions = 16
    num_arm_actions = 6
    num_costs = 2


@configclass
class LocoObservationsCfg:
    """Loco-Manipulation observation groups in Isaac Lab manager form."""

    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(
            func=go2w_mdp.loco_policy_proprio,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS),
            },
        )
        priv = ObsTerm(func=go2w_mdp.loco_privileged_obs, params={})
        proprio_history = ObsTerm(
            func=go2w_mdp.loco_policy_proprio,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS),
            },
            history_length=10,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        proprio = ObsTerm(
            func=go2w_mdp.loco_policy_proprio,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS),
            },
        )
        priv = ObsTerm(func=go2w_mdp.loco_privileged_obs, params={})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2wPiperRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: LocoObservationsCfg = LocoObservationsCfg()
    loco_runner: LocoRunnerEnvCfg = LocoRunnerEnvCfg()
    loco_reward_split: LocoRewardSplitCfg = LocoRewardSplitCfg()
    loco_costs: LocoP3OCostCfg = LocoP3OCostCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = GO2W_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # Match the legged-gym control cadence: 2 physics steps at 0.005 s.
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        # Stage 1 control split from Loco Go2wPiperCfg:
        # - 12 leg joints use position targets with action_scale=0.25.
        # - 4 foot wheel joints use velocity targets with action_scale_vel=10.0.
        # - Piper arm consumes 6 policy dimensions but is held at the default pose.
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            scale=LOCO_CONTROL.leg_action_scale,
            use_default_offset=True,
        )
        self.actions.wheel_vel = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=[".*_foot_joint"],
            scale=LOCO_CONTROL.wheel_action_scale_vel,
            use_default_offset=True,
        )
        self.actions.arm_hold = FixedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            action_dim=LOCO_CONTROL.num_arm_actions,
        )

        # Velocity commands from the gym task: no lateral command, heading command enabled.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-1.57, 1.57)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 2.0

        # Domain randomization mirrors the imported legged-gym values where Lab has built-in events.
        self.events.push_robot = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.3)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # Rewards and terminations adapted to the Go2W-Piper body names.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*thigh", ".*calf", "link.*"]
        self.rewards.undesired_contacts.weight = -0.1
        self.rewards.dof_torques_l2.weight = -0.0005
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS)
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS)
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_rate_l2.func = go2w_mdp.base_action_rate_l2
        self.rewards.action_rate_l2.params = {"num_base_actions": LOCO_CONTROL.num_base_actions}
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS)

        self.rewards.dof_vel_loco = RewTerm(
            func=go2w_mdp.joint_vel_l2_without_wheels,
            weight=-2.0e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS)},
        )
        self.rewards.dof_acc_loco = RewTerm(
            func=go2w_mdp.joint_acc_l2_without_wheels,
            weight=-2.0e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS)},
        )
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-0.5,
            params={"target_height": 0.4, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.dof_vel_limits = RewTerm(
            func=mdp.joint_vel_limits,
            weight=-LOCO_CONSTRAINTS.dof_vel_limits_scale if LOCO_CONSTRAINTS.enabled else 0.0,
            params={"soft_ratio": 1.0, "asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS)},
        )
        self.rewards.joint_power = RewTerm(
            func=go2w_mdp.joint_power_l1,
            weight=-5.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS)},
        )
        self.rewards.stand_still_loco = RewTerm(
            func=go2w_mdp.stand_still_loco,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS),
            },
        )
        self.rewards.run_still_loco = RewTerm(
            func=go2w_mdp.run_still_loco,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.BASE_JOINTS),
            },
        )
        self.rewards.joint_mirror = RewTerm(
            func=go2w_mdp.joint_mirror_l2,
            weight=-1.0,
            params={
                "mirror_joint_pairs": (
                    ("FL_(hip|thigh|calf)_joint", "FR_(hip|thigh|calf)_joint"),
                    ("RL_(hip|thigh|calf)_joint", "RR_(hip|thigh|calf)_joint"),
                )
            },
        )
        self.rewards.arm_deviation = RewTerm(
            func=go2w_mdp.arm_deviation_l2,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=go2w_mdp.ARM_JOINTS)},
        )
        self.rewards.arm_stage1_zero = RewTerm(
            func=go2w_mdp.zero_arm_reward,
            weight=0.0,
            params={},
        )

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class Go2wPiperRoughEnvCfg_PLAY(Go2wPiperRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

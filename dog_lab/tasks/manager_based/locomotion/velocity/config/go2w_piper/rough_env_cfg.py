# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from dog_lab.assets.robots.go2w_piper import GO2W_PIPER_CFG  # isort: skip


@configclass
class Go2wPiperRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
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

        # Control split from Go2wPiperCfg:
        # - 12 leg joints use position targets with action_scale=0.25.
        # - 4 foot wheel joints use velocity targets with action_scale_vel=10.0.
        # - Piper arm starts as a conservative joint-position action so the first
        #   bring-up path verifies URDF import and joint control before adding IK.
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            scale=0.25,
            use_default_offset=True,
        )
        self.actions.wheel_vel = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=[".*_foot_joint"],
            scale=10.0,
            use_default_offset=True,
        )
        self.actions.arm_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            scale=0.1,
            use_default_offset=True,
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
        self.rewards.dof_acc_l2.weight = -2.0e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -1.0

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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree Go2W with Piper manipulator."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from dog_lab import DOG_LAB_DATA_DIR


GO2W_PIPER_USD_PATH = (
    f"{DOG_LAB_DATA_DIR}/Robots/Unitree/Go2W-Piper/"
    "go2w_piper_description/USD.usd"
)
"""Path to the Go2W-Piper USD asset."""

GO2W_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2W_PIPER_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.67,
            ".*_calf_joint": -1.3,
            ".*_foot_joint": 0.0,
            "joint1": 0.0,
            "joint2": 1.57,
            "joint3": -0.8,
            "joint4": 0.0,
            "joint5": -0.7,
            "joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={
                ".*_hip_joint": 23.7,
                ".*_thigh_joint": 23.7,
                ".*_calf_joint": 35.55,
            },
            velocity_limit={
                ".*_hip_joint": 30.1,
                ".*_thigh_joint": 30.1,
                ".*_calf_joint": 20.07,
            },
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=23.7,
            velocity_limit=30.1,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit=100.0,
            velocity_limit={"joint[1-5]": 5.0, "joint6": 3.0},
            stiffness=400.0,
            damping=20.0,
            friction=0.0,
        ),
    },
)
"""Configuration of the Go2W-Piper mobile manipulator loaded directly from USD."""

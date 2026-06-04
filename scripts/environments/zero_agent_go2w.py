# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="DogLab-Go2W-Piper-Flat-Play-v0",
    help="DogLab task to use as the control/config template.",
)
parser.add_argument(
    "--wheel_test",
    choices=["none", "same", "lr", "rl"],
    default="none",
    help="Manual wheel action test pattern.",
)
parser.add_argument(
    "--wheel_action",
    type=float,
    default=0.0,
    help="Wheel action magnitude for manual wheel test.",
)
parser.add_argument("--debug_steps", type=int, default=0, help="Print wheel diagnostics for N steps.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import dog_lab.tasks  # noqa: F401
import dog_lab.tasks.go2w_piper.mdp as go2w_mdp
from dog_lab import DOG_LAB_DATA_DIR
from dog_lab.tasks.go2w_piper.mdp._helpers import base_roll_pitch, local_joint_ids
from isaaclab_tasks.utils import parse_env_cfg


GO2W_USD_PATH = (
    f"{DOG_LAB_DATA_DIR}/Robots/Unitree/Go2W-Piper/"
    "go2w_piper_description/USD_go2w.usd"
)


def _go2w_only_loco_policy_proprio(env, command_name: str = "base_velocity", asset_cfg=None) -> torch.Tensor:
    """DogLab Loco proprioception with the six Piper joints padded by zeros."""

    asset = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    base_joint_ids, _ = asset.find_joints(go2w_mdp.BASE_JOINTS, preserve_order=True)

    roll_pitch = base_roll_pitch(asset)
    base_ang_vel = asset.data.root_ang_vel_b * 0.25
    joint_err = asset.data.joint_pos[:, base_joint_ids] - asset.data.default_joint_pos[:, base_joint_ids]
    local_wheel_ids = local_joint_ids(asset, base_joint_ids, go2w_mdp.WHEEL_JOINTS)
    if local_wheel_ids:
        joint_err[:, local_wheel_ids] = 0.0
    joint_vel = asset.data.joint_vel[:, base_joint_ids] * 0.05

    arm_pad = torch.zeros(env.num_envs, len(go2w_mdp.ARM_JOINTS), device=env.device)
    actions = env.action_manager.action[:, :16]
    commands = env.command_manager.get_command(command_name)[:, :3]
    ee_goal_local = torch.zeros(env.num_envs, 3, device=env.device)
    return torch.cat(
        (
            roll_pitch,
            base_ang_vel,
            joint_err,
            arm_pad,
            joint_vel,
            arm_pad,
            actions,
            commands,
            ee_goal_local,
        ),
        dim=-1,
    )


def _configure_go2w_only_env(env_cfg) -> None:
    """Use the DogLab control stack with a Go2W-only USD and no arm terms."""

    env_cfg.scene.robot.spawn.usd_path = GO2W_USD_PATH
    env_cfg.scene.robot.actuators.pop("arm", None)
    for joint_name in go2w_mdp.ARM_JOINTS:
        env_cfg.scene.robot.init_state.joint_pos.pop(joint_name, None)

    env_cfg.actions.arm_ik = None
    env_cfg.actions.arm_hold = None

    for group in (env_cfg.observations.policy, env_cfg.observations.critic):
        group.proprio.func = _go2w_only_loco_policy_proprio
        group.proprio.params["asset_cfg"].joint_names = go2w_mdp.BASE_JOINTS
        if hasattr(group, "proprio_history") and group.proprio_history is not None:
            group.proprio_history.func = _go2w_only_loco_policy_proprio
            group.proprio_history.params["asset_cfg"].joint_names = go2w_mdp.BASE_JOINTS

    env_cfg.events.randomize_loco_domain = None
    if hasattr(env_cfg.events, "add_gripper_mass"):
        env_cfg.events.add_gripper_mass = None
    if hasattr(env_cfg.events, "randomize_arm_gains"):
        env_cfg.events.randomize_arm_gains = None

    env_cfg.rewards.arm_deviation = None
    env_cfg.rewards.tracking_ee_cart_world = None
    env_cfg.rewards.tracking_ee_orn = None
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*thigh", ".*calf"]


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_go2w_only_env(env_cfg)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    base_action = None
    try:
        base_action = env.unwrapped.action_manager.get_term("loco_base")
        print("[WHEEL TEST] usd_path:", GO2W_USD_PATH)
        print("[WHEEL TEST] wheel_joint_map:", base_action._wheel_joint_map)
        print("[WHEEL TEST] expected_order:", ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"])
    except Exception as exc:
        print("[WHEEL TEST] failed to get loco_base:", exc)

    debug_step = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            if args_cli.wheel_test != "none" and base_action is not None:
                a = args_cli.wheel_action
                if args_cli.wheel_test == "same":
                    pattern = torch.tensor([a, a, a, a], device=actions.device)
                elif args_cli.wheel_test == "lr":
                    pattern = torch.tensor([a, -a, a, -a], device=actions.device)
                elif args_cli.wheel_test == "rl":
                    pattern = torch.tensor([-a, a, -a, a], device=actions.device)
                actions[:, base_action._wheel_local_ids] = pattern
            # apply actions
            env.step(actions)
            if args_cli.debug_steps > 0 and debug_step < args_cli.debug_steps:
                if args_cli.wheel_test != "none":
                    _print_wheel_debug(env, debug_step, base_action)
                debug_step += 1

    # close the simulator
    env.close()


def _print_wheel_debug(env, step: int, base_action):
    """Print wheel command, measured wheel state, and resulting base motion."""

    if base_action is None:
        return
    robot = env.unwrapped.scene["robot"]
    wheel_joint_ids = list(getattr(base_action, "_wheel_joint_ids", []))
    wheel_names = [base_action._joint_names[i] for i in getattr(base_action, "_wheel_local_ids", [])]
    wheel_target = base_action._joint_vel_target[0].detach().cpu()
    wheel_vel = robot.data.joint_vel[0, wheel_joint_ids].detach().cpu()
    root_lin_vel_b = robot.data.root_lin_vel_b[0].detach().cpu()
    root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu()
    wheel_torque = robot.data.applied_torque[0, wheel_joint_ids].detach().cpu()
    print(
        "[WHEEL TEST]"
        f" step={step}"
        f" pattern={args_cli.wheel_test}"
        f" wheel_action={args_cli.wheel_action:.3f}"
        f" wheel_names={wheel_names}"
        f" wheel_target={['%.3f' % x for x in wheel_target.tolist()]}"
        f" wheel_vel={['%.3f' % x for x in wheel_vel.tolist()]}"
        f" root_lin_vel_b={['%.3f' % x for x in root_lin_vel_b.tolist()]}"
        f" root_yaw_vel_b={root_ang_vel_b[2].item():.3f}"
        f" wheel_torque={['%.3f' % x for x in wheel_torque.tolist()]}"
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

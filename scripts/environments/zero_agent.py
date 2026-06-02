# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--arm_mode",
    type=str,
    default="hold",
    choices=["hold", "ik", "ik_fixed"],
    help="Arm behavior for zero-agent checks.",
)
parser.add_argument(
    "--ik_start",
    type=float,
    nargs=3,
    default=(0.5, 0.3, 0.0),
    metavar=("L", "PITCH", "YAW"),
    help="Initial spherical EE target for IK checks.",
)
parser.add_argument(
    "--ik_goal",
    type=float,
    nargs=3,
    default=(0.5, 0.35, 0.0),
    metavar=("L", "PITCH", "YAW"),
    help="Fixed spherical EE target used by --arm_mode ik_fixed.",
)
parser.add_argument("--ik_traj_time", type=float, default=4.0, help="Seconds to move from ik_start to ik_goal.")
parser.add_argument("--ik_hold_time", type=float, default=1000.0, help="Seconds to hold ik_goal before resampling.")
parser.add_argument("--random_commands", action="store_true", help="Keep the task's random base velocity commands.")
parser.add_argument("--debug_steps", type=int, default=0, help="Print IK diagnostics for N environment steps.")
parser.add_argument(
    "--wheel_test",
    type=str,
    default="none",
    choices=["none", "same", "lr", "rl"],
    help="Manual wheel action test pattern.",
)
parser.add_argument(
    "--wheel_action",
    type=float,
    default=1.0,
    help="Wheel action magnitude for manual wheel test.",
)
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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dog_lab  # noqa: F401
import isaaclab_tasks  # noqa: F401
from dog_lab.tasks.go2w_piper.mdp import ARM_JOINTS
from dog_lab.tasks.go2w_piper.mdp.actions import FixedJointPositionActionCfg
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if not args_cli.random_commands and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.0, 0.0)
        if command_cfg.heading_command:
            command_cfg.ranges.heading = (0.0, 0.0)
        command_cfg.rel_standing_envs = 0.0
    if args_cli.arm_mode == "hold" and hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "arm_ik"):
        env_cfg.actions.arm_ik = None
        env_cfg.actions.arm_hold = FixedJointPositionActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINTS,
            action_dim=6,
        )
    elif args_cli.arm_mode in ("ik", "ik_fixed") and hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "arm_ik"):
        env_cfg.actions.arm_hold = None
        arm_ik_cfg = env_cfg.actions.arm_ik
        arm_ik_cfg.init_pos_start = tuple(args_cli.ik_start)
        arm_ik_cfg.traj_time = args_cli.ik_traj_time
        arm_ik_cfg.hold_time = args_cli.ik_hold_time
        if args_cli.arm_mode == "ik_fixed":
            goal_l, goal_p, goal_y = args_cli.ik_goal
            arm_ik_cfg.init_pos_end = tuple(args_cli.ik_goal)
            arm_ik_cfg.sample_initial_goal = False
            arm_ik_cfg.resample_goals = False
            arm_ik_cfg.pos_l = (goal_l, goal_l)
            arm_ik_cfg.pos_p = (goal_p, goal_p)
            arm_ik_cfg.pos_y = (goal_y, goal_y)
            arm_ik_cfg.delta_orn_r = (0.0, 0.0)
            arm_ik_cfg.delta_orn_p = (0.0, 0.0)
            arm_ik_cfg.delta_orn_y = (0.0, 0.0)
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
        print("[WHEEL TEST] wheel_local_ids:", base_action._wheel_local_ids)
        print("[WHEEL TEST] wheel_names:", [base_action._joint_names[i] for i in base_action._wheel_local_ids])
        print("[WHEEL TEST] expected_order:", ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"])
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
                _print_ik_debug(env, debug_step)
                debug_step += 1

    # close the simulator
    env.close()


def _print_wheel_debug(env, step: int, base_action):
    """Print compact wheel order, command, and resulting base velocity diagnostics."""

    if base_action is None:
        return
    robot = env.unwrapped.scene["robot"]
    wheel_joint_ids = list(getattr(base_action, "_wheel_joint_ids", []))
    wheel_names = [base_action._joint_names[i] for i in getattr(base_action, "_wheel_local_ids", [])]
    wheel_target = base_action._joint_vel_target[0].detach().cpu()
    wheel_vel = robot.data.joint_vel[0, wheel_joint_ids].detach().cpu()
    root_lin_vel_b = robot.data.root_lin_vel_b[0].detach().cpu()
    root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu()
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
    )


def _print_ik_debug(env, step: int):
    """Print compact IK tracking diagnostics for the first environment."""

    try:
        arm = env.unwrapped.action_manager.get_term("arm_ik")
    except KeyError:
        if step == 0:
            print("[DEBUG] arm_ik is disabled; use --arm_mode ik or --arm_mode ik_fixed to test IK.")
        return
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu()
    root_pos = robot.data.root_pos_w[0].detach().cpu()
    root_quat = robot.data.root_quat_w[0]
    root_euler = math_utils.euler_xyz_from_quat(root_quat.unsqueeze(0))
    root_euler = torch.stack(root_euler, dim=-1)[0].detach().cpu()
    ee_pos = robot.data.body_pos_w[0, arm._body_idx].detach().cpu()
    goal_pos = arm.curr_ee_goal_cart_world[0].detach().cpu()
    goal_sphere = arm.curr_ee_goal_sphere[0].detach().cpu()
    ee_goal_local = arm.ee_goal_local_cart[0].detach().cpu()
    err = goal_pos - ee_pos
    joint_pos = robot.data.joint_pos[0, arm._joint_ids].detach().cpu()
    joint_target = arm._joint_pos_target[0].detach().cpu()
    print(
        "[DEBUG]"
        f" step={step}"
        f" goal_timer={arm.goal_timer[0].item():.1f}"
        f" command={['%.3f' % x for x in command.tolist()]}"
        f" root_pos={['%.3f' % x for x in root_pos.tolist()]}"
        f" root_rpy={['%.3f' % x for x in root_euler.tolist()]}"
        f" ee_pos={['%.3f' % x for x in ee_pos.tolist()]}"
        f" goal_pos={['%.3f' % x for x in goal_pos.tolist()]}"
        f" err_norm={err.norm().item():.4f}"
        f" goal_sphere={['%.3f' % x for x in goal_sphere.tolist()]}"
        f" ee_goal_local={['%.3f' % x for x in ee_goal_local.tolist()]}"
        f" arm_joint_pos={['%.3f' % x for x in joint_pos.tolist()]}"
        f" arm_joint_target={['%.3f' % x for x in joint_target.tolist()]}"
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()



# ./isaaclab_ubuntu.sh -p scripts/environments/zero_agent.py --task DogLab-Go2W-Piper-Flat-Play-v0 --num_envs 1 --arm_mode ik_fixed --ik_start 0.5 0.3 0.0 --ik_goal 0.6 0.5 0.5 --ik_traj_time 1.0 --debug_steps 600
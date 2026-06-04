# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

DOG_LAB_ROOT = Path(__file__).resolve().parents[3]
LOCO_RSL_RL_ROOT = DOG_LAB_ROOT / "third_party" / "loco_rsl_rl"
if LOCO_RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(LOCO_RSL_RL_ROOT))

if sys.platform == "win32":
    # Isaac Sim ships HDF5 DLLs used by native sensor plugins. Load its h5py first so
    # IsaacLab's later h5py import reuses the same DLL family instead of conda's.
    import os

    isaac_path = os.environ.get("ISAAC_PATH")
    if isaac_path:
        kit_site_packages = Path(isaac_path) / "kit" / "python" / "Lib" / "site-packages"
        if kit_site_packages.is_dir():
            sys.path.insert(0, str(kit_site_packages))
            import h5py  # noqa: F401
            sys.path.pop(0)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--command_x", type=float, default=1, help="Fixed forward velocity command for play.")
parser.add_argument("--command_y", type=float, default=0.0, help="Fixed lateral velocity command for play.")
parser.add_argument("--command_yaw", type=float, default=0.0, help="Fixed yaw velocity command for play.")
parser.add_argument("--command_heading", type=float, default=0.0, help="Fixed heading command for play.")
parser.add_argument(
    "--use_heading_command",
    action="store_true",
    default=False,
    help="Use heading control for play yaw commands. By default play uses the fixed --command_yaw value.",
)
parser.add_argument(
    "--random_commands",
    action="store_true",
    default=False,
    help="Use the task's training command sampler instead of the fixed Loco play command.",
)
parser.add_argument(
    "--disable_play_arm_commands",
    action="store_true",
    default=False,
    help="Keep the task's default arm EE goal sampler instead of the play-time sampler.",
)
parser.add_argument("--arm_command_traj_time", type=float, default=2, help="Play arm EE command transition time.")
parser.add_argument("--arm_command_hold_time", type=float, default=1, help="Play arm EE command hold time.")
parser.add_argument("--arm_command_l_min", type=float, default=0.5, help="Minimum arm EE spherical radius.")
parser.add_argument("--arm_command_l_max", type=float, default=0.7, help="Maximum arm EE spherical radius.")
parser.add_argument("--arm_command_pitch_min", type=float, default=-0.524, help="Minimum arm EE spherical pitch.")
parser.add_argument("--arm_command_pitch_max", type=float, default=1.047, help="Maximum arm EE spherical pitch.")
parser.add_argument("--arm_command_yaw_min", type=float, default=-1.57, help="Minimum arm EE spherical yaw.")
parser.add_argument("--arm_command_yaw_max", type=float, default=1.57, help="Maximum arm EE spherical yaw.")
parser.add_argument(
    "--arm_command_min_distance",
    type=float,
    default=0.18,
    help="Minimum Cartesian distance between consecutive play arm EE commands.",
)
parser.add_argument(
    "--priv_encoding",
    action="store_true",
    default=False,
    help="Use privileged latent inference instead of the Loco play history encoder.",
)
parser.add_argument("--debug_steps", type=int, default=0, help="Print base/arm diagnostics for N play steps.")
parser.add_argument("--debug_arm", action="store_true", default=False, help="Also print arm IK diagnostics.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

sys.path.insert(0, str(DOG_LAB_ROOT))

import dog_lab  # noqa: F401
import isaaclab_tasks  # noqa: F401
from dog_lab.rl.loco_rsl_rl import LocoRslRlOnPolicyRunnerCfg, LocoRslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: LocoRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if not args_cli.random_commands and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
        command_cfg.ranges.lin_vel_y = (args_cli.command_y, args_cli.command_y)
        command_cfg.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)
        command_cfg.heading_command = args_cli.use_heading_command
        if command_cfg.heading_command:
            command_cfg.ranges.heading = (args_cli.command_heading, args_cli.command_heading)
        command_cfg.rel_standing_envs = 0.0
    if not args_cli.disable_play_arm_commands and hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "arm_ik"):
        arm_cfg = env_cfg.actions.arm_ik
        arm_cfg.traj_time = args_cli.arm_command_traj_time
        arm_cfg.hold_time = args_cli.arm_command_hold_time
        arm_cfg.sample_initial_goal = True
        arm_cfg.resample_goals = True
        arm_cfg.pos_l = (args_cli.arm_command_l_min, args_cli.arm_command_l_max)
        arm_cfg.pos_p = (args_cli.arm_command_pitch_min, args_cli.arm_command_pitch_max)
        arm_cfg.pos_y = (args_cli.arm_command_yaw_min, args_cli.arm_command_yaw_max)
        if hasattr(arm_cfg, "min_resample_goal_distance"):
            arm_cfg.min_resample_goal_distance = args_cli.arm_command_min_distance
        print(
            "[INFO] Play arm EE command sampler:"
            f" traj={arm_cfg.traj_time:.2f}s hold={arm_cfg.hold_time:.2f}s"
            f" pos_l={arm_cfg.pos_l} pos_p={arm_cfg.pos_p} pos_y={arm_cfg.pos_y}"
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = LocoRslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    dt = env.unwrapped.physics_dt

    # reset environment
    obs_result = env.get_observations()
    obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result
    timestep = 0
    debug_step = 0
    if args_cli.debug_steps > 0:
        try:
            base_action = env.unwrapped.action_manager.get_term("loco_base")
            print(
                "[BASE_DEBUG_INIT]"
                f" action_term={base_action.__class__.__name__}"
                f" base_joint_names={base_action._joint_names}"
                f" wheel_joint_map={base_action._wheel_joint_map}"
                f" leg_joint_map={base_action._non_wheel_joint_map}"
                f" position_scale={base_action.cfg.position_scale}"
                f" velocity_scale={base_action.cfg.velocity_scale}"
            )
        except Exception as exc:
            print(f"[BASE_DEBUG_INIT] failed_to_get_loco_base={exc}")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs, hist_encoding=not args_cli.priv_encoding)
            arm_debug_before = None
            if args_cli.debug_arm and args_cli.debug_steps > 0 and debug_step < args_cli.debug_steps:
                unwrapped = env.unwrapped
                try:
                    arm_ik = unwrapped.action_manager.get_term("arm_ik")
                    arm_debug_before = {
                        "registered": arm_ik.__class__.__name__ == "LocoArmIKAction",
                        "class_name": arm_ik.__class__.__name__,
                        "timer": arm_ik.goal_timer[0].clone(),
                        "goal_world": arm_ik.curr_ee_goal_cart_world[0].clone(),
                        "goal_local": arm_ik.ee_goal_local_cart[0].clone(),
                        "goal_sphere": arm_ik.ee_goal_sphere[0].clone(),
                        "curr_goal_sphere": arm_ik.curr_ee_goal_sphere[0].clone(),
                    }
                except Exception as exc:
                    arm_debug_before = {"error": str(exc)}
            # env stepping
            obs, _, leg_rewards, arm_rewards, costs, dones, extras = env.step(actions)
            if args_cli.debug_steps > 0 and debug_step < args_cli.debug_steps:
                unwrapped = env.unwrapped
                robot = unwrapped.scene["robot"]
                try:
                    base_action = unwrapped.action_manager.get_term("loco_base")
                    command = unwrapped.command_manager.get_command("base_velocity")[0, :3]
                    root_lin_vel_b = robot.data.root_lin_vel_b[0]
                    root_ang_vel_b = robot.data.root_ang_vel_b[0]
                    root_z = robot.data.root_pos_w[0, 2]
                    wheel_actions = actions[0, base_action._wheel_local_ids]
                    wheel_vel_target = base_action._joint_vel_target[0]
                    wheel_vel = robot.data.joint_vel[0, base_action._wheel_joint_ids]
                    wheel_pos = robot.data.joint_pos[0, base_action._wheel_joint_ids]
                    wheel_names = [base_action._joint_names[i] for i in base_action._wheel_local_ids]
                    wheel_torque = robot.data.applied_torque[0, base_action._wheel_joint_ids]
                    episode_len = unwrapped.episode_length_buf[0].item()
                    timeout = None
                    if isinstance(extras, dict) and "time_outs" in extras:
                        timeout = bool(extras["time_outs"][0].item())
                    print(
                        "[BASE_DEBUG]"
                        f" step={debug_step}"
                        f" episode_len={episode_len}"
                        f" done={bool(dones[0].item())}"
                        f" timeout={timeout}"
                        f" command={['%.3f' % x for x in command.detach().cpu().tolist()]}"
                        f" root_z={root_z.item():.3f}"
                        f" lin_vel_b={['%.3f' % x for x in root_lin_vel_b.detach().cpu().tolist()]}"
                        f" yaw_vel_b={root_ang_vel_b[2].item():.3f}"
                        f" vel_err_x={(command[0] - root_lin_vel_b[0]).item():.3f}"
                        f" vel_err_y={(command[1] - root_lin_vel_b[1]).item():.3f}"
                        f" yaw_err={(command[2] - root_ang_vel_b[2]).item():.3f}"
                        f" action_absmax={actions[0].abs().max().item():.3f}"
                        f" base_action_absmax={actions[0, : base_action.action_dim].abs().max().item():.3f}"
                        f" wheel_names={wheel_names}"
                        f" wheel_actions={['%.3f' % x for x in wheel_actions.detach().cpu().tolist()]}"
                        f" wheel_vel_target={['%.3f' % x for x in wheel_vel_target.detach().cpu().tolist()]}"
                        f" wheel_vel={['%.3f' % x for x in wheel_vel.detach().cpu().tolist()]}"
                        f" wheel_pos={['%.3f' % x for x in wheel_pos.detach().cpu().tolist()]}"
                        f" leg_reward_step={leg_rewards[0].item():.4f}"
                        f" arm_reward_step={arm_rewards[0].item():.4f}"
                        f" cost_step={costs[0].detach().cpu().flatten().tolist()}"
                        f" wheel_torque={['%.3f' % x for x in wheel_torque.detach().cpu().tolist()]}"
                    )
                except Exception as exc:
                    print(f"[BASE_DEBUG] step={debug_step} error={exc}")
            if args_cli.debug_arm and args_cli.debug_steps > 0 and debug_step < args_cli.debug_steps:
                unwrapped = env.unwrapped
                robot = unwrapped.scene["robot"]
                try:
                    arm_ik = unwrapped.action_manager.get_term("arm_ik")
                    arm_registered = arm_ik.__class__.__name__ == "LocoArmIKAction"
                    ee_body_idx = getattr(arm_ik, "_body_idx")
                    ee_pos = robot.data.body_pos_w[0, ee_body_idx]
                    goal_world = arm_ik.curr_ee_goal_cart_world[0]
                    goal_local = arm_ik.ee_goal_local_cart[0]
                    goal_sphere = arm_ik.ee_goal_sphere[0]
                    curr_goal_sphere = arm_ik.curr_ee_goal_sphere[0]
                    timer = arm_ik.goal_timer[0]
                    target_error = torch.linalg.norm(goal_world - ee_pos).item()
                    goal_world_delta = 0.0
                    goal_local_delta = 0.0
                    goal_sphere_delta = 0.0
                    curr_goal_sphere_delta = 0.0
                    timer_delta = 0.0
                    if arm_debug_before is not None and "error" not in arm_debug_before:
                        goal_world_delta = torch.linalg.norm(goal_world - arm_debug_before["goal_world"]).item()
                        goal_local_delta = torch.linalg.norm(goal_local - arm_debug_before["goal_local"]).item()
                        goal_sphere_delta = torch.linalg.norm(goal_sphere - arm_debug_before["goal_sphere"]).item()
                        curr_goal_sphere_delta = torch.linalg.norm(
                            curr_goal_sphere - arm_debug_before["curr_goal_sphere"]
                        ).item()
                        timer_delta = (timer - arm_debug_before["timer"]).item()
                    process_actions_updated = abs(timer_delta) > 0.0 or goal_world_delta > 1.0e-6
                    print(
                        "[ARM_IK_DEBUG]"
                        f" step={debug_step}"
                        f" registered={arm_registered}"
                        f" class={arm_ik.__class__.__name__}"
                        f" process_actions_updated={process_actions_updated}"
                        f" timer={timer.item():.1f}"
                        f" timer_delta={timer_delta:.1f}"
                        f" goal_world_delta={goal_world_delta:.6f}"
                        f" goal_local_delta={goal_local_delta:.6f}"
                        f" ee_goal_sphere={['%.3f' % x for x in goal_sphere.detach().cpu().tolist()]}"
                        f" curr_ee_goal_sphere={['%.3f' % x for x in curr_goal_sphere.detach().cpu().tolist()]}"
                        f" goal_sphere_delta={goal_sphere_delta:.6f}"
                        f" curr_goal_sphere_delta={curr_goal_sphere_delta:.6f}"
                        f" ee_pos={['%.3f' % x for x in ee_pos.detach().cpu().tolist()]}"
                        f" ee_goal={['%.3f' % x for x in goal_world.detach().cpu().tolist()]}"
                        f" ee_goal_local={['%.3f' % x for x in goal_local.detach().cpu().tolist()]}"
                        f" ee_pos_error={target_error:.4f}"
                    )
                except Exception as exc:
                    before_error = None
                    if arm_debug_before is not None:
                        before_error = arm_debug_before.get("error")
                    print(
                        "[ARM_IK_DEBUG]"
                        f" step={debug_step}"
                        " registered=False"
                        f" before_error={before_error}"
                        f" after_error={exc}"
                    )
            if args_cli.debug_steps > 0 and debug_step < args_cli.debug_steps:
                debug_step += 1
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
# ./isaaclab_ubuntu.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task DogLab-Go2W-Piper-Flat-Play-v0 --checkpoint /home/ymy/isaac_storage/projects/dog/dog_lab/output_total/model_15000.pt --debug_steps 500 --priv_encoding

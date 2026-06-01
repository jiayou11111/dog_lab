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
parser.add_argument("--command_x", type=float, default=1.0, help="Fixed forward velocity command for play.")
parser.add_argument("--command_y", type=float, default=0.0, help="Fixed lateral velocity command for play.")
parser.add_argument("--command_yaw", type=float, default=0.0, help="Fixed yaw velocity command for play.")
parser.add_argument("--command_heading", type=float, default=0.0, help="Fixed heading command for play.")
parser.add_argument(
    "--random_commands",
    action="store_true",
    default=False,
    help="Use the task's training command sampler instead of the fixed Loco play command.",
)
parser.add_argument(
    "--priv_encoding",
    action="store_true",
    default=False,
    help="Use privileged latent inference instead of the Loco play history encoder.",
)
parser.add_argument("--debug_steps", type=int, default=0, help="Print command/action/state diagnostics for N steps.")
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
        if command_cfg.heading_command:
            command_cfg.ranges.heading = (args_cli.command_heading, args_cli.command_heading)
        command_cfg.rel_standing_envs = 0.0

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

    if args_cli.debug_steps > 0:
        try:
            loco_base = env.unwrapped.action_manager.get_term("loco_base")
            wheel_local_ids = list(getattr(loco_base, "_wheel_local_ids", []))
            wheel_names = [loco_base._joint_names[i] for i in wheel_local_ids]
            print("[DEBUG] loco_base action map:")
            for action_id, joint_name in enumerate(loco_base._joint_names):
                marker = " wheel" if action_id in wheel_local_ids else ""
                print(f"[DEBUG]   action[{action_id:02d}] -> {joint_name}{marker}")
            print(f"[DEBUG] wheel_local_ids={wheel_local_ids} wheel_names={wheel_names}")
            print(
                "[DEBUG]"
                f" position_scale={getattr(loco_base.cfg, 'position_scale', None)}"
                f" velocity_scale={getattr(loco_base.cfg, 'velocity_scale', None)}"
            )
        except Exception as exc:
            print(f"[DEBUG] failed to print loco_base action map: {exc}")

    # reset environment
    obs_result = env.get_observations()
    obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result
    timestep = 0
    debug_step = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs, hist_encoding=not args_cli.priv_encoding)
            if args_cli.debug_steps > 0 and debug_step < args_cli.debug_steps:
                unwrapped = env.unwrapped
                robot = unwrapped.scene["robot"]
                command = unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
                root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
                root_vel_b = robot.data.root_lin_vel_b[0].detach().cpu().tolist()
                root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().tolist()
                action0 = actions[0].detach().cpu()
                base_action0 = action0[:16]
                wheel_debug = ""
                try:
                    loco_base = unwrapped.action_manager.get_term("loco_base")
                    wheel_local_ids = list(getattr(loco_base, "_wheel_local_ids", []))
                    wheel_actions = base_action0[wheel_local_ids].tolist()
                    wheel_vel_from_action = (
                        base_action0[wheel_local_ids] * getattr(loco_base.cfg, "velocity_scale", 1.0)
                    ).tolist()
                    prev_wheel_vel_targets = loco_base._joint_vel_target[0].detach().cpu().tolist()
                    wheel_joint_ids = list(getattr(loco_base, "_wheel_joint_ids", []))
                    wheel_joint_vel = robot.data.joint_vel[0, wheel_joint_ids].detach().cpu().tolist()
                    wheel_debug = (
                        f" wheel_actions={['%.3f' % x for x in wheel_actions]}"
                        f" wheel_vel_from_action={['%.3f' % x for x in wheel_vel_from_action]}"
                        f" prev_wheel_vel_targets={['%.3f' % x for x in prev_wheel_vel_targets]}"
                        f" wheel_joint_vel={['%.3f' % x for x in wheel_joint_vel]}"
                    )
                except Exception as exc:
                    wheel_debug = f" wheel_debug_error={exc}"
                print(
                    "[DEBUG]"
                    f" step={debug_step}"
                    f" command={['%.3f' % x for x in command]}"
                    f" root_z={root_pos[2]:.3f}"
                    f" lin_vel_b={['%.3f' % x for x in root_vel_b]}"
                    f" yaw_vel_b={root_ang_vel_b[2]:.3f}"
                    f" action_mean={action0.mean().item():.3f}"
                    f" action_absmax={action0.abs().max().item():.3f}"
                    f" base_action={['%.3f' % x for x in base_action0.tolist()]}"
                    f"{wheel_debug}"
                )
            # env stepping
            obs, _, _, _, _, _, _ = env.step(actions)
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

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from .config import RealDogPiperConfig
from .ee_goal import EEGoalSampler
from .go2w_api import Go2WApi
from .observation import ObservationBuilder
from .piper_api import PiperApi
from .policy import PolicyRunner
from .types import ArmCommand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real Go2W-Piper baseline deployment.")
    parser.add_argument("--actor-path", type=str, default=None)
    parser.add_argument("--hist-encoder-path", type=str, default=None)
    parser.add_argument("--iface", type=str, default="eth0", help="Go2W Unitree SDK2 network interface.")
    parser.add_argument("--piper-can", type=str, default="can0", help="Piper CAN interface.")
    parser.add_argument("--cmd", type=float, nargs=3, default=(0.15, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means until Ctrl-C.")
    parser.add_argument("--dry-run", action="store_true", help="Run policy/observation loop without sending hardware commands.")
    parser.add_argument("--log-every", type=float, default=1.0)
    parser.add_argument("--csv-path", type=str, default="real_dog_piper_dry_obs_action.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RealDogPiperConfig()
    cfg.go2w.network_interface = args.iface
    cfg.piper.can_name = args.piper_can

    policy = PolicyRunner(cfg.policy, actor_path=args.actor_path, hist_encoder_path=args.hist_encoder_path)
    obs_builder = ObservationBuilder(cfg)
    ee_goals = EEGoalSampler(cfg.ee_goal)
    go2w = Go2WApi(cfg.go2w, dry_run=args.dry_run)
    piper = PiperApi(cfg.piper, dry_run=args.dry_run)

    go2w.connect()
    piper.connect()

    commands = np.array(args.cmd, dtype=np.float32)
    start_time = time.time()
    last_log = 0.0
    step = 0
    csv_file = open(args.csv_path, "w", newline="") if args.csv_path else None
    csv_writer = None
    if csv_file is not None:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["step", "time", "cmd_vx", "cmd_vy", "cmd_wz"]
            + [f"obs_{i}" for i in range(cfg.policy.num_proprio)]
            + [f"action_{i}" for i in range(cfg.policy.num_actions)]
            + [f"ee_goal_{i}" for i in range(3)]
        )
    try:
        while args.duration <= 0.0 or time.time() - start_time < args.duration:
            step_start = time.time()
            go2w_state = go2w.read_state()
            piper_state = piper.read_state()
            ee_goal_local = ee_goals.step()

            obs = obs_builder.build(go2w_state, piper_state, commands, ee_goal_local)
            actions = policy.act(obs, obs_builder.history)
            obs_builder.update_last_action(actions)
            now = time.time()
            if csv_writer is not None:
                csv_writer.writerow(
                    [step, now - start_time, *commands.tolist(), *obs.tolist(), *actions.tolist(), *ee_goal_local.tolist()]
                )

            go2w_command = go2w.build_command_from_actions(actions[: cfg.policy.num_base_actions])
            arm_target = piper.solve_ik_position(ee_goal_local, piper_state.joint_pos)

            go2w.send_command(go2w_command)
            piper.send_joint_targets(ArmCommand(joint_pos=arm_target, gripper_m=piper_state.gripper_m))

            # 打印日志
            if now - last_log >= args.log_every:
                mode = "dry-run" if args.dry_run else "real"
                print(
                    f"[{mode}] t={now - start_time:.2f}s "
                    f"cmd={commands.tolist()} "
                    f"action_norm={float(np.linalg.norm(actions)):.3f} "
                    f"ee_goal={ee_goal_local.round(3).tolist()}"
                )
                last_log = now
            step += 1

            sleep_s = cfg.control_dt - (time.time() - step_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file is not None:
            csv_file.close()
        go2w.stop()
        piper.stop()


if __name__ == "__main__":
    main()

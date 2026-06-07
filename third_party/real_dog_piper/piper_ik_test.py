from __future__ import annotations

import argparse
import time

import numpy as np

from .config import RealDogPiperConfig
from .ee_goal import EEGoalSampler
from .piper_api import PiperApi
from .types import ArmCommand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Piper-only IK test.")
    parser.add_argument("--piper-can", type=str, default="can0")
    parser.add_argument("--ee-goal", type=float, nargs=3, default=(0.5, 0.3, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--sample-goal", action="store_true", help="Use the deployment EE goal sampler instead of --ee-goal.")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RealDogPiperConfig()
    cfg.piper.can_name = args.piper_can
    piper = PiperApi(cfg.piper, dry_run=args.dry_run)
    sampler = EEGoalSampler(cfg.ee_goal) if args.sample_goal else None
    fixed_goal = np.array(args.ee_goal, dtype=np.float32)

    piper.connect()
    print(f"[piper] connected can={args.piper_can} dry_run={args.dry_run}")

    start_time = time.time()
    last_log = 0.0
    step = 0
    try:
        while time.time() - start_time < args.duration:
            loop_start = time.time()
            state = piper.read_state()
            ee_goal = sampler.step() if sampler is not None else fixed_goal
            target = piper.solve_ik_position(ee_goal, state.joint_pos)
            piper.send_joint_targets(ArmCommand(joint_pos=target, gripper_m=state.gripper_m))

            now = time.time()
            if now - last_log >= args.log_every:
                print(
                    f"[piper] step={step} t={now - start_time:.2f}s "
                    f"goal={ee_goal.round(4).tolist()} "
                    f"target_rad={target.round(4).tolist()} "
                    f"ee_now={state.ee_pos_local.round(4).tolist()}"
                )
                last_log = now
            step += 1

            sleep_s = cfg.control_dt - (time.time() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        piper.stop()


if __name__ == "__main__":
    main()

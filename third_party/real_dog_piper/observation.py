from __future__ import annotations

import numpy as np

from .config import ARM_JOINT_NAMES, RealDogPiperConfig, WHEEL_INDICES
from .math_utils import euler_from_quat_wxyz
from .types import Go2WState, PiperState


class ObservationBuilder:
    """Build the 71-D proprio block used by dog_lab's Go2W-Piper policy."""

    def __init__(self, cfg: RealDogPiperConfig):
        self.cfg = cfg
        self.default_joint_pos = np.concatenate(
            [cfg.go2w.default_joint_pos, cfg.piper.default_joint_pos],
            axis=0,
        ).astype(np.float32)
        self.last_base_actions = np.zeros(cfg.policy.num_base_actions, dtype=np.float32)
        self.history = np.zeros((cfg.policy.history_len, cfg.policy.num_proprio), dtype=np.float32)

    def update_last_action(self, actions: np.ndarray) -> None:
        self.last_base_actions[:] = np.asarray(actions[: self.cfg.policy.num_base_actions], dtype=np.float32)

    def build(
        self,
        go2w: Go2WState,
        piper: PiperState,
        commands_xyz: np.ndarray,
        ee_goal_local: np.ndarray,
    ) -> np.ndarray:
        base_euler = euler_from_quat_wxyz(go2w.base_quat_wxyz)
        joint_pos = np.concatenate([go2w.joint_pos, piper.joint_pos], axis=0).astype(np.float32)
        joint_vel = np.concatenate([go2w.joint_vel, piper.joint_vel], axis=0).astype(np.float32)
        joint_err = joint_pos - self.default_joint_pos
        joint_err[WHEEL_INDICES] = 0.0

        obs = np.concatenate(
            [
                base_euler[:2],
                go2w.base_ang_vel * self.cfg.obs_ang_vel_scale,
                joint_err,
                joint_vel * self.cfg.obs_dof_vel_scale,
                self.last_base_actions,
                np.asarray(commands_xyz, dtype=np.float32) * self.cfg.command_scale,
                np.asarray(ee_goal_local, dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        if obs.shape[0] != self.cfg.policy.num_proprio:
            raise RuntimeError(
                f"Bad proprio shape {obs.shape[0]}; expected {self.cfg.policy.num_proprio}. "
                f"Check joint order and arm joints ({ARM_JOINT_NAMES})."
            )
        obs = np.clip(obs, -self.cfg.policy.clip_obs, self.cfg.policy.clip_obs)
        self.history = np.concatenate([self.history[1:], obs[None, :]], axis=0)
        return obs

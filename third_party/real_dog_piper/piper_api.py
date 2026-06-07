from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, PiperConfig
from .types import ArmCommand, PiperState


RAD_TO_MILLI_DEG = 180000.0 / math.pi
MILLI_DEG_TO_RAD = math.pi / 180000.0


class PiperApi:
    """Piper CAN API wrapper with feedback, joint control, and simple position IK."""

    def __init__(self, cfg: PiperConfig, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self._piper = None
        self._fk = None
        self._last_state = PiperState(joint_pos=cfg.default_joint_pos.copy(), stamp=time.time())

    def connect(self) -> None:
        if self.dry_run:
            return
        self._ensure_piper_sdk_path()
        try:
            from piper_sdk import C_PiperForwardKinematics, C_PiperInterface_V2
        except ImportError as exc:
            raise ImportError("piper_sdk is required for real Piper control. Run with --dry-run to test offline.") from exc

        self._piper = C_PiperInterface_V2(self.cfg.can_name)
        self._piper.ConnectPort()
        enable_start = time.time()
        while not self._piper.EnablePiper():
            if time.time() - enable_start > self.cfg.enable_timeout_s:
                raise TimeoutError(f"Piper enable timeout on {self.cfg.can_name}")
            time.sleep(0.01)
        self._piper.MotionCtrl_2(0x01, 0x01, self.cfg.move_speed_percent, 0x00)
        self._piper.GripperCtrl(0, 1000, 0x01, 0)
        self._fk = C_PiperForwardKinematics()

    def read_state(self) -> PiperState:
        if self.dry_run:
            self._last_state.stamp = time.time()
            self._last_state.ee_pos_local = self.forward_position(self._last_state.joint_pos)
            return PiperState(
                joint_pos=self._last_state.joint_pos.copy(),
                joint_vel=self._last_state.joint_vel.copy(),
                ee_pos_local=self._last_state.ee_pos_local.copy(),
                ee_rpy_local=self._last_state.ee_rpy_local.copy(),
                gripper_m=self._last_state.gripper_m,
                stamp=self._last_state.stamp,
            )
        if self._piper is None:
            raise RuntimeError("Piper API is not connected.")
        now = time.time()
        joint_msg = self._piper.GetArmJointMsgs().joint_state
        joint_pos = np.array(
            [
                joint_msg.joint_1,
                joint_msg.joint_2,
                joint_msg.joint_3,
                joint_msg.joint_4,
                joint_msg.joint_5,
                joint_msg.joint_6,
            ],
            dtype=np.float32,
        ) * MILLI_DEG_TO_RAD
        dt = max(1.0e-3, now - self._last_state.stamp)
        joint_vel = (joint_pos - self._last_state.joint_pos) / dt
        end_pose = self._piper.GetArmEndPoseMsgs().end_pose
        gripper = self._piper.GetArmGripperMsgs().gripper_state
        state = PiperState(
            joint_pos=joint_pos,
            joint_vel=joint_vel.astype(np.float32),
            ee_pos_local=np.array([end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis], dtype=np.float32) * 1.0e-6,
            ee_rpy_local=np.array([end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis], dtype=np.float32)
            * MILLI_DEG_TO_RAD,
            gripper_m=float(gripper.grippers_angle) * 1.0e-6,
            stamp=now,
        )
        self._last_state = state
        return state

    def solve_ik_position(self, target_pos_local_m: np.ndarray, seed_joint_pos: np.ndarray) -> np.ndarray:
        q = np.asarray(seed_joint_pos, dtype=np.float32).copy()
        target = np.asarray(target_pos_local_m, dtype=np.float32)
        for _ in range(self.cfg.ik_iterations):
            pos = self.forward_position(q)
            err = target - pos
            if np.linalg.norm(err) < self.cfg.ik_position_tolerance_m:
                break
            jac = self._numeric_position_jacobian(q)
            lhs = jac @ jac.T + (self.cfg.ik_damping**2) * np.eye(3, dtype=np.float32)
            dq = jac.T @ np.linalg.solve(lhs, err)
            q += self.cfg.ik_step_scale * dq.astype(np.float32)
            q = np.clip(q, self.cfg.joint_lower, self.cfg.joint_upper)
        return q.astype(np.float32)

    def send_joint_targets(self, command: ArmCommand) -> None:
        q = np.clip(np.asarray(command.joint_pos, dtype=np.float32), self.cfg.joint_lower, self.cfg.joint_upper)
        if self.dry_run:
            now = time.time()
            dt = max(1.0e-3, now - self._last_state.stamp)
            self._last_state.joint_vel = (q - self._last_state.joint_pos) / dt
            self._last_state.joint_pos = q.copy()
            self._last_state.gripper_m = float(command.gripper_m)
            self._last_state.stamp = now
            return
        if self._piper is None:
            raise RuntimeError("Piper API is not connected.")
        milli_deg = np.round(q * RAD_TO_MILLI_DEG).astype(int)
        self._piper.MotionCtrl_2(0x01, 0x01, self.cfg.move_speed_percent, 0x00)
        self._piper.JointCtrl(*[int(v) for v in milli_deg])
        self._piper.GripperCtrl(int(abs(command.gripper_m) * 1.0e6), 1000, 0x01, 0)

    def stop(self) -> None:
        if self.dry_run or self._piper is None:
            return
        self._piper.MotionCtrl_2(0x00, 0x01, 0, 0x00)

    def forward_position(self, q: np.ndarray) -> np.ndarray:
        if self._fk is None:
            self._ensure_piper_sdk_path()
            try:
                from piper_sdk import C_PiperForwardKinematics

                self._fk = C_PiperForwardKinematics()
            except ImportError:
                return np.zeros(3, dtype=np.float32)
        fk = self._fk.CalFK([float(v) for v in q])[-1]
        return np.array(fk[:3], dtype=np.float32) * 1.0e-3

    def _numeric_position_jacobian(self, q: np.ndarray) -> np.ndarray:
        eps = 1.0e-4
        base = self.forward_position(q)
        jac = np.zeros((3, 6), dtype=np.float32)
        for i in range(6):
            q_eps = q.copy()
            q_eps[i] += eps
            jac[:, i] = (self.forward_position(q_eps) - base) / eps
        return jac

    @staticmethod
    def _ensure_piper_sdk_path() -> None:
        sdk_root = PROJECT_ROOT / "piper_sdk"
        if sdk_root.exists() and str(sdk_root) not in sys.path:
            sys.path.insert(0, str(sdk_root))

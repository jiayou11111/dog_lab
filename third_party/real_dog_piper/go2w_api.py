from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Go2WConfig
from .types import Go2WCommand, Go2WState


def _field(obj: Any, name: str) -> Any:
    attr = getattr(obj, name)
    return attr() if callable(attr) else attr


def _assign(obj: Any, name: str, value: Any) -> None:
    attr = getattr(obj, name, None)
    if callable(attr):
        try:
            attr(value)
            return
        except TypeError:
            pass
    setattr(obj, name, value)


def _as_array(values: Any, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        out[i] = float(values[i])
    return out


def _init_channel(channel: Any, *args: Any) -> None:
    if hasattr(channel, "Init"):
        channel.Init(*args)
    elif hasattr(channel, "InitChannel"):
        channel.InitChannel(*args)
    else:
        raise AttributeError(f"Unsupported Unitree channel object: {type(channel)!r}")


@dataclass
class _UnitreeImports:
    ChannelFactoryInitialize: Any
    ChannelPublisher: Any
    ChannelSubscriber: Any
    LowCmd: Any
    LowState: Any
    default_low_cmd: Any | None
    CRC: Any | None


class Go2WApi:
    """Thin Unitree Go2W low-level API wrapper.

    Public joint order is always dog_lab policy order:
    FR hip/thigh/calf/wheel, FL ..., RR ..., RL ...
    """

    def __init__(self, cfg: Go2WConfig, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self._imports: _UnitreeImports | None = None
        self._publisher = None
        self._subscriber = None
        self._low_cmd = None
        self._low_state = None
        self._crc = None
        self._last_dry_state = Go2WState(joint_pos=cfg.default_joint_pos.copy(), stamp=time.time())

    def connect(self) -> None:
        if self.dry_run:
            return
        self._imports = self._load_unitree_sdk()
        try:
            self._imports.ChannelFactoryInitialize(0, self.cfg.network_interface)
        except TypeError:
            self._imports.ChannelFactoryInitialize(self.cfg.network_interface)

        self._low_cmd = self._make_low_cmd()
        self._publisher = self._imports.ChannelPublisher(self.cfg.lowcmd_topic, self._imports.LowCmd)
        _init_channel(self._publisher)
        self._subscriber = self._imports.ChannelSubscriber(self.cfg.lowstate_topic, self._imports.LowState)
        _init_channel(self._subscriber, self._low_state_callback, 1)
        if self._imports.CRC is not None:
            self._crc = self._imports.CRC()

    def read_state(self) -> Go2WState:
        if self.dry_run:
            self._last_dry_state.stamp = time.time()
            return copy.deepcopy(self._last_dry_state)
        if self._low_state is None:
            raise RuntimeError("No Go2W low-state received yet.")
        state = copy.deepcopy(self._low_state)
        imu = _field(state, "imu_state")
        motors = _field(state, "motor_state")
        q = np.zeros(16, dtype=np.float32)
        dq = np.zeros(16, dtype=np.float32)
        tau = np.zeros(16, dtype=np.float32)
        for policy_i, motor_i in enumerate(self.cfg.policy_to_unitree_motor):
            motor = motors[motor_i]
            q[policy_i] = float(_field(motor, "q"))
            dq[policy_i] = float(_field(motor, "dq"))
            tau[policy_i] = float(_field(motor, "tau_est")) if hasattr(motor, "tau_est") else 0.0
        return Go2WState(
            joint_pos=q,
            joint_vel=dq,
            joint_tau=tau,
            base_quat_wxyz=_as_array(_field(imu, "quaternion"), 4),
            base_ang_vel=_as_array(_field(imu, "gyroscope"), 3),
            stamp=time.time(),
        )

    def build_command_from_actions(self, actions: np.ndarray) -> Go2WCommand:
        actions = np.asarray(actions[:16], dtype=np.float32)
        q = self.cfg.default_joint_pos.copy()
        dq = np.zeros(16, dtype=np.float32)
        q += actions * self.cfg.leg_action_scale
        for idx in self.cfg.wheel_indices:
            q[idx] = self.cfg.pos_stop
            dq[idx] = actions[idx] * self.cfg.wheel_action_scale_vel
        return Go2WCommand(q=q, dq=dq, kp=self.cfg.kp.copy(), kd=self.cfg.kd.copy(), tau=np.zeros(16, dtype=np.float32))

    def send_command(self, command: Go2WCommand) -> None:
        if self.dry_run:
            self._last_dry_state.joint_pos[:] = np.where(
                np.isfinite(command.q) & (command.q < 1.0e8),
                command.q,
                self._last_dry_state.joint_pos,
            )
            self._last_dry_state.joint_vel[:] = command.dq
            return
        motors = _field(self._low_cmd, "motor_cmd")
        for policy_i, motor_i in enumerate(self.cfg.policy_to_unitree_motor):
            motor = motors[motor_i]
            _assign(motor, "mode", 0x01)
            _assign(motor, "q", float(command.q[policy_i]))
            _assign(motor, "dq", float(command.dq[policy_i]))
            _assign(motor, "kp", float(command.kp[policy_i]))
            _assign(motor, "kd", float(command.kd[policy_i]))
            _assign(motor, "tau", float(command.tau[policy_i]))
        self._write_low_cmd()

    def stop(self) -> None:
        command = Go2WCommand(
            q=np.full(16, self.cfg.pos_stop, dtype=np.float32),
            dq=np.full(16, self.cfg.vel_stop, dtype=np.float32),
        )
        self.send_command(command)

    def _low_state_callback(self, message: Any) -> None:
        self._low_state = message

    def _make_low_cmd(self) -> Any:
        assert self._imports is not None
        cmd = self._imports.default_low_cmd() if self._imports.default_low_cmd else self._imports.LowCmd()
        try:
            head = _field(cmd, "head")
            head[0], head[1] = 0xFE, 0xEF
        except Exception:
            pass
        for name, value in (("level_flag", 0xFF), ("gpio", 0)):
            if hasattr(cmd, name):
                _assign(cmd, name, value)
        motors = _field(cmd, "motor_cmd")
        for motor in motors:
            _assign(motor, "mode", 0x01)
            _assign(motor, "q", self.cfg.pos_stop)
            _assign(motor, "dq", self.cfg.vel_stop)
            _assign(motor, "kp", 0.0)
            _assign(motor, "kd", 0.0)
            _assign(motor, "tau", 0.0)
        return cmd

    def _write_low_cmd(self) -> None:
        if self._crc is not None and hasattr(self._low_cmd, "crc"):
            try:
                _assign(self._low_cmd, "crc", self._crc.Crc(self._low_cmd))
            except Exception:
                pass
        self._publisher.Write(self._low_cmd)

    @staticmethod
    def _load_unitree_sdk() -> _UnitreeImports:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        except ImportError as exc:
            raise ImportError(
                "unitree_sdk2py is required for real Go2W control. "
                "Install Unitree SDK2 Python bindings or run with --dry-run."
            ) from exc

        default_low_cmd = None
        try:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

            default_low_cmd = unitree_go_msg_dds__LowCmd_
        except Exception:
            pass

        crc_cls = None
        try:
            from unitree_sdk2py.utils.crc import CRC

            crc_cls = CRC
        except Exception:
            pass

        return _UnitreeImports(ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber, LowCmd_, LowState_, default_low_cmd, crc_cls)

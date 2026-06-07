from __future__ import annotations

import numpy as np

from .config import EEGoalConfig
from .math_utils import sphere_to_cart


class EEGoalSampler:
    """Loco-Manipulation style local end-effector goal generator."""

    def __init__(self, cfg: EEGoalConfig, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.traj_steps = max(1, int(cfg.traj_time / cfg.step_dt))
        self.total_steps = max(1, int((cfg.traj_time + cfg.hold_time) / cfg.step_dt))
        self.start_sphere = np.array(cfg.init_pos_start, dtype=np.float32)
        self.goal_sphere = np.array(cfg.init_pos_end, dtype=np.float32)
        self.curr_sphere = self.start_sphere.copy()
        self.timer = 0

    def reset(self) -> None:
        self.start_sphere[:] = np.array(self.cfg.init_pos_start, dtype=np.float32)
        self.goal_sphere[:] = np.array(self.cfg.init_pos_end, dtype=np.float32)
        self.curr_sphere[:] = self.start_sphere
        self.timer = 0

    def step(self) -> np.ndarray:
        t = np.clip(self.timer / self.traj_steps, 0.0, 1.0)
        self.curr_sphere = (1.0 - t) * self.start_sphere + t * self.goal_sphere
        self.timer += 1
        if self.timer > self.total_steps:
            self._resample()
        return sphere_to_cart(self.curr_sphere)

    def _resample(self) -> None:
        self.start_sphere[:] = self.goal_sphere
        low = np.array([self.cfg.pos_l[0], self.cfg.pos_p[0], self.cfg.pos_y[0]], dtype=np.float32)
        high = np.array([self.cfg.pos_l[1], self.cfg.pos_p[1], self.cfg.pos_y[1]], dtype=np.float32)
        self.goal_sphere[:] = self.rng.uniform(low, high).astype(np.float32)
        self.timer = 0

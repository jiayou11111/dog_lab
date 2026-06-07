from __future__ import annotations

import math

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


def euler_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def sphere_to_cart(sphere: np.ndarray) -> np.ndarray:
    length, pitch, yaw = sphere
    radius_xy = length * math.cos(pitch)
    return np.array(
        [radius_xy * math.cos(yaw), radius_xy * math.sin(yaw), length * math.sin(pitch)],
        dtype=np.float32,
    )

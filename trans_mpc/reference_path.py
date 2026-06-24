from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PathConfig, SimulationConfig


@dataclass(frozen=True)
class ReferencePath:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    kappa: np.ndarray
    v_ref: np.ndarray


def generate_double_lane_change(path_cfg: PathConfig, sim_cfg: SimulationConfig, speed: float | None = None) -> ReferencePath:
    x = np.arange(0.0, path_cfg.length + path_cfg.sample_step, path_cfg.sample_step)
    s1 = 0.5 * (1.0 + np.tanh((x - path_cfg.first_shift_x) / path_cfg.transition))
    s2 = 0.5 * (1.0 + np.tanh((x - path_cfg.return_shift_x) / path_cfg.transition))
    y = path_cfg.lane_width * (s1 - s2)

    dy = np.gradient(y, x)
    ddy = np.gradient(dy, x)
    yaw = np.arctan2(dy, np.ones_like(dy))
    kappa = ddy / np.maximum((1.0 + dy**2) ** 1.5, 1.0e-6)
    base_speed = sim_cfg.default_speed if speed is None else speed
    v_ref = np.full_like(x, base_speed, dtype=float)
    kappa_abs = np.abs(kappa)
    active = kappa_abs > 1.0e-6
    if np.any(active):
        speed_cap = np.sqrt(path_cfg.curve_speed_ay_limit / np.maximum(kappa_abs, 1.0e-6))
        v_ref = np.minimum(v_ref, speed_cap)
    return ReferencePath(x=x, y=y, yaw=yaw, kappa=kappa, v_ref=v_ref)


def nearest_path_index(path: ReferencePath, x: float, y: float, start_index: int = 0) -> int:
    start_index = max(0, min(start_index, len(path.x) - 1))
    lookahead_end = min(len(path.x), start_index + 80)
    dx = path.x[start_index:lookahead_end] - x
    dy = path.y[start_index:lookahead_end] - y
    return int(start_index + np.argmin(dx * dx + dy * dy))

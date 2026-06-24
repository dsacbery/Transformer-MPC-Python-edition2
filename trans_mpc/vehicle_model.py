from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import SimulationConfig, VehicleConfig


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    beta: float = 0.0
    ay: float = 0.0
    delta: float = 0.0
    ax: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.yaw, self.vx, self.vy, self.yaw_rate, self.beta, self.ay, self.delta, self.ax],
            dtype=float,
        )


class DynamicBicycleModel:
    def __init__(self, vehicle: VehicleConfig, sim: SimulationConfig):
        self.vehicle = vehicle
        self.sim = sim

    def step(self, state: VehicleState, delta: float, ax: float, mu: float) -> VehicleState:
        cfg = self.vehicle
        dt = self.sim.dt
        vx = max(abs(state.vx), cfg.min_vx)
        delta = float(np.clip(delta, -cfg.max_steer, cfg.max_steer))
        ax = float(np.clip(ax, -cfg.max_ax, cfg.max_ax))
        mu = float(np.clip(mu, 0.05, 1.2))

        alpha_f = delta - math.atan2(state.vy + cfg.lf * state.yaw_rate, vx)
        alpha_r = -math.atan2(state.vy - cfg.lr * state.yaw_rate, vx)

        f_yf = cfg.cf * alpha_f
        f_yr = cfg.cr * alpha_r

        wheelbase = cfg.lf + cfg.lr
        f_zf = cfg.mass * cfg.g * cfg.lr / wheelbase
        f_zr = cfg.mass * cfg.g * cfg.lf / wheelbase
        f_yf = float(np.clip(f_yf, -mu * f_zf, mu * f_zf))
        f_yr = float(np.clip(f_yr, -mu * f_zr, mu * f_zr))

        vy_dot = (f_yf + f_yr) / cfg.mass - state.vx * state.yaw_rate
        yaw_rate_dot = (cfg.lf * f_yf - cfg.lr * f_yr) / cfg.iz
        ay = (f_yf + f_yr) / cfg.mass

        x_dot = state.vx * math.cos(state.yaw) - state.vy * math.sin(state.yaw)
        y_dot = state.vx * math.sin(state.yaw) + state.vy * math.cos(state.yaw)

        next_vx = max(cfg.min_vx, state.vx + ax * dt)
        next_vy = state.vy + vy_dot * dt
        next_yaw_rate = state.yaw_rate + yaw_rate_dot * dt
        next_yaw = _wrap_angle(state.yaw + next_yaw_rate * dt)

        beta = math.atan2(next_vy, max(abs(next_vx), cfg.min_vx))
        return VehicleState(
            x=state.x + x_dot * dt,
            y=state.y + y_dot * dt,
            yaw=next_yaw,
            vx=next_vx,
            vy=next_vy,
            yaw_rate=next_yaw_rate,
            beta=beta,
            ay=ay,
            delta=delta,
            ax=ax,
        )


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


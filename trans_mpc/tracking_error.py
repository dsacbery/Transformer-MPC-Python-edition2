from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .reference_path import ReferencePath, nearest_path_index
from .vehicle_model import VehicleState


@dataclass(frozen=True)
class TrackingError:
    index: int
    e_y: float
    e_psi: float
    e_y_rate: float
    e_psi_rate: float
    kappa_ref: float
    v_ref: float
    yaw_ref: float


class ErrorTracker:
    def __init__(self, dt: float):
        self.dt = dt
        self.last_index = 0
        self.last_e_y = 0.0
        self.last_e_psi = 0.0

    def compute(self, state: VehicleState, path: ReferencePath) -> TrackingError:
        idx = nearest_path_index(path, state.x, state.y, self.last_index)
        ref_yaw = float(path.yaw[idx])
        dx = state.x - float(path.x[idx])
        dy = state.y - float(path.y[idx])
        e_y = -math.sin(ref_yaw) * dx + math.cos(ref_yaw) * dy
        e_psi = wrap_angle(state.yaw - ref_yaw)
        e_y_rate = (e_y - self.last_e_y) / self.dt
        e_psi_rate = wrap_angle(e_psi - self.last_e_psi) / self.dt
        self.last_index = idx
        self.last_e_y = e_y
        self.last_e_psi = e_psi
        return TrackingError(
            index=idx,
            e_y=e_y,
            e_psi=e_psi,
            e_y_rate=e_y_rate,
            e_psi_rate=e_psi_rate,
            kappa_ref=float(path.kappa[idx]),
            v_ref=float(path.v_ref[idx]),
            yaw_ref=ref_yaw,
        )


def build_feature_vector(state: VehicleState, error: TrackingError, delta_rate: float) -> np.ndarray:
    return np.array(
        [
            state.vx,
            state.vy,
            state.yaw_rate,
            state.ay,
            state.beta,
            state.delta,
            delta_rate,
            state.ax,
            error.e_y,
            error.e_psi,
            error.e_y_rate,
            error.e_psi_rate,
            error.kappa_ref,
            error.v_ref,
        ],
        dtype=float,
    )


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


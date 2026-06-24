from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MPCConfig


@dataclass(frozen=True)
class RiskOutput:
    r_low: float
    r_ey: float
    r_stab: float
    k_v: float


@dataclass(frozen=True)
class MPCParameters:
    weights: dict[str, float]
    constraints: dict[str, float]
    v_ref: float
    risk: RiskOutput


class RiskMapper:
    def __init__(self, config: MPCConfig):
        self.config = config
        self._filtered = RiskOutput(0.0, 0.0, 0.0, 1.0)

    def reset(self) -> None:
        self._filtered = RiskOutput(0.0, 0.0, 0.0, 1.0)

    def map(self, risk: RiskOutput, v_ref: float) -> MPCParameters:
        clipped = _clip_risk(risk, self.config.k_v_min)
        filtered = self._filter(clipped)
        self._filtered = filtered

        c = self.config
        weights = {
            "q_y": _sat(c.q_y * (1.0 + 1.2 * filtered.r_ey), c.q_y, c.q_y_max),
            "q_psi": _sat(c.q_psi * (1.0 + 1.0 * filtered.r_ey), c.q_psi, c.q_psi_max),
            "q_beta": _sat(c.q_beta * (1.0 + 2.4 * filtered.r_stab), c.q_beta, c.q_beta_max),
            "q_r": _sat(c.q_r * (1.0 + 2.0 * filtered.r_stab), c.q_r, c.q_r_max),
            "r_delta": _sat(c.r_delta * (1.0 + 1.2 * filtered.r_stab), c.r_delta, c.r_delta_max),
            "r_d_delta": _sat(c.r_d_delta * (1.0 + 2.0 * filtered.r_stab), c.r_d_delta, c.r_d_delta_max),
        }
        constraints = {
            "beta_max": _sat(c.beta_max * (1.0 - c.c_beta * filtered.r_stab), c.beta_min, c.beta_max),
            "yaw_rate_max": _sat(c.yaw_rate_max * (1.0 - c.c_yaw_rate * filtered.r_stab), c.yaw_rate_min, c.yaw_rate_max),
            "ay_max": _sat(c.ay_max * (1.0 - c.c_ay * filtered.r_stab), c.ay_min, c.ay_max),
            "delta_max": c.delta_max,
            "delta_rate_max": _sat(
                c.delta_rate_max * (1.0 - 0.45 * filtered.r_stab),
                c.delta_rate_min,
                c.delta_rate_max,
            ),
        }
        speed_scale = min(filtered.k_v, 1.0 - c.c_v * filtered.r_low)
        v_adapt = float(v_ref) * _sat(speed_scale, c.k_v_min, 1.0)
        return MPCParameters(weights=weights, constraints=constraints, v_ref=v_adapt, risk=filtered)

    def _filter(self, risk: RiskOutput) -> RiskOutput:
        prev = self._filtered
        return RiskOutput(
            r_low=_smooth(prev.r_low, risk.r_low, self.config),
            r_ey=_smooth(prev.r_ey, risk.r_ey, self.config),
            r_stab=_smooth(prev.r_stab, risk.r_stab, self.config),
            k_v=risk.k_v,
        )


def _clip_risk(risk: RiskOutput, k_v_min: float) -> RiskOutput:
    values = np.array([risk.r_low, risk.r_ey, risk.r_stab, risk.k_v], dtype=float)
    if not np.all(np.isfinite(values)):
        return RiskOutput(0.0, 0.0, 0.0, 1.0)
    return RiskOutput(
        r_low=float(np.clip(risk.r_low, 0.0, 1.0)),
        r_ey=float(np.clip(risk.r_ey, 0.0, 1.0)),
        r_stab=float(np.clip(risk.r_stab, 0.0, 1.0)),
        k_v=float(np.clip(risk.k_v, k_v_min, 1.0)),
    )


def _smooth(previous: float, current: float, config: MPCConfig) -> float:
    rho = config.rho_up if current > previous else config.rho_down
    return float(rho * previous + (1.0 - rho) * current)


def _sat(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


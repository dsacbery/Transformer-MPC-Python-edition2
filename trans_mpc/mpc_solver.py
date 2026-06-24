from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .config import MPCConfig, SimulationConfig, VehicleConfig
from .risk_mapper import MPCParameters


@dataclass(frozen=True)
class MPCResult:
    delta: float
    feasible: bool
    status: str


def fixed_mpc_parameters(config: MPCConfig, v_ref: float) -> MPCParameters:
    from .risk_mapper import RiskOutput

    weights = {
        "q_y": config.q_y,
        "q_psi": config.q_psi,
        "q_beta": config.q_beta,
        "q_r": config.q_r,
        "r_delta": config.r_delta,
        "r_d_delta": config.r_d_delta,
    }
    constraints = {
        "beta_max": config.beta_max,
        "yaw_rate_max": config.yaw_rate_max,
        "ay_max": config.ay_max,
        "delta_max": config.delta_max,
        "delta_rate_max": config.delta_rate_max,
    }
    return MPCParameters(weights=weights, constraints=constraints, v_ref=v_ref, risk=RiskOutput(0.0, 0.0, 0.0, 1.0))


def solve_ltv_mpc(
    x0: np.ndarray,
    v_ref: float,
    kappa_seq: np.ndarray,
    previous_delta: float,
    params: MPCParameters,
    vehicle: VehicleConfig,
    sim: SimulationConfig,
    config: MPCConfig,
) -> MPCResult:
    horizon = min(config.horizon, len(kappa_seq))
    if horizon <= 1:
        return MPCResult(delta=float(np.clip(previous_delta, -config.delta_max, config.delta_max)), feasible=False, status="short_horizon")

    v = max(float(v_ref), vehicle.min_vx)
    a_d, b_d, e_d = _discrete_lateral_model(v, vehicle, sim.dt)
    x = cp.Variable((4, horizon + 1))
    u = cp.Variable(horizon)
    cost = 0.0
    constraints = [x[:, 0] == x0]

    w = params.weights
    c = params.constraints
    for i in range(horizon):
        kappa = float(kappa_seq[i])
        r_ref = params.v_ref * kappa
        beta = x[2, i + 1] / v
        cost += w["q_y"] * cp.square(x[0, i + 1])
        cost += w["q_psi"] * cp.square(x[1, i + 1])
        cost += w["q_beta"] * cp.square(beta)
        cost += w["q_r"] * cp.square(x[3, i + 1] - r_ref)
        cost += w["r_delta"] * cp.square(u[i])
        if i == 0:
            cost += w["r_d_delta"] * cp.square(u[i] - previous_delta)
            constraints.append(cp.abs(u[i] - previous_delta) <= c["delta_rate_max"] * sim.dt)
        else:
            cost += w["r_d_delta"] * cp.square(u[i] - u[i - 1])
            constraints.append(cp.abs(u[i] - u[i - 1]) <= c["delta_rate_max"] * sim.dt)
        constraints.extend(
            [
                x[:, i + 1] == a_d @ x[:, i] + b_d.flatten() * u[i] + e_d.flatten() * kappa,
                cp.abs(u[i]) <= c["delta_max"],
                cp.abs(beta) <= c["beta_max"],
                cp.abs(x[3, i + 1]) <= c["yaw_rate_max"],
            ]
        )

    problem = cp.Problem(cp.Minimize(cost), constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False, warm_start=True, max_iter=4000)
    except Exception as exc:  # pragma: no cover - solver-specific safety net
        return MPCResult(delta=float(np.clip(previous_delta, -config.delta_max, config.delta_max)), feasible=False, status=f"exception:{exc}")

    if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
        return MPCResult(delta=float(np.clip(u.value[0], -c["delta_max"], c["delta_max"])), feasible=True, status=str(problem.status))
    return MPCResult(delta=float(np.clip(previous_delta, -config.delta_max, config.delta_max)), feasible=False, status=str(problem.status))


def _discrete_lateral_model(vx: float, vehicle: VehicleConfig, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = vehicle.mass
    iz = vehicle.iz
    lf = vehicle.lf
    lr = vehicle.lr
    cf = vehicle.cf
    cr = vehicle.cr
    v = max(vx, vehicle.min_vx)
    a_c = np.array(
        [
            [0.0, v, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -(cf + cr) / (m * v), ((-lf * cf + lr * cr) / (m * v) - v)],
            [0.0, 0.0, (-lf * cf + lr * cr) / (iz * v), -(lf * lf * cf + lr * lr * cr) / (iz * v)],
        ],
        dtype=float,
    )
    b_c = np.array([[0.0], [0.0], [cf / m], [lf * cf / iz]], dtype=float)
    e_c = np.array([[0.0], [-v], [0.0], [0.0]], dtype=float)
    return np.eye(4) + dt * a_c, dt * b_c, dt * e_c


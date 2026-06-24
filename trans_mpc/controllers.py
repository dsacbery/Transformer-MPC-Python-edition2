from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import MPCConfig, SimulationConfig, VehicleConfig
from .mpc_solver import fixed_mpc_parameters, solve_ltv_mpc
from .reference_path import ReferencePath
from .risk_mapper import MPCParameters, RiskMapper, RiskOutput
from .tracking_error import TrackingError
from .vehicle_model import VehicleState


@dataclass(frozen=True)
class ControlContext:
    state: VehicleState
    error: TrackingError
    path: ReferencePath
    feature_history: np.ndarray
    previous_delta: float
    vehicle_config: VehicleConfig
    sim_config: SimulationConfig
    mpc_config: MPCConfig


@dataclass(frozen=True)
class ControlCommand:
    delta: float
    ax: float = 0.0
    risk: RiskOutput = RiskOutput(0.0, 0.0, 0.0, 1.0)
    mpc_status: str = "not_used"
    mpc_feasible: bool = True


class Controller(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def control(self, context: ControlContext) -> ControlCommand:
        ...


class PIDController:
    name = "PID"

    def __init__(self, kp_y: float = 0.22, kp_psi: float = 1.4, kd_y: float = 0.015):
        self.kp_y = kp_y
        self.kp_psi = kp_psi
        self.kd_y = kd_y

    def reset(self) -> None:
        return None

    def control(self, context: ControlContext) -> ControlCommand:
        e = context.error
        delta = -(self.kp_y * e.e_y + self.kp_psi * e.e_psi + self.kd_y * e.e_y_rate)
        delta = float(np.clip(delta, -context.vehicle_config.max_steer, context.vehicle_config.max_steer))
        return ControlCommand(delta=delta)


class FixedMPCController:
    name = "Fixed MPC"

    def reset(self) -> None:
        return None

    def control(self, context: ControlContext) -> ControlCommand:
        params = fixed_mpc_parameters(context.mpc_config, context.error.v_ref)
        result = _solve_from_context(context, params)
        ax = speed_tracking_accel(context.state.vx, params.v_ref, context.vehicle_config)
        return ControlCommand(delta=result.delta, ax=ax, risk=params.risk, mpc_status=result.status, mpc_feasible=result.feasible)


class RuleRiskMPCController:
    name = "Rule-risk MPC"

    def __init__(self, mapper: RiskMapper | None = None):
        self.mapper = mapper

    def reset(self) -> None:
        if self.mapper is not None:
            self.mapper.reset()

    def control(self, context: ControlContext) -> ControlCommand:
        mapper = self.mapper or RiskMapper(context.mpc_config)
        risk = rule_risk_from_context(context)
        params = mapper.map(risk, context.error.v_ref)
        result = _solve_from_context(context, params)
        ax = speed_tracking_accel(context.state.vx, params.v_ref, context.vehicle_config)
        return ControlCommand(delta=result.delta, ax=ax, risk=params.risk, mpc_status=result.status, mpc_feasible=result.feasible)


class TransformerMPCController:
    name = "Transformer-MPC"

    def __init__(self, model, standardizer, history_len: int, mapper: RiskMapper | None = None):
        self.model = model
        self.standardizer = standardizer
        self.history_len = history_len
        self.mapper = mapper

    def reset(self) -> None:
        if self.mapper is not None:
            self.mapper.reset()

    def control(self, context: ControlContext) -> ControlCommand:
        risk = rule_risk_from_context(context)
        if self.model is not None and self.standardizer is not None and len(context.feature_history) >= self.history_len:
            try:
                import torch

                window = context.feature_history[-self.history_len :]
                window = self.standardizer.transform(window)
                tensor = torch.as_tensor(window[None, :, :], dtype=torch.float32)
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(tensor).cpu().numpy()[0]
                risk = RiskOutput(float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]))
            except Exception:
                risk = rule_risk_from_context(context)

        mapper = self.mapper or RiskMapper(context.mpc_config)
        params = mapper.map(risk, context.error.v_ref)
        result = _solve_from_context(context, params)
        ax = speed_tracking_accel(context.state.vx, params.v_ref, context.vehicle_config)
        return ControlCommand(delta=result.delta, ax=ax, risk=params.risk, mpc_status=result.status, mpc_feasible=result.feasible)


def rule_risk_from_context(context: ControlContext) -> RiskOutput:
    state = context.state
    e = context.error
    r_ey = float(np.clip(abs(e.e_y) / 1.2 + abs(e.e_y_rate) / 8.0, 0.0, 1.0))
    r_stab = float(np.clip(max(abs(state.beta) / 0.10, abs(state.yaw_rate) / 0.75, abs(state.ay) / 6.0), 0.0, 1.0))
    r_low = float(np.clip(0.5 * r_ey + 0.5 * r_stab, 0.0, 1.0))
    k_v = float(np.clip(1.0 - 0.35 * r_low, context.mpc_config.k_v_min, 1.0))
    return RiskOutput(r_low=r_low, r_ey=r_ey, r_stab=r_stab, k_v=k_v)


def speed_tracking_accel(vx: float, v_ref: float, vehicle_config: VehicleConfig) -> float:
    ax = 0.8 * (float(v_ref) - float(vx))
    return float(np.clip(ax, -vehicle_config.max_ax, vehicle_config.max_ax))


def _solve_from_context(context: ControlContext, params: MPCParameters):
    idx = context.error.index
    horizon = context.mpc_config.horizon
    indices = np.clip(np.arange(idx, idx + horizon), 0, len(context.path.kappa) - 1)
    kappa_seq = context.path.kappa[indices]
    x0 = np.array([context.error.e_y, context.error.e_psi, context.state.vy, context.state.yaw_rate], dtype=float)
    return solve_ltv_mpc(
        x0=x0,
        v_ref=max(params.v_ref, context.vehicle_config.min_vx),
        kappa_seq=kappa_seq,
        previous_delta=context.previous_delta,
        params=params,
        vehicle=context.vehicle_config,
        sim=context.sim_config,
        config=context.mpc_config,
    )

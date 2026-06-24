from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import MPCConfig, SimulationConfig, VehicleConfig
from .controllers import ControlContext, Controller
from .reference_path import ReferencePath
from .tracking_error import ErrorTracker, build_feature_vector
from .vehicle_model import DynamicBicycleModel, VehicleState


def run_simulation(
    controller: Controller,
    scenario,
    path: ReferencePath,
    sim_config: SimulationConfig,
    vehicle_config: VehicleConfig,
    mpc_config: MPCConfig | None = None,
) -> dict[str, np.ndarray | bool | str]:
    mpc_config = mpc_config or MPCConfig()
    if hasattr(controller, "reset"):
        controller.reset()

    plant = DynamicBicycleModel(vehicle_config, sim_config)
    tracker = ErrorTracker(sim_config.dt)
    state = VehicleState(y=scenario.initial_y, yaw=scenario.initial_yaw, vx=scenario.speed)
    previous_delta = 0.0
    previous_delta_rate = 0.0
    feature_history: list[np.ndarray] = []
    logs: dict[str, list] = defaultdict(list)

    rng = np.random.default_rng(sim_config.random_seed)
    steps = int(sim_config.max_time / sim_config.dt)
    completed = False

    for step in range(steps):
        error = tracker.compute(state, path)
        noisy_state = _noisy_state(state, scenario.noise_std, rng)
        feature = build_feature_vector(noisy_state, error, previous_delta_rate)
        feature_history.append(feature)
        context = ControlContext(
            state=noisy_state,
            error=error,
            path=path,
            feature_history=np.asarray(feature_history, dtype=float),
            previous_delta=previous_delta,
            vehicle_config=vehicle_config,
            sim_config=sim_config,
            mpc_config=mpc_config,
        )
        command = controller.control(context)
        delta_rate = (command.delta - previous_delta) / sim_config.dt
        mu = scenario.mu_at(state.x)
        ax = command.ax
        if getattr(scenario, "disturbed", False):
            ax += float(rng.normal(0.0, 0.15))
        _append_logs(logs, step * sim_config.dt, state, error, feature, command, delta_rate, mu)
        state = plant.step(state, command.delta, ax, mu)
        previous_delta = command.delta
        previous_delta_rate = delta_rate
        if error.index >= len(path.x) - 3:
            completed = True
            break

    arrays = {key: np.asarray(value) for key, value in logs.items()}
    arrays["features"] = np.asarray(feature_history[: len(arrays.get("time", []))], dtype=float)
    arrays["completed"] = completed
    arrays["controller"] = getattr(controller, "name", controller.__class__.__name__)
    arrays["scenario"] = scenario.name
    return arrays


def _noisy_state(state: VehicleState, noise_std: float, rng: np.random.Generator) -> VehicleState:
    if noise_std <= 0.0:
        return state
    return VehicleState(
        x=state.x,
        y=state.y + float(rng.normal(0.0, noise_std)),
        yaw=state.yaw + float(rng.normal(0.0, noise_std * 0.2)),
        vx=state.vx,
        vy=state.vy + float(rng.normal(0.0, noise_std)),
        yaw_rate=state.yaw_rate,
        beta=state.beta,
        ay=state.ay,
        delta=state.delta,
        ax=state.ax,
    )


def _append_logs(logs, time, state, error, feature, command, delta_rate, mu) -> None:
    logs["time"].append(time)
    logs["x"].append(state.x)
    logs["y"].append(state.y)
    logs["yaw"].append(state.yaw)
    logs["vx"].append(state.vx)
    logs["vy"].append(state.vy)
    logs["yaw_rate"].append(state.yaw_rate)
    logs["beta"].append(state.beta)
    logs["ay"].append(state.ay)
    logs["delta"].append(command.delta)
    logs["delta_rate"].append(delta_rate)
    logs["ax"].append(command.ax)
    logs["e_y"].append(error.e_y)
    logs["e_psi"].append(error.e_psi)
    logs["e_y_rate"].append(error.e_y_rate)
    logs["e_psi_rate"].append(error.e_psi_rate)
    logs["kappa_ref"].append(error.kappa_ref)
    logs["v_ref"].append(error.v_ref)
    logs["mu"].append(mu)
    logs["r_low"].append(command.risk.r_low)
    logs["r_ey"].append(command.risk.r_ey)
    logs["r_stab"].append(command.risk.r_stab)
    logs["k_v"].append(command.risk.k_v)
    logs["mpc_feasible"].append(bool(command.mpc_feasible))
    logs["feature_0"].append(feature[0])

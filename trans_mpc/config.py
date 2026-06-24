from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleConfig:
    mass: float = 1500.0
    iz: float = 2250.0
    lf: float = 1.2
    lr: float = 1.6
    cf: float = 80000.0
    cr: float = 80000.0
    g: float = 9.81
    max_steer: float = 0.50
    max_ax: float = 2.0
    min_vx: float = 0.5


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.05
    max_time: float = 12.0
    history_len: int = 16
    prediction_horizon_steps: int = 20
    default_speed: float = 12.0
    random_seed: int = 7


@dataclass(frozen=True)
class PathConfig:
    length: float = 120.0
    lane_width: float = 3.8
    sample_step: float = 0.5
    first_shift_x: float = 28.0
    return_shift_x: float = 62.0
    transition: float = 8.0
    curve_speed_ay_limit: float = 3.5


@dataclass
class MPCConfig:
    horizon: int = 12
    q_y: float = 8.0
    q_psi: float = 4.0
    q_beta: float = 10.0
    q_r: float = 2.0
    r_delta: float = 0.3
    r_d_delta: float = 25.0
    q_y_max: float = 25.0
    q_psi_max: float = 15.0
    q_beta_max: float = 45.0
    q_r_max: float = 12.0
    r_delta_max: float = 2.5
    r_d_delta_max: float = 100.0
    beta_max: float = 0.12
    beta_min: float = 0.045
    yaw_rate_max: float = 0.75
    yaw_rate_min: float = 0.25
    ay_max: float = 6.0
    ay_min: float = 2.5
    delta_max: float = 0.50
    delta_rate_max: float = 0.45
    delta_rate_min: float = 0.18
    k_v_min: float = 0.45
    rho_up: float = 0.45
    rho_down: float = 0.85
    c_v: float = 0.35
    c_beta: float = 0.45
    c_yaw_rate: float = 0.4
    c_ay: float = 0.35


@dataclass(frozen=True)
class TransformerConfig:
    input_dim: int = 14
    d_model: int = 48
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.05
    k_v_min: float = 0.45


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 16
    quick_epochs: int = 3
    learning_rate: float = 1.0e-3
    val_fraction: float = 0.2

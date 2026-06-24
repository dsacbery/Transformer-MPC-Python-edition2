# Transformer-MPC Python Prototype Design

## Objective

Build a modular Python prototype for a trainable Transformer-MPC vehicle trajectory tracking controller. The first target is to validate the control concept in a Python simulation before moving to CarSim/Simulink.

The prototype uses a dynamic bicycle vehicle model with simplified low-friction tire saturation, a standard double lane change path, trainable Transformer risk prediction, risk-aware adaptive MPC, and baseline comparisons against PID, fixed MPC, and rule-risk adaptive MPC.

## Project Root

All files are placed under:

```text
/Users/dsacbery/Study/code/TRANS
```

## Scope

Included in the first version:

- Dynamic bicycle plant with simplified low-friction saturation.
- Standard double lane change reference path.
- Multi-scenario closed-loop data generation.
- Supervised risk label generation from future vehicle response.
- Trainable PyTorch Transformer risk predictor.
- Risk mapping layer for MPC weights, stability constraints, and speed correction.
- Four-controller comparison:
  - PID
  - Fixed-parameter MPC
  - Rule-risk adaptive MPC
  - Transformer-MPC
- Extreme scenario tests:
  - High-speed double lane change
  - Friction step change
  - Low friction during lane-change segment
  - Initial lateral offset
  - Sensor noise
- Figures, logs, and metrics tables.

Excluded from the first version:

- CarSim/Simulink integration.
- Real vehicle deployment.
- External road-friction preview.
- Direct Transformer output of steering or acceleration.
- Online estimation of true road friction.

## Module Layout

Use 17 functional Python files, excluding empty `__init__.py` files:

```text
TRANS/
  trans_mpc/
    config.py
    vehicle_model.py
    reference_path.py
    tracking_error.py
    scenario_manager.py
    mpc_solver.py
    controllers.py
    risk_mapper.py
    risk_labels.py
    transformer_model.py
    dataset.py
    simulator.py
    metrics.py
    plotting.py

  scripts/
    generate_dataset.py
    train_transformer.py
    run_experiments.py
```

### Module Responsibilities

`config.py`

Defines dataclass-based configuration for vehicle parameters, simulation step size, MPC weights and constraints, Transformer hyperparameters, training settings, and output paths.

`vehicle_model.py`

Implements the dynamic bicycle model and simplified low-friction tire saturation. The plant uses friction internally to generate realistic low-adhesion response, but the Transformer does not receive true friction as an online input.

`reference_path.py`

Generates the standard double lane change path and computes reference yaw, curvature, and reference speed.

`tracking_error.py`

Computes lateral error, heading error, error rates, beta, lateral acceleration, and model input features.

`scenario_manager.py`

Defines training and test scenarios, including speed, friction profile, low-friction zone, initial offset, and noise.

`mpc_solver.py`

Builds and solves the linear time-varying MPC QP with OSQP through CVXPY. It supports adaptive weights, adaptive stability bounds, input constraints, and fallback behavior when the solver is infeasible.

`controllers.py`

Provides a unified controller interface for PID, fixed MPC, rule-risk adaptive MPC, and Transformer-MPC.

`risk_mapper.py`

Clips, filters, and maps risk outputs into MPC weights, stability constraints, steering-rate limits, and reference speed correction. If Transformer output is invalid, it falls back to rule risk.

`risk_labels.py`

Creates training labels `r_low`, `r_ey`, `r_stab`, and `k_v` using future-window vehicle response. Labels are based on tracking error growth, stability margin, control saturation, speed, and curvature. True friction is not used as a target.

`transformer_model.py`

Defines the PyTorch Transformer encoder and prediction heads for `r_low`, `r_ey`, `r_stab`, and `k_v`.

`dataset.py`

Builds time-window datasets, feature normalization, train/validation split, and PyTorch dataloaders.

`simulator.py`

Runs closed-loop simulation for a selected scenario and controller.

`metrics.py`

Computes lateral error RMSE, maximum lateral error, heading error RMSE, beta/yaw-rate/lateral-acceleration maxima, steering smoothness, solver failures, completion status, and timing.

`plotting.py`

Generates comparison figures for trajectory, lateral error, stability variables, risk prediction, and steering.

Script files:

- `scripts/generate_dataset.py`: runs multi-scenario closed-loop simulations and saves training data.
- `scripts/train_transformer.py`: trains and saves the Transformer model.
- `scripts/run_experiments.py`: runs the four-controller comparison and exports figures/tables.

## Data Flow

### Stage 1: Dataset Generation

```text
scenario sampling
  -> dynamic bicycle plant
  -> PID / fixed MPC / disturbed closed-loop simulation
  -> log vehicle state, control input, tracking error, reference curvature, reference speed
  -> future-window risk label generation
  -> save dataset and scaler
```

### Stage 2: Transformer Training

```text
history window X[k-L+1:k]
  -> Transformer encoder
  -> risk heads
  -> r_low, r_ey, r_stab, k_v
  -> supervised multi-task loss
  -> save best_transformer.pt
```

### Stage 3: Closed-Loop Comparison

```text
test scenario
  -> PID
  -> fixed MPC
  -> rule-risk adaptive MPC
  -> Transformer-MPC
  -> common metrics and plots
```

## Transformer Inputs

The online input window includes:

```text
vx
vy
yaw_rate
ay
beta
delta
delta_rate
ax
e_y
e_psi
e_y_rate
e_psi_rate
kappa_ref
v_ref
```

The online input must not include:

```text
true friction mu
future friction
external road-friction preview
features directly computed from true friction
```

## Transformer Outputs

The model outputs:

```text
r_low, r_ey, r_stab, k_v
```

Meanings:

- `r_low`: overall low-friction and instability risk.
- `r_ey`: future lateral error growth risk.
- `r_stab`: stability risk based on beta, yaw rate, and lateral acceleration margins.
- `k_v`: reference speed scaling coefficient.

## Risk Labeling

Labels are generated from future-window response:

```text
r_ey   <- future max |e_y| against speed-curvature adaptive threshold
r_stab <- future max beta/yaw_rate/ay stability margin
r_low  <- weighted combination of r_ey, r_stab, and control saturation risk
k_v    <- safe speed ratio from risk and curvature
```

All risk labels are clipped to stable numerical ranges. `r_low`, `r_ey`, and `r_stab` are in `[0, 1]`. `k_v` is in `[k_min, 1]`.

## Risk Mapping

Risk outputs pass through:

```text
clip
  -> fast-rise/slow-fall filter
  -> adaptive MPC parameter mapping
```

Mapping effects:

- `r_ey` increases lateral error and heading error weights.
- `r_stab` increases stability penalties and tightens beta, yaw-rate, and lateral-acceleration constraints.
- `r_low` and `k_v` reduce reference speed.
- High stability risk increases steering-rate penalty and may tighten steering-rate bounds.

The mapping layer preserves lower and upper bounds on all MPC weights, constraints, and speed correction values.

## Controllers

PID:

Uses lateral and heading error feedback to produce steering. This provides a simple traditional baseline.

Fixed MPC:

Uses the same LTV-MPC solver with fixed nominal weights and constraints.

Rule-risk Adaptive MPC:

Computes risk from current error and stability variables using explicit rules, then uses the same risk mapper.

Transformer-MPC:

Uses the trained Transformer to predict risk from historical input windows, then uses the same risk mapper and MPC solver.

## Experiments

Baseline scenario:

```text
standard double lane change
medium speed
uniform high friction
```

Extreme scenarios:

```text
high-speed double lane change
friction step change after entering the second lane
low friction during lane-change segment
initial lateral offset
sensor noise
```

## Outputs

Generated outputs:

```text
outputs/
  figures/
    trajectory_comparison.png
    lateral_error_comparison.png
    yaw_beta_stability.png
    risk_prediction.png
    steering_comparison.png
  tables/
    metrics_summary.csv
  logs/
    experiment_log.csv
```

## Verification

Use the existing virtual environment:

```text
/Users/dsacbery/Study/code/.venv/bin/python
```

Verification commands:

```text
python scripts/generate_dataset.py
python scripts/train_transformer.py
python scripts/run_experiments.py
```

Expected checks:

- Generated dataset has valid sample count, input dimension, and bounded labels.
- Training saves `best_transformer.pt` and produces decreasing validation loss over the initial epochs.
- Experiments run all four controllers.
- Figures and metrics tables are generated.
- MPC infeasibility is logged and handled.
- Invalid Transformer output falls back to rule-risk MPC.

## Known Constraints

The first version is a control-concept prototype, not a high-fidelity vehicle simulator. Tire saturation is simplified to make low-friction behavior visible and controllable. The intended next step is to use the same module boundaries to replace the Python plant with CarSim/Simulink or another high-fidelity vehicle model.

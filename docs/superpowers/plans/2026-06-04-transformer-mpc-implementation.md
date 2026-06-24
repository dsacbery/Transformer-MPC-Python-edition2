# Transformer-MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trainable Python Transformer-MPC prototype for double lane change tracking with PID, fixed MPC, rule-risk MPC, and Transformer-MPC comparison.

**Architecture:** The project is a small Python package under `trans_mpc/` with focused modules for vehicle dynamics, reference paths, tracking errors, scenarios, MPC solving, controllers, risk labeling, Transformer training, simulation, metrics, and plotting. Three scripts run dataset generation, training, and experiments. Tests use standard-library `unittest` because `pytest` is not installed in the available virtual environment.

**Tech Stack:** Python 3.14 from `/Users/dsacbery/Study/code/.venv/bin/python`, NumPy, SciPy, Matplotlib, PyTorch, CVXPY, OSQP, scikit-learn, pandas, joblib, unittest.

---

## File Structure

Create:

- `/Users/dsacbery/Study/code/TRANS/trans_mpc/__init__.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/config.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/vehicle_model.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/reference_path.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/tracking_error.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/scenario_manager.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/mpc_solver.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/controllers.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/risk_mapper.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/risk_labels.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/transformer_model.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/dataset.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/simulator.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/metrics.py`
- `/Users/dsacbery/Study/code/TRANS/trans_mpc/plotting.py`
- `/Users/dsacbery/Study/code/TRANS/scripts/generate_dataset.py`
- `/Users/dsacbery/Study/code/TRANS/scripts/train_transformer.py`
- `/Users/dsacbery/Study/code/TRANS/scripts/run_experiments.py`
- `/Users/dsacbery/Study/code/TRANS/tests/test_core.py`
- `/Users/dsacbery/Study/code/TRANS/README.md`

No git commits will be created because `/Users/dsacbery/Study/code` is not a git repository.

## Task 1: Package Skeleton and Core Tests

**Files:**

- Create: `tests/test_core.py`
- Create: `trans_mpc/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_core.py` with tests that import the planned modules and verify the public behavior:

```python
import math
import unittest
import numpy as np


class CoreBehaviorTests(unittest.TestCase):
    def test_double_lane_change_has_curvature_and_lane_shift(self):
        from trans_mpc.config import PathConfig, SimulationConfig
        from trans_mpc.reference_path import generate_double_lane_change

        path = generate_double_lane_change(PathConfig(), SimulationConfig())
        self.assertGreater(len(path.x), 50)
        self.assertGreater(float(np.max(path.y)), 3.0)
        self.assertTrue(np.all(np.isfinite(path.kappa)))

    def test_vehicle_step_keeps_state_finite_under_low_friction(self):
        from trans_mpc.config import VehicleConfig, SimulationConfig
        from trans_mpc.vehicle_model import VehicleState, DynamicBicycleModel

        model = DynamicBicycleModel(VehicleConfig(), SimulationConfig())
        state = VehicleState(vx=12.0)
        for _ in range(20):
            state = model.step(state, delta=0.08, ax=0.0, mu=0.25)
        self.assertTrue(np.all(np.isfinite(state.as_vector())))
        self.assertLess(abs(state.beta), 0.6)

    def test_risk_labels_are_bounded(self):
        from trans_mpc.risk_labels import compute_future_risk_labels

        logs = {
            "e_y": np.array([0.0, 0.1, 0.3, 0.5, 0.8]),
            "beta": np.array([0.0, 0.01, 0.03, 0.04, 0.06]),
            "yaw_rate": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            "ay": np.array([0.0, 0.5, 1.0, 2.0, 3.0]),
            "delta": np.array([0.0, 0.02, 0.04, 0.06, 0.08]),
            "v_ref": np.full(5, 12.0),
            "kappa_ref": np.full(5, 0.03),
        }
        labels = compute_future_risk_labels(logs, horizon=3)
        for key in ("r_low", "r_ey", "r_stab", "k_v"):
            self.assertIn(key, labels)
            self.assertTrue(np.all(np.isfinite(labels[key])))
        self.assertTrue(np.all((labels["r_low"] >= 0.0) & (labels["r_low"] <= 1.0)))
        self.assertTrue(np.all((labels["k_v"] >= 0.45) & (labels["k_v"] <= 1.0)))

    def test_risk_mapper_outputs_safe_parameters(self):
        from trans_mpc.config import MPCConfig
        from trans_mpc.risk_mapper import RiskMapper, RiskOutput

        mapper = RiskMapper(MPCConfig())
        params = mapper.map(RiskOutput(r_low=1.4, r_ey=0.8, r_stab=1.2, k_v=0.2), v_ref=15.0)
        self.assertGreaterEqual(params.v_ref, 15.0 * mapper.config.k_v_min)
        self.assertLessEqual(params.v_ref, 15.0)
        self.assertGreater(params.weights["q_beta"], mapper.config.q_beta)
        self.assertLessEqual(params.constraints["beta_max"], mapper.config.beta_max)

    def test_transformer_forward_shape_and_bounds(self):
        import torch
        from trans_mpc.transformer_model import RiskTransformer

        model = RiskTransformer(input_dim=14, d_model=32, nhead=4, num_layers=1)
        out = model(torch.zeros(3, 8, 14))
        self.assertEqual(tuple(out.shape), (3, 4))
        self.assertTrue(torch.all(out[:, :3] >= 0.0))
        self.assertTrue(torch.all(out[:, :3] <= 1.0))
        self.assertTrue(torch.all(out[:, 3] >= 0.45))
        self.assertTrue(torch.all(out[:, 3] <= 1.0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest tests.test_core -v
```

Expected: tests fail because `trans_mpc` modules do not exist yet.

## Task 2: Core Dynamics, Path, Risk Labels, and Risk Mapping

**Files:**

- Create: `trans_mpc/config.py`
- Create: `trans_mpc/vehicle_model.py`
- Create: `trans_mpc/reference_path.py`
- Create: `trans_mpc/risk_labels.py`
- Create: `trans_mpc/risk_mapper.py`

- [ ] **Step 1: Implement dataclass configs**

Define `VehicleConfig`, `SimulationConfig`, `PathConfig`, `MPCConfig`, `TransformerConfig`, and `TrainingConfig` with bounded defaults for the first prototype.

- [ ] **Step 2: Implement vehicle and path modules**

Implement `VehicleState`, `DynamicBicycleModel.step`, `ReferencePath`, and `generate_double_lane_change`.

- [ ] **Step 3: Implement risk labels and risk mapper**

Implement future-window risk labels and bounded risk-to-MPC parameter mapping.

- [ ] **Step 4: Run tests and verify GREEN for core behavior**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest tests.test_core -v
```

Expected: tests pass except modules that still depend on the Transformer if it is not implemented yet.

## Task 3: Transformer Model and Dataset Utilities

**Files:**

- Create: `trans_mpc/transformer_model.py`
- Create: `trans_mpc/dataset.py`

- [ ] **Step 1: Implement RiskTransformer**

Use a lightweight Transformer encoder with bounded output heads for `r_low`, `r_ey`, `r_stab`, and `k_v`.

- [ ] **Step 2: Implement dataset windowing and standardization**

Create a `WindowedRiskDataset`, feature standardizer helpers, and train/validation split function.

- [ ] **Step 3: Run core tests**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest tests.test_core -v
```

Expected: all core tests pass.

## Task 4: Tracking, Scenarios, MPC, Controllers, and Simulator

**Files:**

- Create: `trans_mpc/tracking_error.py`
- Create: `trans_mpc/scenario_manager.py`
- Create: `trans_mpc/mpc_solver.py`
- Create: `trans_mpc/controllers.py`
- Create: `trans_mpc/simulator.py`

- [ ] **Step 1: Implement tracking error and scenario manager**

Compute closest reference point, lateral/heading error, local curvature, friction profiles, and scenario presets.

- [ ] **Step 2: Implement MPC solver**

Use CVXPY/OSQP for a compact LTV lateral MPC with fixed and adaptive parameter support. Return safe fallback steering if infeasible.

- [ ] **Step 3: Implement controllers**

Provide PID, fixed MPC, rule-risk MPC, and Transformer-MPC classes with the same `control(context)` style.

- [ ] **Step 4: Implement simulator**

Run closed-loop simulation and log all features required for dataset, metrics, and plotting.

- [ ] **Step 5: Run a smoke simulation**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python - <<'PY'
from trans_mpc.config import *
from trans_mpc.reference_path import generate_double_lane_change
from trans_mpc.scenario_manager import make_default_scenarios
from trans_mpc.controllers import PIDController
from trans_mpc.simulator import run_simulation
cfg = SimulationConfig(max_time=4.0)
path = generate_double_lane_change(PathConfig(), cfg)
scenario = make_default_scenarios()[0]
result = run_simulation(PIDController(), scenario, path, cfg, VehicleConfig())
print(len(result["time"]), result["completed"])
assert len(result["time"]) > 5
PY
```

Expected: prints a positive sample count and no traceback.

## Task 5: Metrics, Plotting, and Scripts

**Files:**

- Create: `trans_mpc/metrics.py`
- Create: `trans_mpc/plotting.py`
- Create: `scripts/generate_dataset.py`
- Create: `scripts/train_transformer.py`
- Create: `scripts/run_experiments.py`
- Create: `README.md`

- [ ] **Step 1: Implement metrics and plotting**

Generate metrics summary and comparison plots for trajectory, lateral error, stability, risks, and steering.

- [ ] **Step 2: Implement dataset generation script**

Run multi-scenario simulations, compute labels, window features, save `data/dataset.npz`, and save `data/scaler.pkl`.

- [ ] **Step 3: Implement training script**

Train the Transformer for a small default number of epochs, save `checkpoints/best_transformer.pt`, and save loss curves.

- [ ] **Step 4: Implement experiment script**

Run PID, fixed MPC, rule-risk MPC, and Transformer-MPC on selected scenarios, save figures and metrics tables.

- [ ] **Step 5: Run end-to-end verification**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python scripts/generate_dataset.py --quick
/Users/dsacbery/Study/code/.venv/bin/python scripts/train_transformer.py --quick
/Users/dsacbery/Study/code/.venv/bin/python scripts/run_experiments.py --quick
```

Expected: commands finish without traceback and produce dataset, checkpoint, metrics, and figures.

## Self-Review

Spec coverage:

- Dynamic bicycle model: Task 2 and Task 4.
- Standard double lane change: Task 2.
- Multi-scenario data generation: Task 4 and Task 5.
- Trainable Transformer: Task 3 and Task 5.
- Four-controller comparison: Task 4 and Task 5.
- Risk labels and mapping: Task 2.
- Figures and tables: Task 5.

Placeholder scan:

- The plan contains no `TBD`, `TODO`, or undefined task names.

Type consistency:

- Risk outputs are consistently named `r_low`, `r_ey`, `r_stab`, `k_v`.
- Feature input dimension is consistently 14.
- Scripts use the package modules created in earlier tasks.

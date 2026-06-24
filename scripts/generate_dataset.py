from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trans_mpc.config import MPCConfig, PathConfig, SimulationConfig, VehicleConfig
from trans_mpc.controllers import FixedMPCController, PIDController, RuleRiskMPCController
from trans_mpc.dataset import FeatureStandardizer, build_windowed_arrays
from trans_mpc.reference_path import generate_double_lane_change
from trans_mpc.risk_labels import compute_future_risk_labels
from trans_mpc.risk_mapper import RiskMapper
from trans_mpc.scenario_manager import make_training_scenarios, make_version2_training_scenarios
from trans_mpc.simulator import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a small dataset generation pass.")
    args = parser.parse_args()

    sim_cfg = SimulationConfig(max_time=3.5 if args.quick else 13.0)
    vehicle_cfg = VehicleConfig()
    mpc_cfg = MPCConfig(horizon=8 if args.quick else 12)
    scenarios = make_training_scenarios(quick=True) if args.quick else make_version2_training_scenarios()
    controllers = [PIDController()]
    if not args.quick:
        controllers.extend([FixedMPCController(), RuleRiskMPCController(RiskMapper(mpc_cfg))])

    runs: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    for scenario in scenarios:
        path = generate_double_lane_change(PathConfig(), sim_cfg, speed=scenario.speed)
        for controller in controllers:
            result = run_simulation(controller, scenario, path, sim_cfg, vehicle_cfg, mpc_cfg)
            labels = compute_future_risk_labels(result, horizon=sim_cfg.prediction_horizon_steps)
            runs.append((result["features"], labels))
            print(f"generated {scenario.name} / {controller.name}: {len(result['time'])} steps")

    raw_features = np.vstack([features for features, _labels in runs])
    standardizer = FeatureStandardizer.fit(raw_features)

    windows = []
    targets = []
    for features, labels in runs:
        x, y = build_windowed_arrays(standardizer.transform(features), labels, sim_cfg.history_len)
        if len(x):
            windows.append(x)
            targets.append(y)

    x_all = np.vstack(windows).astype(np.float32)
    y_all = np.vstack(targets).astype(np.float32)

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / "dataset.npz", windows=x_all, targets=y_all)
    joblib.dump(standardizer, data_dir / "scaler.pkl")
    print(f"saved dataset: windows={x_all.shape}, targets={y_all.shape}")


if __name__ == "__main__":
    main()

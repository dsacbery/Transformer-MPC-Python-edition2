from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trans_mpc.config import MPCConfig, PathConfig, SimulationConfig, TransformerConfig, VehicleConfig
from trans_mpc.controllers import FixedMPCController, PIDController, RuleRiskMPCController, TransformerMPCController
from trans_mpc.metrics import save_metrics
from trans_mpc.plotting import save_comparison_plots
from trans_mpc.reference_path import generate_double_lane_change
from trans_mpc.risk_mapper import RiskMapper
from trans_mpc.scenario_manager import make_default_scenarios, make_version2_evaluation_scenarios
from trans_mpc.simulator import run_simulation
from trans_mpc.transformer_model import RiskTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a short experiment pass.")
    args = parser.parse_args()

    sim_cfg = SimulationConfig(max_time=3.5 if args.quick else 13.0)
    vehicle_cfg = VehicleConfig()
    mpc_cfg = MPCConfig(horizon=8 if args.quick else 12)
    model, scaler = _load_model_and_scaler()

    controllers = [
        PIDController(),
        FixedMPCController(),
        RuleRiskMPCController(RiskMapper(mpc_cfg)),
        TransformerMPCController(model, scaler, sim_cfg.history_len, RiskMapper(mpc_cfg)),
    ]
    scenarios = make_default_scenarios()[:2] if args.quick else make_version2_evaluation_scenarios()

    results = []
    for scenario in scenarios:
        path = generate_double_lane_change(PathConfig(), sim_cfg, speed=scenario.speed)
        for controller in controllers:
            result = run_simulation(controller, scenario, path, sim_cfg, vehicle_cfg, mpc_cfg)
            results.append(result)
            print(f"ran {scenario.name} / {controller.name}: completed={result['completed']}")

    output_dir = ROOT / "outputs"
    save_metrics(results, output_dir / "tables" / "metrics_summary.csv")
    _save_experiment_log(results, output_dir / "logs" / "experiment_log.csv")
    save_comparison_plots(results, output_dir / "figures")
    print(f"saved outputs under {output_dir}")


def _load_model_and_scaler():
    ckpt_path = ROOT / "checkpoints" / "best_transformer.pt"
    scaler_path = ROOT / "data" / "scaler.pkl"
    if not ckpt_path.exists() or not scaler_path.exists():
        return None, None
    cfg = TransformerConfig()
    model = RiskTransformer(cfg.input_dim, cfg.d_model, cfg.nhead, cfg.num_layers, cfg.dropout, cfg.k_v_min)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    scaler = joblib.load(scaler_path)
    return model, scaler


def _save_experiment_log(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    keys = [
        "time",
        "x",
        "y",
        "vx",
        "e_y",
        "e_psi",
        "beta",
        "yaw_rate",
        "ay",
        "delta",
        "r_low",
        "r_ey",
        "r_stab",
        "k_v",
        "mu",
        "mpc_feasible",
    ]
    for result in results:
        n = len(result["time"])
        for i in range(n):
            row = {"scenario": result["scenario"], "controller": result["controller"]}
            for key in keys:
                row[key] = result[key][i]
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()

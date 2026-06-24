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
        from trans_mpc.config import SimulationConfig, VehicleConfig
        from trans_mpc.vehicle_model import DynamicBicycleModel, VehicleState

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

    def test_windowed_dataset_shapes(self):
        from trans_mpc.dataset import build_windowed_arrays

        features = np.arange(10 * 14, dtype=float).reshape(10, 14)
        labels = {
            "r_low": np.linspace(0.0, 1.0, 10),
            "r_ey": np.linspace(0.1, 0.9, 10),
            "r_stab": np.linspace(0.2, 0.8, 10),
            "k_v": np.linspace(1.0, 0.7, 10),
        }
        x, y = build_windowed_arrays(features, labels, history_len=4)
        self.assertEqual(x.shape, (7, 4, 14))
        self.assertEqual(y.shape, (7, 4))
        np.testing.assert_array_equal(x[0], features[:4])
        np.testing.assert_allclose(y[0], [labels["r_low"][3], labels["r_ey"][3], labels["r_stab"][3], labels["k_v"][3]])

    def test_pid_smoke_simulation_produces_logs(self):
        from trans_mpc.config import PathConfig, SimulationConfig, VehicleConfig
        from trans_mpc.controllers import PIDController
        from trans_mpc.reference_path import generate_double_lane_change
        from trans_mpc.scenario_manager import Scenario
        from trans_mpc.simulator import run_simulation

        sim_cfg = SimulationConfig(max_time=2.0, default_speed=8.0)
        path = generate_double_lane_change(PathConfig(length=40.0), sim_cfg, speed=8.0)
        scenario = Scenario(name="smoke", speed=8.0, mu_high=0.9, mu_low=0.9)
        result = run_simulation(PIDController(), scenario, path, sim_cfg, VehicleConfig())
        self.assertGreater(len(result["time"]), 5)
        self.assertEqual(len(result["time"]), len(result["e_y"]))
        self.assertTrue(np.all(np.isfinite(result["features"])))

    def test_metrics_summary_contains_expected_fields(self):
        from trans_mpc.metrics import summarize_result

        result = {
            "controller": "test",
            "scenario": "unit",
            "completed": True,
            "e_y": np.array([0.0, 0.5, -0.25]),
            "e_psi": np.array([0.0, 0.1, -0.1]),
            "beta": np.array([0.0, 0.02, -0.03]),
            "yaw_rate": np.array([0.1, 0.2, 0.3]),
            "ay": np.array([1.0, 2.0, 3.0]),
            "delta_rate": np.array([0.0, 0.1, -0.1]),
            "mpc_feasible": np.array([True, False, True]),
            "time": np.array([0.0, 0.05, 0.10]),
        }
        summary = summarize_result(result)
        self.assertEqual(summary["controller"], "test")
        self.assertEqual(summary["scenario"], "unit")
        self.assertEqual(summary["mpc_failures"], 1)
        self.assertAlmostEqual(summary["max_abs_e_y"], 0.5)
        self.assertGreater(summary["rmse_e_y"], 0.0)

    def test_speed_tracking_accel_commands_deceleration_for_lower_reference(self):
        from trans_mpc.config import VehicleConfig
        from trans_mpc.controllers import speed_tracking_accel

        ax = speed_tracking_accel(vx=15.0, v_ref=10.0, vehicle_config=VehicleConfig())
        self.assertLess(ax, 0.0)
        self.assertGreaterEqual(ax, -VehicleConfig().max_ax)

    def test_find_true_intervals_keeps_contiguous_regions(self):
        from trans_mpc.advantage_analysis import find_true_intervals

        mask = np.array([False, True, True, False, True, True, True])
        time = np.arange(len(mask), dtype=float) * 0.1
        x = np.arange(len(mask), dtype=float)
        intervals = find_true_intervals(mask, time, x, min_steps=2)
        self.assertEqual(intervals, [(0.1, 0.2, 1.0, 2.0, 2), (0.4, 0.6, 4.0, 6.0, 3)])

    def test_version2_scenarios_cover_richer_conditions(self):
        from trans_mpc.scenario_manager import make_version2_evaluation_scenarios, make_version2_training_scenarios

        train = make_version2_training_scenarios()
        eval_scenarios = make_version2_evaluation_scenarios()
        self.assertGreaterEqual(len(train), 10)
        self.assertGreaterEqual(len(eval_scenarios), 8)
        self.assertLessEqual(len(eval_scenarios), 10)
        self.assertEqual(len({s.name for s in train}), len(train))
        self.assertEqual(len({s.name for s in eval_scenarios}), len(eval_scenarios))
        self.assertTrue(any(s.speed >= 17.0 for s in eval_scenarios))
        self.assertTrue(any(s.mu_low <= 0.24 for s in eval_scenarios))
        self.assertTrue(any(s.noise_std > 0.0 for s in eval_scenarios))
        self.assertTrue(any(s.disturbed for s in eval_scenarios))
        self.assertTrue(any(s.initial_y != 0.0 or s.initial_yaw != 0.0 for s in eval_scenarios))

    def test_version2_training_defaults_are_moderate_full(self):
        from trans_mpc.config import TrainingConfig
        from trans_mpc.scenario_manager import make_version2_evaluation_scenarios, make_version2_training_scenarios

        self.assertGreaterEqual(TrainingConfig().epochs, 16)
        self.assertGreaterEqual(len(make_version2_training_scenarios()), len(make_version2_evaluation_scenarios()))

    def test_select_zoom_scenario_prefers_high_transformer_risk(self):
        import pandas as pd

        from scripts.generate_trans_result_figures import select_zoom_scenario

        df = pd.DataFrame(
            [
                {"scenario": "low", "controller": "Transformer-MPC", "time": 0.0, "r_low": 0.2, "r_stab": 0.1},
                {"scenario": "high", "controller": "Transformer-MPC", "time": 0.0, "r_low": 0.7, "r_stab": 0.8},
                {"scenario": "high", "controller": "PID", "time": 0.0, "r_low": 0.0, "r_stab": 0.0},
            ]
        )
        self.assertEqual(select_zoom_scenario(df), "high")

    def test_reference_line_helpers_add_baseline_and_dlc_path(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from trans_mpc.config import PathConfig, SimulationConfig
        from trans_mpc.reference_lines import add_double_lane_change_reference, add_zero_baseline

        fig, ax = plt.subplots()
        add_zero_baseline(ax)
        add_double_lane_change_reference(ax, PathConfig(), SimulationConfig())
        labels = [line.get_label() for line in ax.get_lines()]
        plt.close(fig)

        self.assertIn("0 baseline", labels)
        self.assertIn("DLC reference path", labels)

    def test_fixed_mpc_tracks_baseline_under_version2_defaults(self):
        from trans_mpc.config import MPCConfig, PathConfig, SimulationConfig, VehicleConfig
        from trans_mpc.controllers import FixedMPCController
        from trans_mpc.metrics import summarize_result
        from trans_mpc.reference_path import generate_double_lane_change
        from trans_mpc.scenario_manager import Scenario
        from trans_mpc.simulator import run_simulation

        sim_cfg = SimulationConfig(max_time=13.0)
        scenario = Scenario(name="baseline", speed=12.0, mu_high=0.92, mu_low=0.92)
        path = generate_double_lane_change(PathConfig(), sim_cfg, speed=scenario.speed)
        result = run_simulation(FixedMPCController(), scenario, path, sim_cfg, VehicleConfig(), MPCConfig())
        summary = summarize_result(result)

        self.assertTrue(summary["completed"])
        self.assertEqual(summary["mpc_failures"], 0)
        self.assertLess(summary["rmse_e_y"], 0.05)
        self.assertLess(summary["max_abs_e_y"], 0.10)

    def test_double_lane_change_reference_speed_is_curvature_limited_at_high_speed(self):
        from trans_mpc.config import PathConfig, SimulationConfig
        from trans_mpc.reference_path import generate_double_lane_change

        path = generate_double_lane_change(PathConfig(), SimulationConfig(default_speed=17.0), speed=17.0)
        self.assertAlmostEqual(float(path.v_ref[0]), 17.0)
        self.assertLess(float(path.v_ref.min()), 16.0)
        self.assertTrue(float(path.v_ref.min()) > 10.0)

    def test_fixed_mpc_completes_high_speed_dlc_under_version2_defaults(self):
        from trans_mpc.config import MPCConfig, PathConfig, SimulationConfig, VehicleConfig
        from trans_mpc.controllers import FixedMPCController
        from trans_mpc.metrics import summarize_result
        from trans_mpc.reference_path import generate_double_lane_change
        from trans_mpc.scenario_manager import Scenario
        from trans_mpc.simulator import run_simulation

        sim_cfg = SimulationConfig(max_time=13.0)
        scenario = Scenario(name="high_speed", speed=17.0, mu_high=0.66, mu_low=0.33, low_mu_start=34.0, low_mu_end=96.0)
        path = generate_double_lane_change(PathConfig(), sim_cfg, speed=scenario.speed)
        result = run_simulation(FixedMPCController(), scenario, path, sim_cfg, VehicleConfig(), MPCConfig())
        summary = summarize_result(result)

        self.assertTrue(summary["completed"])
        self.assertLess(summary["max_abs_e_y"], 40.0)
        self.assertLess(summary["mpc_failures"], 100)


if __name__ == "__main__":
    unittest.main()

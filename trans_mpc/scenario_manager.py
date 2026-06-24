from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    speed: float
    mu_high: float = 0.9
    mu_low: float = 0.3
    low_mu_start: float = 1.0e9
    low_mu_end: float = 1.0e9
    initial_y: float = 0.0
    initial_yaw: float = 0.0
    noise_std: float = 0.0
    disturbed: bool = False

    def mu_at(self, x: float) -> float:
        if self.low_mu_start <= x <= self.low_mu_end:
            return self.mu_low
        return self.mu_high


def make_default_scenarios() -> list[Scenario]:
    return [
        Scenario(name="baseline_high_mu", speed=12.0, mu_high=0.9, mu_low=0.9),
        Scenario(name="friction_step", speed=14.0, mu_high=0.9, mu_low=0.28, low_mu_start=38.0, low_mu_end=120.0),
        Scenario(name="lane_change_low_mu", speed=13.0, mu_high=0.9, mu_low=0.25, low_mu_start=24.0, low_mu_end=70.0),
        Scenario(name="high_speed_dlc", speed=17.0, mu_high=0.65, mu_low=0.35, low_mu_start=35.0, low_mu_end=90.0),
        Scenario(name="offset_noise", speed=12.0, mu_high=0.75, mu_low=0.35, low_mu_start=35.0, low_mu_end=85.0, initial_y=-0.8, noise_std=0.01),
    ]


def make_training_scenarios(quick: bool = False) -> list[Scenario]:
    base = [
        Scenario(name="train_pid_high_mu", speed=10.0, mu_high=0.9, mu_low=0.9),
        Scenario(name="train_step_mid", speed=12.0, mu_high=0.9, mu_low=0.35, low_mu_start=35.0, low_mu_end=100.0),
        Scenario(name="train_step_low", speed=14.0, mu_high=0.85, mu_low=0.25, low_mu_start=32.0, low_mu_end=100.0),
        Scenario(name="train_lane_low", speed=13.0, mu_high=0.9, mu_low=0.3, low_mu_start=24.0, low_mu_end=72.0),
        Scenario(name="train_offset", speed=11.0, mu_high=0.8, mu_low=0.32, low_mu_start=38.0, low_mu_end=95.0, initial_y=0.7),
        Scenario(name="train_noisy", speed=12.0, mu_high=0.75, mu_low=0.30, low_mu_start=36.0, low_mu_end=90.0, noise_std=0.015),
    ]
    return base[:3] if quick else base


def make_version2_training_scenarios() -> list[Scenario]:
    return [
        Scenario(name="v2_train_high_mu_10", speed=10.0, mu_high=0.95, mu_low=0.95),
        Scenario(name="v2_train_high_mu_12", speed=12.0, mu_high=0.90, mu_low=0.90),
        Scenario(name="v2_train_step_mid_12", speed=12.0, mu_high=0.90, mu_low=0.36, low_mu_start=34.0, low_mu_end=105.0),
        Scenario(name="v2_train_step_low_14", speed=14.0, mu_high=0.85, mu_low=0.26, low_mu_start=32.0, low_mu_end=108.0),
        Scenario(name="v2_train_lane_entry_low_13", speed=13.0, mu_high=0.90, mu_low=0.28, low_mu_start=22.0, low_mu_end=70.0),
        Scenario(name="v2_train_lane_mid_low_15", speed=15.0, mu_high=0.80, mu_low=0.30, low_mu_start=38.0, low_mu_end=88.0),
        Scenario(name="v2_train_high_speed_17", speed=17.0, mu_high=0.68, mu_low=0.34, low_mu_start=35.0, low_mu_end=95.0),
        Scenario(name="v2_train_offset_y", speed=12.0, mu_high=0.78, mu_low=0.32, low_mu_start=36.0, low_mu_end=94.0, initial_y=0.7),
        Scenario(name="v2_train_offset_yaw", speed=12.0, mu_high=0.82, mu_low=0.34, low_mu_start=36.0, low_mu_end=92.0, initial_yaw=0.035),
        Scenario(name="v2_train_noisy", speed=13.0, mu_high=0.78, mu_low=0.30, low_mu_start=35.0, low_mu_end=92.0, noise_std=0.012),
        Scenario(name="v2_train_disturbed", speed=14.0, mu_high=0.74, mu_low=0.29, low_mu_start=34.0, low_mu_end=90.0, disturbed=True),
        Scenario(name="v2_train_recovery", speed=13.0, mu_high=0.88, mu_low=0.27, low_mu_start=28.0, low_mu_end=62.0),
    ]


def make_version2_evaluation_scenarios() -> list[Scenario]:
    return [
        Scenario(name="baseline_high_mu_full", speed=12.0, mu_high=0.92, mu_low=0.92),
        Scenario(name="mild_friction_step_full", speed=13.0, mu_high=0.90, mu_low=0.38, low_mu_start=36.0, low_mu_end=115.0),
        Scenario(name="severe_friction_step_full", speed=14.0, mu_high=0.88, mu_low=0.24, low_mu_start=34.0, low_mu_end=115.0),
        Scenario(name="lane_change_low_mu_full", speed=13.5, mu_high=0.88, mu_low=0.26, low_mu_start=23.0, low_mu_end=74.0),
        Scenario(name="high_speed_dlc_full", speed=17.0, mu_high=0.66, mu_low=0.33, low_mu_start=34.0, low_mu_end=96.0),
        Scenario(name="offset_noise_full", speed=12.5, mu_high=0.76, mu_low=0.31, low_mu_start=35.0, low_mu_end=93.0, initial_y=-0.75, noise_std=0.012),
        Scenario(name="yaw_offset_disturbed_full", speed=13.0, mu_high=0.78, mu_low=0.30, low_mu_start=34.0, low_mu_end=92.0, initial_yaw=0.04, disturbed=True),
        Scenario(name="low_mu_recovery_full", speed=14.0, mu_high=0.90, mu_low=0.27, low_mu_start=26.0, low_mu_end=64.0),
        Scenario(name="extreme_high_speed_low_mu_full", speed=18.0, mu_high=0.62, mu_low=0.23, low_mu_start=32.0, low_mu_end=90.0, initial_y=0.35, noise_std=0.008),
    ]

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import MPCConfig
from .reference_lines import (
    add_double_lane_change_reference,
    add_reference_hline,
    add_unit_interval_reference,
    add_zero_baseline,
)


def save_comparison_plots(results: list[dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_scenario[str(result.get("scenario", "scenario"))].append(result)

    _plot_xy(by_scenario, output_dir / "trajectory_comparison.png")
    _plot_time_series(by_scenario, "e_y", "Lateral error [m]", output_dir / "lateral_error_comparison.png")
    _plot_time_series(by_scenario, "delta", "Steering [rad]", output_dir / "steering_comparison.png")
    _plot_risk(by_scenario, output_dir / "risk_prediction.png")
    _plot_stability(by_scenario, output_dir / "yaw_beta_stability.png")


def _make_axes(by_scenario):
    n = max(1, len(by_scenario))
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n), squeeze=False)
    return fig, axes[:, 0]


def _plot_xy(by_scenario, path: Path) -> None:
    fig, axes = _make_axes(by_scenario)
    for ax, (scenario, rows) in zip(axes, by_scenario.items()):
        add_double_lane_change_reference(ax)
        for row in rows:
            ax.plot(row["x"], row["y"], label=row.get("controller", "controller"))
        ax.set_title(f"Trajectory - {scenario}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_time_series(by_scenario, key: str, ylabel: str, path: Path) -> None:
    fig, axes = _make_axes(by_scenario)
    mpc_cfg = MPCConfig()
    for ax, (scenario, rows) in zip(axes, by_scenario.items()):
        for row in rows:
            ax.plot(row["time"], row[key], label=row.get("controller", "controller"))
        add_zero_baseline(ax)
        if key == "delta":
            add_reference_hline(ax, mpc_cfg.delta_max, "+delta limit")
            add_reference_hline(ax, -mpc_cfg.delta_max, "-delta limit")
        ax.set_title(f"{ylabel} - {scenario}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_risk(by_scenario, path: Path) -> None:
    fig, axes = _make_axes(by_scenario)
    for ax, (scenario, rows) in zip(axes, by_scenario.items()):
        for row in rows:
            if "Transformer" in row.get("controller", "") or "Rule" in row.get("controller", ""):
                ax.plot(row["time"], row["r_low"], label=f"{row.get('controller')} r_low")
        add_unit_interval_reference(ax)
        ax.set_title(f"Risk - {scenario}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("risk")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_stability(by_scenario, path: Path) -> None:
    fig, axes = _make_axes(by_scenario)
    mpc_cfg = MPCConfig()
    for ax, (scenario, rows) in zip(axes, by_scenario.items()):
        for row in rows:
            ax.plot(row["time"], row["beta"], label=f"{row.get('controller')} beta")
        add_zero_baseline(ax)
        add_reference_hline(ax, mpc_cfg.beta_max, "+beta limit")
        add_reference_hline(ax, -mpc_cfg.beta_max, "-beta limit")
        ax.set_title(f"Beta stability - {scenario}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("beta [rad]")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reference_lines import add_reference_hline, add_zero_baseline


CRITERIA = {
    "tracking_abs_e_y": "Tracking |e_y| lower",
    "beta_stability": "Beta lower",
    "ay_stability": "Lateral acceleration lower",
    "composite_score": "Composite score lower",
}


def composite_score(frame: pd.DataFrame) -> np.ndarray:
    return (
        np.abs(frame["e_y"].to_numpy()) / 1.5
        + np.abs(frame["beta"].to_numpy()) / 0.10
        + np.abs(frame["ay"].to_numpy()) / 8.0
        + 0.15 * np.abs(frame["delta"].to_numpy()) / 0.5
    )


def find_true_intervals(mask: np.ndarray, time: np.ndarray, x: np.ndarray, min_steps: int = 3) -> list[tuple[float, float, float, float, int]]:
    intervals: list[tuple[float, float, float, float, int]] = []
    start = None
    for i, active in enumerate(mask):
        if bool(active) and start is None:
            start = i
        is_last = i == len(mask) - 1
        if start is not None and ((not bool(active)) or is_last):
            end = i if bool(active) and is_last else i - 1
            steps = end - start + 1
            if steps >= min_steps:
                intervals.append(
                    (
                        round(float(time[start]), 3),
                        round(float(time[end]), 3),
                        round(float(x[start]), 3),
                        round(float(x[end]), 3),
                        int(steps),
                    )
                )
            start = None
    return intervals


def analyze_transformer_advantage(log_df: pd.DataFrame, min_steps: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    interval_rows = []
    for scenario in sorted(log_df["scenario"].unique()):
        by_controller = {
            controller: group.sort_values("time").reset_index(drop=True)
            for controller, group in log_df[log_df["scenario"] == scenario].groupby("controller")
        }
        transformer = by_controller.get("Transformer-MPC")
        if transformer is None:
            continue
        for baseline in ("PID", "Fixed MPC", "Rule-risk MPC"):
            base = by_controller.get(baseline)
            if base is None:
                continue
            n = min(len(transformer), len(base))
            trans = transformer.iloc[:n].reset_index(drop=True)
            other = base.iloc[:n].reset_index(drop=True)
            advantages = {
                "tracking_abs_e_y": np.abs(other["e_y"].to_numpy()) - np.abs(trans["e_y"].to_numpy()),
                "beta_stability": np.abs(other["beta"].to_numpy()) - np.abs(trans["beta"].to_numpy()),
                "ay_stability": np.abs(other["ay"].to_numpy()) - np.abs(trans["ay"].to_numpy()),
                "composite_score": composite_score(other) - composite_score(trans),
            }
            for key, advantage in advantages.items():
                mask = advantage > 0.0
                summary_rows.append(
                    {
                        "scenario": scenario,
                        "baseline": baseline,
                        "criterion": key,
                        "criterion_label": CRITERIA[key],
                        "advantage_pct": float(mask.mean() * 100.0),
                        "mean_advantage": float(np.mean(advantage)),
                        "mean_positive_advantage": float(np.mean(advantage[mask])) if np.any(mask) else 0.0,
                        "max_advantage": float(np.max(advantage)),
                    }
                )
                for t0, t1, x0, x1, steps in find_true_intervals(
                    mask,
                    trans["time"].to_numpy(),
                    trans["x"].to_numpy(),
                    min_steps=min_steps,
                ):
                    interval_rows.append(
                        {
                            "scenario": scenario,
                            "baseline": baseline,
                            "criterion": key,
                            "criterion_label": CRITERIA[key],
                            "t_start": t0,
                            "t_end": t1,
                            "x_start_transformer": x0,
                            "x_end_transformer": x1,
                            "steps": steps,
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(interval_rows)


def save_advantage_figures(log_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_composite_scores(log_df, output_dir / "advantage_composite_score.png")
    _plot_advantage_percentages(summary_df, output_dir / "advantage_percentage_bars.png")


def _plot_composite_scores(log_df: pd.DataFrame, path: Path) -> None:
    scenarios = sorted(log_df["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 4 * len(scenarios)), squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        scenario_df = log_df[log_df["scenario"] == scenario]
        for controller, group in scenario_df.groupby("controller"):
            group = group.sort_values("time")
            ax.plot(group["time"], composite_score(group), label=controller)
        transformer = scenario_df[scenario_df["controller"] == "Transformer-MPC"].sort_values("time")
        low_mu = transformer["mu"] < 0.5
        if low_mu.any():
            ax.axvspan(float(transformer.loc[low_mu, "time"].min()), float(transformer.loc[low_mu, "time"].max()), color="orange", alpha=0.12, label="low friction")
        add_zero_baseline(ax)
        ax.set_title(f"Composite tracking-stability score - {scenario}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("lower is better")
        ax.grid(True)
        ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_advantage_percentages(summary_df: pd.DataFrame, path: Path) -> None:
    filtered = summary_df[summary_df["criterion"].isin(["tracking_abs_e_y", "beta_stability", "ay_stability", "composite_score"])]
    labels = sorted(filtered["criterion_label"].unique())
    scenarios = sorted(filtered["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 4.5 * len(scenarios)), squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        sub = filtered[filtered["scenario"] == scenario]
        x = np.arange(len(labels))
        width = 0.25
        for i, baseline in enumerate(["PID", "Fixed MPC", "Rule-risk MPC"]):
            values = []
            for label in labels:
                row = sub[(sub["baseline"] == baseline) & (sub["criterion_label"] == label)]
                values.append(float(row["advantage_pct"].iloc[0]) if len(row) else 0.0)
            ax.bar(x + (i - 1) * width, values, width=width, label=f"vs {baseline}")
        ax.set_title(f"Transformer-MPC advantage percentage - {scenario}")
        ax.set_ylabel("% of aligned time steps")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 100)
        add_reference_hline(ax, 50.0, "50% parity line")
        ax.grid(True, axis="y")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)

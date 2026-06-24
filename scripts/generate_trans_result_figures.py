from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trans_mpc.advantage_analysis import composite_score
from trans_mpc.config import MPCConfig, TransformerConfig
from trans_mpc.reference_lines import (
    add_history_index_references,
    add_identity_reference,
    add_reference_hline,
    add_unit_interval_reference,
    add_zero_baseline,
)
from trans_mpc.transformer_model import RiskTransformer


OUT = ROOT / "outputs" / "figures" / "trans_result"
LOG_PATH = ROOT / "outputs" / "logs" / "experiment_log.csv"
METRICS_PATH = ROOT / "outputs" / "tables" / "metrics_summary.csv"
ADV_PATH = ROOT / "outputs" / "tables" / "advantage_summary.csv"
HISTORY_PATH = ROOT / "checkpoints" / "training_history.csv"
DATASET_PATH = ROOT / "data" / "dataset.npz"
SCALER_PATH = ROOT / "data" / "scaler.pkl"
CKPT_PATH = ROOT / "checkpoints" / "best_transformer.pt"

FEATURE_NAMES = [
    "vx",
    "vy",
    "yaw_rate",
    "ay",
    "beta",
    "delta",
    "delta_rate",
    "ax",
    "e_y",
    "e_psi",
    "e_y_rate",
    "e_psi_rate",
    "kappa_ref",
    "v_ref",
]

TARGET_NAMES = ["r_low", "r_ey", "r_stab", "k_v"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_df = pd.read_csv(LOG_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)
    advantage_df = pd.read_csv(ADV_PATH)
    plot_risk_prediction_timeline(log_df)
    plot_mpc_parameter_adaptation(log_df)
    plot_high_risk_zoom(log_df)
    plot_composite_score(log_df)
    plot_advantage_bars(advantage_df)
    plot_training_prediction_effect()
    plot_feature_saliency()
    print(f"saved figures under {OUT}")


def plot_risk_prediction_timeline(log_df: pd.DataFrame) -> None:
    cfg = MPCConfig()
    scenarios = sorted(log_df["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 4 * len(scenarios)), squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        row = _controller(log_df, scenario, "Transformer-MPC")
        ax.plot(row["time"], row["r_low"], label="r_low", linewidth=2)
        ax.plot(row["time"], row["r_ey"], label="r_ey", linewidth=1.7)
        ax.plot(row["time"], row["r_stab"], label="r_stab", linewidth=1.7)
        ax.plot(row["time"], row["k_v"], label="k_v", linewidth=1.7)
        _shade_low_mu(ax, row)
        add_unit_interval_reference(ax)
        add_reference_hline(ax, cfg.k_v_min, "k_v_min", color="0.2", linestyle="-.")
        ax2 = ax.twinx()
        ax2.plot(row["time"], np.abs(row["beta"]), "--", color="black", alpha=0.55, label="|beta|")
        add_reference_hline(ax2, cfg.beta_max, "beta limit", color="0.1", linestyle=":")
        ax.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("|beta| [rad]")
        ax.set_title(f"Transformer risk outputs and stability response - {scenario}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("risk / speed scale")
        ax.grid(True)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(OUT / "01_risk_prediction_timeline.png", dpi=180)
    plt.close(fig)


def plot_mpc_parameter_adaptation(log_df: pd.DataFrame) -> None:
    cfg = MPCConfig()
    scenarios = sorted(log_df["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 4 * len(scenarios)), squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        row = _controller(log_df, scenario, "Transformer-MPC")
        r_stab = row["r_stab"].to_numpy()
        r_low = row["r_low"].to_numpy()
        k_v = row["k_v"].to_numpy()
        v0 = float(row["vx"].iloc[0])
        q_beta = np.clip(cfg.q_beta * (1.0 + 2.4 * r_stab), cfg.q_beta, cfg.q_beta_max)
        beta_max = np.clip(cfg.beta_max * (1.0 - cfg.c_beta * r_stab), cfg.beta_min, cfg.beta_max)
        r_d_delta = np.clip(cfg.r_d_delta * (1.0 + 2.0 * r_stab), cfg.r_d_delta, cfg.r_d_delta_max)
        v_ref_adapt = v0 * np.clip(np.minimum(k_v, 1.0 - cfg.c_v * r_low), cfg.k_v_min, 1.0)
        ax.plot(row["time"], q_beta / cfg.q_beta, label="Q_beta / Q_beta0")
        ax.plot(row["time"], r_d_delta / cfg.r_d_delta, label="R_dDelta / R_dDelta0")
        ax.plot(row["time"], beta_max / cfg.beta_max, label="beta_max / beta_max0")
        ax.plot(row["time"], v_ref_adapt / v0, label="v_ref_adapt / v_ref0")
        _shade_low_mu(ax, row)
        add_zero_baseline(ax)
        add_reference_hline(ax, 1.0, "nominal value")
        add_reference_hline(ax, cfg.k_v_min, "minimum speed scale", color="0.2", linestyle="-.")
        ax.set_title(f"Risk-to-MPC parameter adaptation - {scenario}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("normalized parameter")
        ax.grid(True)
        ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(OUT / "02_mpc_parameter_adaptation.png", dpi=180)
    plt.close(fig)


def plot_high_risk_zoom(log_df: pd.DataFrame) -> None:
    cfg = MPCConfig()
    scenario = select_zoom_scenario(log_df)
    transformer_full = _controller(log_df, scenario, "Transformer-MPC")
    if len(transformer_full):
        risk = transformer_full["r_low"].abs() + transformer_full["r_stab"].abs()
        peak_time = float(transformer_full.loc[risk.idxmax(), "time"])
        t_start = max(0.0, peak_time - 1.5)
        t_end = peak_time + 1.5
        sub = log_df[(log_df["scenario"] == scenario) & (log_df["time"] >= t_start) & (log_df["time"] <= t_end)].copy()
    else:
        sub = log_df[log_df["scenario"] == scenario].copy()
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for controller, group in sub.groupby("controller"):
        group = group.sort_values("time")
        axes[0].plot(group["time"], group["e_y"], label=controller)
        axes[1].plot(group["time"], group["beta"], label=controller)
        axes[2].plot(group["time"], group["ay"], label=controller)
        axes[3].plot(group["time"], group["delta"], label=controller)
    transformer = sub[sub["controller"] == "Transformer-MPC"].sort_values("time")
    for ax in axes:
        _shade_low_mu(ax, transformer)
        add_zero_baseline(ax)
        ax.grid(True)
    add_reference_hline(axes[1], cfg.beta_max, "+beta limit")
    add_reference_hline(axes[1], -cfg.beta_max, "-beta limit")
    add_reference_hline(axes[2], cfg.ay_max, "+ay limit")
    add_reference_hline(axes[2], -cfg.ay_max, "-ay limit")
    add_reference_hline(axes[3], cfg.delta_max, "+delta limit")
    add_reference_hline(axes[3], -cfg.delta_max, "-delta limit")
    axes[0].set_ylabel("e_y [m]")
    axes[1].set_ylabel("beta [rad]")
    axes[2].set_ylabel("ay [m/s^2]")
    axes[3].set_ylabel("delta [rad]")
    axes[3].set_xlabel("time [s]")
    axes[0].set_title(f"High-risk zoom: {scenario}")
    axes[0].legend(loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(OUT / "03_high_risk_zoom.png", dpi=180)
    plt.close(fig)


def plot_composite_score(log_df: pd.DataFrame) -> None:
    scenarios = sorted(log_df["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 4 * len(scenarios)), squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        for controller, group in log_df[log_df["scenario"] == scenario].groupby("controller"):
            group = group.sort_values("time")
            ax.plot(group["time"], composite_score(group), label=controller)
        transformer = _controller(log_df, scenario, "Transformer-MPC")
        _shade_low_mu(ax, transformer)
        add_zero_baseline(ax)
        ax.set_title(f"Composite tracking-stability score - {scenario}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("lower is better")
        ax.grid(True)
        ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(OUT / "04_composite_score.png", dpi=180)
    plt.close(fig)


def plot_advantage_bars(advantage_df: pd.DataFrame) -> None:
    filtered = advantage_df[advantage_df["criterion"].isin(["tracking_abs_e_y", "beta_stability", "ay_stability", "composite_score"])]
    labels = ["Tracking |e_y| lower", "Beta lower", "Lateral acceleration lower", "Composite score lower"]
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
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylim(0, 100)
        add_reference_hline(ax, 50.0, "50% parity line")
        ax.grid(True, axis="y")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "05_advantage_percentage_bars.png", dpi=180)
    plt.close(fig)


def plot_training_prediction_effect() -> None:
    data = np.load(DATASET_PATH)
    history = pd.read_csv(HISTORY_PATH)
    model = _load_model()
    windows = torch.as_tensor(data["windows"], dtype=torch.float32)
    targets = data["targets"]
    with torch.no_grad():
        pred = model(windows[: min(200, len(windows))]).numpy()
    true = targets[: len(pred)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0, 0].plot(history["epoch"], history["val_loss"], marker="o", label="validation")
    axes[0, 0].set_title("Training and validation loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("SmoothL1 loss")
    add_zero_baseline(axes[0, 0])
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    for ax, idx, name in [(axes[0, 1], 0, "r_low"), (axes[1, 0], 2, "r_stab")]:
        ax.scatter(true[:, idx], pred[:, idx], s=18, alpha=0.65)
        add_identity_reference(ax)
        ax.set_title(f"Prediction vs label: {name}")
        ax.set_xlabel("label")
        ax.set_ylabel("prediction")
        ax.grid(True)

    n = min(100, len(pred))
    axes[1, 1].plot(true[:n, 0], label="label r_low")
    axes[1, 1].plot(pred[:n, 0], label="pred r_low")
    axes[1, 1].set_title("Risk sequence alignment")
    axes[1, 1].set_xlabel("sample index")
    axes[1, 1].set_ylabel("r_low")
    add_unit_interval_reference(axes[1, 1])
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "06_training_prediction_effect.png", dpi=180)
    plt.close(fig)


def plot_feature_saliency() -> None:
    data = np.load(DATASET_PATH)
    windows = data["windows"]
    targets = data["targets"]
    idx = int(np.argmax(targets[:, 0] + targets[:, 2]))
    model = _load_model()
    x = torch.tensor(windows[idx : idx + 1], dtype=torch.float32, requires_grad=True)
    y = model(x)[0, 0] + model(x)[0, 2]
    y.backward()
    saliency = x.grad.detach().abs().numpy()[0].T
    saliency = saliency / max(float(saliency.max()), 1e-9)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    im = ax.imshow(saliency, aspect="auto", cmap="magma", origin="lower")
    ax.set_yticks(np.arange(len(FEATURE_NAMES)))
    ax.set_yticklabels(FEATURE_NAMES)
    ax.set_xlabel("history step")
    ax.set_title("Feature-time gradient saliency for r_low + r_stab")
    add_history_index_references(ax, saliency.shape[1])
    fig.colorbar(im, ax=ax, label="normalized |gradient|")
    fig.tight_layout()
    fig.savefig(OUT / "07_feature_time_saliency_heatmap.png", dpi=180)
    plt.close(fig)


def _load_model() -> RiskTransformer:
    cfg = TransformerConfig()
    model = RiskTransformer(cfg.input_dim, cfg.d_model, cfg.nhead, cfg.num_layers, cfg.dropout, cfg.k_v_min)
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def select_zoom_scenario(log_df: pd.DataFrame) -> str:
    transformer = log_df[log_df["controller"] == "Transformer-MPC"].copy()
    if transformer.empty:
        return str(sorted(log_df["scenario"].unique())[0])
    transformer["risk_score"] = transformer["r_low"].abs() + transformer["r_stab"].abs()
    scores = transformer.groupby("scenario")["risk_score"].max().sort_values(ascending=False)
    return str(scores.index[0])


def _controller(log_df: pd.DataFrame, scenario: str, controller: str) -> pd.DataFrame:
    return log_df[(log_df["scenario"] == scenario) & (log_df["controller"] == controller)].sort_values("time").reset_index(drop=True)


def _shade_low_mu(ax, row: pd.DataFrame) -> None:
    if len(row):
        low = row["mu"] < 0.5
        if low.any():
            ax.axvspan(float(row.loc[low, "time"].min()), float(row.loc[low, "time"].max()), color="orange", alpha=0.12, label="low friction")


if __name__ == "__main__":
    main()

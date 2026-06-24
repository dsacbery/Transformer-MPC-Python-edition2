from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import PathConfig, SimulationConfig
from .reference_path import generate_double_lane_change


BASELINE_KWARGS = {"color": "0.25", "linestyle": "--", "linewidth": 1.0, "alpha": 0.75}
REFERENCE_PATH_KWARGS = {"color": "black", "linestyle": "--", "linewidth": 2.0, "alpha": 0.85}
BOUNDARY_KWARGS = {"color": "0.35", "linestyle": ":", "linewidth": 1.0, "alpha": 0.65}


def add_zero_baseline(ax, axis: str = "y", label: str = "0 baseline") -> None:
    if axis == "y":
        ax.axhline(0.0, label=_dedupe_label(ax, label), **BASELINE_KWARGS)
    elif axis == "x":
        ax.axvline(0.0, label=_dedupe_label(ax, label), **BASELINE_KWARGS)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def add_reference_hline(ax, value: float, label: str, **kwargs) -> None:
    style = {**BOUNDARY_KWARGS, **kwargs}
    ax.axhline(float(value), label=_dedupe_label(ax, label), **style)


def add_reference_vline(ax, value: float, label: str, **kwargs) -> None:
    style = {**BOUNDARY_KWARGS, **kwargs}
    ax.axvline(float(value), label=_dedupe_label(ax, label), **style)


def add_unit_interval_reference(ax, include_zero: bool = True) -> None:
    if include_zero:
        add_reference_hline(ax, 0.0, "risk lower bound")
    add_reference_hline(ax, 1.0, "risk upper bound")


def add_double_lane_change_reference(ax, path_cfg: PathConfig | None = None, sim_cfg: SimulationConfig | None = None, speed: float | None = None) -> None:
    cfg = path_cfg or PathConfig()
    path = generate_double_lane_change(cfg, sim_cfg or SimulationConfig(), speed=speed)
    ax.plot(path.x, path.y, label=_dedupe_label(ax, "DLC reference path"), **REFERENCE_PATH_KWARGS)
    add_reference_hline(ax, 0.0, "lane center y=0", color="0.45", linestyle=":", linewidth=1.0, alpha=0.7)
    add_reference_hline(ax, cfg.lane_width, "lane offset reference", color="0.45", linestyle=":", linewidth=1.0, alpha=0.7)


def add_identity_reference(ax) -> None:
    ax.plot([0.0, 1.0], [0.0, 1.0], label=_dedupe_label(ax, "ideal y=x"), color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    add_reference_hline(ax, 0.0, "risk lower bound")
    add_reference_hline(ax, 1.0, "risk upper bound")
    add_reference_vline(ax, 0.0, "label lower bound")
    add_reference_vline(ax, 1.0, "label upper bound")


def add_history_index_references(ax, history_steps: int | None = None) -> None:
    add_reference_vline(ax, 0.0, "history start")
    if history_steps is not None and history_steps > 1:
        add_reference_vline(ax, history_steps - 1, "history end")


def add_low_mu_span(ax, time: Iterable[float], mu: Iterable[float], threshold: float = 0.5) -> None:
    time_arr = np.asarray(list(time), dtype=float)
    mu_arr = np.asarray(list(mu), dtype=float)
    if time_arr.size == 0 or mu_arr.size == 0:
        return
    low = mu_arr < threshold
    if low.any():
        ax.axvspan(float(time_arr[low].min()), float(time_arr[low].max()), color="orange", alpha=0.12, label=_dedupe_label(ax, "low friction"))


def _dedupe_label(ax, label: str) -> str:
    existing = {line.get_label() for line in ax.get_lines()}
    existing.update(collection.get_label() for collection in ax.collections)
    existing.update(patch.get_label() for patch in ax.patches)
    return "_nolegend_" if label in existing else label

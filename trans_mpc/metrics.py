from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def summarize_result(result: dict) -> dict[str, float | int | str | bool]:
    e_y = _array(result, "e_y")
    e_psi = _array(result, "e_psi")
    beta = _array(result, "beta")
    yaw_rate = _array(result, "yaw_rate")
    ay = _array(result, "ay")
    delta_rate = _array(result, "delta_rate")
    feasible = np.asarray(result.get("mpc_feasible", np.ones_like(e_y, dtype=bool)), dtype=bool)
    time = _array(result, "time")
    return {
        "controller": str(result.get("controller", "")),
        "scenario": str(result.get("scenario", "")),
        "completed": bool(result.get("completed", False)),
        "duration_s": float(time[-1]) if len(time) else 0.0,
        "rmse_e_y": _rmse(e_y),
        "max_abs_e_y": _max_abs(e_y),
        "rmse_e_psi": _rmse(e_psi),
        "max_abs_beta": _max_abs(beta),
        "max_abs_yaw_rate": _max_abs(yaw_rate),
        "max_abs_ay": _max_abs(ay),
        "rms_delta_rate": _rmse(delta_rate),
        "mpc_failures": int(np.sum(~feasible)),
    }


def summarize_results(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([summarize_result(result) for result in results])


def save_metrics(results: list[dict], output_path: str | Path) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = summarize_results(results)
    df.to_csv(output_path, index=False)
    return df


def _array(result: dict, key: str) -> np.ndarray:
    return np.asarray(result.get(key, []), dtype=float)


def _rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def _max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


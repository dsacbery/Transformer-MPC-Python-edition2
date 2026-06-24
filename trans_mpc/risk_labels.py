from __future__ import annotations

import numpy as np


def compute_future_risk_labels(logs: dict[str, np.ndarray], horizon: int = 20, k_v_min: float = 0.45) -> dict[str, np.ndarray]:
    n = len(logs["e_y"])
    r_ey = np.zeros(n)
    r_stab = np.zeros(n)
    r_low = np.zeros(n)
    k_v = np.ones(n)

    e_y = np.asarray(logs["e_y"], dtype=float)
    beta = np.asarray(logs["beta"], dtype=float)
    yaw_rate = np.asarray(logs["yaw_rate"], dtype=float)
    ay = np.asarray(logs["ay"], dtype=float)
    delta = np.asarray(logs["delta"], dtype=float)
    v_ref = np.asarray(logs["v_ref"], dtype=float)
    kappa_ref = np.asarray(logs["kappa_ref"], dtype=float)

    for i in range(n):
        j = min(n, i + max(1, horizon))
        v = max(float(v_ref[i]), 0.1)
        kappa = abs(float(kappa_ref[i]))
        ey_threshold = np.clip(0.35 + 0.025 * v + 3.5 * kappa, 0.45, 1.6)
        yaw_threshold = np.clip(abs(v * kappa) + 0.35, 0.35, 0.95)
        ay_threshold = np.clip(v * v * kappa + 2.5, 2.5, 7.0)

        ey_risk = np.max(np.abs(e_y[i:j])) / ey_threshold
        beta_risk = np.max(np.abs(beta[i:j])) / 0.10
        yaw_risk = np.max(np.abs(yaw_rate[i:j])) / yaw_threshold
        ay_risk = np.max(np.abs(ay[i:j])) / ay_threshold
        control_risk = np.max(np.abs(delta[i:j])) / 0.50

        r_ey[i] = np.clip(ey_risk, 0.0, 1.0)
        r_stab[i] = np.clip(max(beta_risk, yaw_risk, ay_risk), 0.0, 1.0)
        r_low[i] = np.clip(0.45 * r_ey[i] + 0.45 * r_stab[i] + 0.10 * np.clip(control_risk, 0.0, 1.0), 0.0, 1.0)
        curvature_risk = np.clip(v * v * kappa / 7.0, 0.0, 1.0)
        k_v[i] = np.clip(1.0 - 0.40 * r_low[i] - 0.15 * curvature_risk, k_v_min, 1.0)

    return {"r_low": r_low, "r_ey": r_ey, "r_stab": r_stab, "k_v": k_v}


"""Dose-response summaries without cherry-picking a magnitude."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def dose_response_metrics(dose, observed_change) -> dict[str, float]:
    x = np.asarray(dose, dtype=float)
    y = np.asarray(observed_change, dtype=float)
    if len(x) < 2:
        return {"monotonic_rho": float("nan"), "linear_slope": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    rho = spearmanr(x, y).statistic
    return {
        "monotonic_rho": float(rho),
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
    }

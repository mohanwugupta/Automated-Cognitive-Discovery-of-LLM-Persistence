"""Compare independently predicted and observed causal logit changes."""

from __future__ import annotations

import numpy as np


def causal_prediction_metrics(predicted, observed) -> dict[str, float]:
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if not len(x):
        return {
            key: float("nan")
            for key in ("correlation", "slope", "intercept", "rmse", "sign_agreement")
        }
    if len(x) > 1 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else float("nan")
    else:
        slope, intercept, correlation = float("nan"), float("nan"), float("nan")
    nonzero = np.abs(x) > 1e-12
    return {
        "correlation": correlation,
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": float(np.sqrt(np.mean(np.square(y - x)))),
        "sign_agreement": (
            float(np.mean(np.sign(x[nonzero]) == np.sign(y[nonzero])))
            if nonzero.any()
            else float("nan")
        ),
    }

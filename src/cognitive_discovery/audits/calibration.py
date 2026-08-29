"""Calibration diagnostics for untouched frozen-model predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitive_discovery.models.fitting import regression_metrics


def _summary(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    design = np.column_stack((np.ones(len(observed)), observed))
    intercept, slope = np.linalg.lstsq(design, predicted, rcond=None)[0]
    residual = observed - predicted
    return {
        **regression_metrics(observed, predicted),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "residual_sd": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "rows": len(observed),
    }


def calibration_tables(
    predictions: pd.DataFrame,
    *,
    observed_column="persistence_logit",
    prediction_column="predicted_persistence_logit",
):
    """Return overall calibration, prediction-decile metrics, and task metrics.

    Predictions are never recalibrated; the fitted line is diagnostic only.
    """

    required = {observed_column, prediction_column, "task_family"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"calibration inputs missing: {sorted(missing)}")
    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "group": "all",
                **_summary(
                    predictions[observed_column], predictions[prediction_column]
                ),
            }
        ]
    )
    decile_frame = predictions.copy()
    unique = decile_frame[prediction_column].nunique()
    bins = min(10, max(1, unique))
    decile_frame["prediction_decile"] = pd.qcut(
        decile_frame[prediction_column], q=bins, labels=False, duplicates="drop"
    )
    deciles = pd.DataFrame(
        [
            {
                "scope": "prediction_decile",
                "group": int(decile) + 1,
                **_summary(part[observed_column], part[prediction_column]),
                "prediction_min": float(part[prediction_column].min()),
                "prediction_max": float(part[prediction_column].max()),
            }
            for decile, part in decile_frame.groupby("prediction_decile", observed=True)
        ]
    )
    per_task = pd.DataFrame(
        [
            {
                "scope": "task",
                "group": task,
                **_summary(part[observed_column], part[prediction_column]),
            }
            for task, part in predictions.groupby("task_family")
        ]
    )
    return overall, deciles, per_task

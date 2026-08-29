from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from cognitive_discovery.models.fitting import regression_metrics, task_macro_metrics
from cognitive_discovery.audits.calibration import calibration_tables


def evaluate_frozen_model(records: pd.DataFrame, model_path: str | Path):
    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle)
    prediction = model.predict(records)
    output = records[
        ["condition_id", "paired_condition_id", "task_family", "persistence_logit"]
    ].copy()
    output["predicted_persistence_logit"] = prediction
    overall, _, _ = calibration_tables(output)
    calibration = overall.iloc[0].to_dict()
    return output, {
        **regression_metrics(records.persistence_logit, prediction),
        **task_macro_metrics(records, prediction),
        "calibration_intercept": float(calibration["calibration_intercept"]),
        "calibration_slope": float(calibration["calibration_slope"]),
        "rmse": float(calibration["rmse"]),
        "mae": float(calibration["mae"]),
        "residual_sd": float(calibration["residual_sd"]),
    }

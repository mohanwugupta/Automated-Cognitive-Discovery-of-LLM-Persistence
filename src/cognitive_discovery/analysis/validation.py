from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from cognitive_discovery.models.fitting import regression_metrics, task_macro_metrics


def evaluate_frozen_model(records: pd.DataFrame, model_path: str | Path):
    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle)
    prediction = model.predict(records)
    output = records[["condition_id", "paired_condition_id", "task_family", "persistence_logit"]].copy()
    output["predicted_persistence_logit"] = prediction
    return output, {
        **regression_metrics(records.persistence_logit, prediction),
        **task_macro_metrics(records, prediction),
    }


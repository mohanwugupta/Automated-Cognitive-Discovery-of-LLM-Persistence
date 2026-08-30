"""Specificity summaries for task, labels, controls, and policy geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def squared_correlation(left, right) -> float:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1] ** 2)


def task_eta_squared(projection, tasks) -> float:
    frame = pd.DataFrame({"projection": projection, "task": tasks}).dropna()
    if frame.empty:
        return float("nan")
    grand = frame.projection.mean()
    total = float(np.square(frame.projection - grand).sum())
    between = float(
        sum(
            len(part) * (part.projection.mean() - grand) ** 2
            for _, part in frame.groupby("task")
        )
    )
    return between / total if total > 0 else float("nan")


def specificity_table(
    frame: pd.DataFrame, *, projection_column="projection"
) -> pd.DataFrame:
    controls = (
        "persistence_logit",
        "success_evidence",
        "continuation_value",
        "disengagement_value",
        "continuation_cost",
        "response_mapping_sign",
    )
    rows = [
        {
            "control": column,
            "shared_variance": squared_correlation(
                frame[projection_column], frame[column]
            ),
        }
        for column in controls
        if column in frame
    ]
    rows.append(
        {
            "control": "task_identity",
            "shared_variance": task_eta_squared(
                frame[projection_column], frame.task_family
            ),
        }
    )
    return pd.DataFrame(rows)

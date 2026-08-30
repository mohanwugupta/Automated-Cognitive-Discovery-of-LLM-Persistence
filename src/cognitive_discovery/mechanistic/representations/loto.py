"""Strict leave-one-task-out representation validation."""

from __future__ import annotations

import pandas as pd

from .ridge_probe import RidgeProbe, probe_metrics


def leave_one_task_out_probe(
    frame: pd.DataFrame,
    *,
    feature_columns,
    target_column: str,
    alpha: float = 1.0,
    rank: int = 1,
) -> pd.DataFrame:
    rows = []
    tasks = sorted(frame.task_family.astype(str).unique())
    for task in tasks:
        train = frame[frame.task_family.astype(str) != task]
        test = frame[frame.task_family.astype(str) == task]
        if train.empty or test.empty:
            continue
        probe = RidgeProbe(alpha=alpha, rank=rank).fit(
            train[list(feature_columns)], train[target_column]
        )
        metrics = probe_metrics(
            test[target_column], probe.predict(test[list(feature_columns)])
        )
        rows.append(
            {
                "heldout_task": task,
                "source_tasks": ",".join(
                    sorted(train.task_family.astype(str).unique())
                ),
                "target_normalization_used": False,
                **metrics,
            }
        )
    return pd.DataFrame(rows)

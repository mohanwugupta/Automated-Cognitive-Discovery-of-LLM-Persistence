"""Strict zero-shot leave-one-task-family-out evaluation."""

from __future__ import annotations

import pandas as pd

from cognitive_discovery.models.fitting import fit_model, regression_metrics


def leave_one_task_out(frame: pd.DataFrame, *, models, sharing="fully_shared") -> pd.DataFrame:
    rows = []
    for heldout in sorted(frame.task_family.astype(str).unique()):
        source = frame[frame.task_family.astype(str) != heldout].reset_index(drop=True)
        target = frame[frame.task_family.astype(str) == heldout].reset_index(drop=True)
        if source.empty or target.empty:
            continue
        for model in models:
            fit = fit_model(source, model, sharing=sharing)
            prediction = fit.predict(target)
            rows.append(
                {
                    "heldout_task": heldout,
                    "model": model,
                    "sharing": sharing,
                    "source_tasks": source.task_family.nunique(),
                    "source_rows": len(source),
                    "target_rows": len(target),
                    **regression_metrics(target.persistence_logit, prediction),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "heldout_task"]).reset_index(drop=True)


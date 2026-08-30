"""Projection-mediated intervention summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mediation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"task_family", "target", "layer", "total_effect", "mediated_effect"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"patching frame is missing: {sorted(missing)}")
    data = frame.copy()
    if "control" not in data:
        data["control"] = "candidate"
    data["projection_mediated_intervention_fraction"] = np.where(
        np.abs(data.total_effect) > 1e-10,
        data.mediated_effect / data.total_effect,
        np.nan,
    )
    return data.groupby(
        ["task_family", "target", "layer", "control"], as_index=False
    ).agg(
        total_effect=("total_effect", "mean"),
        mediated_effect=("mediated_effect", "mean"),
        projection_mediated_intervention_fraction=(
            "projection_mediated_intervention_fraction",
            "mean",
        ),
        contrasts=("total_effect", "size"),
    )

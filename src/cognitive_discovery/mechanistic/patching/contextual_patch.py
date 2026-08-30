"""Comparison of raw and context-relevant projection patches."""

from __future__ import annotations

import pandas as pd


def contextual_patch_advantage(frame: pd.DataFrame) -> pd.DataFrame:
    contextual = frame[frame.contrast_family == "contextual_history"]
    if "control" in contextual:
        contextual = contextual[contextual.control == "candidate"]
    means = (
        contextual.groupby(["task_family", "target"], as_index=False)
        .mediated_effect.mean()
        .pivot(index="task_family", columns="target", values="mediated_effect")
        .reset_index()
    )
    if {"outcome_history", "contextual_outcome_history"} <= set(means):
        means["contextual_minus_raw"] = (
            means.contextual_outcome_history - means.outcome_history
        )
    return means

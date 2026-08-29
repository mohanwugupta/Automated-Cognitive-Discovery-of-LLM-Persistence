"""Registered decision rule that permits observational equivalence."""

from __future__ import annotations

import pandas as pd


def resolve_theory_outcome(
    bootstrap_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    *,
    equivalence_margin: float = 0.02,
    minimum_r2: float = 0.0,
) -> dict:
    overall = bootstrap_summary[
        (bootstrap_summary.scope == "overall") & (bootstrap_summary.group == "all")
    ]
    if len(overall) != 1:
        raise ValueError("theory decision requires one overall paired-bootstrap row")
    row = overall.iloc[0]
    if len(model_metrics) and float(model_metrics.r2.max()) < float(minimum_r2):
        return {
            "outcome": "third_architecture",
            "winner": None,
            "reason": "both frozen theories failed the registered predictive floor",
        }
    if float(row.ci_low) > 0:
        return {
            "outcome": "latent_context",
            "winner": "latent_context",
            "reason": "latent context has lower paired squared error",
        }
    if float(row.ci_high) < 0:
        return {
            "outcome": "dual_history",
            "winner": "dual_history",
            "reason": "dual history has lower paired squared error",
        }
    if abs(float(row.mean_delta_error_dh_minus_lc)) <= float(equivalence_margin):
        return {
            "outcome": "observational_equivalence",
            "winner": None,
            "reason": "paired interval crosses zero within the registered equivalence margin",
        }
    return {
        "outcome": "unresolved",
        "winner": None,
        "reason": "paired interval crosses zero outside the equivalence margin",
    }

"""Prediction disagreement between leading cognitive architectures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model


def disagreement_scores(
    observed: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    architecture_a: str = "dual_history",
    architecture_b: str = "latent_context",
    prediction_a=None,
    prediction_b=None,
) -> pd.DataFrame:
    if prediction_a is None or prediction_b is None:
        fit_a = fit_hierarchical_model(observed, architecture_a, variant="M4")
        fit_b = fit_hierarchical_model(observed, architecture_b, variant="M4")
        prediction_a = fit_a.predict(candidates, include_random=True)
        prediction_b = fit_b.predict(candidates, include_random=True)
    prediction_a = np.asarray(prediction_a, dtype=float)
    prediction_b = np.asarray(prediction_b, dtype=float)
    difference = np.abs(prediction_a - prediction_b)
    rank = pd.Series(difference).rank(method="average", pct=True).to_numpy(dtype=float)
    result = candidates[["paired_condition_id", "semantic_hash", "task_family"]].copy()
    result[f"prediction_{architecture_a}"] = prediction_a
    result[f"prediction_{architecture_b}"] = prediction_b
    result["disagreement_raw"] = difference
    result["disagreement_score"] = rank
    return result

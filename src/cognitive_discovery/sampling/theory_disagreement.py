"""Frozen-theory disagreement scores for Round-3 candidates."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


OUTCOME_COLUMNS = {
    "persistence_logit",
    "p_continue",
    "p_disengage",
    "sampled_action",
    "terminated",
}


def assert_outcome_free_candidates(candidates: pd.DataFrame) -> None:
    leaked = OUTCOME_COLUMNS & set(candidates)
    if leaked:
        raise ValueError(f"Round-3 outcomes leaked into candidate scoring: {sorted(leaked)}")


def _rank01(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    if series.nunique() <= 1:
        return np.full(len(series), 0.5)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def theory_disagreement_scores(
    candidates: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    assert_outcome_free_candidates(candidates)
    if len(predictions) < 2:
        raise ValueError("disagreement requires at least two frozen theories")
    arrays = {
        name: np.asarray(values, dtype=float) for name, values in predictions.items()
    }
    if any(len(values) != len(candidates) for values in arrays.values()):
        raise ValueError("prediction length disagrees with candidate pool")
    matrix = np.column_stack(list(arrays.values()))
    result = candidates[["paired_condition_id", "semantic_hash", "task_family"]].copy()
    for name, values in arrays.items():
        result[f"prediction_{name}"] = values
    result["disagreement_variance"] = np.var(matrix, axis=1)
    for left, right in combinations(sorted(arrays), 2):
        result[f"absolute_difference_{left}_{right}"] = np.abs(
            arrays[left] - arrays[right]
        )
    if {"dual_history", "latent_context"} <= set(arrays):
        primary = np.abs(arrays["dual_history"] - arrays["latent_context"])
    else:
        primary = np.max(matrix, axis=1) - np.min(matrix, axis=1)
    result["disagreement_raw"] = primary
    result["disagreement_score"] = _rank01(primary)
    return result

"""Marginal, pairwise, and task-by-factor undercoverage scores."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


def _factor_columns(frame):
    return tuple(
        sorted(
            column
            for column in frame
            if column.startswith("factor_")
            and not column.startswith("factor_available_")
        )
    )


def _cell(value):
    return "__missing__" if pd.isna(value) else str(value)


def _rank01(values):
    series = pd.Series(np.asarray(values, dtype=float))
    if series.nunique() <= 1:
        return np.full(len(series), 0.5)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def coverage_scores(observed: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    factors = tuple(
        sorted(set(_factor_columns(observed)) | set(_factor_columns(candidates)))
    )
    marginal = Counter()
    pairwise = Counter()
    task_factor = Counter()
    for _, row in observed.iterrows():
        values = [(factor, _cell(row.get(factor))) for factor in factors]
        marginal.update(values)
        task_factor.update(
            (str(row.task_family), factor, value) for factor, value in values
        )
        pairwise.update(
            (left, lv, right, rv) for (left, lv), (right, rv) in combinations(values, 2)
        )
    rows = []
    for _, row in candidates.iterrows():
        values = [(factor, _cell(row.get(factor))) for factor in factors]
        marginal_score = np.mean(
            [1.0 / np.sqrt(1.0 + marginal[value]) for value in values]
        )
        task_score = np.mean(
            [
                1.0 / np.sqrt(1.0 + task_factor[(str(row.task_family), factor, value)])
                for factor, value in values
            ]
        )
        pairs = [
            (left, lv, right, rv) for (left, lv), (right, rv) in combinations(values, 2)
        ]
        pair_score = np.mean([1.0 / np.sqrt(1.0 + pairwise[value]) for value in pairs])
        rows.append((marginal_score, pair_score, task_score))
    result = candidates[["paired_condition_id", "semantic_hash", "task_family"]].copy()
    result["coverage_marginal"] = [row[0] for row in rows]
    result["coverage_pairwise"] = [row[1] for row in rows]
    result["coverage_task_factor"] = [row[2] for row in rows]
    raw = result[
        ["coverage_marginal", "coverage_pairwise", "coverage_task_factor"]
    ].mean(axis=1)
    result["coverage_score"] = _rank01(raw)
    return result

"""Task-macro marginal and prespecified second-order behavioral effects."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from cognitive_discovery.models.features import FeatureEncoder


def marginal_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    factor_columns = sorted(
        column
        for column in frame
        if column.startswith("factor_") and not column.startswith("factor_available_")
    )
    for factor in factor_columns:
        available = frame[frame[factor].notna()]
        for level, part in available.groupby(factor, sort=True):
            task_means = part.groupby("task_family").persistence_logit.mean()
            centered = []
            for task, value in task_means.items():
                baseline = available[available.task_family == task].persistence_logit.mean()
                centered.append(float(value - baseline))
            rows.append(
                {
                    "factor": factor.removeprefix("factor_"),
                    "level": level,
                    "task_macro_effect": float(np.mean(centered)),
                    "task_macro_absolute_effect": float(np.mean(np.abs(centered))),
                    "tasks": int(part.task_family.nunique()),
                    "observations": len(part),
                }
            )
    return pd.DataFrame(rows).sort_values(
        "task_macro_absolute_effect", ascending=False
    ).reset_index(drop=True)


def interaction_effects(frame: pd.DataFrame, *, maximum_factors: int = 10) -> pd.DataFrame:
    factors = sorted(
        column
        for column in frame
        if column.startswith("factor_") and not column.startswith("factor_available_")
    )[:maximum_factors]
    if len(factors) < 2:
        return pd.DataFrame(columns=["interaction", "coefficient", "absolute_coefficient"])
    candidates = tuple(f"{left}*{right}" for left, right in combinations(factors, 2))
    encoder = FeatureEncoder((*factors, *candidates))
    matrix, names = encoder.fit_transform(frame)
    from sklearn.linear_model import RidgeCV

    fit = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0), cv=5).fit(
        matrix, frame.persistence_logit
    )
    rows = []
    for name, coefficient in zip(names, fit.coef_):
        if "*" not in name or name.endswith("__available"):
            continue
        rows.append(
            {
                "interaction": name,
                "coefficient": float(coefficient),
                "absolute_coefficient": abs(float(coefficient)),
            }
        )
    return pd.DataFrame(rows).sort_values("absolute_coefficient", ascending=False)


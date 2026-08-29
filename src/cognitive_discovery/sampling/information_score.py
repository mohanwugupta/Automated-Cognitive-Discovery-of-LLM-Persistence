"""Bootstrap parameter-uncertainty score for legal candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model


def _rank01(values):
    series = pd.Series(np.asarray(values, dtype=float))
    if series.nunique() <= 1:
        return np.full(len(series), 0.5)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def information_scores(
    observed: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    architecture: str = "dual_history",
    bootstraps: int = 32,
    seed: int = 0,
    ensemble_predictions=None,
) -> pd.DataFrame:
    predictions = [] if ensemble_predictions is None else list(ensemble_predictions)
    if ensemble_predictions is None:
        group_column = (
            "paired_condition_id" if "paired_condition_id" in observed else "episode_id"
        )
        groups = np.asarray(sorted(observed[group_column].astype(str).unique()))
        rng = np.random.default_rng(int(seed))
        for _ in range(int(bootstraps)):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            pieces = [
                observed[observed[group_column].astype(str) == group]
                for group in sampled
            ]
            bootstrap = pd.concat(pieces, ignore_index=True)
            fit = fit_hierarchical_model(
                bootstrap, architecture, variant="M4", alphas=(1.0, 10.0)
            )
            predictions.append(fit.predict(candidates, include_random=True))
    variance = (
        np.var(np.vstack(predictions), axis=0, ddof=1)
        if len(predictions) > 1
        else np.zeros(len(candidates))
    )
    result = candidates[["paired_condition_id", "semantic_hash", "task_family"]].copy()
    result["information_variance"] = variance
    result["information_score"] = _rank01(variance)
    return result

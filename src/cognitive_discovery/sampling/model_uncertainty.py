"""Bootstrap frozen-model uncertainty over candidate predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model

from .theory_disagreement import assert_outcome_free_candidates


def _rank01(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    if series.nunique() <= 1:
        return np.full(len(series), 0.5)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def model_uncertainty_scores(
    observed: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    architectures: tuple[str, ...],
    variant: str = "M4",
    bootstraps: int = 32,
    seed: int = 0,
    ensemble_predictions: dict[str, list[np.ndarray]] | None = None,
) -> pd.DataFrame:
    assert_outcome_free_candidates(candidates)
    ensembles = {name: [] for name in architectures}
    if ensemble_predictions is not None:
        ensembles = {name: list(ensemble_predictions[name]) for name in architectures}
    else:
        group = "paired_condition_id" if "paired_condition_id" in observed else "episode_id"
        groups = np.asarray(sorted(observed[group].astype(str).unique()))
        rng = np.random.default_rng(int(seed))
        for _ in range(int(bootstraps)):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            bootstrap = pd.concat(
                [observed[observed[group].astype(str) == value] for value in sampled],
                ignore_index=True,
            )
            for architecture in architectures:
                fit = fit_hierarchical_model(
                    bootstrap,
                    architecture,
                    variant=variant,
                    alphas=(1.0, 10.0),
                )
                ensembles[architecture].append(fit.predict(candidates))
    per_model = []
    result = candidates[["paired_condition_id", "semantic_hash", "task_family"]].copy()
    for architecture in architectures:
        values = ensembles[architecture]
        variance = (
            np.var(np.vstack(values), axis=0, ddof=1)
            if len(values) > 1
            else np.zeros(len(candidates))
        )
        result[f"uncertainty_variance_{architecture}"] = variance
        per_model.append(variance)
    mean_variance = np.mean(np.column_stack(per_model), axis=1)
    result["uncertainty_raw"] = mean_variance
    result["uncertainty_score"] = _rank01(mean_variance)
    return result

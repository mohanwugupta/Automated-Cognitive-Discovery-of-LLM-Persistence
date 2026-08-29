"""Leakage-safe zero/few-shot task adaptation with frozen population terms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from cognitive_discovery.models.fitting import regression_metrics

from .random_effects import fit_hierarchical_model


def _semantic_groups(frame: pd.DataFrame) -> np.ndarray:
    if "paired_condition_id" in frame:
        return frame.paired_condition_id.astype(str).to_numpy()
    if "episode_id" in frame:
        return frame.episode_id.astype(str).to_numpy()
    return np.asarray([f"row-{index}" for index in range(len(frame))])


def evaluate_few_shot_adaptation(
    frame: pd.DataFrame,
    *,
    architecture: str,
    sample_sizes=(0, 1, 4, 8, 16, 32, 64),
    seed: int = 0,
    variant: str = "M4",
    adaptation_alpha: float = 10.0,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
) -> pd.DataFrame:
    """Fit population terms on source tasks and only ``u_t`` on target shots."""

    rows = []
    for heldout in sorted(frame.task_family.astype(str).unique()):
        source = frame[frame.task_family.astype(str) != heldout].reset_index(drop=True)
        target = frame[frame.task_family.astype(str) == heldout].reset_index(drop=True)
        population = fit_hierarchical_model(
            source, architecture, variant=variant, alphas=alphas
        )
        if heldout in population.outcome_tasks:
            raise RuntimeError("held-out task outcomes leaked into population fit")
        base = population.predict(target, include_random=False)
        x = population.feature_matrix(target)
        groups = _semantic_groups(target)
        unique = np.asarray(sorted(set(groups)))
        rng = np.random.default_rng(int(seed) + sum(map(ord, heldout)))
        rng.shuffle(unique)
        for requested in sample_sizes:
            n = min(int(requested), len(unique))
            adaptation_groups = set(unique[:n])
            adaptation = np.asarray([value in adaptation_groups for value in groups])
            evaluation = ~adaptation
            if not evaluation.any():
                evaluation = np.ones(len(target), dtype=bool)
            if n:
                residual = target.persistence_logit.to_numpy(dtype=float) - base
                effect = (
                    Ridge(alpha=float(adaptation_alpha), fit_intercept=False)
                    .fit(x[adaptation], residual[adaptation])
                    .coef_
                )
            else:
                effect = np.zeros(x.shape[1], dtype=float)
            prediction = base + x @ effect
            metrics = regression_metrics(
                target.persistence_logit.to_numpy(dtype=float)[evaluation],
                prediction[evaluation],
            )
            rows.append(
                {
                    "architecture": architecture,
                    "variant": variant,
                    "heldout_task": heldout,
                    "requested_semantic_conditions": int(requested),
                    "adaptation_semantic_conditions": n,
                    "adaptation_rows": int(adaptation.sum()),
                    "evaluation_rows": int(evaluation.sum()),
                    "population_outcome_tasks": ";".join(population.outcome_tasks),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)

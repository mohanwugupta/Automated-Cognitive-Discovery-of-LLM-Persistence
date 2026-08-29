"""Bootstrap partial-pooling intervals for task-specific parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model


def bootstrap_parameter_intervals(
    frame: pd.DataFrame,
    *,
    architecture: str,
    bootstraps: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    group_column = (
        "paired_condition_id" if "paired_condition_id" in frame else "episode_id"
    )
    groups = np.asarray(sorted(frame[group_column].astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    estimates = []
    for iteration in range(int(bootstraps)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pieces = [frame[frame[group_column].astype(str) == group] for group in sampled]
        bootstrap = pd.concat(pieces, ignore_index=True)
        fit = fit_hierarchical_model(bootstrap, architecture, variant="M4")
        part = fit.task_parameters()
        part["bootstrap"] = iteration
        estimates.append(part)
    values = pd.concat(estimates, ignore_index=True)
    rows = []
    for keys, part in values.groupby(["architecture", "task_family", "parameter"]):
        estimate = part.estimate.to_numpy(dtype=float)
        rows.append(
            {
                "architecture": keys[0],
                "task_family": keys[1],
                "parameter": keys[2],
                "estimate_mean": float(np.mean(estimate)),
                "interval_low": float(np.quantile(estimate, 0.025)),
                "interval_high": float(np.quantile(estimate, 0.975)),
                "probability_positive": float(np.mean(estimate > 0)),
            }
        )
    return pd.DataFrame(rows)

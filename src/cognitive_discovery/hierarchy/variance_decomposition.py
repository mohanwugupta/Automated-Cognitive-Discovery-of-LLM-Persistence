"""Task-parameter variance and sign-consistency summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def variance_decomposition(fit) -> pd.DataFrame:
    if fit.variant != "M4":
        raise ValueError("ontology variance decomposition requires an M4 fit")
    parameters = fit.task_parameters()
    rows = []
    for name, part in parameters.groupby("parameter", sort=False):
        ontology = part.ontology_effect.to_numpy(dtype=float)
        total = part.estimate.to_numpy(dtype=float)
        random = part.random_effect.to_numpy(dtype=float)
        total_variance = float(np.var(total))
        ontology_variance = float(np.var(ontology))
        rows.append(
            {
                "architecture": fit.architecture,
                "parameter": name,
                "ontology_variance": ontology_variance,
                "random_effect_variance": float(np.var(random)),
                "total_task_variance": total_variance,
                # Intentionally uncapped: covariance can make this exceed one.
                "ontology_fraction_q": (
                    ontology_variance / total_variance
                    if total_variance > 1e-12
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def parameter_sign_consistency(parameters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (architecture, name), part in parameters.groupby(
        ["architecture", "parameter"], sort=False
    ):
        values = part.estimate.to_numpy(dtype=float)
        rows.append(
            {
                "architecture": architecture,
                "parameter": name,
                "tasks": len(values),
                "positive_fraction": float(np.mean(values > 0)),
                "negative_fraction": float(np.mean(values < 0)),
                "sign_consistent": bool(np.all(values > 0) or np.all(values < 0)),
            }
        )
    return pd.DataFrame(rows)

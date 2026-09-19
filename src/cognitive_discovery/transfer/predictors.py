"""Preregistered, held-out-target prediction of directed causal transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cognitive_discovery.hierarchy.task_descriptors import DESCRIPTOR_NAMES, TASK_DESCRIPTORS


PREDICTOR_FAMILIES = {
    "behavioral": ("behavioral_similarity",),
    "computational": ("same_theory", "parameter_similarity"),
    "ontology": ("ontology_similarity",),
    "counterfactual": ("counterfactual_similarity",),
    "neural": ("subspace_overlap", "depth_difference", "rank_difference"),
}


def _cosine(left, right) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 1e-12 else 0.0


def build_task_pair_predictors(
    *, model: str, theory: str, tasks, behavioral_profiles: pd.DataFrame,
    parameter_profiles: pd.DataFrame, counterfactuals: pd.DataFrame,
    controller_registry: pd.DataFrame, geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the preregistered compact predictor families before modeling."""

    def profiles(frame, value_column):
        return frame.pivot(index="task_family", columns="feature", values=value_column).fillna(0.0)
    behavior = profiles(behavioral_profiles, "value")
    parameters = profiles(parameter_profiles, "value")
    cf = counterfactuals.groupby("task_family").predicted_counterfactual_effect.agg(
        mean="mean", sd="std", q25=lambda value: value.quantile(.25),
        median="median", q75=lambda value: value.quantile(.75),
    ).fillna(0.0)
    controllers = controller_registry.set_index("task_family")
    geometry_lookup = {}
    for row in geometry.itertuples():
        geometry_lookup[(str(row.source_task_i), str(row.source_task_j))] = float(row.subspace_overlap)
        geometry_lookup[(str(row.source_task_j), str(row.source_task_i))] = float(row.subspace_overlap)
    rows = []
    for source in tasks:
        for target in tasks:
            ontology_source = [TASK_DESCRIPTORS[source][name] for name in DESCRIPTOR_NAMES]
            ontology_target = [TASK_DESCRIPTORS[target][name] for name in DESCRIPTOR_NAMES]
            rows.append({
                "model": model, "theory": theory, "source_task": source, "target_task": target,
                "behavioral_similarity": _cosine(behavior.loc[source], behavior.loc[target]),
                "same_theory": 1.0,
                "parameter_similarity": _cosine(parameters.loc[source], parameters.loc[target]),
                "ontology_similarity": _cosine(ontology_source, ontology_target),
                "counterfactual_similarity": _cosine(cf.loc[source], cf.loc[target]),
                "subspace_overlap": 1.0 if source == target else geometry_lookup.get((source, target), np.nan),
                "depth_difference": abs(float(controllers.loc[source, "relative_depth"]) - float(controllers.loc[target, "relative_depth"])),
                "rank_difference": abs(float(controllers.loc[source, "rank"]) - float(controllers.loc[target, "rank"])),
            })
    return pd.DataFrame(rows)


def predictor_columns(frame, families) -> list[str]:
    columns = [column for family in families for column in PREDICTOR_FAMILIES[family]]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"frozen transfer predictors are missing: {sorted(missing)}")
    return columns


def fit_heldout_target_predictor(frame: pd.DataFrame, *, families=("behavioral", "computational", "ontology", "counterfactual", "neural"), alpha=10.0):
    """Evaluate one preregistered ridge model with leave-one-target-task CV."""

    required = {"source_task", "target_task", "global_cfr"}
    if required - set(frame.columns):
        raise ValueError("transfer predictor table lacks source/target/global_cfr")
    columns = predictor_columns(frame, families)
    rows, coefficients = [], []
    for heldout in sorted(frame.target_task.astype(str).unique()):
        train = frame[frame.target_task.astype(str) != heldout]
        test = frame[frame.target_task.astype(str) == heldout]
        if train.empty or test.empty:
            continue
        scaler = StandardScaler().fit(train[columns])
        model = Ridge(alpha=float(alpha)).fit(scaler.transform(train[columns]), train.global_cfr)
        prediction = model.predict(scaler.transform(test[columns]))
        for row, predicted in zip(test.itertuples(), prediction):
            rows.append({"source_task": row.source_task, "target_task": row.target_task, "observed_global_cfr": row.global_cfr, "predicted_global_cfr": float(predicted), "heldout_target": heldout})
        for name, value in zip(columns, model.coef_):
            coefficients.append({"heldout_target": heldout, "predictor": name, "coefficient": float(value), "alpha": float(alpha)})
    return pd.DataFrame(rows), pd.DataFrame(coefficients)

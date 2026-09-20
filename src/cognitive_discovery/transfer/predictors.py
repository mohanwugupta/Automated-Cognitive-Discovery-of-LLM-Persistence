"""Preregistered, held-out-target prediction of directed causal transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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

FROZEN_FAMILIES = (
    "behavioral_computational", "task_structure", "counterfactual_geometry",
    "neural_similarity",
)
FORBIDDEN_OUTCOME_TOKENS = {
    "global_cfr", "bootstrap_low", "bootstrap_high", "transfer_status",
    "source_valid", "cfr",
}


def predictor_spec_hash(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_predictor_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != "transfer-predictors-v1":
        raise ValueError("unsupported frozen transfer-predictor specification")
    if spec.get("frozen_before_off_diagonal") is not True:
        raise ValueError("transfer predictors must be frozen before off-diagonal execution")
    if set(spec.get("families", {})) != set(FROZEN_FAMILIES):
        raise ValueError("the four frozen transfer-predictor families are required")
    serialized = json.dumps(spec["families"], sort_keys=True).lower()
    for token in FORBIDDEN_OUTCOME_TOKENS:
        if token in serialized:
            raise ValueError(f"transfer predictor outcome leakage: {token}")
    structure = spec.get("task_structure", {})
    if not structure.get("binary_descriptors") or len(structure.get("tasks", {})) != 7:
        raise ValueError("the seven-task structure ontology must be explicit")
    required = set(structure["binary_descriptors"]) | {"temporal_horizon", "choice_structure"}
    for task, values in structure["tasks"].items():
        if required - set(values):
            raise ValueError(f"task structure is incomplete for {task}")
        if values["temporal_horizon"] not in structure["temporal_horizon_levels"]:
            raise ValueError(f"invalid temporal horizon for {task}")
        if values["choice_structure"] not in structure["choice_structure_levels"]:
            raise ValueError(f"invalid choice structure for {task}")
    return spec


def build_task_structure_predictors(spec: dict) -> pd.DataFrame:
    """Deterministically encode directed task pairs without transfer outcomes."""

    structure = spec["task_structure"]
    tasks = structure["tasks"]
    descriptors = structure["binary_descriptors"]
    horizons = structure["temporal_horizon_levels"]
    rows = []
    for source in sorted(tasks):
        for target in sorted(tasks):
            left, right = tasks[source], tasks[target]
            matches = [float(left[name] == right[name]) for name in descriptors]
            rows.append({
                "source_task": source,
                "target_task": target,
                "binary_descriptor_similarity": float(np.mean(matches)),
                "temporal_horizon_similarity": 1.0 - abs(
                    float(horizons[left["temporal_horizon"]])
                    - float(horizons[right["temporal_horizon"]])
                ) / 2.0,
                "choice_structure_match": float(
                    left["choice_structure"] == right["choice_structure"]
                ),
            })
    return pd.DataFrame(rows)


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

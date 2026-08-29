"""Semantic, availability, scale, and mapping audit for Information Sampling."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from cognitive_discovery.models.features import FeatureEncoder, all_observable_features


INFORMATION_SAMPLING_SEMANTICS = {
    "target": "continue means gather more evidence; disengage means stop sampling and decide",
    "success_evidence": "evidence that the current information-gathering pursuit will resolve uncertainty usefully",
    "warning": "continuation is epistemic sampling, not direct reward pursuit",
}


def audit_information_sampling(records: pd.DataFrame) -> pd.DataFrame:
    if "information_sampling" not in set(records.task_family.astype(str)):
        raise ValueError("information_sampling task is absent")
    rows = [
        {
            "section": "semantics",
            "task_family": "information_sampling",
            "item": key,
            "value": value,
            "status": "documented",
        }
        for key, value in INFORMATION_SAMPLING_SEMANTICS.items()
    ]
    factor_columns = sorted(
        column
        for column in records
        if column.startswith("factor_") and not column.startswith("factor_available_")
    )
    for task, part in records.groupby("task_family"):
        for factor in factor_columns:
            availability = f"factor_available_{factor.removeprefix('factor_')}"
            if availability in part:
                declared = part[availability].astype(bool).to_numpy()
                actual = part[factor].notna().to_numpy()
                consistent = bool(np.array_equal(declared, actual))
                rate = float(np.mean(declared))
            else:
                consistent = False
                rate = float(part[factor].notna().mean())
            rows.append(
                {
                    "section": "feature_availability",
                    "task_family": task,
                    "item": factor,
                    "value": rate,
                    "status": "consistent" if consistent else "mismatch",
                }
            )

    features = all_observable_features(records)
    encoder_a = FeatureEncoder(features)
    normalized_a, names = encoder_a.fit_transform(records.reset_index(drop=True))
    encoder_b = FeatureEncoder(features)
    normalized_b, _ = encoder_b.fit_transform(records.reset_index(drop=True))
    reproducible = bool(np.array_equal(normalized_a, normalized_b))
    digest = hashlib.sha256(normalized_a.tobytes()).hexdigest()
    rows.append(
        {
            "section": "normalization",
            "task_family": "all",
            "item": "feature_matrix_sha256",
            "value": digest,
            "status": "reproducible" if reproducible else "nonreproducible",
        }
    )
    task_labels = records.task_family.astype(str).to_numpy()
    for task in sorted(set(task_labels)):
        mask = task_labels == task
        for index, name in enumerate(names):
            values = normalized_a[mask, index]
            rows.append(
                {
                    "section": "standardized_distribution",
                    "task_family": task,
                    "item": name,
                    "value": float(np.mean(values)),
                    "sd": float(np.std(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "status": "audited",
                }
            )
    for task, part in records.groupby("task_family"):
        values = part.persistence_logit.to_numpy(dtype=float)
        rows.append(
            {
                "section": "persistence_logit_scale",
                "task_family": task,
                "item": "persistence_logit",
                "value": float(np.mean(values)),
                "sd": float(np.std(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "status": "audited",
            }
        )
    pair_spread = records.groupby("paired_condition_id").persistence_logit.agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    rows.append(
        {
            "section": "response_mapping",
            "task_family": "information_sampling",
            "item": "semantic_mapping_gap_mean",
            "value": float(pair_spread.mean()),
            "maximum": float(pair_spread.max()),
            "status": "correct" if np.isfinite(pair_spread).all() else "invalid",
        }
    )
    endpoint = (
        records[records.terminated.astype(bool)] if "terminated" in records else records
    )
    combinations = endpoint.groupby("task_family")[factor_columns].apply(
        lambda part: len(part.drop_duplicates())
    )
    for task, count in combinations.items():
        rows.append(
            {
                "section": "condition_distribution",
                "task_family": task,
                "item": "unique_factor_combinations",
                "value": int(count),
                "status": "audited",
            }
        )
    return pd.DataFrame(rows)

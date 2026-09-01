"""Semantic-pair clustered bootstrap utilities for stable CFR inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics


PRIMARY_METRICS = (
    "global_cfr",
    "correlation",
    "slope",
    "intercept",
    "rmse",
    "sign_accuracy",
)


def _cluster_samples(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    strata_column: str | None,
    samples: int,
    seed: int,
):
    if cluster_column not in frame:
        raise ValueError(f"bootstrap cluster column is absent: {cluster_column}")
    strata = [None] if strata_column is None else sorted(frame[strata_column].unique())
    rng = np.random.default_rng(int(seed))
    grouped = {}
    for stratum in strata:
        part = frame if stratum is None else frame[frame[strata_column] == stratum]
        clusters = sorted(part[cluster_column].astype(str).unique())
        if not clusters:
            continue
        grouped[stratum] = {
            cluster: part.index[part[cluster_column].astype(str) == cluster].to_numpy()
            for cluster in clusters
        }
    for _ in range(int(samples)):
        indices = []
        for mapping in grouped.values():
            clusters = np.asarray(list(mapping), dtype=object)
            chosen = rng.choice(clusters, size=len(clusters), replace=True)
            indices.extend(index for cluster in chosen for index in mapping[cluster])
        yield np.asarray(indices, dtype=int)


def _metric_row(frame, *, predicted_column, observed_column, threshold):
    return counterfactual_metrics(
        frame[predicted_column].to_numpy(dtype=float),
        frame[observed_column].to_numpy(dtype=float),
        threshold=threshold,
    )


def bootstrap_metric_intervals(
    frame: pd.DataFrame,
    *,
    predicted_column: str = "predicted_counterfactual_effect",
    observed_column: str = "neural_counterfactual_effect",
    cluster_column: str = "contrast_id",
    strata_column: str | None = "task_family",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    threshold: float = 0.10,
) -> tuple[dict, list[dict]]:
    """Return point metrics and percentile CIs resampled by semantic pair."""

    if frame.empty:
        raise ValueError("bootstrap requires at least one counterfactual row")
    point = _metric_row(
        frame,
        predicted_column=predicted_column,
        observed_column=observed_column,
        threshold=threshold,
    )
    draws = {metric: [] for metric in PRIMARY_METRICS}
    reset = frame.reset_index(drop=True)
    for indices in _cluster_samples(
        reset,
        cluster_column=cluster_column,
        strata_column=strata_column,
        samples=samples,
        seed=seed,
    ):
        metrics = _metric_row(
            reset.iloc[indices],
            predicted_column=predicted_column,
            observed_column=observed_column,
            threshold=threshold,
        )
        for metric in PRIMARY_METRICS:
            if np.isfinite(metrics[metric]):
                draws[metric].append(float(metrics[metric]))
    alpha = (1.0 - float(confidence)) / 2.0
    intervals = []
    for metric in PRIMARY_METRICS:
        values = np.asarray(draws[metric], dtype=float)
        intervals.append(
            {
                "metric": metric,
                "estimate": float(point[metric]),
                "ci_lower": (
                    float(np.quantile(values, alpha)) if len(values) else np.nan
                ),
                "ci_upper": (
                    float(np.quantile(values, 1.0 - alpha)) if len(values) else np.nan
                ),
                "bootstrap_valid": int(len(values)),
                "bootstrap_samples": int(samples),
                "confidence": float(confidence),
                "resampling_unit": cluster_column,
                "stratified_by": strata_column,
            }
        )
    return point, intervals


def paired_bootstrap_difference(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
    *,
    metric: str = "global_cfr",
    pair_column: str = "pair_id",
    predicted_column: str = "predicted_counterfactual_effect",
    observed_column: str = "neural_counterfactual_effect",
    cluster_column: str = "contrast_id",
    strata_column: str | None = "task_family",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    threshold: float = 0.10,
) -> dict:
    """Candidate-minus-control CI using identical rows and cluster resamples."""

    keys = [pair_column, cluster_column]
    if strata_column is not None:
        keys.append(strata_column)
    left = candidate[[*keys, predicted_column, observed_column]].rename(
        columns={observed_column: "candidate_observed"}
    )
    right = control[[pair_column, observed_column]].rename(
        columns={observed_column: "control_observed"}
    )
    merged = left.merge(right, on=pair_column, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(control):
        raise ValueError("candidate and control must use identical test rows")

    def difference(part):
        candidate_metrics = counterfactual_metrics(
            part[predicted_column], part.candidate_observed, threshold=threshold
        )
        control_metrics = counterfactual_metrics(
            part[predicted_column], part.control_observed, threshold=threshold
        )
        return float(candidate_metrics[metric] - control_metrics[metric])

    estimate = difference(merged)
    values = []
    reset = merged.reset_index(drop=True)
    for indices in _cluster_samples(
        reset,
        cluster_column=cluster_column,
        strata_column=strata_column,
        samples=samples,
        seed=seed,
    ):
        value = difference(reset.iloc[indices])
        if np.isfinite(value):
            values.append(value)
    values = np.asarray(values, dtype=float)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "metric": metric,
        "estimate_difference": estimate,
        "ci_lower": float(np.quantile(values, alpha)) if len(values) else np.nan,
        "ci_upper": (
            float(np.quantile(values, 1.0 - alpha)) if len(values) else np.nan
        ),
        "bootstrap_valid": int(len(values)),
        "bootstrap_samples": int(samples),
        "rows": int(len(merged)),
        "row_identity_verified": True,
    }


def bootstrap_vsi(
    matched: pd.DataFrame,
    unrelated: list[pd.DataFrame],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    threshold: float = 0.10,
) -> dict:
    """Bootstrap CFR_G(S_X→X) minus the strongest S_X→Z cell."""

    if not unrelated:
        raise ValueError("VSI requires at least one unrelated-variable target")
    frames = [matched, *unrelated]
    point_values = [
        counterfactual_metrics(
            frame.predicted_counterfactual_effect,
            frame.neural_counterfactual_effect,
            threshold=threshold,
        )["global_cfr"]
        for frame in frames
    ]
    estimate = float(point_values[0] - np.nanmax(point_values[1:]))
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(samples)):
        values = []
        for frame in frames:
            clusters = frame[["contrast_id", "task_family"]].drop_duplicates()
            pieces = []
            for task, task_clusters in clusters.groupby("task_family"):
                ids = task_clusters.contrast_id.astype(str).to_numpy()
                chosen = rng.choice(ids, size=len(ids), replace=True)
                task_frame = frame[frame.task_family == task]
                pieces.extend(
                    task_frame[task_frame.contrast_id.astype(str) == cluster]
                    for cluster in chosen
                )
            sampled = pd.concat(pieces, ignore_index=True)
            values.append(
                counterfactual_metrics(
                    sampled.predicted_counterfactual_effect,
                    sampled.neural_counterfactual_effect,
                    threshold=threshold,
                )["global_cfr"]
            )
        value = values[0] - np.nanmax(values[1:])
        if np.isfinite(value):
            draws.append(float(value))
    draws = np.asarray(draws, dtype=float)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "vsi": estimate,
        "ci_lower": float(np.quantile(draws, alpha)) if len(draws) else np.nan,
        "ci_upper": (float(np.quantile(draws, 1.0 - alpha)) if len(draws) else np.nan),
        "bootstrap_valid": int(len(draws)),
        "bootstrap_samples": int(samples),
    }

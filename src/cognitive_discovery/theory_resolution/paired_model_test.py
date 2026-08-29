"""Stratified paired prediction-error tests on untouched Round-3 conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitive_discovery.models.fitting import regression_metrics


def semantic_prediction_errors(
    observations: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    if any(len(values) != len(observations) for values in predictions.values()):
        raise ValueError("frozen prediction length does not match observations")
    scored = observations.copy().reset_index(drop=True)
    for architecture, values in predictions.items():
        scored[f"prediction_{architecture}"] = np.asarray(values, dtype=float)
    aggregations = {
        "task_family": "first",
        "split": "first",
        "sampling_strategy": "first",
        "persistence_logit": "mean",
    }
    for column in scored:
        if column.startswith("context_"):
            aggregations[column] = "first"
    for architecture in predictions:
        aggregations[f"prediction_{architecture}"] = "mean"
    semantic = scored.groupby("paired_condition_id", as_index=False).agg(aggregations)
    for architecture in predictions:
        semantic[f"squared_error_{architecture}"] = (
            semantic.persistence_logit - semantic[f"prediction_{architecture}"]
        ) ** 2
    if {"dual_history", "latent_context"} <= set(predictions):
        semantic["delta_error_dh_minus_lc"] = (
            semantic.squared_error_dual_history
            - semantic.squared_error_latent_context
        )
    semantic["contextual_history"] = semantic.get(
        "context_phase_order", pd.Series([np.nan] * len(semantic))
    ).notna()
    return semantic


def _scope_masks(frame: pd.DataFrame, high_disagreement_ids: set[str]):
    yield "overall", "all", np.ones(len(frame), dtype=bool)
    for task in sorted(frame.task_family.astype(str).unique()):
        yield "task", task, frame.task_family.astype(str).to_numpy() == task
    yield "subset", "contextual_history", frame.contextual_history.astype(bool).to_numpy()
    yield (
        "subset",
        "high_disagreement",
        frame.paired_condition_id.astype(str).isin(high_disagreement_ids).to_numpy(),
    )
    yield (
        "subset",
        "random_reference",
        (frame.sampling_strategy.astype(str) == "random").to_numpy(),
    )


def compare_frozen_theories(
    errors: pd.DataFrame,
    *,
    high_disagreement_ids: set[str] | None = None,
    bootstraps: int = 2_000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "delta_error_dh_minus_lc" not in errors:
        raise ValueError("paired dual-history/latent-context errors are absent")
    if "split" in errors and set(errors.split.astype(str)) != {"model_discrimination"}:
        raise ValueError("frozen comparison may use only the untouched discrimination split")
    high_disagreement_ids = set(high_disagreement_ids or ())
    rng = np.random.default_rng(int(seed))
    draws, summaries = [], []
    for scope, group, mask in _scope_masks(errors, high_disagreement_ids):
        part = errors[mask].reset_index(drop=True)
        if part.empty:
            continue
        task_indices = {
            task: np.flatnonzero(part.task_family.astype(str).to_numpy() == task)
            for task in sorted(part.task_family.astype(str).unique())
        }
        values = []
        for iteration in range(int(bootstraps)):
            indices = np.concatenate(
                [rng.choice(index, size=len(index), replace=True) for index in task_indices.values()]
            )
            value = float(part.delta_error_dh_minus_lc.to_numpy()[indices].mean())
            values.append(value)
            draws.append(
                {
                    "scope": scope,
                    "group": group,
                    "bootstrap": iteration,
                    "mean_delta_error_dh_minus_lc": value,
                }
            )
        values = np.asarray(values, dtype=float)
        summaries.append(
            {
                "scope": scope,
                "group": group,
                "rows": len(part),
                "mean_delta_error_dh_minus_lc": float(
                    part.delta_error_dh_minus_lc.mean()
                ),
                "median_delta_error_dh_minus_lc": float(
                    part.delta_error_dh_minus_lc.median()
                ),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "probability_latent_context_better": float(np.mean(values > 0)),
                "probability_dual_history_better": float(np.mean(values < 0)),
            }
        )
    metrics = []
    for architecture in ("dual_history", "latent_context"):
        prediction = errors[f"prediction_{architecture}"]
        metrics.append(
            {
                "architecture": architecture,
                **regression_metrics(errors.persistence_logit, prediction),
                "rmse": float(
                    np.sqrt(np.mean(errors[f"squared_error_{architecture}"]))
                ),
            }
        )
    return pd.DataFrame(draws), pd.DataFrame(summaries), pd.DataFrame(metrics)


def context_reinstatement_table(errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matched behavioral contrasts and their cue-reliability summary."""

    required = {
        "context_critical_contrast_id",
        "context_a_history_valence",
        "context_b_history_valence",
        "context_context_return",
        "context_cue_reliability",
    }
    if not required <= set(errors):
        return pd.DataFrame(), pd.DataFrame()
    values = {"negative": -1.0, "mixed": 0.0, "positive": 1.0}
    rows = []
    contextual = errors[errors.contextual_history.astype(bool)]
    for contrast_id, part in contextual.groupby("context_critical_contrast_id"):
        if len(part) != 2:
            continue
        context_return = str(part.context_context_return.iloc[0])
        matched_column = (
            "context_a_history_valence"
            if context_return == "A"
            else "context_b_history_valence"
            if context_return == "B"
            else None
        )
        if matched_column is None:
            effect = float("nan")
        else:
            x = part[matched_column].map(values).to_numpy(dtype=float)
            y = part.persistence_logit.to_numpy(dtype=float)
            difference = x[1] - x[0]
            effect = float((y[1] - y[0]) / difference) if abs(difference) > 0 else float("nan")
        rows.append(
            {
                "critical_contrast_id": contrast_id,
                "task_family": part.task_family.iloc[0],
                "context_return": context_return,
                "cue_reliability": part.context_cue_reliability.iloc[0],
                "cue_probability": part.context_cue_probability.iloc[0],
                "change_point": part.context_change_point.iloc[0],
                "reinstatement_slope": effect,
            }
        )
    contrasts = pd.DataFrame(rows)
    if contrasts.empty:
        return contrasts, contrasts
    summary = (
        contrasts.groupby(
            ["task_family", "context_return", "cue_reliability", "change_point"],
            as_index=False,
            dropna=False,
        )
        .reinstatement_slope.agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_reinstatement", "median": "median_reinstatement"})
    )
    return contrasts, summary

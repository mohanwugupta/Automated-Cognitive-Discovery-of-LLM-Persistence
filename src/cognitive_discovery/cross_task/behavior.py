"""Behavioral causal-effect map B[a,t,v]."""

from __future__ import annotations

import numpy as np
import pandas as pd
import json


LEVEL_SCORE = {
    "negative": -1.0, "low": -1.0, "disengage": -1.0,
    "neutral": 0.0, "medium": 0.0, "mixed": 0.0,
    "positive": 1.0, "high": 1.0, "continue": 1.0,
    "changing": -1.0, "stable": 1.0, "new": -1.0, "same": 1.0,
}

FACTOR_COLUMNS = {"progress": "factor_progress_evidence"}


def semantic_persistence_logit(p_continue, p_disengage, *, epsilon: float = 1e-12):
    p_continue = np.clip(np.asarray(p_continue, dtype=float), epsilon, 1.0)
    p_disengage = np.clip(np.asarray(p_disengage, dtype=float), epsilon, 1.0)
    result = np.log(p_continue) - np.log(p_disengage)
    return float(result) if result.ndim == 0 else result


def _effect(values: pd.DataFrame, factor: str) -> tuple[float, float, float]:
    raw = values[factor].dropna()
    numeric = pd.to_numeric(raw, errors="coerce")
    is_numeric = numeric.notna().all()
    levels = list((numeric if is_numeric else raw.astype(str)).unique())
    if len(levels) < 2:
        raise ValueError("causal contrast requires at least two observed levels")
    scores = {level: LEVEL_SCORE.get(level) for level in levels}
    if is_numeric:
        low, high = min(levels), max(levels)
    elif all(score is not None for score in scores.values()):
        low, high = min(levels, key=scores.get), max(levels, key=scores.get)
    else:
        low, high = sorted(levels)[0], sorted(levels)[-1]
    if is_numeric:
        converted = pd.to_numeric(values[factor], errors="coerce")
        low_values = values.loc[np.isclose(converted, low, equal_nan=False), "persistence_logit"]
        high_values = values.loc[np.isclose(converted, high, equal_nan=False), "persistence_logit"]
    else:
        low_values = values.loc[values[factor].astype(str) == str(low), "persistence_logit"]
        high_values = values.loc[values[factor].astype(str) == str(high), "persistence_logit"]
    pooled = pd.concat([low_values, high_values]).std(ddof=1)
    effect = float(high_values.mean() - low_values.mean())
    standardized = float(effect / pooled) if np.isfinite(pooled) and pooled > 0 else np.nan
    return effect, standardized, float((high_values.mean() > low_values.mean()))


def estimate_behavioral_effects(
    observations: pd.DataFrame,
    *,
    variables,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate high-minus-low experimental contrasts without selecting on outcomes."""

    required = {"model", "task_family", "response_mapping", "persistence_logit"}
    missing = required - set(observations)
    if missing:
        raise ValueError(f"behavior observations missing columns: {sorted(missing)}")
    if bootstrap_samples < 100:
        raise ValueError("behavioral uncertainty requires at least 100 bootstrap samples")
    observations = observations.copy()
    def sequence(value):
        if isinstance(value, str):
            return json.loads(value)
        return list(value)
    if "history_actions" in observations:
        observations["factor_action_history"] = observations.history_actions.map(
            lambda value: np.mean([
                1.0 if str(action) == "continue" else -1.0 for action in sequence(value)
            ]) if sequence(value) else np.nan
        )
    if "history_outcomes" in observations:
        observations["factor_outcome_history"] = observations.history_outcomes.map(
            lambda value: np.mean(list(map(float, sequence(value)))) if sequence(value) else np.nan
        )
    context_required = {
        "context_context_return", "context_a_history_outcomes", "context_b_history_outcomes"
    }
    if context_required <= set(observations):
        def contextual(row):
            key = str(row["context_context_return"])
            if key not in {"A", "B"}:
                return np.nan
            values = sequence(row[f"context_{key.lower()}_history_outcomes"])
            return np.mean(list(map(float, values))) if values else np.nan
        observations["factor_contextual_history"] = observations.apply(contextual, axis=1)
    rows = []
    for (model, task), part in observations.groupby(["model", "task_family"], sort=True):
        for variable in variables:
            factor = FACTOR_COLUMNS.get(variable, f"factor_{variable}")
            if factor not in part or part[factor].dropna().nunique() < 2:
                continue
            effect, standardized, _ = _effect(part, factor)
            groups = (
                part["semantic_group"].astype(str).unique()
                if "semantic_group" in part else part.index.astype(str).to_numpy()
            )
            rng = np.random.default_rng(
                int.from_bytes(__import__("hashlib").sha256(
                    f"{seed}:{model}:{task}:{variable}".encode()
                ).digest()[:8], "big")
            )
            samples = []
            group_column = "semantic_group" if "semantic_group" in part else None
            for _ in range(int(bootstrap_samples)):
                drawn = rng.choice(groups, size=len(groups), replace=True)
                if group_column:
                    boot = pd.concat(
                        [part[part[group_column].astype(str) == group] for group in drawn],
                        ignore_index=True,
                    )
                else:
                    boot = part.loc[[int(value) for value in drawn]]
                try:
                    samples.append(_effect(boot, factor)[0])
                except ValueError:
                    continue
            if not samples:
                raise RuntimeError(f"bootstrap lost both levels for {task} × {variable}")
            mapping_effects = []
            for _, mapping_part in part.groupby("response_mapping"):
                try:
                    mapping_effects.append(_effect(mapping_part, factor)[0])
                except ValueError:
                    pass
            mapping_interaction = (
                float(max(mapping_effects) - min(mapping_effects))
                if len(mapping_effects) >= 2 else np.nan
            )
            rows.append({
                "model": model, "task": task, "variable": variable,
                "effect": effect, "standardized_effect": standardized,
                "ci_low": float(np.quantile(samples, .025)),
                "ci_high": float(np.quantile(samples, .975)),
                "direction_consistency": float(np.mean(np.sign(samples) == np.sign(effect))),
                "mapping_interaction": mapping_interaction,
                "n": int(len(part)), "bootstrap_samples": int(len(samples)),
                "endpoint_id": "semantic_persistence_logit_v1",
            })
    return pd.DataFrame(rows)


def estimate_preregistered_interactions(
    observations: pd.DataFrame, interactions
) -> pd.DataFrame:
    """Estimate frozen two-factor interaction coefficients with analytic intervals."""

    rows = []
    for (model, task), part in observations.groupby(["model", "task_family"], sort=True):
        for left, right in interactions:
            left_column = FACTOR_COLUMNS.get(left, f"factor_{left}")
            right_column = FACTOR_COLUMNS.get(right, f"factor_{right}")
            if left_column not in part or right_column not in part:
                continue
            def encoded(series):
                mapped = series.astype(str).map(LEVEL_SCORE)
                return pd.to_numeric(series, errors="coerce").fillna(mapped)
            x, z = encoded(part[left_column]), encoded(part[right_column])
            valid = x.notna() & z.notna() & part.persistence_logit.notna()
            if valid.sum() < 8 or x[valid].nunique() < 2 or z[valid].nunique() < 2:
                continue
            design = np.column_stack((
                np.ones(valid.sum()), x[valid], z[valid], x[valid] * z[valid]
            ))
            y = part.loc[valid, "persistence_logit"].to_numpy(dtype=float)
            coefficient, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficient
            degrees = max(1, len(y) - design.shape[1])
            variance = float(residual @ residual / degrees)
            covariance = variance * np.linalg.pinv(design.T @ design)
            error = float(np.sqrt(max(0.0, covariance[3, 3])))
            rows.append({
                "model": model, "task": task, "variable_a": left,
                "variable_b": right, "interaction": float(coefficient[3]),
                "ci_low": float(coefficient[3] - 1.96 * error),
                "ci_high": float(coefficient[3] + 1.96 * error),
                "n": int(valid.sum()), "endpoint_id": "semantic_persistence_logit_v1",
            })
    return pd.DataFrame(rows)

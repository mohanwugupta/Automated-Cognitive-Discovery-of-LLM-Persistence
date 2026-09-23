"""Outcome-blind deterministic active theory-discrimination acquisition."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _scale(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    return np.zeros_like(values) if high - low <= 1e-12 else (values - low) / (high - low)


def select_active_batch(
    candidates: pd.DataFrame,
    *,
    batch_size: int,
    observed_groups,
    weights: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    if set(weights) != {"coverage", "uncertainty", "disagreement"}:
        raise ValueError("acquisition weights must cover coverage, uncertainty, disagreement")
    if abs(sum(map(float, weights.values())) - 1.0) > 1e-12:
        raise ValueError("acquisition weights must sum to one")
    required = {"condition_id", "semantic_group", *weights}
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"candidate pool missing columns: {sorted(missing)}")
    pool = candidates[
        ~candidates.semantic_group.astype(str).isin(set(map(str, observed_groups)))
    ].copy()
    if len(pool) < batch_size:
        raise ValueError("active candidate pool is smaller than the requested batch")
    pool["acquisition_score"] = sum(
        float(weight) * _scale(pool[name]) for name, weight in weights.items()
    )
    pool["tie_break"] = pool.condition_id.astype(str).map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    return (
        pool.sort_values(["acquisition_score", "tie_break"], ascending=[False, True])
        .head(int(batch_size)).drop(columns="tie_break").reset_index(drop=True)
    )

"""Matched randomization controls for causal-specificity tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def orthonormal_random_subspaces(
    dimension: int, rank: int, count: int, *, seed: int
) -> np.ndarray:
    dimension, rank, count = int(dimension), int(rank), int(count)
    if not (dimension >= rank >= 1 and count >= 1):
        raise ValueError(
            "random subspaces require dimension >= rank >= 1 and count >= 1"
        )
    rng = np.random.default_rng(int(seed))
    return np.stack(
        [
            np.linalg.qr(rng.normal(size=(dimension, rank)), mode="reduced")[0]
            for _ in range(count)
        ]
    )


def pair_row_hash(pair_ids) -> str:
    payload = "\n".join(sorted(map(str, pair_ids))).encode()
    return hashlib.sha256(payload).hexdigest()


def _adaptive_bins(values: pd.Series) -> np.ndarray:
    if len(values) < 4 or values.nunique() < 2:
        return np.zeros(len(values), dtype=int)
    bins = pd.qcut(values.rank(method="first"), q=2, labels=False)
    counts = pd.Series(bins).value_counts()
    return (
        np.asarray(bins, dtype=int)
        if len(counts) > 1 and counts.min() >= 2
        else np.zeros(len(values), dtype=int)
    )


def shuffle_sources(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Derange sources within task, mapping, and adaptive current-state bin."""

    required = {
        "pair_id",
        "source_condition_id",
        "task_family",
        "response_mapping",
        "current_state_score",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"shuffled-source columns are absent: {sorted(missing)}")
    output = frame.copy().reset_index(drop=True)
    output["original_source_condition_id"] = output.source_condition_id
    output["current_state_bin"] = -1
    output["shuffled_source_pair_id"] = None
    rng = np.random.default_rng(int(seed))
    for _, mapping_part in output.groupby(
        ["task_family", "response_mapping"], sort=True
    ):
        bins = _adaptive_bins(mapping_part.current_state_score)
        output.loc[mapping_part.index, "current_state_bin"] = bins
        for _, part in output.loc[mapping_part.index].groupby("current_state_bin"):
            if len(part) < 2:
                raise ValueError("shuffled-source stratum cannot be deranged")
            order = part.index.to_numpy()
            shift = int(rng.integers(1, len(order)))
            donors = np.roll(order, shift)
            output.loc[order, "source_condition_id"] = output.loc[
                donors, "original_source_condition_id"
            ].to_numpy()
            output.loc[order, "shuffled_source_pair_id"] = output.loc[
                donors, "pair_id"
            ].to_numpy()
    if (output.source_condition_id == output.original_source_condition_id).any():
        raise RuntimeError("shuffled-source assignment contains a fixed point")
    return output


def shuffle_targets(
    frame: pd.DataFrame,
    *,
    target_column: str = "predicted_counterfactual_effect",
    seed: int,
) -> pd.DataFrame:
    """Permute semantic targets within task while keeping label mappings together."""

    required = {"contrast_id", "task_family", target_column}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"shuffled-target columns are absent: {sorted(missing)}")
    output = frame.copy()
    output["original_predicted_counterfactual_effect"] = output[target_column]
    output["shuffled_target_contrast_id"] = None
    rng = np.random.default_rng(int(seed))
    for task, part in output.groupby("task_family", sort=True):
        semantic = (
            part.groupby("contrast_id", as_index=False)[target_column]
            .mean()
            .sort_values("contrast_id")
            .reset_index(drop=True)
        )
        if len(semantic) < 2:
            raise ValueError(f"task {task} has too few semantic targets to shuffle")
        shift = int(rng.integers(1, len(semantic)))
        donor = semantic.iloc[np.roll(np.arange(len(semantic)), shift)].reset_index(
            drop=True
        )
        mapping = {
            row.contrast_id: (
                donor.iloc[index].contrast_id,
                donor.iloc[index][target_column],
            )
            for index, row in semantic.iterrows()
        }
        indices = part.index
        output.loc[indices, "shuffled_target_contrast_id"] = [
            mapping[value][0] for value in part.contrast_id
        ]
        output.loc[indices, target_column] = [
            mapping[value][1] for value in part.contrast_id
        ]
    if (
        output.shuffled_target_contrast_id.astype(str) == output.contrast_id.astype(str)
    ).any():
        raise RuntimeError("shuffled-target assignment contains a fixed point")
    return output

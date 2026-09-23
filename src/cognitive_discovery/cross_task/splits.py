"""Deterministic leakage-resistant splits for behavioral and neural analyses."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


def _seed(seed: int, key: tuple[str, ...]) -> int:
    digest = hashlib.sha256(f"{seed}:".encode() + "\x1f".join(key).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_grouped_splits(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    stratify_columns: list[str],
    seed: int,
    fractions=(0.6, 0.2, 0.2),
    names=("train", "selection", "test"),
    output_column: str = "analysis_split",
) -> pd.DataFrame:
    """Assign whole semantic groups within each stratum to three frozen splits."""

    required = set(group_columns) | set(stratify_columns)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"split frame missing columns: {sorted(missing)}")
    if len(fractions) != 3 or len(names) != 3 or abs(sum(fractions) - 1.0) > 1e-12:
        raise ValueError("three split names and fractions summing to one are required")
    result = frame.copy()
    key_column = "__cross_task_group"
    result[key_column] = result[group_columns].astype(str).agg("\x1f".join, axis=1)
    # One group cannot cross scientific strata because that would permit leakage.
    if stratify_columns:
        crossed = result.groupby(key_column)[stratify_columns].nunique().max(axis=1)
        if crossed.gt(1).any():
            raise ValueError("a semantic group crosses requested split strata")
    assignments: dict[str, str] = {}
    grouped = result.groupby(stratify_columns, dropna=False, sort=True) if stratify_columns else [((), result)]
    for stratum, part in grouped:
        if not isinstance(stratum, tuple):
            stratum = (str(stratum),)
        else:
            stratum = tuple(map(str, stratum))
        groups = sorted(part[key_column].unique())
        if len(groups) < 3:
            raise ValueError(f"stratum {stratum} needs at least three independent groups")
        rng = np.random.default_rng(_seed(seed, stratum))
        rng.shuffle(groups)
        selection = max(1, round(len(groups) * float(fractions[1])))
        test = max(1, round(len(groups) * float(fractions[2])))
        if selection + test >= len(groups):
            selection = test = 1
        train = len(groups) - selection - test
        for group in groups[:train]:
            assignments[group] = names[0]
        for group in groups[train:train + selection]:
            assignments[group] = names[1]
        for group in groups[train + selection:]:
            assignments[group] = names[2]
    result[output_column] = result[key_column].map(assignments)
    if result.groupby(key_column)[output_column].nunique().max() != 1:
        raise RuntimeError("semantic group crossed analysis splits")
    return result.drop(columns=[key_column])


def split_hash(frame: pd.DataFrame, *, group_columns: list[str], split_column: str) -> str:
    values = (
        frame[[*group_columns, split_column]].astype(str).drop_duplicates()
        .sort_values([*group_columns, split_column]).to_dict("records")
    )
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assert_untouched_validation(
    acquired_group_ids, validation_group_ids, *, label: str = "active acquisition"
) -> None:
    overlap = set(map(str, acquired_group_ids)) & set(map(str, validation_group_ids))
    if overlap:
        raise ValueError(f"{label} contaminated untouched validation: {sorted(overlap)[:5]}")

"""Shared fail-closed checks for semantic grouping and task holdouts."""

from __future__ import annotations


FAMILIAR_NEURAL_TASKS = frozenset({"bandit", "debugging", "foraging", "solvability"})
HELDOUT_NEURAL_TASKS = frozenset({"effort", "information_sampling", "waiting"})


class SplitValidationError(ValueError):
    pass


def validate_group_coassignment(frame, *, group_columns, split_column: str) -> None:
    missing = (set(group_columns) | {split_column}) - set(frame.columns)
    if missing:
        raise SplitValidationError(f"missing split columns: {sorted(missing)}")
    counts = frame.groupby(list(group_columns), dropna=False)[split_column].nunique(dropna=False)
    leaked = counts[counts > 1]
    if len(leaked):
        raise SplitValidationError(f"semantic group crosses splits: {leaked.index[0]!r}")


def validate_no_semantic_overlap(left, right) -> None:
    overlap = set(left) & set(right)
    if overlap:
        raise SplitValidationError(f"semantic overlap across holdout boundary: {sorted(overlap)[:5]}")

"""Fixed pair/episode-safe discovery, interpolation, and structural splits."""

from __future__ import annotations

import random
from dataclasses import replace

import pandas as pd


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def assign_splits(frame: pd.DataFrame, *, seed: int = 0, config: dict | None = None) -> pd.DataFrame:
    frame = frame.copy()
    required = {"paired_condition_id", "episode_id", "task_family"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"split records missing fields: {sorted(missing)}")
    union = _UnionFind()
    for row in frame.itertuples():
        union.union(f"pair::{row.paired_condition_id}", f"episode::{row.episode_id}")
    frame["_group"] = [union.find(f"pair::{value}") for value in frame.paired_condition_id]
    group_task = frame.groupby("_group").task_family.first().to_dict()
    by_task: dict[str, list[str]] = {}
    for group, task in group_task.items():
        by_task.setdefault(str(task), []).append(group)
    assignments = {}
    rng = random.Random(int(seed))
    structural_groups = set()
    if config:
        holdouts = config.get("splits", {}).get("structural_holdout", ())
        for rule in holdouts:
            if len(rule) != 4:
                raise ValueError(f"structural holdout must have four entries: {rule}")
            left, left_value, right, right_value = rule
            left_column = left if left in frame else f"factor_{left}"
            right_column = right if right in frame else f"factor_{right}"
            if left_column in frame and right_column in frame:
                matched = frame[
                    (frame[left_column].astype(str) == str(left_value))
                    & (frame[right_column].astype(str) == str(right_value))
                ]
                structural_groups.update(matched._group)
    for task in sorted(by_task):
        groups = sorted(by_task[task])
        rng.shuffle(groups)
        n = len(groups)
        n_structural = max(1, round(n * 0.20)) if n >= 3 else 0
        preferred = [group for group in groups if group in structural_groups]
        fallback = [group for group in groups if group not in structural_groups]
        structural = (preferred + fallback)[:n_structural]
        groups = structural + [group for group in groups if group not in structural]
        remaining = n - n_structural
        n_interpolation = max(1, round(n * 0.20)) if remaining >= 2 else 0
        for group in groups[:n_structural]:
            assignments[group] = "structural_test"
        for group in groups[n_structural : n_structural + n_interpolation]:
            assignments[group] = "interpolation_test"
        for group in groups[n_structural + n_interpolation :]:
            assignments[group] = "discovery"
    frame["split"] = frame._group.map(assignments)
    frame = frame.drop(columns="_group")
    if frame.groupby("paired_condition_id").split.nunique().max() > 1:
        raise RuntimeError("paired condition crossed splits")
    if frame.groupby("episode_id").split.nunique().max() > 1:
        raise RuntimeError("episode crossed splits")
    return frame


def assign_condition_splits(conditions, *, seed: int = 0, config: dict | None = None):
    rows = []
    for condition in conditions:
        rows.append(
            {
                "condition_id": condition.condition_id,
                "paired_condition_id": condition.paired_condition_id,
                "episode_id": condition.paired_condition_id,
                "task_family": condition.task_family,
                "continuation_advantage": condition.continuation_advantage,
                **{f"factor_{name}": value for name, value in condition.semantic_factors.items()},
            }
        )
    assigned = assign_splits(pd.DataFrame(rows), seed=seed, config=config)
    lookup = dict(zip(assigned.condition_id, assigned.split))
    return [replace(condition, split=lookup[condition.condition_id]) for condition in conditions]

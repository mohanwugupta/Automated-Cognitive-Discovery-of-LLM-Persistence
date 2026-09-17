"""Prospective deterministic grouped behavioral and neural splits."""

from __future__ import annotations

import hashlib
import json
import random


def _task_rng(seed: int, task: str) -> random.Random:
    digest = hashlib.sha256(f"{int(seed)}:{task}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _assign_three_way(groups, names, fractions, *, seed: int, task: str):
    groups = sorted(map(str, groups))
    _task_rng(seed, task).shuffle(groups)
    n = len(groups)
    if n < 3:
        raise ValueError(f"task {task!r} needs at least three independent groups")
    selection_count = max(1, round(n * float(fractions[1])))
    test_count = max(1, round(n * float(fractions[2])))
    if selection_count + test_count >= n:
        selection_count = test_count = 1
    train_count = n - selection_count - test_count
    assignments = {}
    for group in groups[:train_count]:
        assignments[group] = names[0]
    for group in groups[train_count : train_count + selection_count]:
        assignments[group] = names[1]
    for group in groups[train_count + selection_count :]:
        assignments[group] = names[2]
    return assignments


def assign_behavior_splits(
    frame,
    *,
    group_column: str,
    task_column: str,
    seed: int,
    fractions=(0.6, 0.2, 0.2),
):
    missing = {group_column, task_column} - set(frame.columns)
    if missing:
        raise ValueError(f"behavior split frame missing columns: {sorted(missing)}")
    result = frame.copy()
    group_tasks = result.groupby(group_column)[task_column].nunique()
    if int(group_tasks.max()) != 1:
        raise ValueError("a behavioral semantic group crosses task families")
    assignment = {}
    for task, group in result.groupby(task_column):
        assignment.update(
            _assign_three_way(
                group[group_column].unique(),
                ("behavior_train", "behavior_selection", "behavior_test"),
                fractions,
                seed=seed,
                task=str(task),
            )
        )
    result["behavior_split"] = result[group_column].astype(str).map(assignment)
    if result.groupby(group_column).behavior_split.nunique().max() != 1:
        raise RuntimeError("behavioral group crossed frozen splits")
    return result


def assign_neural_splits(
    frame,
    *,
    group_column: str,
    task_column: str,
    fitting_tasks,
    holdout_tasks,
    seed: int,
    fractions=(0.6, 0.2, 0.2),
):
    missing = {group_column, task_column} - set(frame.columns)
    if missing:
        raise ValueError(f"neural split frame missing columns: {sorted(missing)}")
    fitting_tasks = set(map(str, fitting_tasks))
    holdout_tasks = set(map(str, holdout_tasks))
    if fitting_tasks & holdout_tasks:
        raise ValueError("fitting and task-holdout sets overlap")
    observed = set(map(str, frame[task_column].unique()))
    if observed - fitting_tasks - holdout_tasks:
        raise ValueError("pair manifest contains a task without a prospective assignment")
    result = frame.copy()
    assignment = {}
    for task, group in result.groupby(task_column):
        task = str(task)
        groups = set(map(str, group[group_column].unique()))
        if task in holdout_tasks:
            assignment.update({value: "neural_task_holdout" for value in groups})
        else:
            assignment.update(
                _assign_three_way(
                    groups,
                    ("neural_train", "neural_selection", "neural_test"),
                    fractions,
                    seed=seed,
                    task=task,
                )
            )
    result["neural_split"] = result[group_column].astype(str).map(assignment)
    if result.groupby(group_column).neural_split.nunique().max() != 1:
        raise RuntimeError("mechanistic group crossed frozen splits")
    return result


def split_manifest_hash(frame, group_column: str, split_column: str) -> str:
    values = (
        frame[[group_column, split_column]]
        .astype(str)
        .drop_duplicates()
        .sort_values([group_column, split_column])
        .to_dict("records")
    )
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

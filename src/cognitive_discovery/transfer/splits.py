"""Prospective task-transfer splits and hard leakage audits."""

from __future__ import annotations

import hashlib
import json
import random


def _rng(seed: int, task: str) -> random.Random:
    digest = hashlib.sha256(f"task-transfer-v1:{seed}:{task}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def assign_transfer_splits(frame, *, group_column="neural_group_id", task_column="task_family", seed=26001, fractions=(0.6, 0.2, 0.2)):
    missing = {group_column, task_column} - set(frame.columns)
    if missing:
        raise ValueError(f"transfer frame missing columns: {sorted(missing)}")
    if abs(sum(map(float, fractions)) - 1.0) > 1e-12:
        raise ValueError("transfer split fractions must sum to one")
    result = frame.copy()
    crossing = result.groupby(group_column)[task_column].nunique()
    if len(crossing) and int(crossing.max()) != 1:
        raise ValueError("a counterfactual group crosses task families")
    assignments = {}
    for task, part in result.groupby(task_column):
        groups = sorted(map(str, part[group_column].unique()))
        if len(groups) < 3:
            raise ValueError(f"task {task!r} needs at least three independent groups")
        _rng(seed, str(task)).shuffle(groups)
        n = len(groups)
        n_selection = max(1, round(n * float(fractions[1])))
        n_test = max(1, round(n * float(fractions[2])))
        if n_selection + n_test >= n:
            n_selection = n_test = 1
        n_train = n - n_selection - n_test
        for group in groups[:n_train]: assignments[group] = "transfer_train"
        for group in groups[n_train:n_train + n_selection]: assignments[group] = "transfer_selection"
        for group in groups[n_train + n_selection:]: assignments[group] = "transfer_test"
    result["transfer_split"] = result[group_column].astype(str).map(assignments)
    if result.groupby(group_column).transfer_split.nunique().max() != 1:
        raise RuntimeError("counterfactual group crossed transfer splits")
    return result


def transfer_split_hash(frame, *, group_column="neural_group_id") -> str:
    records = (frame[[group_column, "task_family", "transfer_split"]].astype(str).drop_duplicates().sort_values([group_column]).to_dict("records"))
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit_transfer_leakage(frame, *, source_tasks, train_ids, selection_ids, test_ids, pair_column="pair_id") -> dict:
    """Fail if target outcomes/groups touched fitting or layer/rank selection."""

    source_tasks = set(map(str, source_tasks))
    train_ids, selection_ids, test_ids = map(lambda values: set(map(str, values)), (train_ids, selection_ids, test_ids))
    if train_ids & selection_ids or train_ids & test_ids or selection_ids & test_ids:
        raise ValueError("train, selection, and target-test pair identities overlap")
    lookup = frame.set_index(pair_column)
    unknown = (train_ids | selection_ids | test_ids) - set(map(str, lookup.index))
    if unknown:
        raise ValueError(f"leakage audit references unknown pairs: {sorted(unknown)[:3]}")
    fit = lookup.loc[sorted(train_ids | selection_ids)]
    if not set(fit.task_family.astype(str)) <= source_tasks:
        raise ValueError("target-task data leaked into controller fitting or selection")
    wrong_train = set(lookup.loc[sorted(train_ids)].transfer_split.astype(str)) - {"transfer_train"}
    wrong_selection = set(lookup.loc[sorted(selection_ids)].transfer_split.astype(str)) - {"transfer_selection"}
    wrong_test = set(lookup.loc[sorted(test_ids)].transfer_split.astype(str)) - {"transfer_test"}
    if wrong_train or wrong_selection or wrong_test:
        raise ValueError("pair identity was used outside its frozen transfer split")
    return {"passed": True, "source_tasks": sorted(source_tasks), "n_train": len(train_ids), "n_selection": len(selection_ids), "n_test": len(test_ids)}


def planned_source_sets(tasks, *, include_all_pairs=True, diversity_seed=26002):
    """Create preregistered single, pair, LOTO, and diversity source sets."""

    tasks = tuple(map(str, tasks))
    plans = [{"design": "single", "source_id": task, "source_tasks": [task], "target_tasks": list(tasks)} for task in tasks]
    if include_all_pairs:
        for left_index, left in enumerate(tasks):
            for right in tasks[left_index + 1:]:
                plans.append({"design": "pair", "source_id": f"{left}+{right}", "source_tasks": [left, right], "target_tasks": [task for task in tasks if task not in {left, right}]})
    for heldout in tasks:
        sources = [task for task in tasks if task != heldout]
        plans.append({"design": "loto", "source_id": f"without-{heldout}", "source_tasks": sources, "target_tasks": [heldout]})
        candidates = sources.copy()
        _rng(diversity_seed, heldout).shuffle(candidates)
        for count in range(1, len(candidates) + 1):
            selected = sorted(candidates[:count])
            plans.append({"design": "diversity", "source_id": f"to-{heldout}-n{count}-{'+'.join(selected)}", "source_tasks": selected, "target_tasks": [heldout], "source_task_count": count})
    return plans

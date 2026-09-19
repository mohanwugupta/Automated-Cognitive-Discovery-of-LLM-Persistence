import pandas as pd
import pytest

from cognitive_discovery.transfer.splits import assign_transfer_splits, audit_transfer_leakage, planned_source_sets, transfer_split_hash
from cognitive_discovery.transfer.neural import transfer_target_tasks


def _pairs():
    return pd.DataFrame([{"pair_id": f"{task}-{index}", "neural_group_id": f"{task}-g{index}", "task_family": task} for task in ("bandit", "effort") for index in range(20)])


def test_transfer_splits_are_grouped_deterministic_and_all_tasks_trainable():
    first = assign_transfer_splits(_pairs(), seed=42)
    second = assign_transfer_splits(_pairs().sample(frac=1, random_state=3), seed=42)
    assert transfer_split_hash(first) == transfer_split_hash(second)
    for _, part in first.groupby("task_family"):
        assert set(part.transfer_split) == {"transfer_train", "transfer_selection", "transfer_test"}


def test_target_task_data_cannot_enter_fit_or_selection():
    frame = assign_transfer_splits(_pairs(), seed=42)
    train = frame[(frame.task_family == "bandit") & (frame.transfer_split == "transfer_train")]
    selection = frame[(frame.task_family == "bandit") & (frame.transfer_split == "transfer_selection")]
    test = frame[(frame.task_family == "effort") & (frame.transfer_split == "transfer_test")]
    assert audit_transfer_leakage(frame, source_tasks=["bandit"], train_ids=train.pair_id, selection_ids=selection.pair_id, test_ids=test.pair_id)["passed"]
    leaked = frame[(frame.task_family == "effort") & (frame.transfer_split == "transfer_selection")]
    with pytest.raises(ValueError, match="target-task data leaked"):
        audit_transfer_leakage(frame, source_tasks=["bandit"], train_ids=train.pair_id, selection_ids=leaked.pair_id, test_ids=test.pair_id)


def test_plan_contains_seven_singles_all_pairs_loto_and_diversity():
    tasks = ["a", "b", "c", "d", "e", "f", "g"]
    plans = planned_source_sets(tasks)
    counts = pd.Series([plan["design"] for plan in plans]).value_counts().to_dict()
    assert counts == {"diversity": 42, "pair": 21, "single": 7, "loto": 7}


def test_single_task_transfer_evaluates_diagonal_before_off_diagonal():
    work = {
        "design": "single",
        "source_tasks": ["bandit"],
        "target_tasks": ["bandit", "effort", "waiting"],
    }
    assert transfer_target_tasks(work, "diagonal", source_gate=None) == ["bandit"]
    with pytest.raises(RuntimeError, match="diagonal source gate"):
        transfer_target_tasks(work, "off_diagonal", source_gate=None)
    assert transfer_target_tasks(work, "off_diagonal", source_gate=True) == [
        "effort",
        "waiting",
    ]
    assert transfer_target_tasks(work, "off_diagonal", source_gate=False) == []


def test_positive_diagonal_gate_exposes_only_off_diagonal_targets():
    work = {
        "design": "single",
        "source_tasks": ["bandit"],
        "target_tasks": ["bandit", "effort", "waiting"],
    }
    assert transfer_target_tasks(work, "diagonal", source_gate=None) == ["bandit"]
    assert transfer_target_tasks(work, "off_diagonal", source_gate=True) == [
        "effort",
        "waiting",
    ]

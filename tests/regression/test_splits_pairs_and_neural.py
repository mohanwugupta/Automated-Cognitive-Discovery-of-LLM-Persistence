from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cognitive_discovery.reproducibility.manifest import sha256_file
from cognitive_discovery.reproducibility.splits import (
    FAMILIAR_NEURAL_TASKS,
    HELDOUT_NEURAL_TASKS,
    SplitValidationError,
    validate_group_coassignment,
    validate_no_semantic_overlap,
)


ROOT = Path(__file__).resolve().parents[2]


def test_core_pair_and_prediction_identity_and_counts():
    pair_path = ROOT / "artifacts/causal_mech_v1/counterfactuals/pair_manifest.parquet"
    prediction_path = ROOT / "artifacts/causal_mech_v1/counterfactuals/predicted_effects.parquet"
    assert sha256_file(pair_path) == "6591f8430e52232932e56f79a768409363ddece9c9da0bcd1b6bd089b4e266e4"
    assert sha256_file(prediction_path) == "dc87d567f8560c079f6671c24d25df81f13f6990b795daf9f7ac4f44ef681351"
    pairs = pd.read_parquet(pair_path)
    assert pairs["pair_id"].is_unique
    assert pairs.groupby("pair_split").size().to_dict() == {
        "mech_pair_test": 230,
        "mech_pair_train": 898,
        "mech_pair_validation": 180,
        "mech_task_holdout": 816,
    }
    assert pairs.groupby("target_variable").size().to_dict() == {
        "action_history": 480,
        "contextual_outcome_history": 684,
        "generic_value": 480,
        "outcome_history": 480,
    }
    validate_group_coassignment(pairs, group_columns=["contrast_id"], split_column="pair_split")


def test_neural_task_holdout_semantics_are_frozen():
    pairs = pd.read_parquet(ROOT / "artifacts/causal_mech_v1/counterfactuals/pair_manifest.parquet")
    selected = pairs[pairs["selected_for_neural"]]
    familiar = set(selected[selected.pair_split != "mech_task_holdout"].task_family)
    heldout = set(selected[selected.pair_split == "mech_task_holdout"].task_family)
    assert familiar == FAMILIAR_NEURAL_TASKS
    assert heldout == HELDOUT_NEURAL_TASKS
    assert familiar.isdisjoint(heldout)


def test_abstraction_split_and_semantic_pair_identity():
    frame = pd.read_parquet(ROOT / "artifacts/abstraction_discovery_v1/design/contrast_manifest.parquet")
    assert frame.groupby("pair_split").size().to_dict() == {
        "abstraction_test": 560,
        "abstraction_train": 1680,
        "abstraction_validation": 560,
    }
    assert frame["semantic_contrast_id"].nunique() == 1400
    assert frame["pair_id"].nunique() == 2800
    validate_group_coassignment(
        frame, group_columns=["semantic_contrast_id"], split_column="pair_split"
    )


def test_qwen_wordings_and_mappings_stay_in_the_same_split():
    frame = pd.read_csv(ROOT / "paper/generated/qwen_confirmation_v2/interventions.csv.gz")
    validate_group_coassignment(
        frame,
        group_columns=["task_family", "background", "pattern"],
        split_column="pair_split",
    )
    frozen = frame[frame.method == "frozen"]
    assert frozen.groupby(["pair_split", "wording"]).size().to_dict() == {
        ("mech_pair_test", "original"): 64,
        ("mech_pair_test", "rephrased"): 64,
        ("mech_task_holdout", "original"): 48,
        ("mech_task_holdout", "rephrased"): 48,
    }


def test_split_validators_reject_leakage():
    frame = pd.DataFrame({"semantic": ["a", "a"], "split": ["train", "test"]})
    with pytest.raises(SplitValidationError, match="crosses splits"):
        validate_group_coassignment(frame, group_columns=["semantic"], split_column="split")
    with pytest.raises(SplitValidationError, match="overlap"):
        validate_no_semantic_overlap({"a", "b"}, {"b", "c"})

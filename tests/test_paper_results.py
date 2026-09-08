"""Protect against a label permutation being mistaken for a target-value null."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("paper_results", Path(__file__).parents[1] / "scripts/paper_results.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_constant_within_task_shuffle_is_uninformative():
    frame = pd.DataFrame({"contrast_id": ["a", "b", "c", "d"],
                          "task_family": ["A", "A", "B", "B"],
                          "original_predicted_counterfactual_effect": [1., 1., 2., 2.],
                          "predicted_counterfactual_effect": [1., 1. + 1e-15, 2., 2.]})
    assert not module.audit_shuffle(frame)["informative_permutation"]


def test_changed_targets_are_counted_and_nonfinite_rejected():
    frame = pd.DataFrame({"contrast_id": ["a", "b"], "task_family": ["A", "A"],
                          "original_predicted_counterfactual_effect": [1., 2.],
                          "predicted_counterfactual_effect": [2., 1.]})
    assert module.audit_shuffle(frame)["changed_target_rows"] == 2
    frame.loc[0, "predicted_counterfactual_effect"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        module.audit_shuffle(frame)


def test_committed_audit_does_not_hide_candidate_or_missing_necessity(tmp_path):
    result = module.build(tmp_path)
    assert len(result["controllers"]) == 3
    assert result["necessity_jobs"] == 0
    assert len(result["provenance"]) >= 9

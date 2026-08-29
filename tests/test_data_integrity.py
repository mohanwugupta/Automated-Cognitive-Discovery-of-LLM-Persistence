import pandas as pd
import pytest

from cognitive_discovery.data.splits import assign_splits
from cognitive_discovery.data.validation import assert_no_future_leakage, validate_records


def _frame():
    rows = []
    for index in range(40):
        pair = f"pair-{index}"
        for mapping in ("forward", "reverse"):
            rows.append(
                {
                    "condition_id": f"{pair}-{mapping}",
                    "paired_condition_id": pair,
                    "episode_id": pair,
                    "task_family": "bandit" if index % 2 else "foraging",
                    "continuation_advantage": (index % 5) - 2,
                    "step": 0,
                    "terminated": True,
                    "p_continue": 0.5,
                    "p_disengage": 0.5,
                    "persistence_logit": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_split_assignment_keeps_pairs_and_episodes_together():
    split = assign_splits(_frame(), seed=33)
    assert set(split.split) == {"discovery", "interpolation_test", "structural_test"}
    assert split.groupby("paired_condition_id").split.nunique().max() == 1
    assert split.groupby("episode_id").split.nunique().max() == 1


def test_future_leakage_is_rejected():
    assert_no_future_leakage(["factor_cost", "history_outcome_1"])
    with pytest.raises(ValueError, match="future leakage"):
        assert_no_future_leakage(["factor_cost", "subsequent_reward"])


def test_post_termination_state_is_rejected():
    frame = _frame().iloc[:2].copy()
    frame.loc[:, "episode_id"] = "same"
    frame.loc[:, "step"] = [0, 1]
    with pytest.raises(ValueError, match="post-termination"):
        validate_records(frame)


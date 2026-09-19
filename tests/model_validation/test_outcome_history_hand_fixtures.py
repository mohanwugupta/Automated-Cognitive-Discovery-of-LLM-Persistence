import pandas as pd
import pytest

from cognitive_discovery.models.features import _series
from validation.reference_models.outcome_history_reference import outcome_trace


@pytest.mark.parametrize(("history", "expected"), [([1], 1), ([1, -1], -.3), ([1, -1, 1], .79)])
def test_outcome_history_hand_fixture(history, expected):
    assert outcome_trace(history) == pytest.approx(expected)
    assert _series(pd.DataFrame({"history_outcomes": [history]}), "history_outcome_kernel")[0] == pytest.approx(expected)


def test_empty_outcome_history_is_missing():
    assert pd.isna(_series(pd.DataFrame({"history_outcomes": [[]]}), "history_outcome_kernel")[0])

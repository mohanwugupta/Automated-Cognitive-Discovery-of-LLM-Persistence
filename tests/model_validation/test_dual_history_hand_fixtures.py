import pandas as pd
import pytest

from cognitive_discovery.models.features import _series
from validation.reference_models.dual_history_reference import features


def test_dual_history_action_and_outcome_ordering():
    row = {"history_actions": ["continue", "disengage", "continue"], "history_outcomes": [-1, 0, 1]}
    expected = features({
        **row,
        **{name: 0 for name in (
            "factor_continuation_value", "factor_disengagement_value", "factor_continuation_cost",
            "factor_progress_evidence", "factor_success_evidence", "factor_uncertainty",
            "factor_prior_investment", "factor_controllability", "factor_environmental_stability",
            "factor_goal_continuity",
        )},
    })
    frame = pd.DataFrame([row])
    assert _series(frame, "history_action_1")[0] == 1
    assert _series(frame, "history_action_kernel")[0] == pytest.approx(expected["history_action_kernel"])
    assert _series(frame, "history_outcome_1")[0] == 1
    assert _series(frame, "history_outcome_kernel")[0] == pytest.approx(expected["history_outcome_kernel"])

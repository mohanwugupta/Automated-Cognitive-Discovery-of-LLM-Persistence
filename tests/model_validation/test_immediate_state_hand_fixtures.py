import pandas as pd

from cognitive_discovery.models.features import _series
from validation.reference_models.immediate_state_reference import features


def test_immediate_state_hand_fixture():
    row = {
        "factor_continuation_value": 2, "factor_disengagement_value": -1,
        "factor_continuation_cost": 1, "factor_progress_evidence": .5,
        "factor_success_evidence": -.5, "factor_uncertainty": 1,
        "factor_prior_investment": 0, "factor_controllability": -1,
        "factor_environmental_stability": 1, "factor_goal_continuity": -1,
    }
    reference = features(row)
    frame = pd.DataFrame([row])
    assert {name: _series(frame, name)[0] for name in reference} == reference

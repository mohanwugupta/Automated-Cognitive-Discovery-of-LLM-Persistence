import pandas as pd
import pytest
from pathlib import Path

from cognitive_discovery.models.features import _series
from validation.reference_models import (
    dual_history_reference, immediate_state_reference,
    latent_context_reference, outcome_history_reference,
)


def _row():
    return {
        "factor_continuation_value": 1, "factor_disengagement_value": -1,
        "factor_continuation_cost": .5, "factor_progress_evidence": -.5,
        "factor_success_evidence": 1, "factor_uncertainty": -1,
        "factor_prior_investment": 1, "factor_controllability": .5,
        "factor_environmental_stability": 1, "factor_goal_continuity": -1,
        "history_actions": ["continue", "disengage", "continue"],
        "history_outcomes": [-1, 0, 1], "context_context_return": "B",
        "context_cue_probability": .8, "context_change_point": "stable",
        "context_a_history_outcomes": [1, 1, 0],
        "context_b_history_outcomes": [-1, -1, 1],
    }


@pytest.mark.parametrize("reference", [immediate_state_reference, outcome_history_reference, dual_history_reference, latent_context_reference])
def test_reference_features_equal_production_features(reference):
    row = _row()
    expected = reference.features(row)
    frame = pd.DataFrame([row])
    for name, value in expected.items():
        assert _series(frame, name)[0] == pytest.approx(value)


def test_reference_counterfactual_sign_is_source_minus_base():
    base, source = _row(), _row()
    source["history_outcomes"] = [1, 1, 1]
    parameters = {"history_outcome_kernel": 2.0}
    assert outcome_history_reference.counterfactual(base, source, parameters) > 0


def test_reference_implementations_do_not_import_production_code():
    directory = Path(__file__).resolve().parents[2] / "validation/reference_models"
    for path in directory.glob("*_reference.py"):
        assert "cognitive_discovery" not in path.read_text(encoding="utf-8")

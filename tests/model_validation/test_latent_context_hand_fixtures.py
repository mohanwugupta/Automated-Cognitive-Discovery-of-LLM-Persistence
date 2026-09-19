import pandas as pd
import pytest

from cognitive_discovery.models.features import _series
from validation.reference_models.latent_context_reference import contextual_trace


def _row(return_context="A", change="stable"):
    return {
        "history_outcomes": [-1, 1], "context_context_return": return_context,
        "context_cue_probability": .8, "context_change_point": change,
        "context_a_history_outcomes": [1, 1], "context_b_history_outcomes": [-1, -1],
    }


@pytest.mark.parametrize(("context", "change"), [("A", "stable"), ("B", "stable"), ("A", "change_point")])
def test_latent_context_switch_and_change_point(context, change):
    row = _row(context, change)
    observed = _series(pd.DataFrame([row]), "context_relevant_outcome_kernel")[0]
    assert observed == pytest.approx(contextual_trace(row))

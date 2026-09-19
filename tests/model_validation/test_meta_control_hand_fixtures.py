import pandas as pd

from cognitive_discovery.models.features import _series


def test_meta_control_interaction_fixtures():
    frame = pd.DataFrame({"factor_continuation_value": [2.], "factor_controllability": [-1.], "factor_continuation_cost": [.5], "factor_uncertainty": [1.]})
    assert _series(frame, "factor_continuation_value*factor_controllability")[0] == -2
    assert _series(frame, "factor_continuation_cost*factor_uncertainty")[0] == .5

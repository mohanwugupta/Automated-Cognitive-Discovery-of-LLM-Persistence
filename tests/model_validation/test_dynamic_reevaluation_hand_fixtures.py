import pandas as pd

from cognitive_discovery.models.features import _series


def test_continuation_advantage_sign_convention():
    frame = pd.DataFrame({"factor_continuation_value": [3.], "factor_disengagement_value": [1.], "factor_continuation_cost": [.5]})
    assert _series(frame, "continuation_advantage")[0] == 1.5

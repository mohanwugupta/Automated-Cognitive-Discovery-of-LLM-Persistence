import pandas as pd
import pytest

from cognitive_discovery.models.features import _series


def test_latent_motivation_proxy_fixture():
    frame = pd.DataFrame({"factor_continuation_value": [2.], "factor_continuation_cost": [.5], "factor_progress_evidence": [1.], "history_outcomes": [[-1., 1.]]})
    # R = .7*(-1) + 1 = .3; M = 2 - .5 + 1 + .7*.3
    assert _series(frame, "latent_state")[0] == pytest.approx(2.71)

import pandas as pd
import pytest

from cognitive_discovery.models.features import _series


def test_action_encoding_and_decay():
    frame = pd.DataFrame({"history_actions": [["continue", "disengage", "continue"]]})
    assert _series(frame, "history_action_1")[0] == 1
    assert _series(frame, "history_action_kernel")[0] == pytest.approx(.49 - .7 + 1)

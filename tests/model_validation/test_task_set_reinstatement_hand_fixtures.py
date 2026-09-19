import pandas as pd

from cognitive_discovery.models.features import _series


def test_task_set_reinstatement_interaction_fixture():
    frame = pd.DataFrame({"history_action_kernel": [2.], "factor_goal_continuity": [-1.]})
    assert _series(frame, "history_action_kernel*factor_goal_continuity")[0] == -2

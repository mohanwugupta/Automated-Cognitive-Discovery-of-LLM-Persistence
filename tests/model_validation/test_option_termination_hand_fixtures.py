from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS


def test_option_termination_has_only_frozen_proxy_terms():
    assert COGNITIVE_MODELS["option_termination"].features == (
        "continuation_advantage", "factor_progress_evidence",
        "factor_goal_continuity", "history_action_kernel",
    )

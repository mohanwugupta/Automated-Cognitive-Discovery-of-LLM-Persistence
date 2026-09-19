from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS


def test_generic_sequential_control_excludes_current_value_terms():
    features = COGNITIVE_MODELS["generic_sequential_choice"].features
    assert features == (
        "history_action_1", "history_action_kernel", "history_outcome_1",
        "history_outcome_kernel", "factor_environmental_stability",
    )

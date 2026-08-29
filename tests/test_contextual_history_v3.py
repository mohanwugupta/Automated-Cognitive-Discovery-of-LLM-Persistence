import json

import numpy as np

from cognitive_discovery.data.storage import records_frame
from cognitive_discovery.design.manifests import semantic_hash
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.experiments.contextual_history import (
    compile_contextual_history_design,
    validate_contextual_history,
)
from cognitive_discovery.experiments.registry import get_renderer
from cognitive_discovery.models.features import FeatureEncoder
from cognitive_discovery.participants.base import DeterministicParticipant
from cognitive_discovery.sampling.candidate_pool import (
    condition_frame,
    generate_theory_candidate_pool,
)


def _representatives(conditions):
    return [condition for condition in conditions if condition.condition_id.endswith("-m0")]


def test_contextual_design_has_order_separate_histories_and_matched_contrasts(theory_config):
    conditions = compile_contextual_history_design(
        theory_config, n_conditions=60, seed=41
    )
    assert len(conditions) == 120
    representatives = _representatives(conditions)
    groups = {}
    for condition in representatives:
        context = condition.contextual_history
        validate_contextual_history(context)
        assert tuple(context["phase_order"]) == ("A1", "B", "A2")
        assert tuple(context["a_history_outcomes"])
        assert tuple(context["b_history_outcomes"])
        assert condition.history.outcomes == tuple(context["b_history_outcomes"])
        assert condition.paired_condition_id.endswith(semantic_hash(condition)[:16])
        groups.setdefault(context["critical_contrast_id"], []).append(condition)
    assert set(map(len, groups.values())) == {2}
    for contrast in groups.values():
        left, right = contrast
        assert left.semantic_factors == right.semantic_factors
        assert left.environment_seed == right.environment_seed
        assert left.response_mapping == right.response_mapping
        assert left.contextual_history["a_history_outcomes"] == right.contextual_history[
            "b_history_outcomes"
        ]
        assert left.contextual_history["b_history_outcomes"] == right.contextual_history[
            "a_history_outcomes"
        ]
    assert {
        condition.contextual_history["cue_reliability"] for condition in representatives
    } == {"low", "medium", "high"}


def test_contextual_renderer_uses_operational_domain_rule(theory_config):
    conditions = compile_contextual_history_design(
        theory_config, n_conditions=6, seed=42
    )
    for condition in _representatives(conditions):
        prompt = get_renderer(condition.task_family).render(condition).messages[0]["content"]
        assert "Environmental rule:" in prompt
        assert "outcome-generating mechanism" in prompt
        assert "Phase A1" in prompt and "Phase B" in prompt and "Phase A2" in prompt


def test_context_relevant_feature_uses_reliability_weighted_matching(theory_config):
    conditions = compile_contextual_history_design(
        theory_config, n_conditions=200, seed=43
    )
    frame = condition_frame(conditions)
    candidate = frame[
        (frame.context_context_return == "A")
        & (frame.context_change_point == "same_environment")
        & (frame.context_cue_reliability == "high")
        & (frame.context_a_history_valence != frame.context_b_history_valence)
    ].head(4)
    assert len(candidate)
    encoder = FeatureEncoder(
        ("history_outcome_kernel", "context_relevant_outcome_kernel")
    )
    matrix, names = encoder.fit_transform(candidate.reset_index(drop=True))
    assert names[0] == "history_outcome_kernel"
    # The two scientific features are separately encoded and not identical.
    assert not np.allclose(matrix[:, 0], matrix[:, 2])


def test_collection_preserves_context_fields_without_future_information(theory_config):
    conditions = compile_contextual_history_design(
        theory_config, n_conditions=6, seed=44
    )
    frame = records_frame(collect_conditions(conditions, DeterministicParticipant()))
    assert frame.context_phase_order.map(json.loads).map(tuple).map(
        lambda value: value == ("A1", "B", "A2")
    ).all()
    assert frame.context_a_history_outcomes.notna().all()
    assert frame.context_b_history_outcomes.notna().all()
    assert not any(column.startswith("subsequent_") for column in frame)


def test_combined_candidate_pool_is_unique_and_cross_domain(theory_config):
    empty_observed = condition_frame(
        compile_contextual_history_design(theory_config, n_conditions=6, seed=99)
    ).iloc[0:0]
    conditions, frame = generate_theory_candidate_pool(
        theory_config,
        empty_observed,
        candidate_count=120,
        contextual_fraction=0.5,
        seed=100,
    )
    assert len(conditions) == 240
    assert len(frame) == frame.semantic_hash.nunique() == 120
    assert {value.split(":")[1] for value in frame.domain} == {"original", "contextual"}

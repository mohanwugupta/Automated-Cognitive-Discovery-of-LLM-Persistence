import numpy as np

from cognitive_discovery.experiments.registry import get_renderer
from cognitive_discovery.mechanistic.dataset.matched_conditions import (
    compile_mechanistic_conditions,
    validate_matched_conditions,
)
from cognitive_discovery.mechanistic.targets.behavioral_targets import (
    compute_condition_targets,
)
from cognitive_discovery.models.features import FeatureEncoder
from cognitive_discovery.sampling.candidate_pool import condition_frame


def _representatives(records):
    return [row for row in records if row.condition.condition_id.endswith("-m0")]


def test_matched_mechanistic_dataset_is_isolated_and_renderable(mechanistic_config):
    records = compile_mechanistic_conditions(mechanistic_config, smoke=True)
    validate_matched_conditions(records)
    assert len({row.contrast_id for row in records}) == 35
    assert {row.contrast_family for row in records} == {
        "outcome_history",
        "contextual_history",
        "raw_history_control",
        "action_history",
        "current_value_control",
    }
    for record in records:
        trial = get_renderer(record.condition.task_family).render(record.condition)
        assert trial.prompt_hash


def test_contextual_pairs_hold_raw_history_and_raw_controls_dissociate(
    mechanistic_config,
):
    records = _representatives(
        compile_mechanistic_conditions(mechanistic_config, smoke=True)
    )
    groups = {}
    for row in records:
        groups.setdefault(row.contrast_id, []).append(row)
    for pair in groups.values():
        plus = next(row for row in pair if row.contrast_member == 1)
        minus = next(row for row in pair if row.contrast_member == -1)
        if plus.contrast_family == "contextual_history":
            assert plus.targets["outcome_history"] == minus.targets["outcome_history"]
            assert (
                plus.targets["contextual_outcome_history"]
                > minus.targets["contextual_outcome_history"]
            )
        if plus.contrast_family == "raw_history_control":
            raw_delta = (
                plus.targets["outcome_history"] - minus.targets["outcome_history"]
            )
            contextual_delta = (
                plus.targets["contextual_outcome_history"]
                - minus.targets["contextual_outcome_history"]
            )
            assert np.isclose(contextual_delta / raw_delta, 0.05)


def test_frozen_contextual_target_matches_behavioral_feature_encoder(
    mechanistic_config,
):
    records = compile_mechanistic_conditions(mechanistic_config, smoke=True)
    contextual = next(
        row.condition for row in records if row.condition.contextual_history
    )
    frame = condition_frame([contextual])
    encoder = FeatureEncoder(
        ("history_outcome_kernel", "context_relevant_outcome_kernel")
    )
    raw, names = encoder._raw(frame)
    targets = compute_condition_targets(contextual)
    assert names[0] == "history_outcome_kernel"
    assert np.isclose(raw[0, 0], targets["outcome_history"])
    assert np.isclose(raw[0, 2], targets["contextual_outcome_history"])

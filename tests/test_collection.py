import math

from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.participants.base import DeterministicParticipant


def test_behavior_collection_outputs_semantic_probabilities(discovery_config):
    design = compile_design(discovery_config, n_conditions=5, seed=5)
    records = collect_conditions(
        design,
        DeterministicParticipant("deterministic/test"),
        model_revision="test-revision",
    )
    assert len(records) == 10
    for row in records:
        assert math.isclose(row.p_continue + row.p_disengage, 1.0)
        assert math.isclose(
            row.persistence_logit, math.log(row.p_continue / row.p_disengage)
        )
        assert row.semantic_continue_token in {"X", "Y"}
        assert row.prompt_hash
        assert row.model_revision == "test-revision"
        assert not hasattr(row, "hidden_states")


def test_mapping_reversal_does_not_reverse_semantic_probability(discovery_config):
    pair = compile_design(discovery_config, n_conditions=1, seed=12)
    records = collect_conditions(pair, DeterministicParticipant())
    assert abs(records[0].p_continue - records[1].p_continue) < 0.30


def test_expanded_collection_builds_ordered_history_prefixes(discovery_config):
    design = compile_design(discovery_config, n_conditions=40, seed=14)
    pair = next(
        design[index : index + 2]
        for index in range(0, 80, 2)
        if design[index].history.length == 3
    )
    records = collect_conditions(
        pair,
        DeterministicParticipant(),
        expand_history_prefixes=True,
    )
    assert len(records) == 8
    for episode_records in (
        records[::2],
        records[1::2],
    ):
        assert [row.step for row in episode_records] == [0, 1, 2, 3]
        assert [len(row.history_actions) for row in episode_records] == [0, 1, 2, 3]
        assert [row.terminated for row in episode_records] == [False, False, False, True]

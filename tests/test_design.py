from collections import Counter, defaultdict

from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.design.manifests import semantic_hash
from cognitive_discovery.ontology.factors import TASK_FACTOR_AVAILABILITY


def test_design_is_counterbalanced_and_pair_safe(discovery_config):
    rows = compile_design(discovery_config, n_conditions=84, seed=17)
    assert len(rows) == 168

    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row.paired_condition_id].append(row)
    assert all(len(pair) == 2 for pair in by_pair.values())

    for pair in by_pair.values():
        left, right = pair
        assert left.response_mapping != right.response_mapping
        assert left.environment_seed == right.environment_seed
        assert left.semantic_factors == right.semantic_factors
        assert left.history == right.history

    task_counts = Counter(row.task_family for row in rows[::2])
    assert max(task_counts.values()) - min(task_counts.values()) <= 1


def test_unavailable_factors_are_missing_not_zero(discovery_config):
    rows = compile_design(discovery_config, n_conditions=70, seed=9)
    for row in rows:
        available = TASK_FACTOR_AVAILABILITY[row.task_family]
        for factor, value in row.semantic_factors.items():
            assert row.factor_available[factor] == (factor in available)
            if factor not in available:
                assert value is None


def test_core_crossing_receives_balanced_coverage(discovery_config):
    rows = compile_design(discovery_config, n_conditions=280, seed=123)[::2]
    task_mapping = Counter(
        (row.task_family, row.response_mapping.mapping_id) for row in rows
    )
    assert max(task_mapping.values()) - min(task_mapping.values()) <= 1
    for task in TASK_FACTOR_AVAILABILITY:
        task_rows = [row for row in rows if row.task_family == task]
        assert len({row.continuation_advantage for row in task_rows}) >= 3
        assert len({row.history.valence for row in task_rows}) >= 3


def test_independent_design_excludes_discovery_conditions(discovery_config):
    discovery = compile_design(discovery_config, n_conditions=140, seed=72)
    excluded = {semantic_hash(row) for row in discovery[::2]}
    validation = compile_design(
        discovery_config,
        n_conditions=70,
        seed=73,
        design_id="validation",
        exclude_semantic_hashes=excluded,
    )
    assert not excluded & {semantic_hash(row) for row in validation[::2]}

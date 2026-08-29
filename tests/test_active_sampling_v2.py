from collections import Counter

import numpy as np
import pandas as pd

from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.sampling.active_mixture import select_active_mixture
from cognitive_discovery.sampling.candidate_pool import (
    condition_frame,
    generate_candidate_pool,
    observed_semantic_hashes,
)
from cognitive_discovery.sampling.coverage_score import coverage_scores
from cognitive_discovery.sampling.disagreement_score import disagreement_scores
from cognitive_discovery.sampling.information_score import information_scores


def _candidate_table():
    return pd.DataFrame(
        {
            "paired_condition_id": ["a", "b", "c"],
            "semantic_hash": ["ha", "hb", "hc"],
            "task_family": ["bandit"] * 3,
            "factor_continuation_value": ["low", "medium", "high"],
            "factor_continuation_cost": ["low", "low", "high"],
        }
    )


def test_coverage_score_prioritizes_underrepresented_cells():
    observed = pd.DataFrame(
        {
            "task_family": ["bandit"] * 20,
            "factor_continuation_value": ["low"] * 20,
            "factor_continuation_cost": ["low"] * 20,
        }
    )
    result = coverage_scores(observed, _candidate_table())
    assert (
        result.loc[result.paired_condition_id == "c", "coverage_score"].item()
        > result.loc[result.paired_condition_id == "a", "coverage_score"].item()
    )


def test_information_score_selects_known_high_uncertainty_cell():
    candidates = _candidate_table()
    ensemble = [
        np.asarray([0.0, -4.0, 1.0]),
        np.asarray([0.0, 4.0, 1.1]),
        np.asarray([0.0, 0.0, 0.9]),
    ]
    result = information_scores(
        pd.DataFrame(), candidates, ensemble_predictions=ensemble
    )
    assert result.sort_values("information_score").iloc[-1].paired_condition_id == "b"


def test_disagreement_score_selects_known_divergence():
    candidates = _candidate_table()
    result = disagreement_scores(
        pd.DataFrame(),
        candidates,
        prediction_a=[0.0, 0.0, -3.0],
        prediction_b=[0.0, 0.1, 3.0],
    )
    assert result.sort_values("disagreement_score").iloc[-1].paired_condition_id == "c"


def test_active_mixture_has_registered_proportions_and_no_duplicates(discovery_config):
    conditions = compile_design(discovery_config, n_conditions=140, seed=991)
    candidates = condition_frame(conditions)
    candidates["coverage_score"] = np.linspace(0, 1, len(candidates))
    candidates["information_score"] = np.linspace(1, 0, len(candidates))
    candidates["disagreement_score"] = np.roll(np.linspace(0, 1, len(candidates)), 10)
    result = select_active_mixture(
        conditions,
        candidates,
        budget=70,
        excluded_hashes={candidates.iloc[0].semantic_hash},
    )
    counts = Counter(result.selected.sampling_strategy)
    assert counts == {"coverage": 28, "information": 28, "disagreement": 14}
    assert result.selected.semantic_hash.nunique() == 70
    assert candidates.iloc[0].semantic_hash not in set(result.selected.semantic_hash)
    assert len(result.conditions) == 140


def test_candidate_pool_excludes_previously_observed_design(discovery_config):
    previous = condition_frame(
        compile_design(discovery_config, n_conditions=28, seed=1234)
    )
    _, candidates = generate_candidate_pool(
        discovery_config,
        previous,
        budget=7,
        multiplier=10,
        seed=1234,
    )
    assert len(candidates) == 70
    assert set(previous.semantic_hash).isdisjoint(candidates.semantic_hash)


def test_semantic_hashes_are_stable_with_arrow_string_missingness(discovery_config):
    previous = condition_frame(
        compile_design(discovery_config, n_conditions=28, seed=1234)
    )
    factor_columns = [
        column
        for column in previous
        if column.startswith("factor_")
        and not column.startswith("factor_available_")
    ]
    for column in factor_columns:
        previous[column] = previous[column].astype("string[pyarrow]")
    assert observed_semantic_hashes(previous) == set(previous.semantic_hash)
    _, candidates = generate_candidate_pool(
        discovery_config,
        previous,
        budget=7,
        multiplier=10,
        seed=1234,
    )
    assert set(previous.semantic_hash).isdisjoint(candidates.semantic_hash)

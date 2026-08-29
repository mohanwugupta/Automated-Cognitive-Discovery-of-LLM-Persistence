from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cognitive_discovery.experiments.contextual_history import (
    compile_contextual_history_design,
)
from cognitive_discovery.sampling.candidate_pool import condition_frame
from cognitive_discovery.sampling.discrimination_mixture import (
    freeze_discrimination_split,
    select_discrimination_mixture,
)
from cognitive_discovery.sampling.model_uncertainty import model_uncertainty_scores
from cognitive_discovery.sampling.theory_disagreement import (
    theory_disagreement_scores,
)


def test_frozen_disagreement_and_uncertainty_scores_match_synthetic_examples():
    candidates = pd.DataFrame(
        {
            "paired_condition_id": ["a", "b", "c"],
            "semantic_hash": ["ha", "hb", "hc"],
            "task_family": ["bandit"] * 3,
        }
    )
    disagreement = theory_disagreement_scores(
        candidates,
        {
            "dual_history": np.array([0.0, 0.0, -3.0]),
            "latent_context": np.array([0.0, 0.1, 3.0]),
        },
    )
    assert disagreement.sort_values("disagreement_score").iloc[-1].paired_condition_id == "c"
    uncertainty = model_uncertainty_scores(
        pd.DataFrame(),
        candidates,
        architectures=("dual_history", "latent_context"),
        ensemble_predictions={
            "dual_history": [np.array([0, -4, 1]), np.array([0, 4, 1])],
            "latent_context": [np.array([0, -3, 1]), np.array([0, 3, 1])],
        },
    )
    assert uncertainty.sort_values("uncertainty_score").iloc[-1].paired_condition_id == "b"


def test_round3_outcomes_are_rejected_during_scoring():
    candidates = pd.DataFrame(
        {
            "paired_condition_id": ["a"],
            "semantic_hash": ["ha"],
            "task_family": ["bandit"],
            "persistence_logit": [1.0],
        }
    )
    with pytest.raises(ValueError, match="outcomes leaked"):
        theory_disagreement_scores(
            candidates,
            {"dual_history": [0.0], "latent_context": [1.0]},
        )


def test_discrimination_mixture_respects_allocation_domains_and_duplicates(theory_config):
    conditions = compile_contextual_history_design(
        theory_config, n_conditions=90, seed=71
    )
    scores = condition_frame(conditions)
    scores["domain"] = scores.task_family + ":contextual"
    scores["disagreement_score"] = np.linspace(0, 1, len(scores))
    scores["uncertainty_score"] = np.linspace(1, 0, len(scores))
    scores["coverage_score"] = np.roll(np.linspace(0, 1, len(scores)), 9)
    result = select_discrimination_mixture(
        conditions, scores, budget=30, seed=19
    )
    assert Counter(result.selected.sampling_strategy) == {
        "discriminating": 18,
        "coverage": 6,
        "random": 6,
    }
    assert result.selected.semantic_hash.nunique() == 30
    assert set(result.selected.groupby("sampling_strategy").domain.nunique()) == {3}
    assert set(
        result.selected.groupby("context_critical_contrast_id").size().unique()
    ) == {2}
    split = freeze_discrimination_split(result.conditions, fraction=0.25, seed=20)
    by_pair = {}
    for condition in split:
        by_pair.setdefault(condition.paired_condition_id, set()).add(condition.split)
    assert set(map(len, by_pair.values())) == {1}
    assert {next(iter(value)) for value in by_pair.values()} == {
        "round3_train",
        "model_discrimination",
    }
    contextual_splits = {}
    for condition in split:
        contextual_splits.setdefault(
            condition.contextual_history["critical_contrast_id"], set()
        ).add(condition.split)
    assert set(map(len, contextual_splits.values())) == {1}


def test_random_reference_is_independent_of_model_and_coverage_scores(theory_config):
    conditions = compile_contextual_history_design(theory_config, n_conditions=90, seed=72)
    scores = condition_frame(conditions)
    scores["domain"] = scores.task_family + ":contextual"
    scores["disagreement_score"] = np.linspace(0, 1, len(scores))
    scores["uncertainty_score"] = np.linspace(1, 0, len(scores))
    scores["coverage_score"] = np.linspace(0, 1, len(scores))
    first = select_discrimination_mixture(conditions, scores, budget=30, seed=22)
    changed = scores.copy()
    for column in ("disagreement_score", "uncertainty_score", "coverage_score"):
        changed[column] = changed[column].iloc[::-1].to_numpy()
    second = select_discrimination_mixture(conditions, changed, budget=30, seed=22)
    random_first = set(
        first.selected.loc[first.selected.sampling_strategy == "random", "semantic_hash"]
    )
    random_second = set(
        second.selected.loc[second.selected.sampling_strategy == "random", "semantic_hash"]
    )
    assert random_first == random_second

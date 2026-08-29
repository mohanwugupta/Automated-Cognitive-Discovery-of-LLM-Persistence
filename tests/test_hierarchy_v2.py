import numpy as np
import pandas as pd

from cognitive_discovery.analysis.hierarchical_comparison import hierarchical_loto
from cognitive_discovery.hierarchy.few_shot import evaluate_few_shot_adaptation
from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model
from cognitive_discovery.hierarchy.task_descriptors import (
    DESCRIPTOR_NAMES,
    TASK_DESCRIPTORS,
    descriptor_frame,
)
from cognitive_discovery.hierarchy.variance_decomposition import variance_decomposition
from cognitive_discovery.models.fitting import regression_metrics
from cognitive_discovery.models.synthetic import generate_hierarchical_teacher_data


def test_task_descriptor_ontology_is_frozen_and_small():
    frame = descriptor_frame()
    assert len(frame) == 7
    assert 4 <= len(DESCRIPTOR_NAMES) <= 6
    assert set(frame.task_family) == set(TASK_DESCRIPTORS)
    assert set(np.unique(frame[list(DESCRIPTOR_NAMES)].to_numpy())) <= {0, 1}


def test_shared_teacher_is_recovered_by_m1():
    train, test = generate_hierarchical_teacher_data(
        "M1", n_train=700, n_test=280, seed=4, noise=0.02
    )
    fit = fit_hierarchical_model(train, "dual_history", variant="M1")
    assert regression_metrics(test.persistence_logit, fit.predict(test))["r2"] > 0.99


def test_random_effect_teacher_is_recovered_by_m3():
    train, test = generate_hierarchical_teacher_data(
        "M3", n_train=700, n_test=280, seed=4, noise=0.02
    )
    m1 = fit_hierarchical_model(train, "dual_history", variant="M1")
    m3 = fit_hierarchical_model(train, "dual_history", variant="M3")
    r1 = regression_metrics(test.persistence_logit, m1.predict(test))["r2"]
    r3 = regression_metrics(test.persistence_logit, m3.predict(test))["r2"]
    assert r3 > 0.99
    assert r3 > r1 + 0.05


def test_ontology_teacher_is_recovered_zero_shot_by_m4():
    train, test = generate_hierarchical_teacher_data(
        "M4", n_train=700, n_test=280, seed=12, noise=0.02
    )
    frame = pd.concat([train, test], ignore_index=True)
    loto = hierarchical_loto(
        frame, architectures=("dual_history",), variants=("M3", "M4")
    )
    average = loto.groupby("variant").r2.mean()
    assert average["M4"] > average["M3"]
    assert not loto.heldout_outcomes_used_for_population.any()


def test_heldout_outcomes_never_contribute_to_ontology_coefficients():
    train, test = generate_hierarchical_teacher_data(
        "M4", n_train=700, n_test=280, seed=7
    )
    frame = pd.concat([train, test], ignore_index=True)
    heldout = "information_sampling"
    fit = fit_hierarchical_model(
        frame[frame.task_family != heldout], "dual_history", variant="M4"
    )
    assert heldout not in fit.outcome_tasks
    prediction = fit.predict(frame[frame.task_family == heldout], include_random=False)
    assert np.isfinite(prediction).all()


def test_few_shot_updates_only_target_random_effect():
    train, test = generate_hierarchical_teacher_data(
        "M4", n_train=700, n_test=280, seed=17
    )
    frame = pd.concat([train, test], ignore_index=True)
    result = evaluate_few_shot_adaptation(
        frame,
        architecture="dual_history",
        sample_sizes=(0, 4),
        seed=3,
    )
    for row in result.itertuples():
        assert row.heldout_task not in row.population_outcome_tasks.split(";")
    assert set(result.requested_semantic_conditions) == {0, 4}


def test_variance_decomposition_reports_raw_uncapped_fraction():
    train, _ = generate_hierarchical_teacher_data("M4", n_train=700, n_test=140, seed=9)
    fit = fit_hierarchical_model(train, "dual_history", variant="M4")
    result = variance_decomposition(fit)
    assert "ontology_fraction_q" in result
    assert len(result) == len(fit.feature_names)

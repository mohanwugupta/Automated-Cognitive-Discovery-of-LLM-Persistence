import numpy as np
import pandas as pd

from cognitive_discovery.audits.hierarchy_uncertainty import paired_hierarchy_bootstrap
from cognitive_discovery.audits.teacher_failures import classify_teacher_checks
from cognitive_discovery.theory_resolution.equivalence import resolve_theory_outcome
from cognitive_discovery.theory_resolution.paired_model_test import (
    compare_frozen_theories,
    semantic_prediction_errors,
)


def _synthetic_errors(winner: str, seed=1):
    rng = np.random.default_rng(seed)
    observed = rng.normal(size=240)
    if winner == "dual_history":
        dual = observed + rng.normal(0, 0.05, len(observed))
        context = observed + rng.normal(0, 0.8, len(observed))
    elif winner == "latent_context":
        dual = observed + rng.normal(0, 0.8, len(observed))
        context = observed + rng.normal(0, 0.05, len(observed))
    else:
        dual = context = observed + rng.normal(0, 0.1, len(observed))
    frame = pd.DataFrame(
        {
            "paired_condition_id": [f"p-{index}" for index in range(len(observed))],
            "task_family": np.resize(["bandit", "foraging", "debugging"], len(observed)),
            "split": "model_discrimination",
            "sampling_strategy": np.resize(
                ["discriminating", "coverage", "random"], len(observed)
            ),
            "persistence_logit": observed,
        }
    )
    return semantic_prediction_errors(
        frame, {"dual_history": dual, "latent_context": context}
    )


def _decision(errors):
    _, summary, metrics = compare_frozen_theories(
        errors, bootstraps=200, seed=12
    )
    return resolve_theory_outcome(summary, metrics, equivalence_margin=0.02)


def test_synthetic_direct_history_favors_dual_history():
    assert _decision(_synthetic_errors("dual_history"))["outcome"] == "dual_history"


def test_synthetic_latent_context_favors_latent_context():
    assert _decision(_synthetic_errors("latent_context"))["outcome"] == "latent_context"


def test_equivalent_models_report_equivalence_not_arbitrary_winner():
    decision = _decision(_synthetic_errors("equivalent"))
    assert decision["outcome"] == "observational_equivalence"
    assert decision["winner"] is None


def test_all_teacher_failures_are_surfaced_and_classified():
    checks = pd.DataFrame(
        {
            "teacher": ["fitted_dual_history", "fitted_latent_context"],
            "model": ["linear_interactions", "gru"],
            "sharing": ["fully_shared", "task_embedding"],
            "r2": [0.7, 0.99],
            "threshold": [0.98, 0.98],
            "passed": [False, True],
        }
    )
    result = classify_teacher_checks(checks, expected_checks=2)
    assert len(result) == 2
    assert result.loc[~result.passed, "failure_category"].item() == "sharing_mismatch"
    assert result.loc[~result.passed, "failure_reason"].str.len().gt(0).all()


def test_m2_m3_m4_bootstrap_uses_one_paired_semantic_split():
    frame = pd.DataFrame(
        {
            "paired_condition_id": [f"p-{index}" for index in range(30)],
            "task_family": np.resize(["bandit", "foraging", "debugging"], 30),
            "persistence_logit": np.linspace(-1, 1, 30),
        }
    )

    class FixedFit:
        def __init__(self, offset):
            self.offset = offset

        def predict(self, values):
            return values.persistence_logit.to_numpy() + self.offset

    fits = {
        ("dual_history", variant): FixedFit(offset)
        for variant, offset in (("M2", 0.0), ("M3", 0.1), ("M4", -0.1))
    }
    draws, summary = paired_hierarchy_bootstrap(
        frame,
        fits,
        architectures=("dual_history",),
        bootstraps=20,
        seed=8,
    )
    assert draws.paired_semantic_bootstrap.all()
    assert draws.split_digest.nunique() == summary.split_digest.nunique() == 1
    assert set(zip(summary.variant_a, summary.variant_b)) == {
        ("M2", "M3"),
        ("M2", "M4"),
        ("M3", "M4"),
    }

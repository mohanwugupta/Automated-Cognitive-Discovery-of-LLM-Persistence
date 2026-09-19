from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from cognitive_discovery.model_validation.pipeline import (
    implementation_checks,
    serialize_identifiability,
    validation_decision,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(
    (ROOT / "configs/validation/computational_models_v1.yaml").read_text(
        encoding="utf-8"
    )
)


def _recovery_matrix():
    probabilities = {
        "dual_history": {"dual_history": 1.0, "outcome_history": 0.0, "latent_context": 0.0},
        "latent_context": {"dual_history": 0.0, "outcome_history": 0.0, "latent_context": 1.0},
        "outcome_history": {"dual_history": 0.6, "outcome_history": 0.4, "latent_context": 0.0},
    }
    rows = []
    for teacher, candidates in probabilities.items():
        for candidate, probability in candidates.items():
            rows.append(
                {
                    "teacher_model": teacher,
                    "candidate_model": candidate,
                    "recovery_probability": probability,
                    "teacher_recovered": teacher == candidate,
                }
            )
    return pd.DataFrame(rows)


def test_implementation_failure_blocks_but_identifiability_ambiguity_continues():
    failed = validation_decision(
        implementation_passed=False, identifiability_status="well_identified", smoke=False
    )
    assert failed == {
        "implementation_validity": "fail",
        "identifiability_status": "well_identified",
        "pipeline_permission": "stop",
    }
    ambiguous = validation_decision(
        implementation_passed=True, identifiability_status="partial", smoke=False
    )
    assert ambiguous == {
        "implementation_validity": "pass",
        "identifiability_status": "partial",
        "pipeline_permission": "continue",
    }


def test_ambiguous_recovery_matrix_serializes_pairwise_diagnostics():
    result = serialize_identifiability(_recovery_matrix(), CONFIG)
    assert result["blocking"] is False
    assert result["identifiability_status"] == "partial"
    pair = next(
        row
        for row in result["pairs"]
        if row["teacher_model"] == "outcome_history"
        and row["confused_with"] == "dual_history"
    )
    assert pair["teacher_recovery_probability"] == 0.4
    assert pair["confusion_probability"] == 0.6
    assert pair["identifiability_status"] == "nested_or_equivalent"
    assert set(pair) == {
        "teacher_model",
        "confused_with",
        "teacher_recovery_probability",
        "confusion_probability",
        "identifiability_status",
        "notes",
    }


def test_runtime_implementation_validity_checks_pass_core_fixtures():
    result = implementation_checks()
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["counterfactual_sign_convention"] == "source_minus_base"


def test_validation_config_splits_blocking_implementation_from_nonblocking_identity():
    assert CONFIG["implementation_gate"]["blocking"] is True
    assert CONFIG["identifiability_gate"]["blocking"] is False
    assert CONFIG["behavioral_survivor_set"] == {
        "metric": "selection_mse",
        "equivalence_rule": "absolute_score_margin",
        "equivalence_margin": 0.02,
    }


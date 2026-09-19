from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from cognitive_discovery.replication.config import ReplicationConfigError, load_replication_config
from cognitive_discovery.replication.neural import THEORY_TARGET, mechanism_candidate_id
from cognitive_discovery.replication.report import build_replication_report
from cognitive_discovery.replication.survivors import (
    SurvivorSetError,
    select_behavioral_survivors,
    validate_frozen_survivor_set,
)


ROOT = Path(__file__).resolve().parents[1]


def _select(scores, margin=0.02):
    return select_behavioral_survivors(
        scores,
        metric="selection_mse",
        direction="minimize",
        equivalence_rule="absolute_score_margin",
        equivalence_margin=margin,
    )


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"dual_history": 0.10, "outcome_history": 0.20}, ["dual_history"]),
        ({"dual_history": 0.10, "outcome_history": 0.11}, ["dual_history", "outcome_history"]),
        (
            {"dual_history": 0.10, "outcome_history": 0.11, "latent_context": 0.12},
            ["dual_history", "latent_context", "outcome_history"],
        ),
        (
            {"dual_history": 0.10, "outcome_history": 0.11, "latent_context": 0.30},
            ["dual_history", "outcome_history"],
        ),
    ],
)
def test_behavioral_survivor_rule_retains_only_behaviorally_equivalent_models(scores, expected):
    result = _select(scores)
    assert result["behavioral_survivor_set"] == expected
    assert result["selected_models"] == expected
    assert result["selected_model"] == result["best_model"]
    assert result["behavioral_theory_status"] == (
        "resolved" if len(expected) == 1 else "unresolved"
    )
    assert result["neural_results_used"] is False


def test_neural_outcomes_cannot_alter_survivor_membership():
    scores = {"dual_history": 0.10, "outcome_history": 0.11}
    before = _select(scores)
    neural_results = {"dual_history": -0.4, "outcome_history": 0.9}
    assert neural_results  # separate evidential axis, never accepted by _select
    after = _select(scores)
    assert after["behavioral_survivor_set"] == before["behavioral_survivor_set"]
    assert after["membership_sha256"] == before["membership_sha256"]
    with pytest.raises(TypeError):
        select_behavioral_survivors(
            scores,
            metric="selection_mse",
            direction="minimize",
            equivalence_rule="absolute_score_margin",
            equivalence_margin=0.02,
            neural_results=neural_results,
        )


def test_survivor_rule_and_membership_are_frozen_before_neural_execution(tmp_path):
    record = _select({"dual_history": 0.10, "outcome_history": 0.11})
    model_root = tmp_path / "behavior/models"
    frozen_root = model_root / "frozen_models"
    frozen_root.mkdir(parents=True)
    selection = {
        "behavioral_survivor_set": record["behavioral_survivor_set"],
        "selected_models": record["selected_models"],
    }
    (model_root / "selected_models.json").write_text(json.dumps(selection))
    (frozen_root / "frozen_architectures.json").write_text(
        json.dumps(record["behavioral_survivor_set"])
    )
    manifest = {
        **record,
        "frozen_before_neural_execution": True,
    }
    manifest_path = model_root / "frozen_survivor_set.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    provenance = {
        "behavioral_survivor_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "behavioral_survivor_rule_sha256": record["rule_sha256"],
        "behavioral_survivor_set_sha256": record["membership_sha256"],
    }
    assert validate_frozen_survivor_set(tmp_path, provenance)[
        "behavioral_survivor_set"
    ] == record["behavioral_survivor_set"]

    selection["behavioral_survivor_set"] = ["outcome_history"]
    (model_root / "selected_models.json").write_text(json.dumps(selection))
    with pytest.raises(SurvivorSetError, match="membership changed"):
        validate_frozen_survivor_set(tmp_path, provenance)


def test_each_survivor_has_separate_target_and_search_namespace():
    survivors = ["dual_history", "outcome_history", "latent_context"]
    assert all(theory in THEORY_TARGET for theory in survivors)
    candidate_ids = [mechanism_candidate_id(theory, 16, 8) for theory in survivors]
    assert len(set(candidate_ids)) == len(survivors)
    assert all(theory in candidate for theory, candidate in zip(survivors, candidate_ids))
    config = yaml.safe_load(
        (ROOT / "configs/replication/default.yaml").read_text(encoding="utf-8")
    )
    assert config["endpoint_id"] == "cognitive_counterfactual_recovery"


def test_legacy_and_canonical_survivor_margins_cannot_silently_diverge():
    with pytest.raises(ReplicationConfigError, match="compatibility alias"):
        load_replication_config(
            ROOT / "configs/replication/default.yaml",
            model_id="example/model",
            revision="a" * 40,
            adapter="qwen",
            overrides={
                "behavioral_survivor_set": {"equivalence_margin": 0.03}
            },
        )


def test_report_keeps_behavioral_and_neural_evidence_separate_by_survivor():
    report = build_replication_report(
        {
            "model": {"id": "example", "revision": "a" * 40, "adapter": "qwen"},
            "interface": {"passed": True},
            "behavior": {
                "passed": True,
                "theory_status": "unresolved",
                "best_model": "dual_history",
                "behavioral_survivor_set": ["dual_history", "outcome_history"],
            },
            "mechanism": {
                "controllers": [
                    {"theory": "dual_history", "layer": 10, "rank": 2},
                    {"theory": "outcome_history", "layer": 12, "rank": 2},
                ],
                "generalization": [
                    {
                        "theory": theory,
                        "split": split,
                        "global_cfr": value,
                        "cfr_ci_low": 0.1,
                        "cfr_ci_high": 0.4,
                        "correlation": 0.5,
                    }
                    for theory, value in (("dual_history", 0.3), ("outcome_history", 0.2))
                    for split in ("neural_test", "neural_task_holdout")
                ],
                "endpoint_id": "cognitive_counterfactual_recovery",
                "metric_id": "global_cfr_v1",
            },
            "specificity": {
                "random_control_passed": True,
                "by_theory": {
                    "dual_history": {"passed": True},
                    "outcome_history": {"passed": False},
                },
            },
        },
        {},
    )
    assert "Behavioral theory uniquely resolved:    NO / UNRESOLVED" in report
    assert "| dual_history | not_run | L10/rank2 | 0.3 [0.1, 0.4] | 0.3 [0.1, 0.4] | not_run | not_run | pass |" in report
    assert "| outcome_history | not_run | L12/rank2 | 0.2 [0.1, 0.4] | 0.2 [0.1, 0.4] | not_run | not_run | fail |" in report
    assert "neural CFR cannot change its membership" in report

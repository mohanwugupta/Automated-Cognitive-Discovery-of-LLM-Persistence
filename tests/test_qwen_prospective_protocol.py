from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cognitive_discovery.replication.harness import (
    initialize_replication,
    load_prospective_run_spec,
)
from cognitive_discovery.replication.report import build_replication_report


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/replication/qwen_prospective_run.yaml"


def test_qwen_run_spec_freezes_identity_freshness_endpoint_and_clean_baseline():
    spec = load_prospective_run_spec(ROOT, SPEC)
    assert spec["analysis_id"] == "qwen-prospective"
    assert spec["interpretation"] == "pipeline_self_replication"
    assert spec["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert spec["endpoint_id"] == "cognitive_counterfactual_recovery"
    assert spec["metric_id"] == "global_cfr_v1"
    assert spec["baseline"]["require_git_dirty_false"] is True
    assert spec["baseline"]["require_full_cpu_tests"] is True
    assert spec["gates"]["mutable_after_baseline"] is False
    assert all(value is False for value in spec["freshness"].values())
    assert set(spec["excluded_endpoints"]) == {
        "natural_effect_recovery",
        "fresh_context_extension",
    }


def test_qwen_run_spec_rejects_any_historical_reuse(tmp_path):
    value = SPEC.read_text(encoding="utf-8").replace(
        "reuse_historical_neural_splits: false",
        "reuse_historical_neural_splits: true",
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(value, encoding="utf-8")
    with pytest.raises(ValueError, match="historical reuse"):
        load_prospective_run_spec(ROOT, path)


def test_qwen_report_contains_descriptive_historical_table_and_unsmoothed_fork():
    results = {
        "model": {"id": "Qwen/Qwen3.5-4B", "revision": "a" * 40, "adapter": "qwen"},
        "interface": {"passed": True, "selected": "yes_no", "approved_task_count": 7},
        "behavior": {
            "passed": True,
            "replication_status": "pass",
            "theory_status": "resolved",
            "best_model": "dual_history",
            "behavioral_survivor_set": ["dual_history"],
            "heldout_by_theory": {"dual_history": {"r2": 0.8, "mse": 0.1}},
        },
        "mechanism": {
            "endpoint_id": "cognitive_counterfactual_recovery",
            "metric_id": "global_cfr_v1",
            "controllers": [{"theory": "dual_history", "layer": 28, "rank": 2}],
            "generalization": [
                {
                    "theory": "dual_history", "split": split,
                    "task_family": None, "global_cfr": cfr,
                    "cfr_ci_low": low, "cfr_ci_high": high, "correlation": corr,
                }
                for split, cfr, low, high, corr in (
                    ("neural_test", 0.5, 0.2, 0.7, 0.5),
                    ("neural_task_holdout", -0.1, -0.3, 0.1, -0.1),
                )
            ],
        },
        "specificity": {"random_control_passed": False, "by_theory": {}},
        "historical_comparison": {
            "behavioral_theory_status": "unresolved",
            "surviving_theories": ["dual_history", "latent_context", "outcome_history"],
            "selected_layer_rank": "L28/rank2 primary",
            "familiar_task_cfr": 0.968265,
            "whole_task_cfr": 0.872806,
            "specificity": "partial",
            "context_only": True,
        },
    }
    report = build_replication_report(results, {"git_commit": "b" * 40})
    assert "Historical Qwen | Prospective Qwen" in report
    assert "0.968265" in report
    assert "Outcome B — prospective Qwen loses whole-task transfer" in report
    assert "historical pipeline/design" in report
    assert "Do not repair or tune this outcome" in report


def test_expected_primary_output_directories_are_named_in_protocol():
    spec = load_prospective_run_spec(ROOT, SPEC)
    assert spec["required_output_directories"] == [
        "interface",
        "behavior",
        "frozen_theories",
        "counterfactuals",
        "mechanism",
        "specificity",
    ]


def test_prospective_initialization_records_clean_commit_lock_and_preflight(
    tmp_path, monkeypatch
):
    commit = "c" * 40
    monkeypatch.setattr(
        "cognitive_discovery.replication.harness._git",
        lambda root, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    lock_hash = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "prospective-baseline-preflight-v1",
                "git_commit": commit,
                "git_dirty": False,
                "uv_lock_sha256": lock_hash,
                "cpu_test_status": "pass",
                "cpu_test_command": "PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "qwen-prospective"
    initialize_replication(
        root=ROOT,
        config_path=ROOT / "configs/replication/default.yaml",
        model_id="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        adapter="qwen",
        tokenizer_id="Qwen/Qwen3.5-4B",
        tokenizer_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        output=output,
        run_spec_path=SPEC,
        baseline_preflight_path=preflight,
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["git_commit"] == commit
    assert provenance["git_dirty"] is False
    assert provenance["environment"]["lockfile_sha256"] == lock_hash
    assert provenance["run_kind"] == "pipeline_self_replication"
    assert provenance["gates_mutable_after_baseline"] is False

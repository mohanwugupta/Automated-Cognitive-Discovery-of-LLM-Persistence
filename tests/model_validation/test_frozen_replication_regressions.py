import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("directory", "theory", "test_cfr", "holdout_cfr"),
    [
        ("gemma-4-12b-restart-20260918-142538", "latent_context", 0.3900876596104764, -0.014981347829789105),
        ("llama-3.1-8b-restart-20260918-142501", "latent_context", 0.8286623182758288, -4.6092019523694265),
    ],
)
def test_completed_replication_metrics_remain_frozen(directory, theory, test_cfr, holdout_cfr):
    path = ROOT / "replications" / directory / "mechanism/generalization_metrics.csv"
    if not path.is_file():
        pytest.skip("lightweight checkout does not carry the completed replication")
    frame = pd.read_csv(path)
    aggregate = frame[(frame.theory == theory) & frame.task_family.isna()].set_index("split")
    assert aggregate.loc["neural_test", "global_cfr"] == pytest.approx(test_cfr)
    assert aggregate.loc["neural_task_holdout", "global_cfr"] == pytest.approx(holdout_cfr)
    assert set(aggregate.endpoint_id) == {"cognitive_counterfactual_recovery"}
    assert set(aggregate.metric_id) == {"global_cfr_v1"}


def test_external_benchmark_boundary_is_unresolved_but_nonblocking():
    text = (ROOT / "validation/external_benchmark/BENCHMARK_SPEC.md").read_text()
    assert "OWNER-FROZEN" in text
    assert "OWNER_REQUIRED" not in text
    assert (ROOT / "validation/external_benchmark/DATASET_MANIFEST.json").is_file()
    assert not (ROOT / "validation/external_benchmark/results.csv").exists()
    status = json.loads(
        (ROOT / "validation/external_benchmark/benchmark_execution_status.json").read_text()
    )
    assert status["execution_status"] == "owner_review"
    assert status["gate_b_status"] == "not_evaluable"
    assert status["implementation_equivalence"] == "pass"
    assert status["benchmark_claim_status"] == "unresolved"
    assert status["benchmark_claim_permission"] == "stop"
    assert status["pipeline_permission"] == "continue"
    assert status["converged_map_fits"] == 858
    assert status["positive_definite_exact_hessians"] == 852
    assert status["failed_exact_hessians"] == 6
    assert status["primary_contrasts_computed"] is False
    assert status["bootstrap_computed"] is False
    assert status["model_ranking_computed"] is False
    assert status["results_csv_emitted"] is False
    assert {row["subject_id"] for row in status["failure_records"]} == {22, 55, 71, 105}


def test_boundary_v2_requirements_are_universal_and_prohibit_repairs():
    text = (
        ROOT
        / "validation/external_benchmark/V2_BOUNDARY_METHOD_REQUIREMENTS.md"
    ).read_text()
    assert "method_not_selected" in text
    assert "every active" in text
    assert "all 143" in text
    for prohibited in (
        "Hessian jitter",
        "eigenvalue flooring",
        "absolute determinants",
        "participant exclusion",
        "deletion of a boundary parameter",
    ):
        assert prohibited in text

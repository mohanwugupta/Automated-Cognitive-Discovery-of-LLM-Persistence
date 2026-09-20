import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cognitive_discovery.transfer.config import load_transfer_config
from cognitive_discovery.transfer.matrix import (
    RESPONSE_MAPPINGS,
    classify_transfer,
    evaluate_source_validity,
    unavailable_transfer_rows,
    write_matrix_artifacts,
)
from cognitive_discovery.transfer.predictors import (
    build_task_structure_predictors,
    load_predictor_spec,
    predictor_spec_hash,
)


ROOT = Path(__file__).resolve().parents[2]


def _metric(mapping, cfr=.4, low=.1, high=.7, correlation=.5, random_p=.01):
    return {
        "model": "qwen", "theory": "dual_history", "source_task": "bandit",
        "target_task": "bandit", "response_mapping": mapping,
        "layer": 10, "rank": 2, "n_train": 20, "n_selection": 8, "n_test": 8,
        "global_cfr": cfr, "bootstrap_low": low, "bootstrap_high": high,
        "correlation": correlation, "slope": .8, "random_mean": .01,
        "random_max": .08, "random_p": random_p,
        "controller_hash": "a" * 64, "split_hash": "b" * 64,
        "endpoint_id": "cognitive_counterfactual_recovery",
        "metric_id": "global_cfr_v1", "status": "available",
    }


def test_source_validity_is_five_part_and_mapping_robust():
    config = load_transfer_config(ROOT / "configs/transfer/v1.yaml")
    rule = config["source_validity"]
    pooled = _metric("pooled")
    mappings = [_metric("continue_x", .42), _metric("continue_y", .38)]
    result = evaluate_source_validity(pooled, mappings, rule)
    assert result["source_valid"] is True
    assert all(result["criteria"].values())
    assert result["mapping_gap"] == pytest.approx(.04)

    failed_random = evaluate_source_validity({**pooled, "random_p": .2}, mappings, rule)
    assert failed_random["source_valid"] is False
    assert failed_random["source_status"] == "invalid_source_controller"
    assert failed_random["criteria"]["beats_matched_random_subspace_null"] is False

    failed_mapping = evaluate_source_validity(
        pooled, [_metric("continue_x", .5), _metric("continue_y", -.1)], rule
    )
    assert failed_mapping["source_valid"] is False
    assert failed_mapping["criteria"]["response_mapping_robustness_passes"] is False


def test_config_freezes_mapping_gate_and_predictor_spec():
    config = load_transfer_config(ROOT / "configs/transfer/v1.yaml")
    assert tuple(config["response_mappings"]) == RESPONSE_MAPPINGS
    assert config["source_validity"]["random_p_max"] == .05
    assert config["source_validity"]["response_mapping"]["maximum_global_cfr_gap"] == .25
    assert config["source_validity"]["response_mapping"]["require_positive_each_mapping"] is True
    spec = load_predictor_spec(ROOT / config["predictors"]["spec_path"])
    assert set(spec["families"]) == {
        "behavioral_computational", "task_structure", "counterfactual_geometry", "neural_similarity"
    }
    assert len(predictor_spec_hash(spec)) == 64
    assert set(spec["task_structure"]["tasks"]) == set(config["tasks"])
    first = build_task_structure_predictors(spec)
    second = build_task_structure_predictors(spec)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 49
    assert not ({"global_cfr", "transfer_status", "source_valid"} & set(first))


def test_predictor_spec_rejects_transfer_outcome_leakage(tmp_path):
    source = ROOT / "configs/transfer/transfer_predictors_v1.yaml"
    text = source.read_text(encoding="utf-8").replace(
        "source: frozen_task_ontology", "source: global_cfr", 1
    )
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="outcome leakage"):
        load_predictor_spec(path)


def test_invalid_sources_are_unavailable_for_each_mapping_not_zero():
    rows = unavailable_transfer_rows(
        model="qwen", theory="dual_history", source_task="bandit",
        target_tasks=["effort"], layer=10, rank=2, n_train=20,
        n_selection=8, controller_hash="a" * 64, split_hash="b" * 64,
    )
    assert set(rows.response_mapping) == {"pooled", *RESPONSE_MAPPINGS}
    assert set(rows.transfer_status) == {"unavailable"}
    assert rows.global_cfr.isna().all()


def test_transfer_status_preserves_uncertainty():
    assert classify_transfer(_metric("pooled"), source_valid=True) == "positive_transfer"
    assert classify_transfer(_metric("pooled", cfr=-.4, low=-.7, high=-.1), source_valid=True) == "negative_transfer"
    assert classify_transfer(_metric("pooled", cfr=.1, low=-.2, high=.3), source_valid=True) == "inconclusive"
    assert classify_transfer(_metric("pooled"), source_valid=False) == "unavailable"


def test_required_diagonal_and_matrix_artifacts_are_mapping_indexed(tmp_path):
    rows = []
    for source in ("bandit", "effort"):
        for target in ("bandit", "effort"):
            for mapping in ("pooled", *RESPONSE_MAPPINGS):
                row = _metric(mapping)
                row.update(source_task=source, target_task=target)
                rows.append(row)
    metrics = pd.DataFrame(rows)
    gates = pd.DataFrame([
        {"model": "qwen", "theory": "dual_history", "source_task": task,
         "source_valid": True, "source_status": "valid_source_controller",
         "mapping_gap": .04, "mapping_sign_consistent": True,
         "response_mapping_robustness_passes": True}
        for task in ("bandit", "effort")
    ])
    controllers = pd.DataFrame([
        {"model": "qwen", "theory": "dual_history", "task": task,
         "layer": 10, "rank": 2, "controller_hash": "a" * 64,
         "split_hash": "b" * 64, "n_train": 20, "n_selection": 8}
        for task in ("bandit", "effort")
    ])
    write_matrix_artifacts(tmp_path, metrics, controllers, gates, task_order=["bandit", "effort"])
    expected = [
        "diagonal/controller_manifest.csv", "diagonal/diagonal_results.csv",
        "diagonal/source_validity.csv", "diagonal/response_mapping_results.csv",
        "matrix/transfer_long.csv", "matrix/transfer_summary.csv",
        "matrix/transfer_matrix_dual_history_continue_x.csv",
        "matrix/transfer_matrix_dual_history_continue_y.csv",
    ]
    assert all((tmp_path / path).is_file() for path in expected)
    long = pd.read_csv(tmp_path / "matrix/transfer_long.csv")
    assert set(long.response_mapping) == set(RESPONSE_MAPPINGS)
    assert set(long.transfer_status) == {"positive_transfer"}

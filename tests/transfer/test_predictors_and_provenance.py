import pandas as pd
import pytest

from cognitive_discovery.transfer.predictors import PREDICTOR_FAMILIES, fit_heldout_target_predictor
from cognitive_discovery.transfer.provenance import validate_transfer_provenance


def test_predictor_validation_holds_out_complete_target_tasks():
    rows = []
    for source in ("a", "b", "c"):
        for target in ("a", "b", "c"):
            row = {"source_task": source, "target_task": target, "global_cfr": .2}
            for columns in PREDICTOR_FAMILIES.values():
                for column in columns: row[column] = float(source == target)
            rows.append(row)
    predictions, coefficients = fit_heldout_target_predictor(pd.DataFrame(rows))
    assert set(predictions.heldout_target) == {"a", "b", "c"}
    assert all(predictions.target_task == predictions.heldout_target)
    assert not coefficients.empty


def test_final_provenance_rejects_dirty_worktree():
    record = {
        "schema_version": "task-transfer-provenance-v1", "git_commit": "a" * 40,
        "git_dirty": True, "diff_sha256": "b" * 64, "model": {"id": "m", "revision": "c" * 40},
        "tokenizer": {"id": "m", "revision": "c" * 40}, "adapter": {"name": "test", "version": "v1"}, "environment_lock_sha256": "d" * 64,
        "specification_hashes": {"x": "e" * 64}, "pair_manifest_sha256": "f" * 64,
        "counterfactual_pair_sha256": "2" * 64, "counterfactual_prediction_sha256": "3" * 64,
        "replication_neural_split_sha256": "4" * 64, "transfer_split_sha256": "1" * 64,
        "source_validity_rule_sha256": "5" * 64, "transfer_predictor_spec_sha256": "6" * 64,
        "endpoint_id": "cognitive_counterfactual_recovery",
        "metric_id": "global_cfr_v1", "seeds": {"x": 1}, "output_root": "/tmp/x",
    }
    validate_transfer_provenance(record, final=False)
    with pytest.raises(ValueError, match="clean worktree"):
        validate_transfer_provenance(record, final=True)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cognitive_discovery.cross_task.acquisition import select_active_batch
from cognitive_discovery.cross_task.behavior import (
    estimate_behavioral_effects,
    semantic_persistence_logit,
)
from cognitive_discovery.cross_task.causal import validate_causal_transfer_rows
from cognitive_discovery.cross_task.compatibility import load_compatibility_matrix
from cognitive_discovery.cross_task.config import load_cross_task_config
from cognitive_discovery.cross_task.provenance import validate_final_provenance
from cognitive_discovery.cross_task.representation import (
    evaluate_frozen_decoder,
    fit_source_decoder,
    run_representational_transfer,
)
from cognitive_discovery.cross_task.splits import assign_grouped_splits
from cognitive_discovery.cross_task.pipeline import initialize_run, prepare_shared_design


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_config_has_three_pinned_models_and_distinct_evidence_levels():
    config = load_cross_task_config(ROOT / "configs/discovery/cross_task_v1.yaml")
    assert [model["key"] for model in config["models"]] == ["qwen", "gemma", "llama"]
    assert all(len(model["revision"]) == 40 for model in config["models"])
    assert config["endpoints"]["behavior"] == "semantic_persistence_logit_v1"
    assert config["endpoints"]["representation"] == "frozen_source_decoding_v1"
    assert config["endpoints"]["causal"] == "cognitive_counterfactual_recovery"
    assert config["artifact_policy"]["save_full_activations"] is False


def test_compatibility_matrix_is_rectangular_but_does_not_fabricate_manipulations():
    matrix = load_compatibility_matrix(
        ROOT / "configs/discovery/task_variable_compatibility_v1.yaml"
    )
    assert len(matrix) == 7 * 13
    assert matrix.groupby(["task", "variable"]).size().eq(1).all()
    contextual = matrix[matrix.variable == "contextual_history"].set_index("task")
    assert contextual.loc["bandit", "manipulable"]
    assert not contextual.loc["solvability", "manipulable"]
    unavailable = matrix[~matrix.manipulable]
    assert unavailable.levels.map(len).eq(0).all()
    assert unavailable.notes.str.len().gt(0).all()


def test_semantic_logit_and_mapping_pair_grouping_are_frozen():
    assert semantic_persistence_logit(0.8, 0.2) == pytest.approx(np.log(4.0))
    rows = []
    for task in ("bandit", "effort"):
        for group in range(12):
            for mapping in ("continue_x", "continue_y"):
                rows.append({"task_family": task, "semantic_group": f"{task}-{group}", "response_mapping": mapping})
    split = assign_grouped_splits(
        pd.DataFrame(rows), group_columns=["semantic_group"],
        stratify_columns=["task_family"], seed=17,
    )
    assert split.groupby("semantic_group").analysis_split.nunique().max() == 1
    assert set(split.analysis_split) == {"train", "selection", "test"}


def test_behavioral_effect_recovers_direction_and_mapping_interaction():
    rows = []
    for index in range(40):
        for level, sign in (("low", -1.0), ("high", 1.0)):
            for mapping, offset in (("continue_x", 0.0), ("continue_y", 0.05)):
                rows.append({
                    "model": "qwen", "task_family": "bandit",
                    "condition_id": f"{index}-{level}-{mapping}",
                    "semantic_group": f"{index}-{level}",
                    "response_mapping": mapping, "analysis_split": "test",
                    "factor_continuation_value": level,
                    "persistence_logit": 0.75 * sign + offset,
                })
    effects = estimate_behavioral_effects(
        pd.DataFrame(rows), variables=["continuation_value"],
        bootstrap_samples=200, seed=91,
    )
    row = effects.iloc[0]
    assert row.effect > 1.4
    assert row.ci_low > 0
    assert abs(row.mapping_interaction) < 0.1
    assert row.endpoint_id == "semantic_persistence_logit_v1"


def test_active_acquisition_is_deterministic_and_excludes_observed_groups():
    frame = pd.DataFrame({
        "condition_id": [f"c{i}" for i in range(8)],
        "semantic_group": [f"g{i}" for i in range(8)],
        "coverage": np.arange(8), "uncertainty": np.arange(8)[::-1],
        "disagreement": [0, 1] * 4,
    })
    kwargs = dict(
        batch_size=3, observed_groups={"g0", "g1"},
        weights={"coverage": .4, "uncertainty": .4, "disagreement": .2}, seed=7,
    )
    first = select_active_batch(frame, **kwargs)
    second = select_active_batch(frame.sample(frac=1, random_state=4), **kwargs)
    assert first.condition_id.tolist() == second.condition_id.tolist()
    assert not set(first.semantic_group) & {"g0", "g1"}


def test_source_decoder_is_frozen_for_cross_task_evaluation():
    x = np.arange(30, dtype=float).reshape(10, 3)
    y = x[:, 0] - 0.5 * x[:, 1]
    decoder = fit_source_decoder(
        x, y, model="qwen", source_task="bandit", variable="progress",
        layer=8, semantic_group_ids=[f"s{i}" for i in range(10)], alpha=1.0,
    )
    before = decoder.coefficients.copy()
    result = evaluate_frozen_decoder(
        decoder, x + 1, y, target_task="effort",
        semantic_group_ids=[f"t{i}" for i in range(10)],
    )
    np.testing.assert_array_equal(before, decoder.coefficients)
    assert result["fit_task"] == "bandit"
    assert result["target_task"] == "effort"
    assert result["refit_on_target"] is False


def test_representation_stage_writes_pairwise_loto_and_sharing_outputs(tmp_path):
    rows = []
    for task_index, task in enumerate(("bandit", "effort")):
        for split in ("train", "test"):
            for group in range(5):
                for mapping in ("continue_x", "continue_y"):
                    value = float(group + task_index)
                    rows.append({
                        "model": "qwen", "task_family": task, "variable": "progress",
                        "layer": 8, "semantic_group": f"{task}-{split}-{group}",
                        "response_mapping": mapping, "analysis_split": split,
                        "target": value, "feature_0": value,
                        "feature_1": value * .5 + (mapping == "continue_y") * .01,
                    })
    result = run_representational_transfer(pd.DataFrame(rows), output=tmp_path)
    assert result["decoder_count"] == 2
    assert result["loto_rows"] == 2
    assert result["sharing_rows"] == 4
    assert all((tmp_path / name).is_file() for name in (
        "within_task.csv", "transfer_long.csv", "loto_transfer.csv",
        "shared_private_models.csv", "decoder_manifest.csv", "neural_split_manifest.json",
    ))


def test_causal_transfer_rejects_target_refit_and_basis_hash_drift():
    valid = pd.DataFrame([{
        "model": "qwen", "source_task": "bandit", "target_task": "effort",
        "variable": "progress", "basis_sha256": "a" * 64,
        "registered_basis_sha256": "a" * 64, "fit_tasks": '["bandit"]',
        "endpoint_id": "cognitive_counterfactual_recovery",
        "metric_id": "global_cfr_v1", "global_cfr": .3,
    }])
    validate_causal_transfer_rows(valid)
    with pytest.raises(ValueError, match="target-task refitting"):
        validate_causal_transfer_rows(valid.assign(fit_tasks='["bandit", "effort"]'))
    with pytest.raises(ValueError, match="basis drift"):
        validate_causal_transfer_rows(valid.assign(basis_sha256="b" * 64))


def test_final_provenance_fails_closed_on_dirty_or_unknown_identity(tmp_path):
    record = {
        "schema_version": "cross-task-provenance-v1", "git_commit": "a" * 40,
        "git_dirty": False, "environment_lock_sha256": "b" * 64,
        "config_sha256": "c" * 64, "compatibility_sha256": "d" * 64,
        "design_sha256": "e" * 64, "behavior_split_sha256": "f" * 64,
        "computational_model_spec_sha256": "1" * 64,
        "active_acquisition_config_sha256": "2" * 64,
        "scientific_freeze_sha256": "3" * 64,
        "representation_target_manifest_sha256": "4" * 64,
        "models": [
            {"key": key, "revision": revision, "tokenizer_revision": revision}
            for key, revision in (("qwen", "1" * 40), ("gemma", "2" * 40), ("llama", "3" * 40))
        ],
        "behavioral_survivor_hashes": {key: "5" * 64 for key in ("qwen", "gemma", "llama")},
        "neural_split_hashes": {key: "6" * 64 for key in ("qwen", "gemma", "llama")},
        "subspace_controller_hashes": {key: ["7" * 64] for key in ("qwen", "gemma", "llama")},
        "endpoint_ids": {
            "behavior": "semantic_persistence_logit_v1",
            "representation": "frozen_source_decoding_v1",
            "causal": "cognitive_counterfactual_recovery",
            "causal_metric": "global_cfr_v1",
        },
        "seeds": {"design": 1},
    }
    validate_final_provenance(record)
    with pytest.raises(ValueError, match="clean worktree"):
        validate_final_provenance({**record, "git_dirty": True})
    bad = json.loads(json.dumps(record))
    bad["models"][0]["revision"] = None
    with pytest.raises(ValueError, match="immutable model revision"):
        validate_final_provenance(bad)


def test_smoke_design_is_shared_hash_complete_and_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = ROOT / "configs/discovery/cross_task_v1.yaml"
    initialize_run(ROOT, first, config, final=False)
    initialize_run(ROOT, second, config, final=False)
    one = prepare_shared_design(ROOT, first, smoke=True)
    two = prepare_shared_design(ROOT, second, smoke=True)
    assert one["jsonl_sha256"] == two["jsonl_sha256"]
    assert one["behavior_split_sha256"] == two["behavior_split_sha256"]
    assert one["response_mapping_pairing_valid"] is True
    assert all((first / model / "provenance.json").is_file() for model in ("qwen", "gemma", "llama"))
    assert all((first / model / "stage_state.json").is_file() for model in ("qwen", "gemma", "llama"))
    provenance = json.loads((first / "provenance.json").read_text())
    assert len(provenance["design_sha256"]) == 64
    assert len(provenance["behavior_split_sha256"]) == 64


def test_della_wrappers_use_gpu_only_for_model_forward_jobs():
    gpu = (ROOT / "slurm/run_cross_task_discovery_gpu.slurm").read_text()
    cpu = (ROOT / "slurm/run_cross_task_discovery_cpu.slurm").read_text()
    submit = (ROOT / "scripts/submit_cross_task_discovery.sh").read_text()
    assert "#SBATCH --gres=gpu:1" in gpu
    assert "#SBATCH --partition=gpu" not in gpu
    assert "set +u" in gpu and "set -u" in gpu
    assert "#SBATCH --gres" not in cpu
    assert "afterok:$interface" in submit
    assert "afterok:$behavior" in submit

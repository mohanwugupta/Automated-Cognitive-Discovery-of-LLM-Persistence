from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_discovery.reproducibility.identities import (
    CORE_QWEN_CONTROLLER_SHA256,
    BehavioralObjectDescriptor,
    NeuralObjectDescriptor,
    model_descriptor,
    require_model_revision,
    verify_behavioral_registry,
    verify_neural_object,
)
from cognitive_discovery.reproducibility.manifest import load_canonical_manifest
from cognitive_discovery.reproducibility.provenance import (
    ProvenanceError,
    validate_run_provenance,
    write_run_provenance,
)


ROOT = Path(__file__).resolve().parents[2]


def test_model_revisions_are_exact_or_explicitly_unknown():
    manifest = load_canonical_manifest(ROOT / "canonical_manifest.yaml")
    core = model_descriptor(manifest, "qwen_core")
    assert core.checkpoint == "Qwen/Qwen3.5-4B"
    assert core.revision is None
    assert core.identity_status == "historical_unknown"
    with pytest.raises(ValueError, match="historical_unknown"):
        require_model_revision(core)

    qwen = require_model_revision(model_descriptor(manifest, "qwen_confirmation"))
    assert qwen.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    llama = require_model_revision(model_descriptor(manifest, "llama_replication"))
    assert llama.revision == "0e9e39f249a16976918f6564b8830bc894c89659"


def test_primary_qwen_neural_object_fails_closed_on_layer_rank_or_hash():
    descriptor = NeuralObjectDescriptor(
        path="artifacts/causal_mech_v1/representations/alignments/"
        "outcome_history__dual_history__L28__rank2__shared__all_tasks.safetensors",
        sha256=CORE_QWEN_CONTROLLER_SHA256,
        layer=28,
        rank=2,
        target_definition="dual_history:outcome_history",
    )
    verify_neural_object(descriptor, root=ROOT, expected_layer=28, expected_rank=2)
    with pytest.raises(ValueError, match="layer"):
        verify_neural_object(descriptor, root=ROOT, expected_layer=30, expected_rank=2)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_neural_object(
            NeuralObjectDescriptor(**{**descriptor.__dict__, "sha256": "0" * 64}),
            root=ROOT,
            expected_layer=28,
            expected_rank=2,
        )


def test_behavioral_registry_binds_architecture_files_and_training_identity():
    registry_path = ROOT / "artifacts/causal_specificity_v2/frozen_models/behavioral_hash.json"
    descriptors = verify_behavioral_registry(registry_path, root=ROOT)
    assert set(descriptors) == {"dual_history", "latent_context", "outcome_history"}
    assert descriptors["dual_history"].model_sha256 == (
        "656639e358adf65bf2079d924f9e6dcadd6f2549508957743effd3aa6dd39ca3"
    )
    assert descriptors["dual_history"].training_condition_sha256 == (
        "87dbeac0efddc575bc86c1b6ab359c36284ff8f173cd117d85325d9fb79a2b25"
    )
    with pytest.raises(ValueError, match="architecture"):
        BehavioralObjectDescriptor(
            architecture="qwen_hidden_state",
            model_path="x",
            model_sha256="0" * 64,
            in_memory_sha256="0" * 64,
            training_condition_sha256="0" * 64,
        ).validate()


def test_new_run_provenance_is_complete_and_written_atomically(tmp_path):
    complete = {
        "schema_version": "run-provenance-v1",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "model": {"checkpoint": "Qwen/Qwen3.5-4B", "revision": "b" * 40},
        "seeds": [1, 2],
        "dataset_hashes": {"conditions": "c" * 64},
        "pair_hash": "d" * 64,
        "split_hash": "e" * 64,
        "behavioral_object": {"sha256": "f" * 64},
        "neural_object": {
            "sha256": "1" * 64,
            "layer": 28,
            "rank": 2,
            "target_definition": "dual_history:outcome_history",
        },
        "endpoint_id": "cognitive_counterfactual_recovery",
        "metric_id": "global_cfr_v1",
        "config_hash": "2" * 64,
        "output_directory": "artifacts/new_run",
    }
    validate_run_provenance(complete)
    output = tmp_path / "run_provenance.json"
    write_run_provenance(output, complete)
    assert json.loads(output.read_text()) == complete

    incomplete = dict(complete)
    incomplete.pop("pair_hash")
    with pytest.raises(ProvenanceError, match="pair_hash"):
        validate_run_provenance(incomplete)
    unknown_revision = json.loads(json.dumps(complete))
    unknown_revision["model"]["revision"] = None
    with pytest.raises(ProvenanceError, match="revision"):
        validate_run_provenance(unknown_revision)

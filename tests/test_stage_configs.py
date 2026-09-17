from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_discovery.reproducibility.manifest import load_canonical_manifest
from cognitive_discovery.reproducibility.provenance import validate_run_provenance
from cognitive_discovery.run import prepare_rerun
from cognitive_discovery.stages.registry import (
    EXPECTED_CONFIG_STAGES,
    StageConfigError,
    load_all_stage_configs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_every_canonical_experiment_has_one_valid_explicit_config():
    manifest = load_canonical_manifest(ROOT / "canonical_manifest.yaml")
    configs = load_all_stage_configs(ROOT, manifest)
    assert set(configs) == set(EXPECTED_CONFIG_STAGES)
    for stage_id, config in configs.items():
        assert config["stage_id"] == stage_id
        assert config["model"]
        assert config["endpoint_id"]
        assert config["metric_id"]
        assert config["split_design"]
        assert config["behavioral_object"]
        assert config["neural_object"]
        assert config["seeds"]
        assert config["output"]["overwrite_frozen"] is False


def test_claim_navigation_config_paths_exist():
    manifest = load_canonical_manifest(ROOT / "canonical_manifest.yaml")
    for claim in manifest["claims"].values():
        config_path = claim["config_path"]
        if config_path is not None:
            assert (ROOT / config_path).is_file()


def test_guarded_rerun_dry_run_requires_pinned_identity_and_safe_output():
    config_path = ROOT / "configs/canonical/qwen_fixed_setting_stability_v1.yaml"
    config, output, command, provenance = prepare_rerun(
        root=ROOT,
        stage_id="qwen_fixed_setting_stability_v1",
        config_path=config_path,
        output_override=None,
    )
    assert config["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert output == ROOT / "artifacts/reruns/qwen_fixed_setting_stability_v1"
    assert str(output) in command
    validate_run_provenance(provenance)

    with pytest.raises(StageConfigError, match="frozen canonical"):
        prepare_rerun(
            root=ROOT,
            stage_id="qwen_fixed_setting_stability_v1",
            config_path=config_path,
            output_override=Path("artifacts/qwen_runpod_v1/unsafe"),
        )
    with pytest.raises(StageConfigError, match="frozen canonical"):
        prepare_rerun(
            root=ROOT,
            stage_id="qwen_fixed_setting_stability_v1",
            config_path=config_path,
            output_override=Path("artifacts/causal_mech_v1/unsafe"),
        )


def test_historical_unknown_and_missing_raw_runs_fail_closed():
    with pytest.raises(StageConfigError, match="not safely rerunnable"):
        prepare_rerun(
            root=ROOT,
            stage_id="causal_mech_v1",
            config_path=ROOT / "configs/canonical/causal_mech_v1.yaml",
            output_override=None,
        )
    with pytest.raises(StageConfigError, match="raw condition manifest"):
        prepare_rerun(
            root=ROOT,
            stage_id="qwen_confirmation_v2",
            config_path=ROOT / "configs/canonical/qwen_confirmation_v2.yaml",
            output_override=None,
        )


def test_not_applicable_provenance_objects_are_explicit_not_fake_hashes():
    record = {
        "schema_version": "run-provenance-v1",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "model": {"checkpoint": "model", "revision": "b" * 40},
        "seeds": [1],
        "dataset_hashes": {"design": "c" * 64},
        "pair_hash": "not_applicable",
        "split_hash": "not_applicable",
        "behavioral_object": {"status": "not_applicable"},
        "neural_object": {"status": "not_applicable"},
        "endpoint_id": "measurement_validity",
        "metric_id": "interface_validity_gate_v2",
        "config_hash": "d" * 64,
        "output_directory": "artifacts/reruns/test",
    }
    validate_run_provenance(record)
    bad = dict(record)
    bad["neural_object"] = {"status": "not_applicable", "sha256": "e" * 64}
    with pytest.raises(ValueError, match="must not contain"):
        validate_run_provenance(bad)

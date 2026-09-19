from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cognitive_discovery.model_validation.external_benchmark import (
    BenchmarkAcquisitionError,
    acquire_benchmark_dataset,
    validate_factual_dataset,
)


ROOT = Path(__file__).resolve().parents[2]


def _factual_fixture():
    return pd.DataFrame(
        [
            {
                "subjectid": participant,
                "type": 1 + trial % 4,
                "reward": 10 if trial % 2 else -10,
                "choice": 1 + trial % 8,
                "false_start": 0,
                "rt": 500,
                "key": 33 if trial % 2 else 36,
            }
            for participant in range(1, 144)
            for trial in range(192)
        ]
    )


def test_owner_frozen_spec_and_config_have_no_placeholder_decisions():
    spec = (ROOT / "validation/external_benchmark/BENCHMARK_SPEC.md").read_text(
        encoding="utf-8"
    )
    assert "OWNER-FROZEN" in spec
    assert "OWNER_REQUIRED" not in spec
    assert "Do not change these thresholds after seeing benchmark results" in spec
    config = yaml.safe_load(
        (ROOT / "configs/validation/external_benchmark_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == "owner_frozen"
    assert config["dataset"]["version_doi"] == "10.6084/m9.figshare.10042319.v3"
    assert config["subset"] == {
        "experiment": "authors_web_experiment",
        "context": "factual",
        "participants": 143,
        "sessions": 2,
        "trials_per_session": 96,
        "additional_participant_exclusions": False,
        "additional_trial_filtering": False,
    }
    assert config["bootstrap"] == {
        "pairing": "participant",
        "samples": 2000,
        "interval": "percentile_95",
        "seed": 23001,
    }


def test_committed_manifest_freezes_all_figshare_file_identities():
    manifest = json.loads(
        (ROOT / "validation/external_benchmark/DATASET_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["article_version"] == 3
    assert manifest["license"] == "CC BY 4.0"
    assert {row["name"] for row in manifest["files"]} == {
        "factual.csv",
        "counterfactual.csv",
        "Read me.txt",
    }
    assert all(len(row["sha256"]) == 64 for row in manifest["files"])
    assert all(len(row["md5"]) == 32 for row in manifest["files"])


def test_factual_contract_preserves_all_participants_trials_and_sessions(tmp_path):
    path = tmp_path / "factual.csv"
    _factual_fixture().to_csv(path, index=False)
    result = validate_factual_dataset(path)
    assert result["rows"] == 27_456
    assert result["participants"] == 143
    assert result["sessions_per_participant"] == 2
    assert result["trials_per_session"] == 96
    assert result["additional_rows_removed"] == 0


def test_factual_contract_fails_closed_on_unregistered_filtering(tmp_path):
    path = tmp_path / "factual.csv"
    _factual_fixture().iloc[:-1].to_csv(path, index=False)
    with pytest.raises(BenchmarkAcquisitionError, match="27,456"):
        validate_factual_dataset(path)


def test_fetcher_is_dry_run_by_default(tmp_path):
    output = tmp_path / "raw"
    result = acquire_benchmark_dataset(root=ROOT, output=output)
    assert result["mode"] == "dry-run"
    assert result["article_version"] == 3
    assert result["version_doi"] == "10.6084/m9.figshare.10042319.v3"
    assert not output.exists()

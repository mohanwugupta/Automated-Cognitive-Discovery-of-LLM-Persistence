from __future__ import annotations

import copy
from pathlib import Path
import subprocess

import pytest

from cognitive_discovery.reproducibility.manifest import (
    ALLOWED_REPLAY_STATUSES,
    CanonicalManifestError,
    load_canonical_manifest,
    validate_canonical_manifest,
    verify_manifest_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]


def test_canonical_manifest_schema_dag_and_hashes():
    manifest = load_canonical_manifest(ROOT / "canonical_manifest.yaml")
    summary = validate_canonical_manifest(manifest, root=ROOT)
    assert summary.stage_count == 22
    assert summary.edge_count > 20
    verified = verify_manifest_artifacts(manifest, root=ROOT)
    assert len(verified) == 43


def test_manifest_fails_closed_on_cycle_unknown_status_and_model():
    original = load_canonical_manifest(ROOT / "canonical_manifest.yaml")

    cyclic = copy.deepcopy(original)
    cyclic["stages"]["discovery_v1"]["depends_on"] = ["paper_evidence_replay"]
    with pytest.raises(CanonicalManifestError, match="cycle"):
        validate_canonical_manifest(cyclic, root=ROOT)

    bad_status = copy.deepcopy(original)
    bad_status["stages"]["discovery_v1"]["status"] = "looks_good"
    with pytest.raises(CanonicalManifestError, match="status"):
        validate_canonical_manifest(bad_status, root=ROOT)

    bad_model = copy.deepcopy(original)
    bad_model["stages"]["discovery_v1"]["model_ref"] = "invented_model"
    with pytest.raises(CanonicalManifestError, match="model_ref"):
        validate_canonical_manifest(bad_model, root=ROOT)


def test_every_stage_has_explicit_replay_availability():
    manifest = load_canonical_manifest(ROOT / "canonical_manifest.yaml")
    assert all(
        stage.get("replay_status") in ALLOWED_REPLAY_STATUSES
        for stage in manifest["stages"].values()
    )


def test_all_hashed_evidence_is_git_tracked_for_clean_clone():
    manifest = load_canonical_manifest(ROOT / "canonical_manifest.yaml")
    tracked = set(
        subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    )
    referenced = {
        output["path"]
        for stage in manifest["stages"].values()
        for output in stage.get("outputs", [])
        if output.get("sha256") is not None
    }
    assert referenced <= tracked

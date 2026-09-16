from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cognitive_discovery.reproducibility.claims import (
    audit_target_shuffle,
    resolve_necessity_status,
    validate_claim_gate_consistency,
)
from cognitive_discovery.reproducibility.manifest import load_canonical_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_owner_decisions_override_ambiguous_legacy_booleans():
    manifest = load_canonical_manifest(ROOT / "canonical_manifest.yaml")
    validate_claim_gate_consistency(manifest, root=ROOT)
    assert manifest["stages"]["theory_resolution_v1"]["resolution_status"] == "unresolved"
    assert manifest["stages"]["level5b_necessity_v2"]["status"] == "not_run"
    assert manifest["stages"]["abstraction_discovery_v1"]["identity_status"] == "unresolved"
    assert manifest["stages"]["ood_free_generation_v1"]["status"] == "boundary"
    assert manifest["stages"]["mechanistic_manifest_v1"]["status"] == "canonical"
    assert manifest["stages"]["mechanistic_probe_steering_v1"]["interpretation_status"] == "legacy"


def test_empty_necessity_jobs_resolve_to_not_run_not_failed():
    jobs = json.loads(
        (ROOT / "artifacts/causal_specificity_v2/frozen_models/necessity_jobs.json").read_text()
    )
    assert resolve_necessity_status(jobs) == "not_run"


def test_target_preserving_shuffles_are_uninformative():
    frame = pd.read_csv(ROOT / "artifacts/causal_specificity_v2/controls/shuffled_target.csv")
    audits = {
        artifact_id: audit_target_shuffle(group)
        for artifact_id, group in frame.groupby("artifact_id")
    }
    assert audits["das_001_outcome_history_dual_history_L28_r2"].status == "uninformative_control"
    assert audits["das_002_outcome_history_outcome_history_L30_r2"].status == "uninformative_control"
    assert audits["das_001_outcome_history_dual_history_L28_r2"].changed_rows == 0
    assert audits["das_002_outcome_history_outcome_history_L30_r2"].changed_rows == 0
    assert audits["das_000_contextual_outcome_history_latent_context_L30_r8"].changed_rows == 7


def test_abstraction_and_ood_are_not_promoted_by_saved_gate_fields():
    abstraction = json.loads((ROOT / "artifacts/abstraction_discovery_v1/gates.json").read_text())
    assert abstraction["causal_abstraction"]["winner"] == "E"
    assert abstraction["causal_abstraction"]["passed"] is False
    ood = json.loads((ROOT / "artifacts/ood_free_generation_v1/gates.json").read_text())
    assert ood["ood_generalization"]["passed"] is False
    assert ood["ood_generalization"]["outcome"] == "ood_generalization_not_established"

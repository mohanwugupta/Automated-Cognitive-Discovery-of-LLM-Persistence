"""Owner-approved interpretations of ambiguous legacy gates and controls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class ClaimConsistencyError(ValueError):
    pass


@dataclass(frozen=True)
class ShuffleAudit:
    rows: int
    changed_rows: int
    maximum_absolute_change: float
    status: str


def audit_target_shuffle(frame, *, tolerance: float = 1e-12) -> ShuffleAudit:
    required = {"predicted_counterfactual_effect", "original_predicted_counterfactual_effect"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"shuffle audit missing columns: {sorted(missing)}")
    shuffled = np.asarray(frame["predicted_counterfactual_effect"], dtype=float)
    original = np.asarray(frame["original_predicted_counterfactual_effect"], dtype=float)
    if not (np.isfinite(shuffled).all() and np.isfinite(original).all()):
        raise ValueError("shuffle audit requires finite target values")
    change = np.abs(shuffled - original)
    changed = int(np.sum(change > tolerance))
    return ShuffleAudit(
        rows=len(frame),
        changed_rows=changed,
        maximum_absolute_change=float(change.max(initial=0.0)),
        status="informative_control" if changed else "uninformative_control",
    )


def resolve_necessity_status(jobs: Any) -> str:
    if isinstance(jobs, dict):
        jobs = jobs.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("necessity jobs must be a list or a mapping containing jobs")
    return "not_run" if not jobs else "evaluated"


def validate_claim_gate_consistency(manifest: Mapping[str, Any], *, root: str | Path) -> None:
    root = Path(root)
    stages = manifest["stages"]
    decision = json.loads(
        (root / "artifacts/theory_resolution_v1/discrimination/theory_decision.json").read_text()
    )
    if not (
        decision.get("outcome") == "unresolved"
        and decision.get("winner") is None
        and stages["theory_resolution_v1"].get("resolution_status") == "unresolved"
    ):
        raise ClaimConsistencyError("theory-resolution claim/gate mismatch")
    jobs = json.loads(
        (root / "artifacts/causal_specificity_v2/frozen_models/necessity_jobs.json").read_text()
    )
    if resolve_necessity_status(jobs) != stages["level5b_necessity_v2"]["status"]:
        raise ClaimConsistencyError("Level-5B status must be not_run when no job exists")
    abstraction = json.loads((root / "artifacts/abstraction_discovery_v1/gates.json").read_text())
    abstraction_gate = abstraction["causal_abstraction"]
    if not (
        abstraction_gate["winner"] == "E"
        and abstraction_gate["passed"] is False
        and stages["abstraction_discovery_v1"].get("identity_status") == "unresolved"
    ):
        raise ClaimConsistencyError("abstraction E description was promoted to identity")
    ood = json.loads((root / "artifacts/ood_free_generation_v1/gates.json").read_text())
    if not (
        stages["ood_free_generation_v1"]["status"] == "boundary"
        and ood["ood_generalization"]["passed"] is False
    ):
        raise ClaimConsistencyError("OOD boundary status conflicts with its gate")
    if stages["mechanistic_manifest_v1"]["status"] != "canonical":
        raise ClaimConsistencyError("mechanistic matched manifest must remain canonical")
    if stages["mechanistic_probe_steering_v1"].get("interpretation_status") != "legacy":
        raise ClaimConsistencyError("mechanistic probe/steering interpretation must remain legacy")

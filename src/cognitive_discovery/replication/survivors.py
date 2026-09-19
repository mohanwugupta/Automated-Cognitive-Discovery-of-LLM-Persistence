"""Behavior-only survivor selection and pre-neural immutability checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .provenance import canonical_hash


class SurvivorSetError(RuntimeError):
    """Raised when a frozen behavioral survivor set is missing or mutated."""


def select_behavioral_survivors(
    scores: Mapping[str, float],
    *,
    metric: str,
    direction: str,
    equivalence_rule: str,
    equivalence_margin: float,
    eligible_models: Sequence[str] | None = None,
) -> dict:
    """Freeze plausible theories using behavioral scores only.

    The current preregistered rule is an absolute score margin.  Deliberately no
    neural-result argument exists: downstream causal evidence cannot affect this
    selection.
    """

    if direction not in {"minimize", "maximize"}:
        raise ValueError("behavioral survivor metric direction must be minimize or maximize")
    if equivalence_rule != "absolute_score_margin":
        raise ValueError("unsupported behavioral survivor equivalence rule")
    if float(equivalence_margin) < 0:
        raise ValueError("behavioral survivor equivalence margin must be non-negative")
    eligible = sorted(set(eligible_models if eligible_models is not None else scores))
    if not eligible:
        raise ValueError("at least one behaviorally eligible model is required")
    missing = sorted(set(eligible) - set(scores))
    if missing:
        raise ValueError(f"behavioral scores are absent for: {missing}")
    normalized = {name: float(scores[name]) for name in eligible}
    if direction == "minimize":
        best_model = min(eligible, key=lambda name: (normalized[name], name))
        deltas = {name: normalized[name] - normalized[best_model] for name in eligible}
    else:
        best_model = min(eligible, key=lambda name: (-normalized[name], name))
        deltas = {name: normalized[best_model] - normalized[name] for name in eligible}
    survivors = sorted(
        name
        for name in eligible
        if deltas[name] <= float(equivalence_margin) + 1e-12
    )
    rule = {
        "metric": str(metric),
        "direction": direction,
        "equivalence_rule": equivalence_rule,
        "equivalence_margin": float(equivalence_margin),
        "evidence_axis": "behavioral_only",
        "neural_results_used": False,
    }
    membership = {
        "best_model": best_model,
        "behavioral_survivor_set": survivors,
        "eligible_models": eligible,
        "scores": normalized,
        "deltas_from_best": deltas,
        "rule_sha256": canonical_hash(rule),
    }
    return {
        "schema_version": "behavioral-survivor-set-v1",
        **rule,
        **membership,
        "membership_sha256": canonical_hash(membership),
        "behavioral_theory_status": "resolved" if len(survivors) == 1 else "unresolved",
        # Backward-compatible fields. selected_model means behavioral best, not a
        # neural winner; selected_models is the complete frozen survivor set.
        "selected_model": best_model,
        "selected_models": survivors,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_survivor_set(output: str | Path, provenance: Mapping) -> dict:
    """Fail closed if survivor membership or its rule changed after freezing."""

    output = Path(output)
    manifest_path = output / "behavior/models/frozen_survivor_set.json"
    selection_path = output / "behavior/models/selected_models.json"
    architecture_path = output / "behavior/models/frozen_models/frozen_architectures.json"
    for path in (manifest_path, selection_path, architecture_path):
        if not path.is_file():
            raise SurvivorSetError(f"frozen behavioral survivor artifact is absent: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    architectures = sorted(map(str, json.loads(architecture_path.read_text(encoding="utf-8"))))
    survivors = sorted(map(str, manifest.get("behavioral_survivor_set", [])))
    selected = sorted(
        map(
            str,
            selection.get("behavioral_survivor_set", selection.get("selected_models", [])),
        )
    )
    if not survivors or survivors != selected or survivors != architectures:
        raise SurvivorSetError("behavioral survivor membership changed after freezing")
    expected_file = provenance.get("behavioral_survivor_manifest_sha256")
    if _file_sha256(manifest_path) != expected_file:
        raise SurvivorSetError("frozen behavioral survivor manifest hash changed")
    if manifest.get("rule_sha256") != provenance.get("behavioral_survivor_rule_sha256"):
        raise SurvivorSetError("behavioral survivor rule changed after freezing")
    if manifest.get("membership_sha256") != provenance.get("behavioral_survivor_set_sha256"):
        raise SurvivorSetError("behavioral survivor membership hash changed")
    if manifest.get("neural_results_used") is not False:
        raise SurvivorSetError("survivor set is not behavior-only")
    return manifest

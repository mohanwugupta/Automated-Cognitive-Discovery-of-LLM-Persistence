"""Hash-complete provenance for clean prospective discovery runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def git_identity(root: str | Path) -> dict:
    root = Path(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip())
    return {"git_commit": commit, "git_dirty": dirty}


def validate_final_provenance(record: dict) -> None:
    required_hashes = {
        "environment_lock_sha256", "config_sha256", "compatibility_sha256",
        "design_sha256", "behavior_split_sha256",
    }
    required = {
        "schema_version", "git_commit", "git_dirty", "models", "seeds",
        "computational_model_spec_sha256", "active_acquisition_config_sha256",
        "scientific_freeze_sha256", "behavioral_survivor_hashes",
        "representation_target_manifest_sha256", "neural_split_hashes",
        "subspace_controller_hashes", "endpoint_ids", *required_hashes,
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"final provenance missing: {sorted(missing)}")
    if record["schema_version"] != "cross-task-provenance-v1":
        raise ValueError("provenance schema drift")
    if record["git_dirty"] is not False:
        raise ValueError("final evidence requires a clean worktree")
    if not HEX40.fullmatch(str(record["git_commit"])):
        raise ValueError("provenance requires an immutable git commit")
    for name in required_hashes:
        if not HEX64.fullmatch(str(record[name])):
            raise ValueError(f"invalid provenance hash: {name}")
    for name in (
        "computational_model_spec_sha256", "active_acquisition_config_sha256",
        "scientific_freeze_sha256", "representation_target_manifest_sha256",
    ):
        if not HEX64.fullmatch(str(record[name])):
            raise ValueError(f"invalid provenance hash: {name}")
    model_keys = {str(model.get("key")) for model in record["models"]}
    for name in (
        "behavioral_survivor_hashes", "neural_split_hashes", "subspace_controller_hashes"
    ):
        value = record[name]
        if set(value) != model_keys:
            raise ValueError(f"{name} must identify all three models")
        if name != "subspace_controller_hashes" and any(
            not HEX64.fullmatch(str(item)) for item in value.values()
        ):
            raise ValueError(f"invalid per-model provenance hash: {name}")
        if name == "subspace_controller_hashes" and any(
            not items or any(not HEX64.fullmatch(str(item)) for item in items)
            for items in value.values()
        ):
            raise ValueError("every model requires frozen subspace/controller hashes")
    for model in record["models"]:
        if not HEX40.fullmatch(str(model.get("revision", ""))):
            raise ValueError("every model needs an immutable model revision")
        if not HEX40.fullmatch(str(model.get("tokenizer_revision", model.get("revision", "")))):
            raise ValueError("every tokenizer needs an immutable model revision")
    if not record["seeds"]:
        raise ValueError("provenance requires explicit seeds")
    if record["endpoint_ids"] != {
        "behavior": "semantic_persistence_logit_v1",
        "representation": "frozen_source_decoding_v1",
        "causal": "cognitive_counterfactual_recovery",
        "causal_metric": "global_cfr_v1",
    }:
        raise ValueError("final provenance endpoint identity drift")


def write_provenance(path: str | Path, record: dict, *, final: bool) -> Path:
    if final:
        validate_final_provenance(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

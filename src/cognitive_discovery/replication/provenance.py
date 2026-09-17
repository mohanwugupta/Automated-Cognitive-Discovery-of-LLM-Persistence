"""Progressive, fail-closed provenance for new model replications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")


class ReplicationProvenanceError(ValueError):
    pass


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ReplicationProvenanceError(f"{name} must be an exact SHA-256")


def _base_validate(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "git_commit",
        "git_dirty",
        "model",
        "tokenizer",
        "adapter",
        "environment",
        "seeds",
        "behavioral_design_hash",
        "behavioral_split_hash",
        "frozen_model_hashes",
        "counterfactual_pair_manifest_hash",
        "neural_split_hash",
        "layer_rank_grid",
        "selected_controller_hash",
        "endpoint_id",
        "metric_id",
        "config_hash",
        "output_path",
    }
    missing = required - set(record)
    if missing:
        raise ReplicationProvenanceError(f"missing replication provenance: {sorted(missing)}")
    if record["schema_version"] != "replication-provenance-v1":
        raise ReplicationProvenanceError("unsupported replication provenance schema")
    if not isinstance(record["git_commit"], str) or not REVISION.fullmatch(record["git_commit"]):
        raise ReplicationProvenanceError("git_commit must be an exact commit")
    if not isinstance(record["git_dirty"], bool):
        raise ReplicationProvenanceError("git_dirty must be boolean")
    for name in ("model", "tokenizer"):
        value = record[name]
        if not isinstance(value, dict) or not value.get("id"):
            raise ReplicationProvenanceError(f"{name} identity is incomplete")
        if not REVISION.fullmatch(str(value.get("revision", ""))):
            raise ReplicationProvenanceError(f"{name} revision must be immutable")
    adapter = record["adapter"]
    if not isinstance(adapter, dict) or not adapter.get("name") or not adapter.get("version"):
        raise ReplicationProvenanceError("adapter name and version are required")
    environment = record["environment"]
    if not isinstance(environment, dict):
        raise ReplicationProvenanceError("environment identity is required")
    _sha(environment.get("lockfile_sha256"), "environment.lockfile_sha256")
    if not isinstance(record["seeds"], dict) or not record["seeds"]:
        raise ReplicationProvenanceError("all seeds must be recorded")
    _sha(record["behavioral_design_hash"], "behavioral_design_hash")
    _sha(record["config_hash"], "config_hash")
    if record["endpoint_id"] != "cognitive_counterfactual_recovery":
        raise ReplicationProvenanceError("replication endpoint changed from cognitive recovery")
    if record["metric_id"] != "global_cfr_v1":
        raise ReplicationProvenanceError("replication metric changed from global_cfr_v1")
    grid = record["layer_rank_grid"]
    if not isinstance(grid, dict) or not grid.get("relative_depths") or not grid.get("ranks"):
        raise ReplicationProvenanceError("layer/rank grid must be frozen before execution")
    if not record["output_path"]:
        raise ReplicationProvenanceError("output_path is required")
    encoded = json.dumps(record, sort_keys=True, default=str).lower()
    if "historical_unknown" in encoded:
        raise ReplicationProvenanceError("new replications cannot contain historical unknowns")


def validate_replication_provenance(
    record: Mapping[str, Any], *, required_for_stage: str
) -> None:
    _base_validate(record)
    frozen_behavior_stages = {
        "counterfactuals",
        "mechanism",
        "generalization",
        "specificity",
        "abstraction",
        "ood",
        "report",
    }
    neural_stages = {
        "mechanism",
        "generalization",
        "specificity",
        "abstraction",
        "ood",
        "report",
    }
    if required_for_stage in frozen_behavior_stages:
        hashes = record["frozen_model_hashes"]
        if not isinstance(hashes, dict) or not hashes:
            raise ReplicationProvenanceError("frozen behavioral model hashes are required")
        for name, digest in hashes.items():
            _sha(digest, f"frozen_model_hashes.{name}")
        _sha(record["behavioral_split_hash"], "behavioral_split_hash")
    if required_for_stage in neural_stages:
        _sha(
            record["counterfactual_pair_manifest_hash"],
            "counterfactual_pair_manifest_hash",
        )
        _sha(record["neural_split_hash"], "neural_split_hash")
    if required_for_stage in {"generalization", "specificity", "abstraction", "ood", "report"}:
        _sha(record["selected_controller_hash"], "selected_controller_hash")


def write_replication_provenance(path: str | Path, record: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path

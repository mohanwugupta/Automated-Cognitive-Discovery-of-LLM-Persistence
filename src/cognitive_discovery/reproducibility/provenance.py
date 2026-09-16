"""Schema and atomic writer for provenance captured by new runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .identities import EndpointID, MetricID


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "git_commit",
        "git_dirty",
        "model",
        "seeds",
        "dataset_hashes",
        "pair_hash",
        "split_hash",
        "behavioral_object",
        "neural_object",
        "endpoint_id",
        "metric_id",
        "config_hash",
        "output_directory",
    }
)


class ProvenanceError(ValueError):
    pass


def _require_sha(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProvenanceError(f"{field} must be a lowercase SHA-256")


def validate_run_provenance(record: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ProvenanceError(f"missing provenance fields: {sorted(missing)}")
    if record["schema_version"] != "run-provenance-v1":
        raise ProvenanceError("unsupported run provenance schema")
    if not isinstance(record["git_commit"], str) or not _GIT_COMMIT.fullmatch(record["git_commit"]):
        raise ProvenanceError("git_commit must be a full lowercase commit hash")
    if not isinstance(record["git_dirty"], bool):
        raise ProvenanceError("git_dirty must be boolean")
    model = record["model"]
    if not isinstance(model, dict) or not model.get("checkpoint") or not model.get("revision"):
        raise ProvenanceError("new runs require a model checkpoint and resolved revision")
    if not isinstance(record["seeds"], list) or not record["seeds"]:
        raise ProvenanceError("new runs require at least one seed")
    if not isinstance(record["dataset_hashes"], dict) or not record["dataset_hashes"]:
        raise ProvenanceError("dataset_hashes must be non-empty")
    for name, digest in record["dataset_hashes"].items():
        _require_sha(digest, f"dataset_hashes.{name}")
    for field in ("pair_hash", "split_hash", "config_hash"):
        _require_sha(record[field], field)
    behavior = record["behavioral_object"]
    neural = record["neural_object"]
    if not isinstance(behavior, dict):
        raise ProvenanceError("behavioral_object must be a mapping")
    _require_sha(behavior.get("sha256"), "behavioral_object.sha256")
    if not isinstance(neural, dict):
        raise ProvenanceError("neural_object must be a mapping")
    _require_sha(neural.get("sha256"), "neural_object.sha256")
    for field in ("layer", "rank"):
        if not isinstance(neural.get(field), int) or neural[field] < 1:
            raise ProvenanceError(f"neural_object.{field} must be a positive integer")
    if not neural.get("target_definition"):
        raise ProvenanceError("neural_object.target_definition is required")
    try:
        EndpointID(record["endpoint_id"])
    except ValueError as error:
        raise ProvenanceError("endpoint_id is not canonical") from error
    try:
        MetricID(record["metric_id"])
    except ValueError as error:
        raise ProvenanceError("metric_id is not canonical") from error
    if not isinstance(record["output_directory"], str) or not record["output_directory"]:
        raise ProvenanceError("output_directory is required")


def write_run_provenance(path: str | Path, record: Mapping[str, Any]) -> Path:
    validate_run_provenance(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path

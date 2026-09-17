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


def _require_sha_or_not_applicable(value: Any, field: str) -> None:
    if value == "not_applicable":
        return
    _require_sha(value, field)


def _validate_optional_object(value: Any, field: str, *, neural: bool = False) -> None:
    if not isinstance(value, dict):
        raise ProvenanceError(f"{field} must be a mapping")
    if value.get("status") == "not_applicable":
        if set(value) != {"status"}:
            raise ProvenanceError(f"{field} not_applicable identity must not contain object fields")
        return
    _require_sha(value.get("sha256"), f"{field}.sha256")
    if neural:
        for name in ("layer", "rank"):
            if not isinstance(value.get(name), int) or value[name] < 1:
                raise ProvenanceError(f"{field}.{name} must be a positive integer")
        if not value.get("target_definition"):
            raise ProvenanceError(f"{field}.target_definition is required")


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
    for field in ("pair_hash", "split_hash"):
        _require_sha_or_not_applicable(record[field], field)
    _require_sha(record["config_hash"], "config_hash")
    behavior = record["behavioral_object"]
    neural = record["neural_object"]
    _validate_optional_object(behavior, "behavioral_object")
    _validate_optional_object(neural, "neural_object", neural=True)
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

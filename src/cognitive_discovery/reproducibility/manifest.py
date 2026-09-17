"""Canonical manifest loading, graph validation, and byte-level artifact identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


ALLOWED_STAGE_STATUSES = frozenset(
    {
        "canonical",
        "supporting",
        "boundary",
        "replication",
        "legacy",
        "unresolved",
        "not_run",
        "uninformative_control",
    }
)
ALLOWED_REPLAY_STATUSES = frozenset(
    {"fully_replayable", "compact_replay_only", "external_dependency", "historical_unknown"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CanonicalManifestError(ValueError):
    """Raised when a canonical identity is incomplete or inconsistent."""


@dataclass(frozen=True)
class ManifestSummary:
    stage_count: int
    edge_count: int
    artifact_count: int


def load_canonical_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CanonicalManifestError("canonical manifest must be a mapping")
    return data


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_artifact_path(value: Any, *, stage_id: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CanonicalManifestError(f"stage {stage_id} has an invalid artifact path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise CanonicalManifestError(f"stage {stage_id} artifact path must stay inside the repo")
    return path


def validate_canonical_manifest(
    manifest: Mapping[str, Any], *, root: str | Path | None = None
) -> ManifestSummary:
    required = {
        "schema_version",
        "canonical_base",
        "status_vocabulary",
        "models",
        "stages",
        "claims",
    }
    missing = required - set(manifest)
    if missing:
        raise CanonicalManifestError(f"canonical manifest missing keys: {sorted(missing)}")
    if manifest["schema_version"] != "stage2a-1":
        raise CanonicalManifestError("unsupported canonical manifest schema_version")

    models = manifest["models"]
    stages = manifest["stages"]
    if not isinstance(models, dict) or not models:
        raise CanonicalManifestError("models must be a non-empty mapping")
    if not isinstance(stages, dict) or not stages:
        raise CanonicalManifestError("stages must be a non-empty mapping")

    claims = manifest["claims"]
    expected_claims = {f"C{index:02d}" for index in range(1, 14)}
    if not isinstance(claims, dict) or set(claims) != expected_claims:
        raise CanonicalManifestError("claims must define exactly C01 through C13")

    for model_id, model in models.items():
        if not isinstance(model, dict) or not model.get("checkpoint"):
            raise CanonicalManifestError(f"model {model_id} must identify its checkpoint")
        revision = model.get("revision")
        if revision is None:
            if model.get("identity_status") != "historical_unknown":
                raise CanonicalManifestError(
                    f"model {model_id} has a null revision without historical_unknown status"
                )
        elif not isinstance(revision, str) or not revision:
            raise CanonicalManifestError(f"model {model_id} has an invalid revision")

    edge_count = 0
    artifact_count = 0
    for stage_id, stage in stages.items():
        if not isinstance(stage, dict):
            raise CanonicalManifestError(f"stage {stage_id} must be a mapping")
        if stage.get("status") not in ALLOWED_STAGE_STATUSES:
            raise CanonicalManifestError(f"stage {stage_id} has unknown status {stage.get('status')!r}")
        if stage.get("replay_status") not in ALLOWED_REPLAY_STATUSES:
            raise CanonicalManifestError(
                f"stage {stage_id} lacks a valid replay_status: {stage.get('replay_status')!r}"
            )
        model_ref = stage.get("model_ref")
        if model_ref is not None and model_ref not in models:
            raise CanonicalManifestError(f"stage {stage_id} has unknown model_ref {model_ref!r}")
        dependencies = stage.get("depends_on", [])
        if not isinstance(dependencies, list):
            raise CanonicalManifestError(f"stage {stage_id} depends_on must be a list")
        edge_count += len(dependencies)
        for dependency in dependencies:
            if dependency not in stages:
                raise CanonicalManifestError(
                    f"stage {stage_id} has unknown dependency {dependency!r}"
                )
        outputs = stage.get("outputs", [])
        if not isinstance(outputs, list):
            raise CanonicalManifestError(f"stage {stage_id} outputs must be a list")
        for output in outputs:
            if not isinstance(output, dict):
                raise CanonicalManifestError(f"stage {stage_id} output must be a mapping")
            _relative_artifact_path(output.get("path"), stage_id=stage_id)
            digest = output.get("sha256")
            if digest is not None and (not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
                raise CanonicalManifestError(f"stage {stage_id} has an invalid SHA-256")
            if digest is None and not (
                output.get("identity_status") or stage.get("availability") or stage.get("identity_status")
            ):
                raise CanonicalManifestError(
                    f"stage {stage_id} has an unhashed output without an availability explanation"
                )
            artifact_count += 1

    active: list[str] = []
    complete: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in complete:
            return
        if stage_id in active:
            cycle = " -> ".join([*active, stage_id])
            raise CanonicalManifestError(f"dependency cycle detected: {cycle}")
        active.append(stage_id)
        for dependency in stages[stage_id].get("depends_on", []):
            visit(dependency)
        active.pop()
        complete.add(stage_id)

    for stage_id in stages:
        visit(stage_id)

    for claim_id, claim in claims.items():
        if not isinstance(claim, dict):
            raise CanonicalManifestError(f"claim {claim_id} must be a mapping")
        analysis_id = claim.get("analysis_id")
        if analysis_id not in stages:
            raise CanonicalManifestError(
                f"claim {claim_id} has unknown analysis_id {analysis_id!r}"
            )
        supporting = claim.get("supporting_stage_ids")
        if not isinstance(supporting, list) or any(stage not in stages for stage in supporting):
            raise CanonicalManifestError(f"claim {claim_id} has invalid supporting_stage_ids")
        for field in ("endpoint_id", "code_path", "artifact_paths", "figure_ref"):
            if not claim.get(field):
                raise CanonicalManifestError(f"claim {claim_id} lacks {field}")
        if not isinstance(claim["artifact_paths"], list):
            raise CanonicalManifestError(f"claim {claim_id} artifact_paths must be a list")
        for value in claim["artifact_paths"]:
            _relative_artifact_path(value, stage_id=f"claim {claim_id}")
        _relative_artifact_path(claim["code_path"], stage_id=f"claim {claim_id}")
        config_path = claim.get("config_path")
        if config_path is not None:
            _relative_artifact_path(config_path, stage_id=f"claim {claim_id}")

    if root is not None:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise CanonicalManifestError(f"repository root does not exist: {root_path}")

    return ManifestSummary(len(stages), edge_count, artifact_count)


def stage_dependency_order(manifest: Mapping[str, Any], stage_id: str) -> list[str]:
    """Return the selected stage and its transitive dependencies in execution order."""

    validate_canonical_manifest(manifest)
    if stage_id not in manifest["stages"]:
        raise CanonicalManifestError(f"unknown stage {stage_id!r}")
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(current: str) -> None:
        if current in visited:
            return
        for dependency in manifest["stages"][current].get("depends_on", []):
            visit(dependency)
        visited.add(current)
        ordered.append(current)

    visit(stage_id)
    return ordered


def verify_stage_artifacts(
    manifest: Mapping[str, Any], *, root: str | Path, stage_id: str
) -> dict[str, str]:
    """Verify hashed outputs for one stage, without implicitly replaying its dependencies."""

    validate_canonical_manifest(manifest, root=root)
    if stage_id not in manifest["stages"]:
        raise CanonicalManifestError(f"unknown stage {stage_id!r}")
    root_path = Path(root)
    verified: dict[str, str] = {}
    for output in manifest["stages"][stage_id].get("outputs", []):
        expected = output.get("sha256")
        if expected is None:
            continue
        relative = _relative_artifact_path(output["path"], stage_id=stage_id)
        path = root_path / relative
        if not path.is_file():
            raise CanonicalManifestError(f"missing canonical artifact: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise CanonicalManifestError(
                f"artifact SHA-256 mismatch for {relative}: {observed} != {expected}"
            )
        verified[relative.as_posix()] = observed
    return verified


def verify_manifest_artifacts(
    manifest: Mapping[str, Any], *, root: str | Path
) -> dict[str, str]:
    root = Path(root)
    validate_canonical_manifest(manifest, root=root)
    verified: dict[str, str] = {}
    for stage_id, stage in manifest["stages"].items():
        for output in stage.get("outputs", []):
            expected = output.get("sha256")
            if expected is None:
                continue
            relative = _relative_artifact_path(output["path"], stage_id=stage_id)
            path = root / relative
            if not path.is_file():
                raise CanonicalManifestError(f"missing canonical artifact: {relative}")
            observed = sha256_file(path)
            if observed != expected:
                raise CanonicalManifestError(
                    f"artifact SHA-256 mismatch for {relative}: {observed} != {expected}"
                )
            verified[relative.as_posix()] = observed
    return verified

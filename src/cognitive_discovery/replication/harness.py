"""Initialization, planning, and safe resume for model replications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import yaml

from .adapters import adapter_registry
from .config import load_replication_config
from .provenance import canonical_hash, validate_replication_provenance, write_replication_provenance
from .state import ReplicationRunState


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _run_id(model_id: str, revision: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return f"{slug}-{revision[:8]}"


def _validate_output(root: Path, output: Path, *, must_not_exist: bool) -> Path:
    output = output.resolve()
    root = root.resolve()
    canonical_roots = [root / "artifacts", root / "paper/generated"]
    allowed_replications = (root / "artifacts/replications").resolve()
    if root in output.parents:
        for frozen in canonical_roots:
            frozen = frozen.resolve()
            if (output == frozen or frozen in output.parents) and not (
                output == allowed_replications or allowed_replications in output.parents
            ):
                raise ValueError("replication output would overlap historical canonical artifacts")
    if must_not_exist and output.exists():
        raise FileExistsError(f"replication output already exists: {output}")
    return output


def _load(
    *,
    config_path: Path,
    model_id: str,
    revision: str,
    adapter: str,
    tokenizer_id: str | None,
    tokenizer_revision: str | None,
    overrides: Mapping[str, Any] | None,
):
    return load_replication_config(
        config_path,
        model_id=model_id,
        revision=revision,
        adapter=adapter,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        overrides=overrides,
    )


def plan_replication(
    *,
    root: str | Path,
    config_path: str | Path,
    model_id: str,
    revision: str,
    adapter: str,
    output: str | Path,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    output = _validate_output(root, Path(output), must_not_exist=False)
    config = _load(
        config_path=config_path,
        model_id=model_id,
        revision=revision,
        adapter=adapter,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        overrides=overrides,
    )
    adapter_value = adapter_registry.create(
        adapter,
        model_id=model_id,
        revision=revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    return {
        "schema_version": "replication-plan-v1",
        "mode": "dry-run",
        "run_id": _run_id(model_id, revision),
        "model": config["model"],
        "adapter_version": adapter_value.adapter_version,
        "endpoint_id": config["endpoint_id"],
        "metric_id": config["metric_id"],
        "relative_depths": config["mechanism"]["relative_depths"],
        "ranks": config["mechanism"]["ranks"],
        "stages": list(ReplicationRunState.new("plan").stages),
        "output": str(output),
        "writes_performed": False,
    }


def initialize_replication(
    *,
    root: str | Path,
    config_path: str | Path,
    model_id: str,
    revision: str,
    adapter: str,
    output: str | Path,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    output = _validate_output(root, Path(output), must_not_exist=True)
    config = _load(
        config_path=config_path,
        model_id=model_id,
        revision=revision,
        adapter=adapter,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        overrides=overrides,
    )
    adapter_value = adapter_registry.create(
        adapter,
        model_id=model_id,
        revision=revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    lockfile = root / "uv.lock"
    if not lockfile.is_file():
        raise FileNotFoundError("uv.lock is required to initialize a new replication")
    output.mkdir(parents=True, exist_ok=False)
    run_id = _run_id(model_id, revision)
    (output / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    state = ReplicationRunState.new(run_id)
    state.write(output / "run_state.json")
    provenance = {
        "schema_version": "replication-provenance-v1",
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "model": {"id": model_id, "revision": revision},
        "tokenizer": {
            "id": tokenizer_id or model_id,
            "revision": tokenizer_revision or revision,
        },
        "adapter": {"name": adapter, "version": adapter_value.adapter_version},
        "environment": {
            "lockfile": "uv.lock",
            "lockfile_sha256": _sha256(lockfile),
        },
        "seeds": config["seeds"],
        "behavioral_design_hash": canonical_hash(
            {
                "tasks": config["task_families"],
                "interface": config["interface"],
                "behavior": config["behavior"],
            }
        ),
        "behavioral_split_hash": "pending",
        "frozen_model_hashes": {},
        "counterfactual_pair_manifest_hash": "pending",
        "neural_split_hash": "pending",
        "layer_rank_grid": {
            "relative_depths": config["mechanism"]["relative_depths"],
            "ranks": config["mechanism"]["ranks"],
        },
        "selected_controller_hash": "pending",
        "endpoint_id": config["endpoint_id"],
        "metric_id": config["metric_id"],
        "config_hash": canonical_hash(config),
        "output_path": str(output),
    }
    validate_replication_provenance(provenance, required_for_stage="interface")
    write_replication_provenance(output / "provenance.json", provenance)
    protocol = {
        "schema_version": "replication-protocol-v1",
        "run_id": run_id,
        "scientific_invariant": "generalization preserves cognitive_counterfactual_recovery",
        "config_hash": provenance["config_hash"],
        "state_path": "run_state.json",
        "provenance_path": "provenance.json",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "output": str(output), "provenance": provenance}

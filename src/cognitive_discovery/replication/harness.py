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


QWEN_PROSPECTIVE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


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


def load_prospective_run_spec(root: str | Path, path: str | Path) -> dict[str, Any]:
    """Validate the owner-frozen Qwen self-replication contract."""

    root, path = Path(root).resolve(), Path(path).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("prospective run spec must be a mapping")
    if value.get("schema_version") != "prospective-replication-run-v1":
        raise ValueError("unsupported prospective run spec")
    if value.get("analysis_id") != "qwen-prospective":
        raise ValueError("prospective analysis_id must be qwen-prospective")
    if value.get("interpretation") != "pipeline_self_replication":
        raise ValueError("Qwen run must be interpreted as pipeline_self_replication")
    model = value.get("model", {})
    expected_model = {
        "id": "Qwen/Qwen3.5-4B",
        "revision": QWEN_PROSPECTIVE_REVISION,
        "tokenizer_id": "Qwen/Qwen3.5-4B",
        "tokenizer_revision": QWEN_PROSPECTIVE_REVISION,
        "adapter": "qwen",
    }
    if model != expected_model:
        raise ValueError("prospective Qwen model/checkpoint identity changed")
    freshness = value.get("freshness", {})
    required_freshness = {
        "reuse_historical_behavioral_models",
        "reuse_historical_das_bases",
        "reuse_historical_layer_rank",
        "reuse_historical_counterfactual_pairs",
        "reuse_historical_neural_splits",
        "reuse_historical_theory_winner",
    }
    if set(freshness) != required_freshness or any(
        freshness[name] is not False for name in required_freshness
    ):
        raise ValueError("prospective Qwen historical reuse is prohibited")
    if value.get("endpoint_id") != "cognitive_counterfactual_recovery":
        raise ValueError("prospective Qwen endpoint drift")
    if value.get("metric_id") != "global_cfr_v1":
        raise ValueError("prospective Qwen metric drift")
    if set(value.get("excluded_endpoints", [])) != {
        "natural_effect_recovery",
        "fresh_context_extension",
    }:
        raise ValueError("prospective Qwen excluded endpoints changed")
    baseline = value.get("baseline", {})
    if not all(
        baseline.get(name) is True
        for name in (
            "require_git_dirty_false",
            "require_full_cpu_tests",
            "require_uv_lock_hash",
        )
    ):
        raise ValueError("prospective Qwen clean baseline requirements changed")
    if value.get("gates", {}).get("mutable_after_baseline") is not False:
        raise ValueError("prospective Qwen gates must be immutable after baseline")
    base = (root / value["base_protocol"]).resolve()
    if not base.is_file():
        raise FileNotFoundError(f"prospective base protocol is absent: {base}")
    history = (root / value["historical_comparison"]["source"]).resolve()
    if not history.is_file():
        raise FileNotFoundError(f"historical context is absent: {history}")
    return value


def _validate_baseline_preflight(
    path: Path,
    *,
    git_commit: str,
    lockfile_sha256: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "prospective-baseline-preflight-v1",
        "git_commit": git_commit,
        "git_dirty": False,
        "uv_lock_sha256": lockfile_sha256,
        "cpu_test_status": "pass",
        "cpu_test_command": "PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider",
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise ValueError(f"prospective baseline preflight mismatch: {name}")
    return value


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
    run_spec_path: str | Path | None = None,
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
    run_spec = (
        load_prospective_run_spec(root, run_spec_path)
        if run_spec_path is not None
        else None
    )
    if run_spec is not None:
        observed = {
            "id": model_id,
            "revision": revision,
            "tokenizer_id": tokenizer_id or model_id,
            "tokenizer_revision": tokenizer_revision or revision,
            "adapter": adapter,
        }
        if observed != run_spec["model"]:
            raise ValueError("CLI model identity differs from the prospective run spec")
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
        "run_spec_sha256": _sha256(Path(run_spec_path)) if run_spec else None,
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
    run_spec_path: str | Path | None = None,
    baseline_preflight_path: str | Path | None = None,
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
    git_commit = _git(root, "rev-parse", "HEAD")
    git_dirty = bool(_git(root, "status", "--porcelain"))
    lockfile_sha256 = _sha256(lockfile)
    run_spec = None
    preflight = None
    if run_spec_path is not None:
        run_spec_path = Path(run_spec_path).resolve()
        run_spec = load_prospective_run_spec(root, run_spec_path)
        expected = run_spec["model"]
        observed = {
            "id": model_id,
            "revision": revision,
            "tokenizer_id": tokenizer_id or model_id,
            "tokenizer_revision": tokenizer_revision or revision,
            "adapter": adapter,
        }
        if observed != expected:
            raise ValueError("CLI model identity differs from the prospective run spec")
        if git_dirty:
            raise ValueError("prospective Qwen requires git_dirty: false")
        if baseline_preflight_path is None:
            raise ValueError("prospective Qwen requires a CPU baseline preflight record")
        preflight = _validate_baseline_preflight(
            Path(baseline_preflight_path).resolve(),
            git_commit=git_commit,
            lockfile_sha256=lockfile_sha256,
        )
    output.mkdir(parents=True, exist_ok=False)
    run_id = _run_id(model_id, revision)
    (output / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    state = ReplicationRunState.new(run_id)
    state.write(output / "run_state.json")
    if run_spec is not None:
        (output / "prospective_run_spec.yaml").write_text(
            run_spec_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (output / "baseline_preflight.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    provenance = {
        "schema_version": "replication-provenance-v1",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "model": {"id": model_id, "revision": revision},
        "tokenizer": {
            "id": tokenizer_id or model_id,
            "revision": tokenizer_revision or revision,
        },
        "adapter": {"name": adapter, "version": adapter_value.adapter_version},
        "environment": {
            "lockfile": "uv.lock",
            "lockfile_sha256": lockfile_sha256,
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
        "behavioral_survivor_rule_sha256": "pending",
        "behavioral_survivor_set_sha256": "pending",
        "behavioral_survivor_manifest_sha256": "pending",
        "behavioral_survivor_set": [],
        "behavioral_theory_status": "pending",
        "behavioral_survivor_set_frozen_before_neural": False,
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
    if run_spec is not None:
        provenance.update(
            {
                "run_kind": "pipeline_self_replication",
                "analysis_id": "qwen-prospective",
                "run_spec_sha256": _sha256(output / "prospective_run_spec.yaml"),
                "baseline_preflight_sha256": _sha256(
                    output / "baseline_preflight.json"
                ),
                "protocol_file_sha256": _sha256(config_path),
                "historical_reuse": run_spec["freshness"],
                "gates_mutable_after_baseline": False,
                "excluded_endpoints": run_spec["excluded_endpoints"],
            }
        )
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

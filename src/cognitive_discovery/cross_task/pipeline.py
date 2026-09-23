"""Artifact-first orchestration for the prospective cross-task program."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from cognitive_discovery.data.storage import write_records
from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.experiments.contextual_history import compile_contextual_history_design
from cognitive_discovery.pipeline import load_config

from .compatibility import load_compatibility_matrix
from .config import load_cross_task_config
from .provenance import canonical_hash, file_hash, git_identity, write_provenance
from .splits import assign_grouped_splits, split_hash


STAGES = (
    "interface", "behavior", "computational", "active_discrimination",
    "hypothesis_freeze", "representation", "causal", "synthesis",
)


def _json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def initialize_run(
    repository: str | Path,
    output: str | Path,
    config_path: str | Path,
    *,
    final: bool,
) -> dict:
    repository, output, config_path = Path(repository).resolve(), Path(output).resolve(), Path(config_path).resolve()
    config = load_cross_task_config(config_path)
    compatibility_path = repository / config["compatibility_matrix"]["path"]
    matrix = load_compatibility_matrix(compatibility_path)
    if output.exists():
        raise FileExistsError(f"cross-task output already exists: {output}")
    identity = git_identity(repository)
    if final and identity["git_dirty"]:
        raise ValueError("final cross-task discovery requires a clean committed worktree")
    for model in [entry["key"] for entry in config["models"]]:
        root = output / model
        for relative in (
            "interface", "behavior", "computational", "representation/persistence_effect_geometry",
            "causal", "synthesis",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)
        _json(root / "stage_state.json", {
            "schema_version": "cross-task-model-state-v1", "model": model,
            "stages": {name: "pending" for name in STAGES},
        })
    (output / "cross_model").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "effective_config.yaml")
    shutil.copyfile(compatibility_path, output / "task_variable_compatibility_v1.yaml")
    matrix.to_csv(output / "task_variable_compatibility_expanded.csv", index=False)
    state = {
        "schema_version": "cross-task-state-v1", "analysis_id": config["analysis_id"],
        "stages": {name: "pending" for name in STAGES},
        "scientific_freeze": None,
    }
    _json(output / "run_state.json", state)
    lockfile = repository / "uv.lock"
    provenance = {
        "schema_version": "cross-task-provenance-v1", **identity,
        "environment_lock_sha256": file_hash(lockfile),
        "config_sha256": file_hash(output / "effective_config.yaml"),
        "compatibility_sha256": file_hash(output / "task_variable_compatibility_v1.yaml"),
        "computational_model_spec_sha256": canonical_hash(config["computational"]),
        "active_acquisition_config_sha256": canonical_hash(config["active_discovery"]),
        "design_sha256": None, "behavior_split_sha256": None,
        "scientific_freeze_sha256": None,
        "behavioral_survivor_hashes": {},
        "representation_target_manifest_sha256": None,
        "neural_split_hashes": {}, "subspace_controller_hashes": {},
        "models": config["models"], "seeds": config["seeds"],
        "endpoint_ids": config["endpoints"], "output_root": str(output),
        "evidence_mode": "final" if final else "development",
        "historical_fits_reused": False, "historical_neural_bases_reused": False,
    }
    write_provenance(output / "provenance.json", provenance, final=False)
    for model in config["models"]:
        _json(output / model["key"] / "provenance.json", {
            "schema_version": "cross-task-model-provenance-v1",
            "model": model, "git_commit": identity["git_commit"],
            "git_dirty": identity["git_dirty"],
            "environment_lock_sha256": provenance["environment_lock_sha256"],
            "config_sha256": provenance["config_sha256"],
            "compatibility_sha256": provenance["compatibility_sha256"],
            "design_sha256": None, "behavior_split_sha256": None,
            "endpoint_ids": config["endpoints"], "seeds": config["seeds"],
            "parent_provenance": "../provenance.json",
        })
    return {"output": str(output), "models": [model["key"] for model in config["models"]], "final": final}


def prepare_shared_design(
    repository: str | Path,
    output: str | Path,
    *,
    smoke: bool = False,
) -> dict:
    repository, output = Path(repository).resolve(), Path(output).resolve()
    config = load_cross_task_config(output / "effective_config.yaml")
    ontology = load_config(repository / "configs/discovery_v2.yaml")
    broad_count = 28 if smoke else int(config["design"]["broad_semantic_conditions"])
    context_count = 12 if smoke else int(config["design"]["contextual_semantic_conditions"])
    broad = compile_design(
        ontology, n_conditions=broad_count, seed=config["seeds"]["design"],
        design_id="cross_task_broad_v1",
    )
    contextual = compile_contextual_history_design(
        ontology, n_conditions=context_count, seed=config["seeds"]["design"] + 101,
        design_id="cross_task_contextual_v1",
    )
    conditions = [*broad, *contextual]
    grouping = pd.DataFrame({
        "condition_id": [condition.condition_id for condition in conditions],
        "task_family": [condition.task_family for condition in conditions],
        "semantic_group": [
            str(condition.contextual_history["critical_contrast_id"])
            if condition.contextual_history else condition.paired_condition_id
            for condition in conditions
        ],
    })
    grouping = assign_grouped_splits(
        grouping, group_columns=["semantic_group"], stratify_columns=["task_family"],
        seed=config["seeds"]["behavior_split"], fractions=config["splits"]["fractions"],
    )
    by_condition = dict(zip(grouping.condition_id, grouping.analysis_split))
    split_names = {"train": "behavior_train", "selection": "behavior_selection", "test": "behavior_test"}
    conditions = [replace(condition, split=split_names[by_condition[condition.condition_id]]) for condition in conditions]
    rows = []
    group_by_id = dict(zip(grouping.condition_id, grouping.semantic_group))
    for condition in conditions:
        row = condition.to_dict()
        row["semantic_group"] = group_by_id[condition.condition_id]
        row["response_mapping_id"] = condition.response_mapping.mapping_id
        row["analysis_split"] = by_condition[condition.condition_id]
        rows.append(row)
    design_root = output / "shared_design"
    design_root.mkdir(parents=True, exist_ok=True)
    table_path = Path(write_records(rows, design_root / "condition_manifest.parquet"))
    jsonl = design_root / "condition_manifest.jsonl"
    jsonl.write_text("".join(json.dumps(row, sort_keys=True, default=list) + "\n" for row in rows), encoding="utf-8")
    split_value = split_hash(grouping, group_columns=["semantic_group"], split_column="analysis_split")
    summary = {
        "schema_version": "cross-task-shared-design-v1", "smoke": smoke,
        "semantic_conditions": len(conditions) // 2, "rendered_conditions": len(conditions),
        "condition_manifest_sha256": file_hash(table_path),
        "jsonl_sha256": file_hash(jsonl), "behavior_split_sha256": split_value,
        "response_mapping_pairing_valid": bool(
            grouping.groupby("semantic_group").size().min() >= 2
        ),
    }
    _json(design_root / "design_summary.json", summary)
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["design_sha256"] = summary["condition_manifest_sha256"]
    provenance["behavior_split_sha256"] = split_value
    write_provenance(provenance_path, provenance, final=False)
    for model in config["models"]:
        path = output / model["key"] / "provenance.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["design_sha256"] = summary["condition_manifest_sha256"]
        record["behavior_split_sha256"] = split_value
        _json(path, record)
    return summary


def mark_model_stage(output: str | Path, model: str, stage: str, status: str, **artifacts) -> None:
    if stage not in STAGES or status not in {"pending", "running", "complete", "blocked"}:
        raise ValueError("invalid model stage transition")
    path = Path(output) / model / "stage_state.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["stages"][stage] = status
    if artifacts:
        value.setdefault("artifacts", {})[stage] = artifacts
    _json(path, value)


def freeze_hypotheses(output: str | Path, manifest: dict) -> dict:
    output = Path(output)
    state_path = output / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["scientific_freeze"] is not None:
        raise RuntimeError("scientific hypotheses are already frozen")
    required = {
        "behavioral_survivors", "representation_targets", "causal_hypotheses",
        "active_discrimination_outcomes",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"scientific freeze lacks fields: {sorted(missing)}")
    model_keys = {"qwen", "gemma", "llama"}
    if set(manifest["behavioral_survivors"]) != model_keys:
        raise ValueError("scientific freeze requires survivor sets for all three models")
    survivor_hashes = {}
    for model in sorted(model_keys):
        path = output / model / "computational/survivor_set.json"
        if not path.is_file():
            raise FileNotFoundError(f"behavioral survivor set is absent: {path}")
        observed = json.loads(path.read_text(encoding="utf-8"))["behavioral_survivor_set"]
        if sorted(map(str, observed)) != sorted(map(str, manifest["behavioral_survivors"][model])):
            raise ValueError(f"freeze manifest differs from {model} behavior-only survivors")
        survivor_hashes[model] = file_hash(path)
    record = {
        "schema_version": "cross-task-scientific-freeze-v1", **manifest,
        "uses_neural_outcomes_for_behavioral_survivors": False,
    }
    record["freeze_sha256"] = canonical_hash(record)
    _json(output / "scientific_freeze.json", record)
    state["scientific_freeze"] = record["freeze_sha256"]
    state["stages"]["hypothesis_freeze"] = "complete"
    _json(state_path, state)
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["scientific_freeze_sha256"] = file_hash(output / "scientific_freeze.json")
    provenance["behavioral_survivor_hashes"] = survivor_hashes
    provenance["representation_target_manifest_sha256"] = canonical_hash(
        manifest["representation_targets"]
    )
    write_provenance(provenance_path, provenance, final=False)
    for model in model_keys:
        mark_model_stage(output, model, "hypothesis_freeze", "complete", freeze_sha256=record["freeze_sha256"])
    return record


def validate_final_run(output: str | Path) -> dict:
    output = Path(output)
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    if not (output / "scientific_freeze.json").is_file():
        raise ValueError("final run lacks a pre-neural scientific freeze")
    from .provenance import validate_final_provenance
    validate_final_provenance(provenance)
    repository = Path(__file__).resolve().parents[3]
    current = git_identity(repository)
    if current != {"git_commit": provenance["git_commit"], "git_dirty": False}:
        raise ValueError("repository identity changed after the prospective freeze")
    expected_files = {
        "environment_lock_sha256": repository / "uv.lock",
        "config_sha256": output / "effective_config.yaml",
        "compatibility_sha256": output / "task_variable_compatibility_v1.yaml",
    }
    for name, path in expected_files.items():
        if file_hash(path) != provenance[name]:
            raise ValueError(f"final provenance artifact drift: {name}")
    return {"valid": True, "provenance_sha256": file_hash(output / "provenance.json")}

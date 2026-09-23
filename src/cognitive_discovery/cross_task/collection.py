"""Fresh three-model interface validation and shared-manifest behavior collection."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from cognitive_discovery.data.storage import write_records
from cognitive_discovery.data.validation import pilot_gate, validate_records
from cognitive_discovery.design.manifests import load_condition_manifest
from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.pipeline import load_config
from cognitive_discovery.replication.adapters import adapter_registry
from cognitive_discovery.replication.interface import select_valid_interface
from cognitive_discovery.replication.workflow import (
    AdapterParticipant,
    _require_full_vocabulary_diagnostics,
)

from .config import load_cross_task_config
from .provenance import file_hash


def _model(config: dict, key: str) -> dict:
    try:
        return next(value for value in config["models"] if value["key"] == key)
    except StopIteration as error:
        raise ValueError(f"unknown cross-task model: {key}") from error


def _adapter(model: dict, model_path: str | Path, *, online: bool):
    path = Path(model_path)
    if not online and not (path / "config.json").is_file():
        raise FileNotFoundError(f"local model path lacks config.json: {path}")
    return adapter_registry.create(
        model["adapter"], model_id=str(model_path), revision=model["revision"],
        tokenizer_id=str(model_path), tokenizer_revision=model["tokenizer_revision"],
        local_files_only=not online,
    ).load()


def validate_model_interface(
    repository: str | Path,
    output: str | Path,
    *,
    model_key: str,
    model_path: str | Path,
    online: bool = False,
    smoke: bool = False,
) -> dict:
    repository, output = Path(repository), Path(output)
    config = load_cross_task_config(output / "effective_config.yaml")
    model = _model(config, model_key)
    adapter = _adapter(model, model_path, online=online)
    ontology = load_config(repository / "configs/discovery_v2.yaml")
    calibration_per_task = 4 if smoke else int(
        config["interfaces"]["calibration_conditions_per_task"]
    )
    validation_per_task = 4 if smoke else int(
        config["interfaces"]["validation_conditions_per_task"]
    )
    calibration = compile_design(
        ontology, n_conditions=calibration_per_task * len(config["tasks"]),
        seed=config["seeds"]["interface_calibration"],
        design_id="cross_task_interface_calibration_v1",
    )
    gates, errors = {}, {}
    stage_root = output / model_key / "interface"
    for candidate in config["interfaces"]["candidates"]:
        try:
            adapter.validate_response_tokens(candidate["labels"])
            frame = validate_records(collect_conditions(
                calibration, AdapterParticipant(adapter, candidate),
                model_revision=model["revision"], sample_actions=True,
                expand_history_prefixes=False,
            ))
            _require_full_vocabulary_diagnostics(frame)
            gates[candidate["id"]] = {
                task: pilot_gate(part, ontology) for task, part in frame.groupby("task_family")
            }
            write_records(frame.to_dict("records"), stage_root / f"calibration_{candidate['id']}.parquet")
        except (ValueError, RuntimeError) as error:
            errors[candidate["id"]] = str(error)
            gates[candidate["id"]] = {
                task: {"approved": False, "checks": {"valid_semantic_tokens": False}, "reason": str(error)}
                for task in config["tasks"]
            }
    selection = select_valid_interface(gates, required_tasks=config["tasks"])
    selected = next((candidate for candidate in config["interfaces"]["candidates"] if candidate["id"] == selection.selected), None)
    validation_gates = {}
    if selection.passed and selected is not None:
        validation_conditions = compile_design(
            ontology, n_conditions=validation_per_task * len(config["tasks"]),
            seed=config["seeds"]["interface_validation"],
            design_id="cross_task_interface_validation_v1",
        )
        validation = validate_records(collect_conditions(
            validation_conditions, AdapterParticipant(adapter, selected),
            model_revision=model["revision"], sample_actions=True,
            expand_history_prefixes=False,
        ))
        _require_full_vocabulary_diagnostics(validation)
        validation_gates = {
            task: pilot_gate(part, ontology)
            for task, part in validation.groupby("task_family")
        }
        write_records(validation.to_dict("records"), stage_root / "validation_observations.parquet")
    passed = bool(
        selection.passed
        and len(validation_gates) == len(config["tasks"])
        and all(gate["approved"] for gate in validation_gates.values())
    )
    record = {
        "schema_version": "cross-task-interface-validation-v1", "model": model,
        "passed": passed, "selected": selection.selected,
        "selected_interface": selected, "calibration_gates": gates,
        "validation_gates": validation_gates, "errors": errors,
        "uses_scientific_outcomes": False,
        "calibration_seed": config["seeds"]["interface_calibration"],
        "validation_seed": config["seeds"]["interface_validation"],
        "overlaps_shared_scientific_design": False,
    }
    stage_root.mkdir(parents=True, exist_ok=True)
    (stage_root / "validation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return record


def collect_shared_behavior(
    repository: str | Path,
    output: str | Path,
    *,
    model_key: str,
    model_path: str | Path,
    online: bool = False,
    smoke: bool = False,
) -> dict:
    del repository
    output = Path(output)
    config = load_cross_task_config(output / "effective_config.yaml")
    model = _model(config, model_key)
    interface_path = output / model_key / "interface/validation.json"
    interface = json.loads(interface_path.read_text(encoding="utf-8"))
    if not interface.get("passed") or not interface.get("selected_interface"):
        raise RuntimeError("behavior collection requires a valid frozen response interface")
    adapter = _adapter(model, model_path, online=online)
    conditions = load_condition_manifest(output / "shared_design/condition_manifest.jsonl")
    frame = validate_records(collect_conditions(
        conditions, AdapterParticipant(adapter, interface["selected_interface"]),
        model_revision=model["revision"], sample_actions=True,
        expand_history_prefixes=False,
    ))
    _require_full_vocabulary_diagnostics(frame)
    frame["model"] = model_key
    manifest_rows = [
        json.loads(line) for line in
        (output / "shared_design/condition_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    semantic_group = {row["condition_id"]: row["semantic_group"] for row in manifest_rows}
    frame["semantic_group"] = frame["condition_id"].astype(str).map(semantic_group)
    if frame.semantic_group.isna().any():
        raise RuntimeError("behavior observations do not match the shared semantic manifest")
    frame["analysis_split"] = frame["split"].map({
        "behavior_train": "train", "behavior_selection": "selection", "behavior_test": "test"
    })
    if frame.analysis_split.isna().any():
        raise RuntimeError("shared condition manifest lost frozen behavioral split identity")
    path = Path(write_records(
        frame.to_dict("records"), output / model_key / "behavior/observations.parquet"
    ))
    shared_manifest = output / "shared_design/condition_manifest.parquet"
    shutil.copyfile(
        shared_manifest, output / model_key / "behavior/condition_manifest.parquet"
    )
    return {
        "model": model_key, "observations": len(frame), "path": str(path),
        "sha256": file_hash(path), "shared_condition_ids": True,
    }

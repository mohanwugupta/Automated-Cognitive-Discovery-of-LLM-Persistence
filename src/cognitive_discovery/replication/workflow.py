"""Stage executors for fresh model behavioral replication and theory freezing.

Neural stages reuse the repository's counterfactual/DAS primitives through the
same adapter boundary; no model-family branch is permitted here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import yaml

from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.data.validation import pilot_gate, validate_records
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.experiments.contextual_history import (
    compile_contextual_history_design,
)
from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model
from cognitive_discovery.pipeline import generate_design, load_config

from .adapters import adapter_registry
from .interface import select_valid_interface
from .provenance import (
    canonical_hash,
    validate_replication_provenance,
    write_replication_provenance,
)
from .splits import assign_behavior_splits, split_manifest_hash
from .state import ReplicationRunState
from .survivors import select_behavioral_survivors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _interface_messages(messages, original_labels, interface):
    """Map an existing counterbalanced X/Y trial to a candidate interface."""

    original_labels = tuple(original_labels)
    candidate_labels = tuple(interface["labels"])
    if len(original_labels) != 2 or len(candidate_labels) != 2:
        raise ValueError("replication interfaces must be binary")
    mapping = dict(zip(original_labels, candidate_labels))
    copied = [dict(message) for message in messages]
    text = copied[-1]["content"]
    if "Choose one:\n" not in text:
        raise ValueError("canonical renderer omitted the binary choice block")
    prefix, tail = text.split("Choose one:\n", maxsplit=1)
    choices = {}
    for line in tail.splitlines():
        if " = " in line:
            label, action = line.split(" = ", maxsplit=1)
            if label in mapping:
                choices[mapping[label]] = action
    if set(choices) != set(candidate_labels):
        raise ValueError("could not map the rendered choice labels to the candidate interface")
    style = interface.get("style", "choice_mapping")
    if style == "yes_no_question":
        yes, no = candidate_labels
        replacement = (
            f"Your two possible actions are to {choices[yes]}, or to {choices[no]}.\n"
            f"Would you choose to {choices[yes]}? Answer only {yes} or {no}."
        )
    else:
        replacement = (
            f"Choose one:\n{candidate_labels[0]} = {choices[candidate_labels[0]]}\n"
            f"{candidate_labels[1]} = {choices[candidate_labels[1]]}\n"
            f"Respond with only {candidate_labels[0]} or {candidate_labels[1]}."
        )
    copied[-1]["content"] = prefix + replacement
    return copied, candidate_labels, mapping


class AdapterParticipant:
    """Behavior-only participant that translates the canonical rendered interface."""

    def __init__(self, adapter, interface):
        self.adapter = adapter
        self.interface = dict(interface)
        self.model_id = adapter.model_id
        self.revision = adapter.revision

    def binary_decision(self, messages, labels, *, positive_label):
        actual, candidate_labels, mapping = _interface_messages(
            messages, labels, self.interface
        )
        positive = mapping[positive_label]
        return self.adapter.get_response_metrics(
            actual,
            candidate_labels,
            positive_label=positive,
        )


def _require_full_vocabulary_diagnostics(frame: pd.DataFrame) -> None:
    """Fail closed rather than converting absent token diagnostics to failures."""

    required = {"p_action_mass_raw", "top_token_is_action"}
    missing = required - set(frame)
    if missing or frame[list(required)].isna().any().any():
        raise RuntimeError(
            "full-vocabulary response diagnostics are missing: "
            f"{sorted(missing) if missing else 'null values'}"
        )
    masses = frame.p_action_mass_raw.to_numpy(dtype=float)
    if not np.isfinite(masses).all() or ((masses < 0.0) | (masses > 1.0)).any():
        raise RuntimeError("full-vocabulary action mass must be finite and lie in [0, 1]")


def _load_run(output: Path):
    config = yaml.safe_load((output / "effective_config.yaml").read_text(encoding="utf-8"))
    state = ReplicationRunState.load(output / "run_state.json")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    if canonical_hash(config) != provenance["config_hash"]:
        raise RuntimeError("effective replication gates changed after initialization")
    if provenance.get("run_kind") == "pipeline_self_replication":
        run_spec = output / "prospective_run_spec.yaml"
        preflight = output / "baseline_preflight.json"
        if not run_spec.is_file() or _sha256(run_spec) != provenance.get("run_spec_sha256"):
            raise RuntimeError("prospective run spec changed after baseline freeze")
        if not preflight.is_file() or _sha256(preflight) != provenance.get(
            "baseline_preflight_sha256"
        ):
            raise RuntimeError("prospective baseline preflight changed")
        repository = Path(__file__).resolve().parents[3]
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        if current_commit != provenance.get("git_commit"):
            raise RuntimeError(
                "repository commit changed after prospective baseline freeze"
            )
        tracked_drift = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "src",
                "scripts",
                "configs",
                "slurm",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=repository,
            check=False,
        )
        if tracked_drift.returncode != 0:
            raise RuntimeError(
                "scientific code, configuration, or environment drifted after "
                "prospective baseline freeze"
            )
    return config, state, provenance


def _adapter(config, *, online: bool):
    model = config["model"]
    value = adapter_registry.create(
        model["adapter"],
        model_id=model["id"],
        revision=model["revision"],
        tokenizer_id=model["tokenizer_id"],
        tokenizer_revision=model["tokenizer_revision"],
        local_files_only=not online,
    )
    return value.load()


def _ontology_config(root: Path, config: dict) -> dict:
    relative = config.get("ontology_config", "configs/discovery_v1.yaml")
    value = load_config(root / relative)
    value["collection"]["expand_history_prefixes"] = False
    value["collection"]["sample_actions"] = True
    interface = config["interface"]
    value["collection"]["maximum_mapping_gap"] = float(
        interface["maximum_mapping_gap"]
    )
    value["collection"]["minimum_logit_sd"] = float(interface["minimum_logit_sd"])
    return value


def _manifest(path: Path, conditions) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(condition.to_dict(), sort_keys=True) + "\n" for condition in conditions),
        encoding="utf-8",
    )
    return path


def execute_interface_stage(root: Path, output: Path, *, online: bool = False) -> dict:
    config, state, provenance = _load_run(output)
    state.start("interface")
    state.write(output / "run_state.json")
    validate_replication_provenance(provenance, required_for_stage="interface")
    adapter = _adapter(config, online=online)
    ontology = _ontology_config(root, config)
    stage_root = output / "interface"
    stage_root.mkdir(parents=True, exist_ok=True)
    task_count = len(config["task_families"])
    calibration_per_task = int(config["interface"].get("calibration_conditions_per_task", 30))
    validation_per_task = int(config["interface"].get("validation_conditions_per_task", 70))
    design, _ = generate_design(
        ontology,
        output=stage_root / "calibration_design",
        conditions=calibration_per_task * task_count,
        seed=int(config["seeds"]["interface_calibration"]),
        design_id="replication_interface_calibration",
    )
    candidate_rows = []
    candidate_gates = {}
    candidate_errors = {}
    for candidate in config["interface"]["candidates"]:
        try:
            adapter.validate_response_tokens(candidate["labels"])
            participant = AdapterParticipant(adapter, candidate)
            frame = validate_records(
                collect_conditions(
                    design,
                    participant,
                    model_revision=config["model"]["revision"],
                    sample_actions=True,
                    expand_history_prefixes=False,
                )
            )
            _require_full_vocabulary_diagnostics(frame)
        except ValueError as error:
            candidate_errors[candidate["id"]] = str(error)
            gates = {
                task: {
                    "approved": False,
                    "checks": {"valid_semantic_tokens": False},
                    "reason": str(error),
                }
                for task in config["task_families"]
            }
            candidate_gates[candidate["id"]] = gates
            for task, gate in gates.items():
                candidate_rows.append(
                    {"interface_id": candidate["id"], "task_family": task, **gate}
                )
            continue
        write_records(frame.to_dict("records"), stage_root / f"calibration_{candidate['id']}.parquet")
        gates = {task: pilot_gate(group, ontology) for task, group in frame.groupby("task_family")}
        candidate_gates[candidate["id"]] = gates
        for task, gate in gates.items():
            candidate_rows.append({"interface_id": candidate["id"], "task_family": task, **gate})
    pd.DataFrame(
        [
            {
                "interface_id": candidate["id"],
                "labels": json.dumps(candidate["labels"]),
                "style": candidate.get("style", "choice_mapping"),
            }
            for candidate in config["interface"]["candidates"]
        ]
    ).to_csv(stage_root / "candidate_interfaces.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(stage_root / "calibration_results.csv", index=False)
    _json(stage_root / "calibration_results.json", candidate_gates)
    selection = select_valid_interface(
        candidate_gates, required_tasks=config["task_families"]
    )
    protocol = {
        "model": config["model"],
        "candidates": config["interface"]["candidates"],
        "selection_uses_scientific_outcomes": False,
        "candidate_errors": candidate_errors,
        "calibration_seed": config["seeds"]["interface_calibration"],
        "validation_seed": config["seeds"]["interface_validation"],
    }
    _json(stage_root / "protocol.json", protocol)
    if not selection.passed:
        result = dataclasses.asdict(selection)
        _json(stage_root / "selected_interface.json", result)
        _json(stage_root / "gates.json", candidate_gates)
        state.complete("interface", outcome="measurement_failure", artifacts={})
        state.write(output / "run_state.json")
        return result
    selected = next(
        candidate
        for candidate in config["interface"]["candidates"]
        if candidate["id"] == selection.selected
    )
    validation_design, _ = generate_design(
        ontology,
        output=stage_root / "validation_design",
        conditions=validation_per_task * task_count,
        seed=int(config["seeds"]["interface_validation"]),
        design_id="replication_interface_validation",
    )
    validation = validate_records(
        collect_conditions(
            validation_design,
            AdapterParticipant(adapter, selected),
            model_revision=config["model"]["revision"],
            sample_actions=True,
            expand_history_prefixes=False,
        )
    )
    _require_full_vocabulary_diagnostics(validation)
    write_records(validation.to_dict("records"), stage_root / "validation_observations.parquet")
    validation_gates = {
        task: pilot_gate(group, ontology) for task, group in validation.groupby("task_family")
    }
    passed = len(validation_gates) == task_count and all(
        gate["approved"] for gate in validation_gates.values()
    )
    result = {
        "selected": selection.selected,
        "labels": selected["labels"],
        "style": selected.get("style", "choice_mapping"),
        "passed": bool(passed),
        "approved_task_count": sum(gate["approved"] for gate in validation_gates.values()),
        "replication_status": "interface_valid" if passed else "measurement_failure",
    }
    _json(stage_root / "gates.json", validation_gates)
    _json(stage_root / "selected_interface.json", result)
    state.complete(
        "interface",
        outcome="pass" if passed else "measurement_failure",
        artifacts={
            "selected_interface": _sha256(stage_root / "selected_interface.json"),
            "gates": _sha256(stage_root / "gates.json"),
        },
    )
    state.write(output / "run_state.json")
    return result


def execute_behavior_stage(root: Path, output: Path, *, online: bool = False) -> dict:
    config, state, provenance = _load_run(output)
    state.start("behavior")
    state.write(output / "run_state.json")
    selected = json.loads((output / "interface/selected_interface.json").read_text())
    if not selected.get("passed"):
        raise RuntimeError("behavior cannot start without a valid measurement interface")
    adapter = _adapter(config, online=online)
    ontology = _ontology_config(root, config)
    stage_root = output / "behavior"
    stage_root.mkdir(parents=True, exist_ok=True)
    broad_conditions, _ = generate_design(
        ontology,
        output=stage_root / "design_source",
        conditions=int(config["behavior"]["broad_conditions"]),
        seed=int(config["seeds"]["behavior_design"]),
        design_id="model_agnostic_replication_behavior",
    )
    contextual_conditions = compile_contextual_history_design(
        ontology,
        n_conditions=int(config["behavior"]["contextual_conditions"]),
        seed=int(config["seeds"]["behavior_design"]) + 101,
        design_id="model_agnostic_replication_contextual_behavior",
    )
    conditions = [*broad_conditions, *contextual_conditions]
    split_frame = pd.DataFrame(
        {
            "condition_id": [condition.condition_id for condition in conditions],
            "semantic_group": [
                (
                    condition.contextual_history["critical_contrast_id"]
                    if condition.contextual_history is not None
                    else condition.paired_condition_id
                )
                for condition in conditions
            ],
            "task_family": [condition.task_family for condition in conditions],
        }
    )
    split_frame = assign_behavior_splits(
        split_frame,
        group_column="semantic_group",
        task_column="task_family",
        seed=int(config["seeds"]["behavior_split"]),
        fractions=tuple(config["behavior"]["split_fractions"].values()),
    )
    split_by_id = dict(zip(split_frame.condition_id, split_frame.behavior_split))
    conditions = [
        dataclasses.replace(condition, split=split_by_id[condition.condition_id])
        for condition in conditions
    ]
    manifest = _manifest(stage_root / "condition_manifest.jsonl", conditions)
    manifest_table = write_records(
        [condition.to_dict() for condition in conditions],
        stage_root / "condition_manifest.parquet",
    )
    participant = AdapterParticipant(adapter, selected)
    observations = validate_records(
        collect_conditions(
            conditions,
            participant,
            model_revision=config["model"]["revision"],
            sample_actions=True,
            expand_history_prefixes=False,
        )
    )
    _require_full_vocabulary_diagnostics(observations)
    observation_path = write_records(
        observations.to_dict("records"), stage_root / "observations.parquet"
    )
    split_hash = split_manifest_hash(split_frame, "semantic_group", "behavior_split")
    split_summary = {
        "schema_version": "behavior-split-v1",
        "seed": config["seeds"]["behavior_split"],
        "sha256": split_hash,
        "counts": split_frame.drop_duplicates("semantic_group").behavior_split.value_counts().to_dict(),
        "group_column": "paired_condition_id_or_contextual_critical_contrast_id",
    }
    _json(stage_root / "split_manifest.json", split_summary)
    gates = {
        task: pilot_gate(group, ontology) for task, group in observations.groupby("task_family")
    }
    _json(
        stage_root / "validity_summary.json",
        {"passed": all(gate["approved"] for gate in gates.values()), "tasks": gates},
    )
    provenance["behavioral_split_hash"] = split_hash
    provenance["behavioral_condition_manifest_hash"] = _sha256(manifest)
    provenance["behavioral_design_hash"] = _sha256(Path(manifest_table))
    write_replication_provenance(output / "provenance.json", provenance)
    state.complete(
        "behavior",
        outcome="pass",
        artifacts={
            "condition_manifest": _sha256(manifest),
            "condition_manifest_table": _sha256(Path(manifest_table)),
            "observations": _sha256(Path(observation_path)),
            "split_manifest": _sha256(stage_root / "split_manifest.json"),
        },
    )
    state.write(output / "run_state.json")
    return {"observations": len(observations), "split_hash": split_hash}


def _model_metrics(frame, prediction, *, phase: str, model: str):
    rows = []
    enriched = frame.assign(prediction=np.asarray(prediction, dtype=float))
    for task, group in [("all", enriched), *list(enriched.groupby("task_family"))]:
        rows.append(
            {
                "phase": phase,
                "model": model,
                "task": task,
                "r2": float(r2_score(group.persistence_logit, group.prediction)),
                "mse": float(mean_squared_error(group.persistence_logit, group.prediction)),
                "n": len(group),
            }
        )
    task_rows = [row for row in rows if row["task"] != "all"]
    rows.append(
        {
            "phase": phase,
            "model": model,
            "task": "task_macro",
            "r2": float(np.mean([row["r2"] for row in task_rows])),
            "mse": float(np.mean([row["mse"] for row in task_rows])),
            "n": len(enriched),
        }
    )
    return rows


def execute_model_comparison_stage(root: Path, output: Path) -> dict:
    del root
    config, state, provenance = _load_run(output)
    state.start("model_comparison")
    state.write(output / "run_state.json")
    frame = read_records(output / "behavior/observations.parquet")
    stage_root = output / "behavior/models"
    candidate_root = stage_root / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    split_names = {
        "behavior_train": "train",
        "behavior_selection": "selection",
        "behavior_test": "test",
    }
    bank = list(config["model_comparison"]["minimum_bank"])
    metrics = []
    models = {}
    train = frame[frame.split == "behavior_train"]
    for name in bank:
        model = fit_hierarchical_model(train, name, variant="M3", alphas=(1.0,))
        models[name] = model
        with (candidate_root / f"{name}.pkl").open("wb") as handle:
            pickle.dump(model, handle, protocol=5)
        if hasattr(model, "task_parameters"):
            model.task_parameters().to_csv(candidate_root / f"{name}_parameters.csv", index=False)
        for split, phase in split_names.items():
            part = frame[frame.split == split]
            prediction = model.predict(part)
            metrics.extend(_model_metrics(part, prediction, phase=phase, model=name))
            pd.DataFrame(
                {
                    "condition_id": part.condition_id,
                    "task_family": part.task_family,
                    "observed": part.persistence_logit,
                    "prediction": prediction,
                }
            ).to_csv(candidate_root / f"{phase}_{name}_predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(stage_root / "model_comparison.csv", index=False)
    metric_frame[metric_frame.task != "all"].to_csv(stage_root / "per_task_metrics.csv", index=False)
    selection = metric_frame[
        (metric_frame.phase == "selection") & (metric_frame.task == "all")
    ].set_index("model")
    test = metric_frame[(metric_frame.phase == "test") & (metric_frame.task == "all")].set_index("model")
    baseline = "immediate_state"
    threshold = float(config["model_comparison"]["minimum_improvement_over_immediate"])
    minimum_r2 = float(config["model_comparison"]["minimum_history_test_r2"])
    history_models = [name for name in bank if name != baseline]
    selection_survivors = [
        name
        for name in history_models
        if float(selection.loc[name, "r2"]) >= float(selection.loc[baseline, "r2"]) + threshold
    ]
    survivor_config = config["behavioral_survivor_set"]
    if not selection_survivors:
        replication = "fail"
        theory_status = "unresolved"
        survivors = []
        best = None
        survivor_record = None
    else:
        survivor_record = select_behavioral_survivors(
            selection.mse.to_dict(),
            metric=survivor_config["metric"],
            direction=survivor_config["direction"],
            equivalence_rule=survivor_config["equivalence_rule"],
            equivalence_margin=float(survivor_config["equivalence_margin"]),
            eligible_models=selection_survivors,
        )
        best = survivor_record["best_model"]
        survivors = survivor_record["behavioral_survivor_set"]
        theory_status = survivor_record["behavioral_theory_status"]
        confirmed = [
            name
            for name in survivors
            if float(test.loc[name, "r2"]) >= minimum_r2
            and float(test.loc[name, "r2"]) >= float(test.loc[baseline, "r2"]) + threshold
        ]
        replication = "pass" if len(confirmed) == len(survivors) else "partial"
    result = {
        "behavioral_replication": replication,
        "behavioral_theory_status": theory_status,
        "best_model": best,
        "behavioral_survivor_set": survivors,
        "selected_model": best,
        "selected_models": survivors,
        "selection_uses_test_for_choice": False,
        "test_is_used_only_for_preregistered_confirmation": True,
        "neural_results_used_for_selection": False,
    }
    if survivor_record is not None:
        result.update(
            {
                "survivor_rule": {
                    key: survivor_record[key]
                    for key in (
                        "metric", "direction", "equivalence_rule",
                        "equivalence_margin", "evidence_axis",
                        "neural_results_used", "rule_sha256",
                    )
                },
                "survivor_scores": survivor_record["scores"],
                "survivor_deltas_from_best": survivor_record["deltas_from_best"],
                "survivor_membership_sha256": survivor_record["membership_sha256"],
            }
        )
    _json(stage_root / "selected_models.json", result)
    state.complete(
        "model_comparison",
        outcome=replication,
        artifacts={
            "comparison": _sha256(stage_root / "model_comparison.csv"),
            "selection": _sha256(stage_root / "selected_models.json"),
        },
    )
    state.write(output / "run_state.json")
    return result


def execute_freeze_theory_stage(root: Path, output: Path) -> dict:
    del root
    config, state, provenance = _load_run(output)
    state.start("freeze_theory")
    state.write(output / "run_state.json")
    stage_root = output / "behavior/models"
    selection = json.loads((stage_root / "selected_models.json").read_text())
    selected = list(
        selection.get("behavioral_survivor_set", selection.get("selected_models", []))
    )
    if not selected:
        state.fail("freeze_theory", reason="no history-sensitive behavioral model survived")
        state.write(output / "run_state.json")
        raise RuntimeError("no behavioral theory can produce a mechanistic counterfactual")
    frame = read_records(output / "behavior/observations.parquet")
    training_ids = sorted(frame.loc[frame.split == "behavior_train", "condition_id"].astype(str))
    training_hash = canonical_hash(training_ids)
    frozen_root = stage_root / "frozen_models"
    frozen_root.mkdir(parents=True, exist_ok=True)
    _json(frozen_root / "frozen_architectures.json", selected)
    survivor_manifest = {
        "schema_version": "behavioral-survivor-set-v1",
        "best_model": selection["best_model"],
        "behavioral_survivor_set": selected,
        "behavioral_theory_status": selection["behavioral_theory_status"],
        **selection["survivor_rule"],
        "scores": selection["survivor_scores"],
        "deltas_from_best": selection["survivor_deltas_from_best"],
        "membership_sha256": selection["survivor_membership_sha256"],
        "frozen_before_neural_execution": True,
    }
    survivor_path = _json(stage_root / "frozen_survivor_set.json", survivor_manifest)
    hashes = {}
    for name in selected:
        source = stage_root / "candidates" / f"{name}.pkl"
        destination = frozen_root / name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / "model.pkl")
        parameter_source = stage_root / "candidates" / f"{name}_parameters.csv"
        if parameter_source.exists():
            shutil.copy2(parameter_source, destination / "parameters.csv")
        else:
            pd.DataFrame().to_csv(destination / "parameters.csv", index=False)
        _json(
            destination / "model_spec.json",
            {
                "architecture": name,
                "variant": "M3",
                "ridge_alphas": [1.0],
                "feature_definition": "canonical cognitive_discovery model registry",
                "fit_seed": config["seeds"]["behavior_fit"],
                "split_hash": provenance["behavioral_split_hash"],
            },
        )
        _json(
            destination / "training_condition_hashes.json",
            {"sha256": training_hash, "condition_ids": training_ids},
        )
        for filename in (
            "model.pkl",
            "parameters.csv",
            "model_spec.json",
            "training_condition_hashes.json",
        ):
            path = destination / filename
            hashes[f"{name}/{filename}"] = _sha256(path)
    provenance["frozen_model_hashes"] = hashes
    provenance["behavioral_survivor_rule_sha256"] = survivor_manifest["rule_sha256"]
    provenance["behavioral_survivor_set_sha256"] = survivor_manifest["membership_sha256"]
    provenance["behavioral_survivor_manifest_sha256"] = _sha256(survivor_path)
    provenance["behavioral_survivor_set"] = selected
    provenance["behavioral_theory_status"] = selection["behavioral_theory_status"]
    provenance["behavioral_survivor_set_frozen_before_neural"] = True
    write_replication_provenance(output / "provenance.json", provenance)
    public_root = output / "frozen_theories"
    public_root.mkdir(parents=True, exist_ok=True)
    _json(
        public_root / "manifest.json",
        {
            **survivor_manifest,
            "canonical_artifact": "behavior/models/frozen_survivor_set.json",
            "frozen_model_root": "behavior/models/frozen_models",
            "frozen_model_hashes": hashes,
        },
    )
    state.complete(
        "freeze_theory",
        outcome=selection["behavioral_theory_status"],
        artifacts=hashes,
    )
    state.write(output / "run_state.json")
    return {
        "best_model": selection["best_model"],
        "behavioral_survivor_set": selected,
        "behavioral_theory_status": selection["behavioral_theory_status"],
        "selected_model": selection["best_model"],
        "selected_models": selected,
        "survivor_rule_sha256": survivor_manifest["rule_sha256"],
        "survivor_set_sha256": survivor_manifest["membership_sha256"],
        "hashes": hashes,
    }


STAGE_EXECUTORS = {
    "interface": execute_interface_stage,
    "behavior": execute_behavior_stage,
    "model_comparison": execute_model_comparison_stage,
    "freeze_theory": execute_freeze_theory_stage,
}


def execute_stage(root: str | Path, output: str | Path, stage: str, *, online=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    executors = dict(STAGE_EXECUTORS)
    if stage in {"counterfactuals", "mechanism", "generalization", "specificity"}:
        from .neural import (
            execute_counterfactual_stage,
            execute_generalization_stage,
            execute_mechanism_stage,
            execute_specificity_stage,
        )

        executors.update(
            {
                "counterfactuals": execute_counterfactual_stage,
                "mechanism": execute_mechanism_stage,
                "generalization": execute_generalization_stage,
                "specificity": execute_specificity_stage,
            }
        )
    elif stage == "report":
        from .finalize import execute_report_stage

        executors["report"] = execute_report_stage
    try:
        executor = executors[stage]
    except KeyError as error:
        raise RuntimeError(
            f"stage {stage!r} is optional or unknown; the core executable stages are "
            "interface, behavior, model_comparison, freeze_theory, counterfactuals, "
            "mechanism, generalization, specificity, and report"
        ) from error
    kwargs = (
        {"online": online}
        if stage in {"interface", "behavior", "mechanism", "generalization", "specificity"}
        else {}
    )
    try:
        return executor(root, output, **kwargs)
    except Exception as error:
        state_path = output / "run_state.json"
        if state_path.exists():
            state = ReplicationRunState.load(state_path)
            if state.stages.get(stage, {}).get("status") == "running":
                state.fail(stage, reason=f"{type(error).__name__}: {error}")
                state.write(state_path)
        raise


def execute_all(root: str | Path, output: str | Path, *, online=False):
    """Execute the prospective core pipeline, skipping completed stages on resume."""

    root, output = Path(root).resolve(), Path(output).resolve()
    results = {}
    core = (
        "interface",
        "behavior",
        "model_comparison",
        "freeze_theory",
        "counterfactuals",
        "mechanism",
        "generalization",
        "specificity",
        "report",
    )
    for stage in core:
        state = ReplicationRunState.load(output / "run_state.json")
        if state.stages[stage]["status"] == "complete":
            continue
        if state.replication_status == "measurement_failure" and stage != "report":
            continue
        results[stage] = execute_stage(root, output, stage, online=online)
        state = ReplicationRunState.load(output / "run_state.json")
        if state.replication_status == "measurement_failure":
            results["report"] = execute_stage(root, output, "report", online=online)
            break
    return results

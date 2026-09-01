"""Counterfactual-first mechanistic discovery orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.experiments.registry import get_renderer
from cognitive_discovery.mechanistic.activations.qwen_runner import (
    MechanisticQwenRunner,
)
from cognitive_discovery.mechanistic.dataset.matched_conditions import (
    load_mechanistic_manifest,
)
from cognitive_discovery.mechanistic.representations.mean_difference import (
    load_directions,
)
from cognitive_discovery.mechanistic.steering.controls import random_directions

from .counterfactuals import (
    FrozenTheoryBank,
    PRIMARY_VARIABLES,
    build_counterfactual_pairs,
    select_neural_pairs,
)
from .das import DASAlignment, audit_learning_splits, load_alignment, save_alignment
from .interventions import (
    interchange_subspace,
    intervention_norm,
    remove_subspace,
    replace_whole_state,
)
from .metrics import (
    counterfactual_metrics,
    counterfactual_recovery,
    counterfactual_specificity,
)
from .synthetic import run_synthetic_validation


ALL_LAYERS = tuple(range(32))
REQUIRED_SPECIFICITY_CONTROLS = (
    "persistence_state",
    "persistence_output",
    "generic_value",
    "task_id",
    "response_mapping",
)
REQUIRED_UNRELATED_TARGETS = ("action_history", "generic_value")


def _root(config: dict, output=None) -> Path:
    return Path(output or config.get("output_root", "artifacts/causal_mech_v1"))


def _theory_root(config: dict, theory_output=None) -> Path:
    return Path(
        theory_output or config.get("theory_output", "artifacts/theory_resolution_v1")
    )


def _source_root(config: dict, mechanistic_output=None) -> Path:
    return Path(
        mechanistic_output
        or config.get("mechanistic_output", "artifacts/mechanistic_v1")
    )


def _json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _record_path(root: Path, relative: str) -> Path:
    path = root / relative
    if path.exists():
        return path
    alternate = path.with_suffix(".csv.gz")
    return alternate if alternate.exists() else path


def _make_runner(config, *, model_path=None, revision=None, online=False):
    return MechanisticQwenRunner.from_pretrained(
        model_path or config["model"],
        revision=revision or config.get("model_revision"),
        local_files_only=not online,
    )


def _condition(record):
    """Return the condition carried by either supported manifest record shape."""

    return getattr(record, "condition", record)


def _trial(record):
    condition = _condition(record)
    return get_renderer(condition.task_family).render(condition)


def _forward(runner, record, *, layer=None, editor=None, capture_layers=None):
    condition = _condition(record)
    trial = _trial(record)
    editors = {int(layer): editor} if layer is not None and editor is not None else None
    if capture_layers is None:
        capture_layers = () if layer is None else (int(layer),)
    return runner.forward(
        list(trial.messages),
        condition.response_mapping.labels,
        positive_label=condition.response_mapping.continue_label,
        editors=editors,
        capture_layers=capture_layers,
    )


def _entry_audit(theory_root: Path, config: dict) -> dict:
    comparison = pd.read_csv(theory_root / "theory/final_model_comparison.csv")
    frozen = json.loads(
        (theory_root / "frozen_models/frozen_architectures.json").read_text(
            encoding="utf-8"
        )
    )
    minimum_r2 = float(config.get("entry_criteria", {}).get("minimum_heldout_r2", 0.6))
    prediction = comparison[comparison.stage == "frozen_prediction"]
    freeze_decisions = pd.read_csv(theory_root / "frozen_models/freeze_decisions.csv")
    rows = []
    for architecture in frozen:
        observed = prediction[prediction.architecture == architecture]
        frozen_row = freeze_decisions[freeze_decisions.architecture == architecture]
        r2 = (
            float(observed.r2.iloc[0])
            if len(observed)
            else (float(frozen_row.macro_r2.iloc[0]) if len(frozen_row) else np.nan)
        )
        specification = theory_root / "frozen_models" / architecture / "model_spec.json"
        training_hashes = (
            theory_root
            / "frozen_models"
            / architecture
            / "training_condition_hashes.json"
        )
        rows.append(
            {
                "architecture": architecture,
                "heldout_r2": r2,
                "strong_heldout_prediction": bool(np.isfinite(r2) and r2 >= minimum_r2),
                "operational_definition_frozen": specification.exists(),
                "training_dataset_frozen": training_hashes.exists(),
                "robustness_across_tasks": True,
                "counterfactual_predictions_required_next": True,
            }
        )
    ready = [
        row
        for row in rows
        if row["strong_heldout_prediction"]
        and row["operational_definition_frozen"]
        and row["training_dataset_frozen"]
    ]
    if not ready:
        raise RuntimeError(
            "no frozen behavioral theory satisfies mechanistic entry criteria"
        )
    return {
        "passed": True,
        "minimum_heldout_r2": minimum_r2,
        "retained_theories": [row["architecture"] for row in ready],
        "theory_ambiguity_retained": len(ready) > 1,
        "architectures": rows,
    }


def prepare_causal_mechanistic_run(
    config: dict,
    *,
    theory_output: str | Path | None = None,
    mechanistic_output: str | Path | None = None,
    output: str | Path | None = None,
):
    """Freeze theory hashes and generate all counterfactuals before neural work."""

    root = _root(config, output)
    theory_root = _theory_root(config, theory_output)
    source_root = _source_root(config, mechanistic_output)
    root.mkdir(parents=True, exist_ok=True)
    entry = _entry_audit(theory_root, config)
    bank = FrozenTheoryBank(theory_root / "frozen_models")
    retained = set(entry["retained_theories"])
    if not retained.issubset(bank.architectures):
        raise ValueError("entry audit and frozen theory bank disagree")
    handoff_root = root / "behavioral_handoff"
    handoff_root.mkdir(parents=True, exist_ok=True)
    _json(handoff_root / "model_hash.json", bank.manifest())
    coefficient_rows = []
    definitions = {}
    for architecture in bank.architectures:
        architecture_root = theory_root / "frozen_models" / architecture
        table = pd.read_csv(architecture_root / "parameters.csv")
        coefficient_rows.extend(table.to_dict("records"))
        definitions[architecture] = json.loads(
            (architecture_root / "model_spec.json").read_text(encoding="utf-8")
        )
    pd.DataFrame(coefficient_rows).to_csv(
        handoff_root / "frozen_coefficients.csv", index=False
    )
    _json(
        handoff_root / "variable_definitions.json",
        {
            "primary_variables": list(PRIMARY_VARIABLES),
            "outcome_history": "Exponentially weighted raw semantic outcome history O_t.",
            "contextual_outcome_history": "Cue-weighted context-relevant outcome history O*_t.",
            "action_history": "Exponentially weighted semantic action history A_t; specificity control.",
            "generic_value": "Current success-evidence intervention; specificity control.",
            "frozen_model_definitions": definitions,
        },
    )
    _json(handoff_root / "entry_criteria.json", entry)

    source_manifest = source_root / "manifests/mechanistic_conditions.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError(
            f"mechanistic condition manifest is absent: {source_manifest}"
        )
    manifest_copy = root / "counterfactuals/mechanistic_conditions.jsonl"
    manifest_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest, manifest_copy)
    records = load_mechanistic_manifest(manifest_copy)
    pairs, predictions = build_counterfactual_pairs(records, bank)
    pairs = select_neural_pairs(pairs, predictions, config)
    pair_path = write_records(
        pairs.to_dict("records"), root / "counterfactuals/pair_manifest.parquet"
    )
    prediction_path = write_records(
        predictions.to_dict("records"),
        root / "counterfactuals/predicted_effects.parquet",
    )
    split_rows = []
    for split, part in pairs.groupby("pair_split"):
        split_rows.append(
            {
                "pair_split": split,
                "pairs": len(part),
                "selected_for_neural": int(part.selected_for_neural.sum()),
                "tasks": sorted(part.task_family.unique()),
                "behavioral_validation_rows": int(part.behavioral_validation_row.sum()),
            }
        )
    _json(root / "counterfactuals/split_audit.json", split_rows)
    synthetic = pd.DataFrame(
        run_synthetic_validation(seed=int(config.get("seed", 73001)))
    )
    synthetic.to_csv(root / "behavioral_handoff/synthetic_validation.csv", index=False)
    if not synthetic.passed.astype(bool).all():
        failed = synthetic.loc[~synthetic.passed.astype(bool), "case"].tolist()
        raise RuntimeError(f"synthetic causal validation failed: {failed}")
    metadata = {
        "protocol_version": config.get(
            "protocol_version", "counterfactual_causal_mechanistic_v1"
        ),
        "git_commit": _git_commit(),
        "model_id": config.get("model", "Qwen/Qwen3.5-4B"),
        "model_revision": config.get("model_revision"),
        "theory_root": str(theory_root),
        "mechanistic_source_root": str(source_root),
        "frozen_theory_hashes": bank.hashes,
        "source_manifest_sha256": _sha256(source_manifest),
        "entry_criteria": entry,
        "theory_targets": config.get("theory_targets", {}),
        "activation_position": "final_prompt_token",
        "layer_index_convention": "zero_based_block_output",
        "neural_learning_splits": ["mech_pair_train", "mech_pair_validation"],
        "scientific_claim_splits": ["mech_pair_test", "mech_task_holdout"],
        "behavioral_validation_used": False,
        "full_activations_saved": False,
    }
    _json(root / "run_metadata.json", metadata)
    return {
        "pair_manifest": pair_path,
        "predicted_effects": prediction_path,
        "pairs": len(pairs),
        "selected_pairs": int(pairs.selected_for_neural.sum()),
        "retained_theories": entry["retained_theories"],
    }


def prepare_diagnostic_information_scan(
    config: dict,
    *,
    mechanistic_output: str | Path | None = None,
    output: str | Path | None = None,
):
    """Import the completed all-layer scan and relabel it as information access."""

    root = _root(config, output)
    source = _source_root(config, mechanistic_output)
    layer_metrics = pd.read_csv(source / "representation/layer_metrics.csv")
    loto = pd.read_csv(source / "representation/loto_metrics.csv")
    specificity = pd.read_csv(source / "representation/specificity.csv")
    layer_metrics["evidence_level"] = 2
    layer_metrics["result_type"] = "information_access"
    layer_metrics["mechanistic_claim_permitted"] = False
    loto["evidence_level"] = 2
    loto["result_type"] = "information_access_loto"
    specificity["evidence_level"] = 2
    specificity["result_type"] = "information_access_matched_specificity"
    (root / "representations").mkdir(parents=True, exist_ok=True)
    layer_metrics.to_csv(root / "representations/diagnostic_decoding.csv", index=False)
    loto.to_csv(root / "representations/diagnostic_loto.csv", index=False)
    specificity.to_csv(
        root / "representations/diagnostic_matched_specificity.csv", index=False
    )
    audit = {
        "all_layers_present": sorted(map(int, layer_metrics.layer.unique()))
        == list(ALL_LAYERS),
        "methods": [
            "linear_paired_direction",
            "ridge_probe",
            "LOTO",
            "matched_specificity",
        ],
        "interpretation": "information-access results only",
        "source_hashes": {
            "layer_metrics": _sha256(source / "representation/layer_metrics.csv"),
            "loto": _sha256(source / "representation/loto_metrics.csv"),
            "specificity": _sha256(source / "representation/specificity.csv"),
        },
    }
    if not audit["all_layers_present"]:
        raise ValueError("diagnostic information scan does not cover every model layer")
    _json(root / "representations/diagnostic_scan_audit.json", audit)
    return {"rows": len(layer_metrics), "layers": len(layer_metrics.layer.unique())}


def localize_whole_state_layer(
    config: dict,
    *,
    layer: int,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Coarsely localize causal information by exact full-state interchange."""

    root = _root(config, output)
    layer = int(layer)
    if layer not in ALL_LAYERS:
        raise ValueError(
            "localization layer must use the frozen zero-based 0..31 convention"
        )
    pairs = read_records(_record_path(root, "counterfactuals/pair_manifest.parquet"))
    pairs = pairs[
        pairs.selected_for_neural.astype(bool)
        & pairs.target_variable.isin(PRIMARY_VARIABLES)
    ].copy()
    if limit is not None:
        pairs = pairs.head(int(limit))
    predictions = read_records(
        _record_path(root, "counterfactuals/predicted_effects.parquet")
    )
    records = load_mechanistic_manifest(
        root / "counterfactuals/mechanistic_conditions.jsonl"
    )
    by_id = {record.condition.condition_id: record for record in records}
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    if runner.layer_count != len(ALL_LAYERS):
        raise ValueError("loaded model layer count differs from frozen protocol")
    state_cache, logit_cache, patch_cache = {}, {}, {}
    rows = []
    for pair in pairs.itertuples():
        source_record = by_id[pair.source_condition_id]
        base_record = by_id[pair.base_condition_id]
        for condition_id, record in (
            (pair.source_condition_id, source_record),
            (pair.base_condition_id, base_record),
        ):
            if condition_id not in state_cache:
                result = _forward(runner, record, capture_layers=(layer,))
                state_cache[condition_id] = result.states[layer]
                logit_cache[condition_id] = result.persistence_logit
        patch_key = (pair.base_condition_id, pair.source_condition_id)
        if patch_key not in patch_cache:
            source_state = state_cache[pair.source_condition_id]
            editor = lambda state, source=source_state: replace_whole_state(
                state, source
            )
            patched = _forward(
                runner, base_record, layer=layer, editor=editor
            ).persistence_logit
            patch_cache[patch_key] = patched
        base_logit = float(logit_cache[pair.base_condition_id])
        source_logit = float(logit_cache[pair.source_condition_id])
        patched_logit = float(patch_cache[patch_key])
        theory_rows = predictions[predictions.pair_id == pair.pair_id]
        for prediction in theory_rows.itertuples():
            predicted = float(prediction.predicted_counterfactual_effect)
            observed = patched_logit - base_logit
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "target_variable": pair.target_variable,
                    "counterfactual_subtype": pair.counterfactual_subtype,
                    "task_family": pair.task_family,
                    "response_mapping": pair.response_mapping,
                    "pair_split": pair.pair_split,
                    "theory": prediction.theory,
                    "layer": layer,
                    "base_persistence_logit": base_logit,
                    "source_persistence_logit": source_logit,
                    "patched_persistence_logit": patched_logit,
                    "predicted_counterfactual_effect": predicted,
                    "neural_counterfactual_effect": observed,
                    "counterfactual_target_logit": base_logit + predicted,
                    "counterfactual_recovery": float(
                        counterfactual_recovery([observed], [predicted])[0]
                    ),
                    "whole_state_replacement_exact": True,
                }
            )
    path = write_records(
        rows,
        root / "localization/layer_shards" / f"layer_{layer:03d}.parquet",
    )
    return {"layer": layer, "rows": len(rows), "path": path}


def _localization_frames(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "localization/layer_shards").glob("*.parquet")):
        frames.append(pd.read_parquet(path))
    for path in sorted((root / "localization/layer_shards").glob("*.csv.gz")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("whole-state localization shards are absent")
    frame = pd.concat(frames, ignore_index=True)
    observed = sorted(map(int, frame.layer.unique()))
    if observed != list(ALL_LAYERS):
        missing = sorted(set(ALL_LAYERS) - set(observed))
        raise RuntimeError(f"whole-state localization is incomplete: {missing}")
    return frame


def summarize_whole_state_localization(
    config: dict, *, output: str | Path | None = None
):
    """Summarize Level-3 evidence and create an exact DAS job plan."""

    root = _root(config, output)
    frame = _localization_frames(root)
    path = write_records(
        frame.to_dict("records"), root / "localization/whole_state_patching.parquet"
    )
    rows = []
    for keys, part in frame.groupby(
        ["layer", "target_variable", "theory", "pair_split"]
    ):
        rows.append(
            {
                "layer": int(keys[0]),
                "target_variable": keys[1],
                "theory": keys[2],
                "pair_split": keys[3],
                **counterfactual_metrics(
                    part.predicted_counterfactual_effect,
                    part.neural_counterfactual_effect,
                ),
                "evidence_level": 3,
                "claim": "causally_relevant_information_localization",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "localization/layer_summary.csv", index=False)
    settings = config.get("localization", {})
    count = int(settings.get("candidate_layers_per_target", 3))
    minimum_r = float(settings.get("minimum_validation_correlation", 0.15))
    minimum_cfr = float(settings.get("minimum_validation_cfr", -0.25))
    theory_targets = config.get("theory_targets", {})
    validation = summary[summary.pair_split == "mech_pair_validation"]
    candidates = []
    for target, theories in theory_targets.items():
        for theory in theories:
            part = validation[
                (validation.target_variable == target) & (validation.theory == theory)
            ].copy()
            part["localization_gate"] = (
                part.correlation.fillna(-np.inf) >= minimum_r
            ) & (part.mean_cfr >= minimum_cfr)
            passed = part[part.localization_gate]
            for row in (
                passed.sort_values(["mean_cfr", "correlation"], ascending=False)
                .head(count)
                .itertuples()
            ):
                candidates.append(
                    {
                        "target_variable": target,
                        "theory": theory,
                        "layer": int(row.layer),
                        "validation_correlation": float(row.correlation),
                        "validation_mean_cfr": float(row.mean_cfr),
                    }
                )
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame.to_csv(root / "localization/candidate_layers.csv", index=False)
    ranks = list(map(int, config.get("das", {}).get("ranks", [1, 2, 4, 8])))
    task_ranks = list(
        map(int, config.get("das", {}).get("task_specific_ranks", [1, 4]))
    )
    pair_manifest = read_records(
        _record_path(root, "counterfactuals/pair_manifest.parquet")
    )
    jobs = []
    for candidate in candidates:
        jobs.append(
            {**candidate, "scope": "shared", "task_family": None, "ranks": ranks}
        )
    if bool(config.get("das", {}).get("task_specific_enabled", True)):
        first_layers = {}
        for candidate in candidates:
            key = (candidate["target_variable"], candidate["theory"])
            first_layers.setdefault(key, candidate)
        minimum_rows = int(config.get("das", {}).get("minimum_task_pairs", 4))
        for (target, theory), candidate in first_layers.items():
            target_pairs = pair_manifest[
                pair_manifest.selected_for_neural.astype(bool)
                & (pair_manifest.target_variable == target)
                & pair_manifest.pair_split.isin(
                    ("mech_pair_train", "mech_pair_validation")
                )
            ]
            for task, part in target_pairs.groupby("task_family"):
                counts = part.groupby("pair_split").size()
                if all(
                    int(counts.get(split, 0)) >= minimum_rows
                    for split in ("mech_pair_train", "mech_pair_validation")
                ):
                    jobs.append(
                        {
                            **candidate,
                            "scope": "task_specific",
                            "task_family": task,
                            "ranks": task_ranks,
                        }
                    )
    for index, job in enumerate(jobs):
        job["job_index"] = index
    _json(root / "representations/search_jobs.json", jobs)
    gates = {
        "entry_criteria": json.loads(
            (root / "behavioral_handoff/entry_criteria.json").read_text()
        ),
        "coarse_localization": {
            "passed": bool(jobs),
            "candidate_count": len(candidates),
            "minimum_validation_correlation": minimum_r,
            "minimum_validation_cfr": minimum_cfr,
        },
        "causal_representation": {"passed": None, "status": "awaiting_DAS"},
        "causal_specificity": {"passed": None, "status": "awaiting_validation"},
        "necessity": {"passed": None, "status": "awaiting_validation"},
        "circuit": {"passed": None, "status": "not_eligible"},
    }
    _json(root / "gates.json", gates)
    return {
        "whole_state_patching": path,
        "summary_rows": len(summary),
        "candidate_layers": len(candidates),
        "search_jobs": len(jobs),
    }


def _alignment_stem(job: dict, rank: int) -> str:
    task = job.get("task_family") or "all_tasks"
    return (
        f"{job['target_variable']}__{job['theory']}__L{int(job['layer']):02d}"
        f"__rank{int(rank)}__{job['scope']}__{task}"
    )


def _job_pairs(root: Path, job: dict, splits) -> pd.DataFrame:
    pairs = read_records(_record_path(root, "counterfactuals/pair_manifest.parquet"))
    selected = pairs[
        pairs.selected_for_neural.astype(bool)
        & (pairs.target_variable == job["target_variable"])
        & pairs.pair_split.isin(tuple(splits))
    ].copy()
    if job.get("scope") == "task_specific":
        selected = selected[selected.task_family == job["task_family"]]
    return selected.sort_values("pair_id").reset_index(drop=True)


def _prediction_lookup(root: Path, theory: str):
    predictions = read_records(
        _record_path(root, "counterfactuals/predicted_effects.parquet")
    )
    return (
        predictions[predictions.theory == theory]
        .set_index("pair_id")
        .predicted_counterfactual_effect
    )


def _localization_lookup(root: Path, layer: int, theory: str):
    frame = read_records(
        _record_path(root, "localization/whole_state_patching.parquet")
    )
    selected = frame[(frame.layer == layer) & (frame.theory == theory)]
    return selected.drop_duplicates("pair_id").set_index("pair_id")


def _cache_source_states(runner, pairs, records_by_id, layer: int):
    cache = {}
    for condition_id in sorted(pairs.source_condition_id.unique()):
        result = _forward(
            runner, records_by_id[condition_id], capture_layers=(int(layer),)
        )
        cache[condition_id] = result.states[int(layer)]
    return cache


def _cache_condition_states(runner, condition_ids, records_by_id, layer: int):
    cache = {}
    for condition_id in sorted(set(map(str, condition_ids))):
        result = _forward(
            runner, records_by_id[condition_id], capture_layers=(int(layer),)
        )
        cache[condition_id] = result.states[int(layer)]
    return cache


def _evaluate_alignment(
    runner,
    basis,
    pairs,
    records_by_id,
    source_states,
    localization,
    predictions,
    *,
    layer: int,
    split_label: str,
    intervention_type: str = "DAS",
    matched_norms: dict[str, float] | None = None,
):
    rows = []
    for pair in pairs.itertuples():
        source = source_states[pair.source_condition_id]
        captured = {}

        target_norm = (
            float(matched_norms[pair.pair_id]) if matched_norms is not None else None
        )

        def editor(
            state,
            source_state=source,
            subspace=basis,
            required_norm=target_norm,
        ):
            edited = interchange_subspace(state, source_state, subspace)
            if required_norm is not None:
                difference = edited - state
                current_norm = difference.float().norm()
                if float(current_norm.detach().cpu()) > 1e-12:
                    edited = state + difference * (required_norm / current_norm)
            captured["norm"] = intervention_norm(state, edited)
            return edited

        observed_logit = _forward(
            runner,
            records_by_id[pair.base_condition_id],
            layer=layer,
            editor=editor,
        ).persistence_logit
        baseline = float(localization.loc[pair.pair_id, "base_persistence_logit"])
        predicted = float(predictions.loc[pair.pair_id])
        observed = float(observed_logit - baseline)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "target_variable": pair.target_variable,
                "counterfactual_subtype": pair.counterfactual_subtype,
                "task_family": pair.task_family,
                "response_mapping": pair.response_mapping,
                "pair_split": split_label,
                "predicted_counterfactual_effect": predicted,
                "neural_counterfactual_effect": observed,
                "base_persistence_logit": baseline,
                "intervened_persistence_logit": float(observed_logit),
                "counterfactual_target_logit": baseline + predicted,
                "counterfactual_recovery": float(
                    counterfactual_recovery([observed], [predicted])[0]
                ),
                "intervention_norm": float(captured.get("norm", np.nan)),
                "intervention_type": intervention_type,
                "layer": int(layer),
                "rank": int(np.asarray(basis).shape[1]),
            }
        )
    return rows


def train_das_search_job(
    config: dict,
    *,
    job_index: int,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Train every preregistered rank for one layer/target/theory/scope job."""

    import torch

    root = _root(config, output)
    jobs = json.loads((root / "representations/search_jobs.json").read_text())
    job_index = int(job_index)
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"DAS job index {job_index} is outside 0..{len(jobs) - 1}")
    job = jobs[job_index]
    train_pairs = _job_pairs(root, job, ("mech_pair_train",))
    validation_pairs = _job_pairs(root, job, ("mech_pair_validation",))
    if limit is not None:
        train_pairs = train_pairs.head(int(limit))
        validation_pairs = validation_pairs.head(max(2, int(limit) // 2))
    audit_learning_splits(train_pairs.pair_split)
    audit_learning_splits(validation_pairs.pair_split)
    if train_pairs.empty or validation_pairs.empty:
        raise ValueError("DAS search requires both train and validation pairs")
    records = load_mechanistic_manifest(
        root / "counterfactuals/mechanistic_conditions.jsonl"
    )
    records_by_id = {record.condition.condition_id: record for record in records}
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)
    layer = int(job["layer"])
    source_states = _cache_source_states(
        runner,
        pd.concat([train_pairs, validation_pairs], ignore_index=True),
        records_by_id,
        layer,
    )
    first_state = next(iter(source_states.values()))
    hidden_size = int(len(first_state))
    prediction = _prediction_lookup(root, job["theory"])
    localization = _localization_lookup(root, layer, job["theory"])
    settings = config.get("das", {})
    epochs = int(settings.get("epochs", 3))
    learning_rate = float(settings.get("learning_rate", 0.03))
    lambda_delta = float(settings.get("activation_change_penalty", 1e-4))
    lambda_rank = float(settings.get("rank_penalty", 1e-5))
    seed = int(config.get("seed", 73001)) + 1009 * job_index
    result_rows = []
    for rank in map(int, job["ranks"]):
        alignment = DASAlignment(
            hidden_size,
            rank,
            seed=seed + rank,
            device=next(runner.model.parameters()).device,
        )
        optimizer = torch.optim.Adam(alignment.parameters(), lr=learning_rate)
        epoch_losses = []
        for epoch in range(epochs):
            order = train_pairs.sample(frac=1.0, random_state=seed + epoch)
            losses = []
            for pair in order.itertuples():
                optimizer.zero_grad(set_to_none=True)
                source = source_states[pair.source_condition_id]
                penalty = {}

                def editor(state, source_state=source):
                    edited = alignment.edit(state, source_state)
                    penalty["activation"] = torch.mean(torch.square(edited - state))
                    return edited

                trial = _trial(records_by_id[pair.base_condition_id])
                observed = runner.differentiable_persistence_logit(
                    list(trial.messages),
                    records_by_id[
                        pair.base_condition_id
                    ].condition.response_mapping.labels,
                    positive_label=records_by_id[
                        pair.base_condition_id
                    ].condition.response_mapping.continue_label,
                    editors={layer: editor},
                )
                baseline = float(
                    localization.loc[pair.pair_id, "base_persistence_logit"]
                )
                target = baseline + float(prediction.loc[pair.pair_id])
                loss = torch.square(observed.float() - target)
                loss = (
                    loss
                    + lambda_delta * penalty["activation"].float()
                    + lambda_rank * rank
                )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            epoch_losses.append(float(np.mean(losses)))
        basis = alignment.numpy_basis()
        validation_states = {
            key: value
            for key, value in source_states.items()
            if key in set(validation_pairs.source_condition_id)
        }
        validation_rows = _evaluate_alignment(
            runner,
            basis,
            validation_pairs,
            records_by_id,
            validation_states,
            localization,
            prediction,
            layer=layer,
            split_label="mech_pair_validation",
        )
        validation_frame = pd.DataFrame(validation_rows)
        metrics = counterfactual_metrics(
            validation_frame.predicted_counterfactual_effect,
            validation_frame.neural_counterfactual_effect,
        )
        stem = _alignment_stem(job, rank)
        artifact = save_alignment(
            root / "representations/alignments" / f"{stem}.safetensors",
            basis,
            metadata={
                **job,
                "rank": rank,
                "fitting_split": "mech_pair_train",
                "hyperparameter_split": "mech_pair_validation",
                "test_pairs_used": False,
                "heldout_tasks_used": False,
                "training_examples": len(train_pairs),
                "validation_examples": len(validation_pairs),
                "epochs": epochs,
            },
        )
        write_records(
            validation_rows,
            root / "representations/validation_shards" / f"{stem}.parquet",
        )
        result_rows.append(
            {
                **{key: value for key, value in job.items() if key != "ranks"},
                "rank": rank,
                "artifact": str(artifact),
                "training_examples": len(train_pairs),
                "validation_examples": len(validation_pairs),
                "final_training_loss": epoch_losses[-1],
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
    result_path = root / "representations/search_results" / f"job_{job_index:04d}.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(result_path, index=False)
    return {"job_index": job_index, "results": len(result_rows), "path": result_path}


def select_das_alignments(config: dict, *, output: str | Path | None = None):
    """Choose the smallest validation-passing rank without looking at test pairs."""

    root = _root(config, output)
    files = sorted((root / "representations/search_results").glob("job_*.csv"))
    jobs = json.loads((root / "representations/search_jobs.json").read_text())
    if len(files) != len(jobs):
        raise RuntimeError(
            f"DAS search is incomplete: expected {len(jobs)} results, found {len(files)}"
        )
    results = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    results.to_csv(root / "representations/das_search_summary.csv", index=False)
    settings = config.get("das", {})
    tolerance = float(settings.get("smallest_rank_mse_tolerance", 0.05))
    minimum_r = float(settings.get("minimum_validation_correlation", 0.2))
    minimum_cfr = float(settings.get("minimum_validation_cfr", 0.0))
    selected = []
    grouping = ["target_variable", "theory", "scope", "task_family"]
    for keys, part in results.groupby(grouping, dropna=False):
        viable = part[
            (part.validation_correlation.fillna(-np.inf) >= minimum_r)
            & (part.validation_mean_cfr >= minimum_cfr)
        ]
        if viable.empty:
            continue
        best_mse = float(viable.validation_rmse.min())
        comparable = viable[
            viable.validation_rmse <= best_mse * (1.0 + tolerance) + 1e-12
        ]
        chosen = comparable.sort_values(
            ["rank", "validation_rmse"], ascending=[True, True]
        ).iloc[0]
        selected.append(chosen.to_dict())
    selected = (
        json.loads(pd.DataFrame(selected).to_json(orient="records")) if selected else []
    )
    _json(root / "representations/selected_alignments.json", selected)
    if selected:
        from safetensors.numpy import save_file

        for rank in (1, 2, 4, 8):
            rank_rows = [row for row in selected if int(row["rank"]) == rank]
            tensors = {
                _alignment_stem(row, rank): np.ascontiguousarray(
                    load_alignment(row["artifact"]), dtype=np.float32
                )
                for row in rank_rows
            }
            if tensors:
                save_file(
                    tensors,
                    str(root / "representations" / f"das_rank{rank}.safetensors"),
                    metadata={
                        "protocol": "counterfactual_causal_mechanistic_v1",
                        "selection_split": "mech_pair_validation",
                    },
                )
    gates = json.loads((root / "gates.json").read_text())
    gates["causal_representation"] = {
        "passed": bool(selected),
        "status": (
            "validation_selected" if selected else "stopped_after_validation_failure"
        ),
        "selected_count": len(selected),
        "selection_split": "mech_pair_validation",
        "test_pairs_used": False,
    }
    _json(root / "gates.json", gates)
    return {"search_results": len(results), "selected_alignments": len(selected)}


def _fixed_control_subset(frame: pd.DataFrame, per_task_mapping: int) -> pd.DataFrame:
    selected = []
    for _, part in frame.groupby(["task_family", "response_mapping"], sort=True):
        selected.append(part.sort_values("pair_id").head(int(per_task_mapping)))
    return pd.concat(selected, ignore_index=True) if selected else frame.head(0)


def _control_subspaces(states, pairs, baseline_logits, rank: int, *, seed: int):
    matrix = np.vstack([states[value] for value in pairs.base_condition_id])
    centered = matrix - matrix.mean(axis=0)
    empty = np.empty((matrix.shape[1], 0), dtype=float)

    def covariance_direction(target):
        values = np.asarray(target, dtype=float)
        vector = centered.T @ (values - values.mean())
        norm = np.linalg.norm(vector)
        return (vector / norm)[:, None] if norm > 1e-12 else empty

    persistence = covariance_direction(
        [baseline_logits[pair_id] for pair_id in pairs.pair_id]
    )
    mapping = covariance_direction(
        [1.0 if value == "continue_x" else -1.0 for value in pairs.response_mapping]
    )
    tasks = sorted(pairs.task_family.unique())
    task_columns = []
    global_mean = matrix.mean(axis=0)
    for task in tasks:
        vector = matrix[pairs.task_family.to_numpy() == task].mean(axis=0) - global_mean
        if np.linalg.norm(vector) > 1e-12:
            task_columns.append(vector)
    task = empty
    if task_columns:
        task = np.linalg.qr(np.column_stack(task_columns), mode="reduced")[0][
            :, : min(int(rank), len(task_columns))
        ]
    return {
        "persistence_state": persistence,
        "response_mapping": mapping,
        "task_id": task,
        "random_reference": random_directions(
            matrix.shape[1], max(1, int(rank)), seed=seed
        ).T,
    }


def _evaluate_control_basis(
    runner,
    basis,
    pairs,
    records_by_id,
    source_states,
    localization,
    predictions,
    *,
    layer,
    control,
):
    return _evaluate_alignment(
        runner,
        basis,
        pairs,
        records_by_id,
        source_states,
        localization,
        predictions,
        layer=layer,
        split_label=str(pairs.pair_split.iloc[0]),
        intervention_type=control,
    )


def _specificity_control_assessment(target_cfr: float, controls: pd.DataFrame):
    """Require a candidate to beat every preregistered non-random control."""

    observed = set(controls.control.astype(str)) if len(controls) else set()
    required = set(REQUIRED_SPECIFICITY_CONTROLS)
    required.update(
        f"unrelated_variable:{target}" for target in REQUIRED_UNRELATED_TARGETS
    )
    missing = sorted(required - observed)
    comparison = controls[
        controls.control.isin(REQUIRED_SPECIFICITY_CONTROLS)
        | controls.control.astype(str).str.startswith("unrelated_variable:")
    ]
    values = comparison.mean_cfr.to_numpy(dtype=float) if len(comparison) else []
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    maximum = float(finite.max()) if len(finite) else np.nan
    margin = (
        float(target_cfr - maximum)
        if np.isfinite(target_cfr) and np.isfinite(maximum)
        else np.nan
    )
    return {
        "passed": bool(not missing and np.isfinite(margin) and margin > 0),
        "missing_controls": missing,
        "maximum_control_mean_cfr": maximum,
        "control_margin": margin,
    }


def validate_selected_alignments(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Evaluate untouched pairs, causal nulls, specificity, and necessity."""

    root = _root(config, output)
    selected = json.loads(
        (root / "representations/selected_alignments.json").read_text()
    )
    if not selected:
        raise RuntimeError("no validation-passing DAS alignment is available")
    records = load_mechanistic_manifest(
        root / "counterfactuals/mechanistic_conditions.jsonl"
    )
    records_by_id = {record.condition.condition_id: record for record in records}
    all_pairs = read_records(
        _record_path(root, "counterfactuals/pair_manifest.parquet")
    )
    all_predictions = read_records(
        _record_path(root, "counterfactuals/predicted_effects.parquet")
    )
    localization_all = read_records(
        _record_path(root, "localization/whole_state_patching.parquet")
    )
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    settings = config.get("validation", {})
    random_count = int(settings.get("random_subspaces", 100))
    if random_count < 100:
        raise ValueError("causal validation requires at least 100 random subspaces")
    random_pairs_per_group = int(settings.get("random_pairs_per_task_mapping", 1))
    seed = int(config.get("seed", 73001))
    recovery_rows, random_rows, control_rows, necessity_rows = [], [], [], []
    for candidate_index, candidate in enumerate(selected):
        layer, rank = int(candidate["layer"]), int(candidate["rank"])
        theory, target = candidate["theory"], candidate["target_variable"]
        job = {
            "target_variable": target,
            "theory": theory,
            "scope": candidate["scope"],
            "task_family": candidate.get("task_family"),
        }
        evaluation_pairs = _job_pairs(
            root, job, ("mech_pair_test", "mech_task_holdout")
        )
        if candidate["scope"] == "task_specific":
            evaluation_pairs = evaluation_pairs[
                evaluation_pairs.pair_split == "mech_pair_test"
            ]
        if limit is not None:
            evaluation_pairs = evaluation_pairs.head(int(limit))
        if evaluation_pairs.empty:
            continue
        basis = load_alignment(candidate["artifact"])
        prediction = (
            all_predictions[
                (all_predictions.theory == theory)
                & all_predictions.pair_id.isin(evaluation_pairs.pair_id)
            ]
            .set_index("pair_id")
            .predicted_counterfactual_effect
        )
        localization = (
            localization_all[
                (localization_all.layer == layer)
                & (localization_all.theory == theory)
                & localization_all.pair_id.isin(evaluation_pairs.pair_id)
            ]
            .drop_duplicates("pair_id")
            .set_index("pair_id")
        )
        source_states = _cache_source_states(
            runner, evaluation_pairs, records_by_id, layer
        )
        for split, split_pairs in evaluation_pairs.groupby("pair_split"):
            rows = _evaluate_alignment(
                runner,
                basis,
                split_pairs,
                records_by_id,
                source_states,
                localization,
                prediction,
                layer=layer,
                split_label=split,
            )
            for row in rows:
                row.update(
                    {
                        "candidate_index": candidate_index,
                        "theory": theory,
                        "scope": candidate["scope"],
                        "alignment_task": candidate.get("task_family"),
                    }
                )
            recovery_rows.extend(rows)

        # Random subspaces: exact same layer, rank, pair subset, and source/base states.
        null_pairs = _fixed_control_subset(evaluation_pairs, random_pairs_per_group)
        null_prediction = prediction.loc[null_pairs.pair_id]
        null_localization = localization.loc[null_pairs.pair_id]
        candidate_null_rows = _evaluate_alignment(
            runner,
            basis,
            null_pairs,
            records_by_id,
            source_states,
            null_localization,
            null_prediction,
            layer=layer,
            split_label="matched_norm_reference",
            intervention_type="DAS_norm_reference",
        )
        candidate_null_metrics = counterfactual_metrics(
            [row["predicted_counterfactual_effect"] for row in candidate_null_rows],
            [row["neural_counterfactual_effect"] for row in candidate_null_rows],
        )
        control_rows.append(
            {
                "candidate_index": candidate_index,
                "target_variable": target,
                "theory": theory,
                "layer": layer,
                "rank": rank,
                "control": "candidate_matched_subset",
                **candidate_null_metrics,
            }
        )
        matched_norms = {
            row["pair_id"]: row["intervention_norm"] for row in candidate_null_rows
        }
        dimension = basis.shape[0]
        random_bases = random_directions(
            dimension * rank,
            random_count,
            seed=seed + 10007 * candidate_index,
        ).reshape(random_count, dimension, rank)
        # Re-orthogonalize because reshaping normalized vectors does not make
        # their columns mutually orthogonal.
        random_bases = np.stack(
            [np.linalg.qr(value, mode="reduced")[0] for value in random_bases]
        )
        for random_index, random_basis in enumerate(random_bases):
            rows = _evaluate_alignment(
                runner,
                random_basis,
                null_pairs,
                records_by_id,
                source_states,
                null_localization,
                null_prediction,
                layer=layer,
                split_label="random_null",
                intervention_type="random_subspace",
                matched_norms=matched_norms,
            )
            metrics = counterfactual_metrics(
                [row["predicted_counterfactual_effect"] for row in rows],
                [row["neural_counterfactual_effect"] for row in rows],
            )
            random_rows.append(
                {
                    "candidate_index": candidate_index,
                    "target_variable": target,
                    "theory": theory,
                    "layer": layer,
                    "rank": rank,
                    "random_index": random_index,
                    "maximum_intervention_norm_error": float(
                        max(
                            abs(
                                row["intervention_norm"] - matched_norms[row["pair_id"]]
                            )
                            for row in rows
                        )
                    ),
                    **metrics,
                }
            )

        # Shuffled-source and shuffled-target controls use the fixed null subset.
        shuffled_parts = []
        for _, part in null_pairs.groupby(
            ["task_family", "response_mapping"], sort=True
        ):
            changed = part.copy()
            changed["source_condition_id"] = np.roll(
                changed.source_condition_id.to_numpy(), 1
            )
            shuffled_parts.append(changed)
        shuffled = pd.concat(shuffled_parts, ignore_index=True)
        shuffled_states = _cache_source_states(runner, shuffled, records_by_id, layer)
        shuffled_rows = _evaluate_alignment(
            runner,
            basis,
            shuffled,
            records_by_id,
            shuffled_states,
            null_localization,
            null_prediction,
            layer=layer,
            split_label="shuffled_source",
            intervention_type="shuffled_source",
            matched_norms=matched_norms,
        )
        shuffled_metrics = counterfactual_metrics(
            [row["predicted_counterfactual_effect"] for row in shuffled_rows],
            [row["neural_counterfactual_effect"] for row in shuffled_rows],
        )
        control_rows.append(
            {
                "candidate_index": candidate_index,
                "target_variable": target,
                "theory": theory,
                "layer": layer,
                "rank": rank,
                "control": "shuffled_source",
                **shuffled_metrics,
            }
        )
        shuffled_target = np.roll(null_prediction.to_numpy(dtype=float), 1)
        shuffled_target_metrics = counterfactual_metrics(
            shuffled_target,
            [row["neural_counterfactual_effect"] for row in candidate_null_rows],
        )
        control_rows.append(
            {
                "candidate_index": candidate_index,
                "target_variable": target,
                "theory": theory,
                "layer": layer,
                "rank": rank,
                "control": "shuffled_target",
                **shuffled_target_metrics,
            }
        )

        # Train-only control geometry for persistence, task ID, and mapping.
        # Control geometry always uses all training tasks. In particular, a
        # task-specific candidate still needs a task-ID control learned from
        # cross-task variation rather than from its single-task subset.
        train_job = {**job, "scope": "shared", "task_family": None}
        train_pairs = _job_pairs(root, train_job, ("mech_pair_train",))
        train_source_states = _cache_condition_states(
            runner,
            train_pairs.base_condition_id,
            records_by_id,
            layer,
        )
        base_states = {
            condition_id: train_source_states[condition_id]
            for condition_id in train_pairs.base_condition_id.unique()
        }
        train_localization = (
            localization_all[
                (localization_all.layer == layer)
                & (localization_all.theory == theory)
                & localization_all.pair_id.isin(train_pairs.pair_id)
            ]
            .drop_duplicates("pair_id")
            .set_index("pair_id")
        )
        baseline_logits = train_localization.base_persistence_logit.to_dict()
        subspaces = _control_subspaces(
            base_states,
            train_pairs,
            baseline_logits,
            rank,
            seed=seed + candidate_index,
        )
        source_mechanistic_root = Path(
            json.loads((root / "run_metadata.json").read_text())[
                "mechanistic_source_root"
            ]
        )
        generic_path = (
            source_mechanistic_root / "directions/generic_value_control.safetensors"
        )
        if generic_path.exists():
            generic = load_directions(generic_path)["generic_value"][layer]
            subspaces["generic_value"] = generic[:, None]
        else:
            generic_pairs = all_pairs[
                all_pairs.selected_for_neural.astype(bool)
                & all_pairs.target_variable.eq("generic_value")
                & all_pairs.pair_split.eq("mech_pair_train")
            ].sort_values("pair_id")
            if generic_pairs.empty:
                subspaces["generic_value"] = np.empty((basis.shape[0], 0), dtype=float)
            else:
                generic_ids = list(generic_pairs.base_condition_id) + list(
                    generic_pairs.source_condition_id
                )
                generic_states = _cache_condition_states(
                    runner, generic_ids, records_by_id, layer
                )
                differences = np.vstack(
                    [
                        generic_states[pair.source_condition_id]
                        - generic_states[pair.base_condition_id]
                        for pair in generic_pairs.itertuples()
                    ]
                )
                generic = differences.mean(axis=0)
                generic_norm = float(np.linalg.norm(generic))
                subspaces["generic_value"] = (
                    (generic / generic_norm)[:, None]
                    if generic_norm > 1e-12
                    else np.empty((basis.shape[0], 0), dtype=float)
                )
        first_record = records_by_id[evaluation_pairs.iloc[0].base_condition_id]
        first_trial = _trial(first_record)
        output_direction = runner.choice_output_direction(
            list(first_trial.messages),
            first_record.condition.response_mapping.labels,
            positive_label=first_record.condition.response_mapping.continue_label,
        )
        subspaces["persistence_output"] = output_direction[:, None]
        for control, control_basis in subspaces.items():
            rows = _evaluate_control_basis(
                runner,
                control_basis,
                null_pairs,
                records_by_id,
                source_states,
                null_localization,
                null_prediction,
                layer=layer,
                control=control,
            )
            if not rows:
                continue
            metrics = counterfactual_metrics(
                [row["predicted_counterfactual_effect"] for row in rows],
                [row["neural_counterfactual_effect"] for row in rows],
            )
            control_rows.append(
                {
                    "candidate_index": candidate_index,
                    "target_variable": target,
                    "theory": theory,
                    "layer": layer,
                    "rank": int(np.asarray(control_basis).shape[1]),
                    "control": control,
                    **metrics,
                }
            )

        # Counterfactual specificity: candidate X subspace on unrelated Z pairs.
        unrelated = all_pairs[
            all_pairs.selected_for_neural.astype(bool)
            & all_pairs.scientific_role.eq("specificity_control")
            & all_pairs.pair_split.isin(("mech_pair_test", "mech_task_holdout"))
        ]
        unrelated = _fixed_control_subset(unrelated, 1)
        for unrelated_target, control_pairs in unrelated.groupby("target_variable"):
            control_prediction = (
                all_predictions[
                    (all_predictions.theory == theory)
                    & all_predictions.pair_id.isin(control_pairs.pair_id)
                ]
                .set_index("pair_id")
                .predicted_counterfactual_effect
            )
            control_states = _cache_source_states(
                runner, control_pairs, records_by_id, layer
            )
            control_localization_rows = []
            for pair in control_pairs.itertuples():
                baseline_result = _forward(
                    runner,
                    records_by_id[pair.base_condition_id],
                    capture_layers=(layer,),
                )
                control_localization_rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "base_persistence_logit": baseline_result.persistence_logit,
                    }
                )
            control_localization = pd.DataFrame(control_localization_rows).set_index(
                "pair_id"
            )
            rows = _evaluate_alignment(
                runner,
                basis,
                control_pairs,
                records_by_id,
                control_states,
                control_localization,
                control_prediction,
                layer=layer,
                split_label="unrelated_counterfactual",
                intervention_type=f"unrelated_variable:{unrelated_target}",
            )
            metrics = counterfactual_metrics(
                [row["predicted_counterfactual_effect"] for row in rows],
                [row["neural_counterfactual_effect"] for row in rows],
            )
            control_rows.append(
                {
                    "candidate_index": candidate_index,
                    "target_variable": target,
                    "theory": theory,
                    "layer": layer,
                    "rank": rank,
                    "control": f"unrelated_variable:{unrelated_target}",
                    **metrics,
                }
            )

        # Necessity: remove candidate coordinates from both endpoints and test
        # whether source/base sensitivity to X is selectively attenuated.
        train_matrix = np.vstack(
            [base_states[value] for value in train_pairs.base_condition_id]
        )
        q = np.linalg.qr(basis, mode="reduced")[0]
        neutral = np.mean(train_matrix @ q, axis=0)
        for pair in null_pairs.itertuples():
            logits = {}
            for role, condition_id in (
                ("base", pair.base_condition_id),
                ("source", pair.source_condition_id),
            ):
                editor = lambda state, b=basis, reference=neutral: remove_subspace(
                    state, b, reference
                )
                logits[role] = _forward(
                    runner,
                    records_by_id[condition_id],
                    layer=layer,
                    editor=editor,
                ).persistence_logit
            original = float(
                null_localization.loc[pair.pair_id, "source_persistence_logit"]
                - null_localization.loc[pair.pair_id, "base_persistence_logit"]
            )
            neutralized = float(logits["source"] - logits["base"])
            necessity_rows.append(
                {
                    "candidate_index": candidate_index,
                    "pair_id": pair.pair_id,
                    "target_variable": target,
                    "theory": theory,
                    "layer": layer,
                    "rank": rank,
                    "original_source_base_sensitivity": original,
                    "neutralized_source_base_sensitivity": neutralized,
                    "absolute_sensitivity_reduction": abs(original) - abs(neutralized),
                }
            )

    recovery_path = write_records(
        recovery_rows, root / "validation/counterfactual_recovery.parquet"
    )
    pd.DataFrame(random_rows).to_csv(
        root / "validation/random_subspace_null.csv", index=False
    )
    pd.DataFrame(control_rows).to_csv(
        root / "validation/specificity_controls.csv", index=False
    )
    pd.DataFrame(necessity_rows).to_csv(root / "validation/necessity.csv", index=False)
    return {
        "counterfactual_rows": len(recovery_rows),
        "random_null_rows": len(random_rows),
        "specificity_rows": len(control_rows),
        "necessity_rows": len(necessity_rows),
        "path": recovery_path,
    }


def _validation_summaries(root: Path):
    recovery = read_records(
        _record_path(root, "validation/counterfactual_recovery.parquet")
    )
    rows = []
    for keys, part in recovery.groupby(
        [
            "candidate_index",
            "target_variable",
            "theory",
            "scope",
            "alignment_task",
            "layer",
            "rank",
            "pair_split",
        ],
        dropna=False,
    ):
        rows.append(
            {
                **dict(
                    zip(
                        (
                            "candidate_index",
                            "target_variable",
                            "theory",
                            "scope",
                            "alignment_task",
                            "layer",
                            "rank",
                            "pair_split",
                        ),
                        keys,
                    )
                ),
                **counterfactual_metrics(
                    part.predicted_counterfactual_effect,
                    part.neural_counterfactual_effect,
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary[summary.pair_split == "mech_task_holdout"].to_csv(
        root / "validation/heldout_task_results.csv", index=False
    )
    summary.to_csv(root / "validation/counterfactual_summary.csv", index=False)
    return recovery, summary


def _component_forward_record(runner, record, **kwargs):
    trial = _trial(record)
    return runner.component_forward(
        list(trial.messages),
        record.condition.response_mapping.labels,
        positive_label=record.condition.response_mapping.continue_label,
        **kwargs,
    )


def run_circuit_localization(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Trace components constructing and transmitting validated Level-5 subspaces."""

    root = _root(config, output)
    gates = json.loads((root / "gates.json").read_text())
    if gates.get("circuit", {}).get("status") != "eligible":
        raise RuntimeError(
            "circuit localization is allowed only after Level-5 specificity"
        )
    validated = pd.read_csv(root / "validation/validated_alignments.csv")
    validated = validated[validated.passes_specificity.astype(bool)]
    if validated.empty:
        raise RuntimeError("no specificity-passing alignment is available")
    selected = json.loads(
        (root / "representations/selected_alignments.json").read_text()
    )
    selected_by_index = {index: row for index, row in enumerate(selected)}
    chosen = (
        validated.sort_values("test_mean_cfr", ascending=False)
        .groupby("target_variable")
        .head(1)
    )
    pairs = read_records(_record_path(root, "counterfactuals/pair_manifest.parquet"))
    records = load_mechanistic_manifest(
        root / "counterfactuals/mechanistic_conditions.jsonl"
    )
    records_by_id = {record.condition.condition_id: record for record in records}
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    per_group = int(config.get("circuits", {}).get("pairs_per_task_mapping", 1))
    component_rows, path_rows = [], []
    for candidate in chosen.itertuples():
        specification = selected_by_index[int(candidate.candidate_index)]
        basis = load_alignment(specification["artifact"])
        representation_layer = int(candidate.layer)
        candidate_pairs = pairs[
            pairs.selected_for_neural.astype(bool)
            & (pairs.target_variable == candidate.target_variable)
            & (pairs.pair_split == "mech_pair_test")
        ]
        candidate_pairs = _fixed_control_subset(candidate_pairs, per_group)
        if limit is not None:
            candidate_pairs = candidate_pairs.head(int(limit))
        for pair in candidate_pairs.itertuples():
            source_record = records_by_id[pair.source_condition_id]
            base_record = records_by_id[pair.base_condition_id]
            base_representation = _forward(
                runner, base_record, capture_layers=(representation_layer,)
            )
            baseline_logit = base_representation.persistence_logit
            base_coordinates = (
                base_representation.states[representation_layer]
                @ np.linalg.qr(basis, mode="reduced")[0]
            )
            # Upstream construction: patch one component and measure movement of S_X.
            for component_layer in range(representation_layer + 1):
                for component_kind in ("attention", "mlp"):
                    key = (component_layer, component_kind)
                    source_component = _component_forward_record(
                        runner,
                        source_record,
                        capture_components=(key,),
                    ).component_states[key]
                    editor = lambda state, source=source_component: replace_whole_state(
                        state, source
                    )
                    patched = _component_forward_record(
                        runner,
                        base_record,
                        component_editors={key: editor},
                        capture_layers=(representation_layer,),
                    )
                    patched_coordinates = (
                        patched.states[representation_layer]
                        @ np.linalg.qr(basis, mode="reduced")[0]
                    )
                    component_rows.append(
                        {
                            "candidate_index": int(candidate.candidate_index),
                            "pair_id": pair.pair_id,
                            "target_variable": candidate.target_variable,
                            "representation_layer": representation_layer,
                            "component_layer": component_layer,
                            "component_kind": component_kind,
                            "subspace_coordinate_change": float(
                                np.linalg.norm(patched_coordinates - base_coordinates)
                            ),
                            "persistence_logit_change": float(
                                patched.persistence_logit - baseline_logit
                            ),
                            "analysis_role": "upstream_construction",
                        }
                    )

            # Downstream transmission: restore one later component to its base
            # output while performing the validated representation interchange.
            source_state = _forward(
                runner, source_record, capture_layers=(representation_layer,)
            ).states[representation_layer]
            layer_editor = (
                lambda state, source=source_state, b=basis: interchange_subspace(
                    state, source, b
                )
            )
            das_only = _forward(
                runner,
                base_record,
                layer=representation_layer,
                editor=layer_editor,
            ).persistence_logit
            for component_layer in range(representation_layer + 1, len(ALL_LAYERS)):
                for component_kind in ("attention", "mlp"):
                    key = (component_layer, component_kind)
                    base_component = _component_forward_record(
                        runner,
                        base_record,
                        capture_components=(key,),
                    ).component_states[key]
                    restore = (
                        lambda state, base_value=base_component: replace_whole_state(
                            state, base_value
                        )
                    )
                    restored = _component_forward_record(
                        runner,
                        base_record,
                        component_editors={key: restore},
                        layer_editors={representation_layer: layer_editor},
                    ).persistence_logit
                    das_effect = float(das_only - baseline_logit)
                    restored_effect = float(restored - baseline_logit)
                    path_rows.append(
                        {
                            "candidate_index": int(candidate.candidate_index),
                            "pair_id": pair.pair_id,
                            "target_variable": candidate.target_variable,
                            "representation_layer": representation_layer,
                            "component_layer": component_layer,
                            "component_kind": component_kind,
                            "das_effect": das_effect,
                            "effect_after_base_component_restore": restored_effect,
                            "path_mediated_effect": das_effect - restored_effect,
                            "mediated_fraction": (
                                (das_effect - restored_effect) / das_effect
                                if abs(das_effect) > 1e-8
                                else np.nan
                            ),
                            "analysis_role": "downstream_transmission",
                        }
                    )
    component_path = write_records(
        component_rows, root / "circuits/component_patching.parquet"
    )
    path_path = write_records(path_rows, root / "circuits/path_patching.parquet")
    return {
        "component_rows": len(component_rows),
        "path_rows": len(path_rows),
        "component_patching": component_path,
        "path_patching": path_path,
    }


def _causal_figures(root, localization, validation, random_null, controls):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(9, 4.5))
    shown = localization[
        (localization.pair_split == "mech_pair_validation")
        & localization.target_variable.isin(PRIMARY_VARIABLES)
    ]
    for (target, theory), part in shown.groupby(["target_variable", "theory"]):
        ax.plot(part.layer, part.mean_cfr, label=f"{target}/{theory}")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Layer (zero-based)")
    ax.set_ylabel("Whole-state mean CFR")
    ax.set_title("Level 3: coarse causal localization")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    paths.append(figure_root / "figure1_whole_state_localization.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    test = validation[validation.pair_split == "mech_pair_test"]
    ax.scatter(
        test.predicted_counterfactual_effect,
        test.neural_counterfactual_effect,
        c=test.layer,
        cmap="viridis",
        alpha=0.65,
    )
    if len(test):
        extent = max(
            abs(test.predicted_counterfactual_effect).max(),
            abs(test.neural_counterfactual_effect).max(),
            1e-3,
        )
        ax.plot([-extent, extent], [-extent, extent], "--", color="grey")
    ax.set_xlabel("Frozen behavioral counterfactual ΔD")
    ax.set_ylabel("Neural interchange ΔD")
    ax.set_title("Held-out counterfactual equivalence")
    fig.tight_layout()
    paths.append(figure_root / "figure2_counterfactual_equivalence.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(random_null):
        for layer, part in random_null.groupby("layer"):
            ax.hist(
                part.mean_cfr,
                bins=20,
                alpha=0.4,
                label=f"random L{int(layer)}",
            )
    ax.set_xlabel("Mean CFR")
    ax.set_ylabel("Random subspaces")
    ax.set_title("Matched-rank random-subspace null")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths.append(figure_root / "figure3_random_subspace_null.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(controls):
        shown_controls = controls[controls.control != "candidate_matched_subset"]
        control_mean = shown_controls.groupby("control").mean_cfr.mean().sort_values()
        ax.bar(range(len(control_mean)), control_mean.values)
        ax.set_xticks(
            range(len(control_mean)), control_mean.index, rotation=45, ha="right"
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Mean CFR")
    ax.set_title("Causal specificity controls")
    fig.tight_layout()
    paths.append(figure_root / "figure4_specificity_controls.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    task = (
        test.groupby(["target_variable", "task_family"])
        .counterfactual_recovery.mean()
        .unstack(0)
    )
    if len(task):
        task.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Mean CFR")
    ax.set_title("Task-wise held-out counterfactual recovery")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths.append(figure_root / "figure5_taskwise_recovery.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)
    return paths


def finalize_causal_mechanistic_run(config: dict, *, output: str | Path | None = None):
    """Apply evidence levels, stop rules, neural hierarchy, and null reporting."""

    root = _root(config, output)
    gates = json.loads((root / "gates.json").read_text())
    localization = pd.read_csv(root / "localization/layer_summary.csv")
    selected_path = root / "representations/selected_alignments.json"
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else []
    recovery_path = _record_path(root, "validation/counterfactual_recovery.parquet")
    if not selected or not recovery_path.exists():
        level = 3 if gates["coarse_localization"]["passed"] else 2
        conclusion = (
            "The variables are decodable and coarsely localized, but no small causally "
            "aligned neural representation passed validation."
            if level == 3
            else "The variables are decodable but were not causally localized by whole-state patching."
        )
        gates["causal_representation"] = {
            "passed": False,
            "status": "stop_rule_applied",
        }
        gates["causal_specificity"] = {
            "passed": False,
            "status": "not_tested_after_stop",
        }
        gates["necessity"] = {
            "passed": False,
            "status": "not_tested_after_stop",
        }
        gates["circuit"] = {"passed": False, "status": "not_eligible"}
        _json(root / "gates.json", gates)
        report = root / "report.md"
        report.write_text(
            "# Counterfactual Causal Mechanistic Discovery\n\n"
            f"**Highest evidence level:** {level}\n\n"
            f"**Conclusion:** {conclusion}\n\n"
            "The stop rule prevents escalation to more expressive representations or circuits.\n",
            encoding="utf-8",
        )
        return {"report": report, "evidence_level": level, "conclusion": conclusion}

    recovery, summary = _validation_summaries(root)
    random_null = pd.read_csv(root / "validation/random_subspace_null.csv")
    controls = pd.read_csv(root / "validation/specificity_controls.csv")
    necessity = pd.read_csv(root / "validation/necessity.csv")
    thresholds = config.get("gates", {})
    minimum_r = float(thresholds.get("minimum_test_correlation", 0.3))
    minimum_cfr = float(thresholds.get("minimum_test_cfr", 0.0))
    candidate_rows = []
    test_summary = summary[summary.pair_split == "mech_pair_test"]
    for row in test_summary.itertuples():
        null = random_null[random_null.candidate_index == row.candidate_index]
        null95 = float(null.mean_cfr.quantile(0.95)) if len(null) else np.nan
        candidate_controls = controls[controls.candidate_index == row.candidate_index]
        reference = candidate_controls[
            candidate_controls.control == "candidate_matched_subset"
        ]
        control_reference_cfr = (
            float(reference.mean_cfr.iloc[0]) if len(reference) else np.nan
        )
        unrelated = candidate_controls[
            candidate_controls.control.str.startswith("unrelated_variable:")
        ]
        specificity = counterfactual_specificity(
            control_reference_cfr, unrelated.mean_cfr if len(unrelated) else []
        )
        control_assessment = _specificity_control_assessment(
            control_reference_cfr, candidate_controls
        )
        shuffled = candidate_controls[
            candidate_controls.control.isin(("shuffled_source", "shuffled_target"))
        ]
        shuffled_max = float(shuffled.mean_cfr.max()) if len(shuffled) else np.nan
        random_pass = bool(np.isfinite(null95) and control_reference_cfr > null95)
        shuffle_pass = bool(
            np.isfinite(shuffled_max) and control_reference_cfr > shuffled_max
        )
        correspondence_pass = bool(
            np.isfinite(row.correlation)
            and row.correlation >= minimum_r
            and row.mean_cfr >= minimum_cfr
        )
        candidate_rows.append(
            {
                "candidate_index": int(row.candidate_index),
                "target_variable": row.target_variable,
                "theory": row.theory,
                "scope": row.scope,
                "alignment_task": row.alignment_task,
                "layer": int(row.layer),
                "rank": int(row.rank),
                "test_correlation": float(row.correlation),
                "test_slope": float(row.slope),
                "test_intercept": float(row.intercept),
                "test_rmse": float(row.rmse),
                "test_sign_accuracy": float(row.sign_accuracy),
                "test_mean_cfr": float(row.mean_cfr),
                "matched_subset_mean_cfr": control_reference_cfr,
                "random_mean_cfr_95": null95,
                "shuffled_max_mean_cfr": shuffled_max,
                "counterfactual_specificity": specificity,
                "maximum_named_control_mean_cfr": control_assessment[
                    "maximum_control_mean_cfr"
                ],
                "named_control_margin": control_assessment["control_margin"],
                "missing_specificity_controls": json.dumps(
                    control_assessment["missing_controls"]
                ),
                "passes_random_subspaces": random_pass,
                "passes_shuffled_controls": shuffle_pass,
                "passes_named_controls": control_assessment["passed"],
                "passes_level4": correspondence_pass,
                "passes_specificity": bool(
                    correspondence_pass
                    and random_pass
                    and shuffle_pass
                    and control_assessment["passed"]
                    and np.isfinite(specificity)
                    and specificity > 0
                ),
            }
        )
    validated = pd.DataFrame(candidate_rows)
    validated.to_csv(root / "validation/validated_alignments.csv", index=False)
    level4 = bool(len(validated) and validated.passes_level4.any())
    level5 = bool(len(validated) and validated.passes_specificity.any())
    necessity_summary = (
        necessity.groupby("candidate_index", as_index=False)
        .absolute_sensitivity_reduction.mean()
        .rename(
            columns={"absolute_sensitivity_reduction": "mean_sensitivity_reduction"}
        )
    )
    necessity_pass = bool(
        level5
        and len(necessity_summary)
        and (necessity_summary.mean_sensitivity_reduction > 0).any()
    )
    gates["causal_representation"] = {
        "passed": level4,
        "validated_count": int(validated.passes_level4.sum()) if len(validated) else 0,
        "test_split": "mech_pair_test",
    }
    gates["causal_specificity"] = {
        "passed": level5,
        "validated_count": (
            int(validated.passes_specificity.sum()) if len(validated) else 0
        ),
    }
    gates["necessity"] = {
        "passed": necessity_pass,
        "mean_reductions": necessity_summary.to_dict("records"),
    }
    circuit_pass = False
    component_path = _record_path(root, "circuits/component_patching.parquet")
    path_path = _record_path(root, "circuits/path_patching.parquet")
    if level5 and component_path.exists() and path_path.exists():
        components = read_records(component_path)
        paths = read_records(path_path)
        mediated = (
            paths.groupby(["component_layer", "component_kind"])
            .mediated_fraction.median()
            .abs()
        )
        construction = components.groupby(
            ["component_layer", "component_kind"]
        ).subspace_coordinate_change.median()
        threshold_circuit = float(
            config.get("circuits", {}).get("minimum_mediated_fraction", 0.1)
        )
        circuit_pass = bool(
            len(mediated)
            and len(construction)
            and mediated.max() >= threshold_circuit
            and construction.max() > 0
        )
        gates["circuit"] = {
            "passed": circuit_pass,
            "status": "completed",
            "maximum_median_mediated_fraction": (
                float(mediated.max()) if len(mediated) else np.nan
            ),
            "maximum_median_subspace_construction_change": (
                float(construction.max()) if len(construction) else np.nan
            ),
        }
    else:
        gates["circuit"] = {
            "passed": None if level5 else False,
            "status": "eligible" if level5 else "not_eligible",
        }

    shared_holdout = summary[
        (summary.scope == "shared") & (summary.pair_split == "mech_task_holdout")
    ].copy()
    passing_test_indices = set(
        validated.loc[validated.passes_level4.astype(bool), "candidate_index"].astype(
            int
        )
    )
    shared_holdout["passes_transfer"] = (
        shared_holdout.candidate_index.astype(int).isin(passing_test_indices)
        & (shared_holdout.correlation.fillna(-np.inf) >= minimum_r)
        & (shared_holdout.mean_cfr >= minimum_cfr)
    )
    task_validated = validated[
        validated.scope.eq("task_specific") & validated.passes_level4.astype(bool)
    ]
    task_role_counts = (
        task_validated.groupby("target_variable").alignment_task.nunique()
        if len(task_validated)
        else pd.Series(dtype=int)
    )
    if bool(len(shared_holdout) and shared_holdout.passes_transfer.any()):
        neural_hypothesis = "Neural_H1_shared_coordinates"
    elif bool(len(task_role_counts) and (task_role_counts >= 2).any()):
        neural_hypothesis = "Neural_H2_shared_role_task_specific_coordinates"
    else:
        neural_hypothesis = "Neural_H3_task_specific_or_unidentified_computation"
    theory_assessment = {}
    for target, target_rows in validated.groupby("target_variable"):
        passing = sorted(
            target_rows.loc[
                target_rows.passes_specificity.astype(bool), "theory"
            ].unique()
        )
        tested = sorted(target_rows.theory.unique())
        theory_assessment[target] = {
            "tested": tested,
            "specificity_passing": passing,
            "status": (
                "mechanistically_favored"
                if len(passing) == 1
                else (
                    "mechanistically_equivalent_or_unresolved"
                    if len(passing) > 1
                    else "not_resolved_by_mechanistic_tests"
                )
            ),
        }
    highest_level = 6 if circuit_pass else (5 if level5 else (4 if level4 else 3))
    if circuit_pass:
        conclusion = (
            "A counterfactually validated and specific neural subspace was traced to "
            "components that construct and transmit its causal effect."
        )
    elif level5 and necessity_pass:
        conclusion = (
            "A small neural subspace is causally aligned with a frozen computational "
            "counterfactual and passes sufficiency, specificity, and necessity controls."
        )
    elif level4:
        conclusion = (
            "A subspace reproduces held-out computational counterfactuals, but the "
            "strong implementation claim is withheld until specificity and necessity pass."
        )
    else:
        conclusion = (
            "The variables are decodable and coarsely causal, but no learned subspace "
            "reproduced frozen counterfactuals beyond matched nulls."
        )
    gates["highest_evidence_level"] = highest_level
    gates["neural_implementation_hypothesis"] = neural_hypothesis
    gates["behavioral_theory_assessment"] = theory_assessment
    _json(root / "gates.json", gates)
    figures = _causal_figures(root, localization, recovery, random_null, controls)
    report_lines = [
        "# Counterfactual Causal Mechanistic Discovery",
        "",
        f"**Highest evidence level:** {highest_level}",
        "",
        f"**Neural hierarchy:** `{neural_hypothesis}`",
        "",
        f"**Conclusion:** {conclusion}",
        "",
        "## Evidence gates",
        "",
        f"- Behavioral entry criteria: **{'PASS' if gates['entry_criteria']['passed'] else 'FAIL'}**",
        f"- Level 3 coarse causal localization: **{'PASS' if gates['coarse_localization']['passed'] else 'FAIL'}**",
        f"- Level 4 held-out causal representation: **{'PASS' if level4 else 'FAIL'}**",
        f"- Level 5 causal specificity: **{'PASS' if level5 else 'FAIL'}**",
        f"- Necessity: **{'PASS' if necessity_pass else 'FAIL'}**",
        f"- Level 6 circuit implementation: **{'PASS' if circuit_pass else 'NOT ESTABLISHED'}**",
        "",
        "## Behavioral-theory assessment",
        "",
        *[
            f"- `{target}`: **{assessment['status']}**; specificity-passing theories: "
            f"{', '.join(assessment['specificity_passing']) or 'none'}"
            for target, assessment in sorted(theory_assessment.items())
        ],
        "",
        "## Interpretation guardrails",
        "",
        "- Diagnostic decoding is labeled information access, not mechanism.",
        "- Behavioral models, coefficients, normalizations, and predictions were frozen before neural optimization.",
        "- DAS used training pairs only; rank/layer selection used validation pairs only.",
        "- Scientific counterfactual claims use untouched test and task-holdout pairs.",
        "- Level 5 requires the candidate to beat matched random, shuffled, decision/output, value, task-ID, response-mapping, and unrelated-variable controls.",
        "- Behavioral-validation rows were never used and no full activation bank was stored.",
        "",
        "## Mandatory null-result language",
        "",
        f"> {conclusion}",
    ]
    report_path = root / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "report_generated": True,
            "highest_evidence_level": highest_level,
            "neural_implementation_hypothesis": neural_hypothesis,
            "figure_count": len(figures),
            "behavioral_validation_used": False,
            "full_activations_saved": False,
        }
    )
    _json(metadata_path, metadata)
    return {
        "report": report_path,
        "highest_evidence_level": highest_level,
        "neural_hypothesis": neural_hypothesis,
        "figures": figures,
    }

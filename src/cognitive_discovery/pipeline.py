"""Artifact-oriented orchestration shared by laptop commands and SLURM jobs."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from cognitive_discovery.analysis.loto import leave_one_task_out
from cognitive_discovery.analysis.model_comparison import run_model_tournament
from cognitive_discovery.analysis.residual_discovery import discover_residuals
from cognitive_discovery.analysis.response_surface import (
    interaction_effects,
    marginal_effects,
)
from cognitive_discovery.analysis.validation import evaluate_frozen_model
from cognitive_discovery.analysis.active_efficiency import compare_active_efficiency
from cognitive_discovery.analysis.hierarchical_comparison import (
    run_hierarchical_analysis,
)
from cognitive_discovery.audits.calibration import calibration_tables
from cognitive_discovery.audits.information_sampling import audit_information_sampling
from cognitive_discovery.data.provenance import build_run_metadata
from cognitive_discovery.data.splits import assign_condition_splits
from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.data.validation import pilot_gate, validate_records
from cognitive_discovery.design.manifests import (
    load_condition_manifest,
    semantic_hash,
    write_design_artifacts,
)
from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.experiments.collection import (
    collect_conditions,
    expanded_render_conditions,
)
from cognitive_discovery.experiments.registry import get_renderer
from cognitive_discovery.participants.base import DeterministicParticipant
from cognitive_discovery.participants.qwen import QwenParticipant
from cognitive_discovery.reporting.report import generate_report
from cognitive_discovery.reporting.figures import generate_figures
from cognitive_discovery.models.flexible.recovery import validate_flexible_ceilings
from cognitive_discovery.models.flexible.recovery import (
    validate_matched_flexible_ceilings,
)
from cognitive_discovery.hierarchy.model_recovery import recover_hierarchy_variants
from cognitive_discovery.hierarchy.task_descriptors import descriptor_frame
from cognitive_discovery.reporting.round2 import (
    generate_round2_figures,
    generate_round2_report,
)
from cognitive_discovery.sampling.active_mixture import select_active_mixture
from cognitive_discovery.sampling.candidate_pool import (
    condition_frame,
    generate_candidate_pool,
    observed_semantic_hashes,
)
from cognitive_discovery.sampling.coverage_score import coverage_scores
from cognitive_discovery.sampling.disagreement_score import disagreement_scores
from cognitive_discovery.sampling.information_score import information_scores


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_output(config: dict, output: str | Path | None = None) -> Path:
    return Path(output or config["output_root"])


def generate_design(
    config: dict,
    *,
    output: str | Path | None = None,
    conditions: int | None = None,
    seed: int | None = None,
    design_id: str = "discovery_v1",
    exclude_semantic_hashes: set[str] | None = None,
    persist: bool = True,
):
    root = resolve_output(config, output)
    design = compile_design(
        config,
        n_conditions=conditions,
        seed=seed,
        design_id=design_id,
        exclude_semantic_hashes=exclude_semantic_hashes,
    )
    design = assign_condition_splits(
        design,
        seed=int(config["split_seed"] if seed is None else seed + 1),
        config=config,
    )
    manifest = None
    if persist:
        manifest = write_design_artifacts(design, config, root / "design")
        split_root = root / "splits"
        split_root.mkdir(parents=True, exist_ok=True)
        for split in ("discovery", "interpolation_test", "structural_test"):
            values = sorted(
                {
                    condition.paired_condition_id
                    for condition in design
                    if condition.split == split
                }
            )
            (split_root / f"{split}.json").write_text(
                json.dumps(values, indent=2) + "\n", encoding="utf-8"
            )
    return design, manifest


def _participant(
    config: dict, *, model_path=None, revision=None, online=False, model_free=False
):
    if model_free:
        return DeterministicParticipant()
    return QwenParticipant.from_pretrained(
        model_path or config["model"],
        revision=revision or config.get("model_revision"),
        local_files_only=not online,
    )


def collect_shard(
    config: dict,
    *,
    output: str | Path | None = None,
    conditions: int | None = None,
    seed: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    model_path=None,
    revision=None,
    online=False,
    model_free=False,
    resume=False,
    manifest: str | Path | None = None,
):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    root = resolve_output(config, output)
    if manifest is not None:
        design = load_condition_manifest(manifest)
        if shard_index == 0:
            write_design_artifacts(design, config, root / "design")
    else:
        design, _ = generate_design(
            config,
            output=root,
            conditions=conditions,
            seed=seed,
            persist=shard_index == 0,
        )
    pair_ids = sorted({condition.paired_condition_id for condition in design})
    selected_pairs = {
        pair_id
        for index, pair_id in enumerate(pair_ids)
        if index % shard_count == shard_index
    }
    selected = [
        condition
        for condition in design
        if condition.paired_condition_id in selected_pairs
    ]
    expanded = bool(config["collection"].get("expand_history_prefixes", False))
    rendered_conditions = (
        list(expanded_render_conditions(selected)) if expanded else selected
    )
    raw_root = root / "behavior/raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    requested_path = raw_root / f"shard-{shard_index:04d}-of-{shard_count:04d}.parquet"
    existing_path = (
        requested_path
        if requested_path.exists()
        else requested_path.with_suffix(".csv.gz")
    )
    if resume and existing_path.exists():
        existing = validate_records(read_records(existing_path))
        expected_ids = {condition.condition_id for condition in rendered_conditions}
        if set(existing.condition_id.astype(str)) != expected_ids:
            raise RuntimeError(
                f"resume shard does not match the compiled design: {existing_path}; use a new output directory"
            )
        expected_model = (
            "deterministic/model-free-smoke"
            if model_free
            else str(model_path or config["model"])
        )
        if set(existing.model.astype(str)) != {expected_model}:
            raise RuntimeError(
                f"resume shard model does not match {expected_model}: {existing_path}"
            )
        return existing_path
    participant = _participant(
        config,
        model_path=model_path,
        revision=revision,
        online=online,
        model_free=model_free,
    )
    records = collect_conditions(
        selected,
        participant,
        model_revision=revision or config.get("model_revision"),
        sample_actions=bool(config["collection"].get("sample_actions", True)),
        expand_history_prefixes=expanded,
    )
    path = write_records(records, requested_path)
    prompt_root = root / "behavior/prompts"
    prompt_root.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_root / f"shard-{shard_index:04d}-of-{shard_count:04d}.jsonl"
    prompt_path.write_text(
        "".join(
            json.dumps(
                {
                    "condition_id": condition.condition_id,
                    "prompt_hash": get_renderer(condition.task_family)
                    .render(condition)
                    .prompt_hash,
                    "messages": get_renderer(condition.task_family)
                    .render(condition)
                    .messages,
                },
                sort_keys=True,
            )
            + "\n"
            for condition in rendered_conditions
        ),
        encoding="utf-8",
    )
    return path


def finalize_collection(
    config: dict,
    *,
    output: str | Path | None = None,
    conditions: int | None = None,
    seed: int | None = None,
    manifest: str | Path | None = None,
):
    root = resolve_output(config, output)
    path_by_shard = {
        path.name.removesuffix(".parquet"): path
        for path in sorted((root / "behavior/raw").glob("shard-*.parquet"))
    }
    for path in sorted((root / "behavior/raw").glob("shard-*.csv.gz")):
        path_by_shard.setdefault(path.name.removesuffix(".csv.gz"), path)
    paths = list(path_by_shard.values())
    if not paths:
        raise FileNotFoundError(
            f"no collection shards found under {root / 'behavior/raw'}"
        )
    frame = pd.concat([read_records(path) for path in paths], ignore_index=True)
    frame = validate_records(frame)
    design = (
        load_condition_manifest(manifest)
        if manifest is not None
        else generate_design(config, output=root, conditions=conditions, seed=seed)[0]
    )
    expected = (
        sum(condition.history.length + 1 for condition in design)
        if config["collection"].get("expand_history_prefixes", False)
        else len(design)
    )
    if len(frame) != expected:
        raise RuntimeError(
            f"collection has {len(frame)} observations; expected {expected}"
        )
    if frame.condition_id.duplicated().any():
        raise RuntimeError("collection shards overlap")
    path = write_records(
        frame.to_dict("records"), root / "behavior/standardized.parquet"
    )
    gates = {
        task: pilot_gate(part.reset_index(drop=True), config)
        for task, part in frame.groupby("task_family")
    }
    validation_root = root / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    (validation_root / "pilot_approval.json").write_text(
        json.dumps(
            {
                "approved": all(item["approved"] for item in gates.values()),
                "tasks": gates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = build_run_metadata(config, design, root=Path.cwd())
    metadata["observed_models"] = sorted(frame.model.dropna().astype(str).unique())
    metadata["observed_model_revisions"] = sorted(
        frame.model_revision.dropna().astype(str).unique()
    )
    (root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frame, path, gates


def _load_standardized(root: Path):
    parquet = root / "behavior/standardized.parquet"
    return read_records(parquet if parquet.exists() else parquet.with_suffix(".csv.gz"))


def analyze_run(config: dict, *, output: str | Path | None = None, smoke=False):
    root = resolve_output(config, output)
    records = validate_records(_load_standardized(root))
    surface_root = root / "analysis/response_surface"
    surface_root.mkdir(parents=True, exist_ok=True)
    marginal_effects(records).to_csv(surface_root / "marginal_effects.csv", index=False)
    interaction_effects(records).to_csv(surface_root / "interactions.csv", index=False)
    flexible_model_root = root / "models/flexible"
    flexible_model_root.mkdir(parents=True, exist_ok=True)
    recovery = validate_flexible_ceilings(config, smoke=smoke)
    recovery.to_csv(flexible_model_root / "synthetic_recovery.csv", index=False)
    if not smoke and not recovery.passed.all():
        failed = recovery.loc[~recovery.passed, "model"].tolist()
        raise RuntimeError(
            f"flexible models failed synthetic teacher recovery and cannot be used as ceilings: {failed}"
        )
    comparison_root = root / "analysis/model_comparison"
    comparison, best_fit = run_model_tournament(
        records, config, output_directory=comparison_root, smoke=smoke
    )
    cognitive_model_root = root / "models/cognitive"
    cognitive_model_root.mkdir(parents=True, exist_ok=True)
    with (cognitive_model_root / "frozen_best.pkl").open("wb") as handle:
        pickle.dump(best_fit, handle)
    comparison[comparison.model.isin(config["models"].get("flexible", []))].to_csv(
        flexible_model_root / "ceiling_metrics.csv", index=False
    )
    best_row = (
        comparison[
            (comparison.split == "interpolation_test")
            & comparison.model.isin(config["models"]["cognitive"])
        ]
        .sort_values("macro_r2", ascending=False)
        .iloc[0]
    )
    loto_root = root / "analysis/loto"
    loto_root.mkdir(parents=True, exist_ok=True)
    loto = leave_one_task_out(
        records[records.split == "discovery"].reset_index(drop=True),
        models=config["models"]["cognitive"],
    )
    loto.to_csv(loto_root / "loto.csv", index=False)
    residual_root = root / "analysis/residual_discovery"
    residual_root.mkdir(parents=True, exist_ok=True)
    residuals = discover_residuals(
        records, base_model=str(best_row.model), seed=int(config["base_seed"])
    )
    residuals.to_csv(residual_root / "interactions.csv", index=False)
    generate_figures(root)
    metadata_path = root / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    report = generate_report(root, metadata)
    return {
        "comparison": comparison,
        "loto": loto,
        "residuals": residuals,
        "report": report,
    }


def run_residual_discovery(
    config: dict,
    *,
    output: str | Path | None = None,
    base_model: str | None = None,
):
    root = resolve_output(config, output)
    if base_model is None:
        best_path = root / "analysis/model_comparison/best_model.txt"
        if not best_path.exists():
            raise FileNotFoundError(
                "fit the cognitive model tournament before residual discovery"
            )
        base_model = best_path.read_text(encoding="utf-8").split()[0]
    records = validate_records(_load_standardized(root))
    result = discover_residuals(
        records, base_model=base_model, seed=int(config["base_seed"])
    )
    residual_root = root / "analysis/residual_discovery"
    residual_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(residual_root / "interactions.csv", index=False)
    return result


def collect_and_evaluate_validation(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    model_free=False,
    conditions: int | None = None,
):
    root = resolve_output(config, output)
    discovery_count = int(config["design"]["discovery_conditions"])
    original = compile_design(
        config, n_conditions=discovery_count, seed=int(config["design_seed"])
    )
    excluded = {semantic_hash(condition) for condition in original[::2]}
    validation_conditions = compile_design(
        config,
        n_conditions=int(conditions or config["validation"]["conditions"]),
        seed=int(config["design_seed"]) + 10_000_000,
        design_id="validation_v1",
        exclude_semantic_hashes=excluded,
    )
    participant = _participant(
        config,
        model_path=model_path,
        revision=revision,
        online=online,
        model_free=model_free,
    )
    observations = collect_conditions(
        validation_conditions,
        participant,
        expand_history_prefixes=bool(
            config["collection"].get("expand_history_prefixes", False)
        ),
    )
    validation_root = root / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    path = write_records(observations, validation_root / "independent_sample.parquet")
    frame = validate_records(read_records(path))
    predictions, metrics = evaluate_frozen_model(
        frame, root / "models/cognitive/frozen_best.pkl"
    )
    predictions.to_csv(validation_root / "predictions.csv", index=False)
    calibration, deciles, per_task = calibration_tables(predictions)
    calibration.to_csv(validation_root / "calibration.csv", index=False)
    deciles.to_csv(validation_root / "prediction_deciles.csv", index=False)
    per_task.to_csv(validation_root / "per_task_metrics.csv", index=False)
    (validation_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata_path = root / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    generate_figures(root)
    generate_report(root, metadata)
    return metrics


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    return write_records(frame.to_dict("records"), path)


def prepare_active_round(
    config: dict,
    *,
    round1_output: str | Path,
    output: str | Path | None = None,
    smoke: bool = False,
):
    """Audit Round 1, fit the hierarchy, and freeze the active manifest."""

    root = resolve_output(config, output)
    round1_root = Path(round1_output)
    records = validate_records(_load_standardized(round1_root))
    hierarchy = run_hierarchical_analysis(
        records, config, output_directory=root / "hierarchy", smoke=smoke
    )
    descriptor_frame().to_csv(root / "hierarchy/task_descriptors.csv", index=False)
    audit_root = root / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    audit_information_sampling(records).to_csv(
        audit_root / "information_sampling.csv", index=False
    )
    recovery = validate_matched_flexible_ceilings(records, config, smoke=smoke)
    recovery.to_csv(audit_root / "teacher_recovery.csv", index=False)
    recovery.to_csv(audit_root / "flexible_ceiling.csv", index=False)
    hierarchy_recovery = recover_hierarchy_variants(
        n_train=140 if smoke else 700,
        n_test=70 if smoke else 280,
        seed=int(config["base_seed"]) + 701,
    )
    hierarchy_recovery.to_csv(audit_root / "hierarchy_recovery.csv", index=False)

    residuals = discover_residuals(
        records,
        base_model=str(hierarchy["best_fit"].architecture),
        base_variant=str(hierarchy["best_fit"].variant),
        seed=int(config["base_seed"]),
        bootstraps=(
            20
            if smoke
            else int(config.get("residual_discovery", {}).get("bootstraps", 200))
        ),
    )
    residual_root = root / "residuals"
    residual_root.mkdir(parents=True, exist_ok=True)
    residuals[
        [
            "interaction",
            "selected_for_validation",
            "stability_probability",
            "selection_rows",
            "selection_group_digest",
        ]
    ].to_csv(residual_root / "candidate_terms.csv", index=False)
    residuals.to_csv(residual_root / "heldout_gains.csv", index=False)

    settings = config.get("active_sampling", {})
    budget = int(
        settings.get("smoke_budget", 70) if smoke else settings.get("budget", 1400)
    )
    multiplier = int(settings.get("candidate_multiplier", 10))
    conditions, candidates = generate_candidate_pool(
        config, records, budget=budget, multiplier=multiplier
    )
    coverage = coverage_scores(records, candidates)
    information = information_scores(
        records[records.split == "discovery"].reset_index(drop=True),
        candidates,
        architecture=str(settings.get("information_architecture", "dual_history")),
        bootstraps=3 if smoke else int(settings.get("information_bootstraps", 32)),
        seed=int(config["base_seed"]),
    )
    disagreement = disagreement_scores(
        records[records.split == "discovery"].reset_index(drop=True),
        candidates,
        architecture_a=str(settings.get("architecture_a", "dual_history")),
        architecture_b=str(settings.get("architecture_b", "latent_context")),
    )
    scores = coverage.merge(
        information.drop(columns=["semantic_hash", "task_family"]),
        on="paired_condition_id",
        validate="one_to_one",
    ).merge(
        disagreement.drop(columns=["semantic_hash", "task_family"]),
        on="paired_condition_id",
        validate="one_to_one",
    )
    selection = select_active_mixture(
        conditions,
        scores,
        budget=budget,
        weights=settings.get(
            "mixture", {"coverage": 0.4, "information": 0.4, "disagreement": 0.2}
        ),
        excluded_hashes=observed_semantic_hashes(records),
    )
    active_root = root / "active_sampling"
    active_root.mkdir(parents=True, exist_ok=True)
    candidate_path = _write_frame(
        condition_frame(conditions), active_root / "candidate_manifest.parquet"
    )
    selected_path = write_records(
        selection.conditions, active_root / "selected_conditions.parquet"
    )
    # JSONL is the canonical collector input and preserves nested dictionaries.
    manifest_path = write_records(
        selection.conditions, active_root / "selected_conditions.jsonl"
    )
    annotated_scores = selection.scores.merge(
        selection.selected[["paired_condition_id", "sampling_strategy"]],
        on="paired_condition_id",
        how="left",
        validate="one_to_one",
    )
    annotated_scores["selected"] = annotated_scores.sampling_strategy.notna()
    score_path = _write_frame(annotated_scores, active_root / "sampling_scores.parquet")
    selection.selected.to_csv(active_root / "selected_allocations.csv", index=False)
    metadata = build_run_metadata(config, selection.conditions, root=Path.cwd())
    metadata.update(
        {
            "round1_output": str(round1_root),
            "active_budget": budget,
            "candidate_multiplier": multiplier,
            "previous_semantic_hashes": len(observed_semantic_hashes(records)),
        }
    )
    (root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generate_round2_figures(root)
    generate_round2_report(root)
    return {
        "manifest": manifest_path,
        "candidate_manifest": candidate_path,
        "selected_conditions": selected_path,
        "scores": score_path,
        "hierarchy": hierarchy,
    }


def update_after_active_round(
    config: dict,
    *,
    round1_output: str | Path,
    active_output: str | Path,
    output: str | Path | None = None,
    smoke: bool = False,
):
    """Refit after active collection and freeze a coverage-random final test."""

    root = resolve_output(config, output)
    round1 = validate_records(_load_standardized(Path(round1_output)))
    active = validate_records(_load_standardized(Path(active_output))).copy()
    active["split"] = "discovery"
    combined = pd.concat([round1, active], ignore_index=True)
    hierarchy = run_hierarchical_analysis(
        combined, config, output_directory=root / "hierarchy", smoke=smoke
    )
    final_count = int(
        config.get("final_validation", {}).get("smoke_conditions", 70)
        if smoke
        else config.get("final_validation", {}).get("conditions", 700)
    )
    final_pool, _ = generate_candidate_pool(
        config,
        combined,
        budget=final_count,
        multiplier=10,
        seed=int(config["design_seed"]) + 30_000_000,
    )
    selected_pair_ids = []
    for condition in final_pool:
        if condition.paired_condition_id not in selected_pair_ids:
            selected_pair_ids.append(condition.paired_condition_id)
        if len(selected_pair_ids) == final_count:
            break
    selected_pair_ids = set(selected_pair_ids)
    final_conditions = [
        condition
        for condition in final_pool
        if condition.paired_condition_id in selected_pair_ids
    ]
    final_conditions = [
        replace(
            condition,
            design_id="final_validation_v2",
            condition_id=condition.condition_id.replace(
                "active_round2_candidates", "final_validation_v2", 1
            ),
            paired_condition_id=condition.paired_condition_id.replace(
                "active_round2_candidates", "final_validation_v2", 1
            ),
            sampling_strategy="coverage_random",
            split="final_validation",
        )
        for condition in final_conditions
    ]
    final_root = root / "final_validation"
    final_root.mkdir(parents=True, exist_ok=True)
    manifest = write_records(final_conditions, final_root / "condition_manifest.jsonl")
    _write_frame(
        condition_frame(final_conditions), final_root / "condition_manifest.parquet"
    )
    generate_round2_figures(root)
    generate_round2_report(root)
    return {"manifest": manifest, "hierarchy": hierarchy}


def evaluate_round2_final(
    config: dict,
    *,
    round1_output: str | Path,
    active_output: str | Path,
    final_output: str | Path,
    output: str | Path | None = None,
):
    """Evaluate the frozen updated hierarchy and active sampling efficiency."""

    root = resolve_output(config, output)
    round1 = validate_records(_load_standardized(Path(round1_output)))
    active = validate_records(_load_standardized(Path(active_output)))
    final = validate_records(_load_standardized(Path(final_output)))
    model_path = root / "hierarchy/best_hierarchical_model.pkl"
    predictions, metrics = evaluate_frozen_model(final, model_path)
    validation_root = root / "final_validation"
    predictions.to_csv(validation_root / "predictions.csv", index=False)
    calibration, deciles, per_task = calibration_tables(predictions)
    calibration.to_csv(validation_root / "calibration.csv", index=False)
    deciles.to_csv(validation_root / "prediction_deciles.csv", index=False)
    per_task.to_csv(validation_root / "per_task_metrics.csv", index=False)
    (validation_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    scores_path = root / "active_sampling/sampling_scores.parquet"
    scores = read_records(
        scores_path if scores_path.exists() else scores_path.with_suffix(".csv.gz")
    )
    if "sampling_strategy" not in scores:
        allocations = pd.read_csv(root / "active_sampling/selected_allocations.csv")
        scores = scores.merge(
            allocations[["paired_condition_id", "sampling_strategy"]],
            on="paired_condition_id",
            how="left",
            validate="one_to_one",
        )
    scores = scores[scores.sampling_strategy.notna()].reset_index(drop=True)
    best = pickle.load(model_path.open("rb"))
    efficiency = compare_active_efficiency(
        round1,
        active,
        final,
        scores,
        architecture=best.architecture,
        budgets=config.get("active_sampling", {}).get(
            "efficiency_budgets", (100, 250, 500, 1000)
        ),
    )
    efficiency.to_csv(root / "active_sampling/budget_curves.csv", index=False)
    generate_round2_figures(root)
    generate_round2_report(root)
    return {"metrics": metrics, "efficiency": efficiency}

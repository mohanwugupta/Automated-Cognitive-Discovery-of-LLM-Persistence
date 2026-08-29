"""Artifact-oriented orchestration shared by laptop commands and SLURM jobs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from cognitive_discovery.analysis.loto import leave_one_task_out
from cognitive_discovery.analysis.model_comparison import run_model_tournament
from cognitive_discovery.analysis.residual_discovery import discover_residuals
from cognitive_discovery.analysis.response_surface import interaction_effects, marginal_effects
from cognitive_discovery.analysis.validation import evaluate_frozen_model
from cognitive_discovery.data.provenance import build_run_metadata
from cognitive_discovery.data.splits import assign_condition_splits
from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.data.validation import pilot_gate, validate_records
from cognitive_discovery.design.manifests import semantic_hash, write_design_artifacts
from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.experiments.collection import collect_conditions, expanded_render_conditions
from cognitive_discovery.experiments.registry import get_renderer
from cognitive_discovery.participants.base import DeterministicParticipant
from cognitive_discovery.participants.qwen import QwenParticipant
from cognitive_discovery.reporting.report import generate_report
from cognitive_discovery.reporting.figures import generate_figures
from cognitive_discovery.models.flexible.recovery import validate_flexible_ceilings


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
                {condition.paired_condition_id for condition in design if condition.split == split}
            )
            (split_root / f"{split}.json").write_text(
                json.dumps(values, indent=2) + "\n", encoding="utf-8"
            )
    return design, manifest


def _participant(config: dict, *, model_path=None, revision=None, online=False, model_free=False):
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
):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    root = resolve_output(config, output)
    design, _ = generate_design(
        config,
        output=root,
        conditions=conditions,
        seed=seed,
        persist=shard_index == 0,
    )
    pair_ids = sorted({condition.paired_condition_id for condition in design})
    selected_pairs = {
        pair_id for index, pair_id in enumerate(pair_ids) if index % shard_count == shard_index
    }
    selected = [
        condition for condition in design if condition.paired_condition_id in selected_pairs
    ]
    expanded = bool(config["collection"].get("expand_history_prefixes", False))
    rendered_conditions = list(expanded_render_conditions(selected)) if expanded else selected
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
        expected_model = "deterministic/model-free-smoke" if model_free else str(
            model_path or config["model"]
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
                    "prompt_hash": get_renderer(condition.task_family).render(condition).prompt_hash,
                    "messages": get_renderer(condition.task_family).render(condition).messages,
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
        raise FileNotFoundError(f"no collection shards found under {root / 'behavior/raw'}")
    frame = pd.concat([read_records(path) for path in paths], ignore_index=True)
    frame = validate_records(frame)
    design, _ = generate_design(config, output=root, conditions=conditions, seed=seed)
    expected = (
        sum(condition.history.length + 1 for condition in design)
        if config["collection"].get("expand_history_prefixes", False)
        else len(design)
    )
    if len(frame) != expected:
        raise RuntimeError(f"collection has {len(frame)} observations; expected {expected}")
    if frame.condition_id.duplicated().any():
        raise RuntimeError("collection shards overlap")
    path = write_records(frame.to_dict("records"), root / "behavior/standardized.parquet")
    gates = {
        task: pilot_gate(part.reset_index(drop=True), config)
        for task, part in frame.groupby("task_family")
    }
    validation_root = root / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    (validation_root / "pilot_approval.json").write_text(
        json.dumps(
            {"approved": all(item["approved"] for item in gates.values()), "tasks": gates},
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
    residuals = discover_residuals(records, base_model=str(best_row.model), seed=int(config["base_seed"]))
    residuals.to_csv(residual_root / "interactions.csv", index=False)
    generate_figures(root)
    metadata_path = root / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    report = generate_report(root, metadata)
    return {"comparison": comparison, "loto": loto, "residuals": residuals, "report": report}


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
            raise FileNotFoundError("fit the cognitive model tournament before residual discovery")
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
    original = compile_design(config, n_conditions=discovery_count, seed=int(config["design_seed"]))
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
    (validation_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    generate_figures(root)
    generate_report(root, metadata)
    return metrics

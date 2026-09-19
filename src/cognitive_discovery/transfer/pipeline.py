"""Prepare, aggregate, and gate restartable task-transfer work units."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from cognitive_discovery.causal_mechanistic.das import load_alignment
from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.replication.provenance import canonical_hash
from cognitive_discovery.replication.survivors import validate_frozen_survivor_set
from cognitive_discovery.model_validation.specifications import validate_specifications

from .config import load_transfer_config
from .geometry import controller_geometry
from .predictors import fit_heldout_target_predictor
from .provenance import file_hash, git_identity, write_provenance
from .splits import assign_transfer_splits, planned_source_sets, transfer_split_hash


OUTPUT_COLUMNS = (
    "model", "theory", "source_task", "target_task", "layer", "rank", "n_train",
    "n_test", "global_cfr", "correlation", "slope", "bootstrap_low",
    "bootstrap_high", "random_p", "endpoint_id", "metric_id", "status",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_transfer(
    replication_root: str | Path,
    output: str | Path,
    config_path: str | Path,
    *,
    final: bool = False,
    scope: str = "single_task_matrix",
) -> dict:
    replication_root, output = Path(replication_root), Path(output)
    config = load_transfer_config(config_path)
    provenance = json.loads((replication_root / "provenance.json").read_text(encoding="utf-8"))
    survivor_manifest = validate_frozen_survivor_set(replication_root, provenance)
    if final and provenance.get("git_dirty") is not False:
        raise ValueError("final transfer evidence requires a clean source replication")
    if provenance["endpoint_id"] != config["endpoint_id"] or provenance["metric_id"] != config["metric_id"]:
        raise ValueError("replication and transfer endpoint/metric identities differ")
    pairs_path = replication_root / "mechanism/pair_manifest.parquet"
    predictions_path = replication_root / "mechanism/counterfactual_predictions.parquet"
    for path in (pairs_path, predictions_path, replication_root / "mechanism/condition_manifest.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(f"transfer input is absent: {path}")
    pairs = read_records(pairs_path)
    if "neural_group_id" not in pairs:
        pairs["neural_group_id"] = pairs.target_variable.astype(str) + ":" + pairs.contrast_id.astype(str)
    pairs = assign_transfer_splits(
        pairs,
        seed=int(config["seeds"]["transfer_split"]),
        fractions=tuple(config["splits"]["fractions"]),
    )
    split_hash = transfer_split_hash(pairs)
    output.mkdir(parents=True, exist_ok=False)
    effective_config = output / "effective_config.yaml"
    shutil.copyfile(config_path, effective_config)
    pair_output = Path(write_records(pairs.to_dict("records"), output / "pair_manifest.parquet"))
    theories = sorted(read_records(predictions_path).theory.astype(str).unique())
    expected_theories = sorted(survivor_manifest["behavioral_survivor_set"])
    if theories != expected_theories:
        raise ValueError(
            "transfer theory index differs from the frozen behavioral survivor set"
        )
    if scope not in {"single_task_matrix", "extended"}:
        raise ValueError("transfer scope must be single_task_matrix or extended")
    work = []
    for theory in theories:
        for plan in planned_source_sets(
            config["tasks"],
            include_all_pairs=bool(config["designs"]["all_pairs"]),
            diversity_seed=int(config["seeds"]["diversity_order"]),
        ):
            if scope == "single_task_matrix" and plan["design"] != "single":
                continue
            work.append({"work_id": f"{theory}__{plan['design']}__{plan['source_id']}", "theory": theory, **plan})
    manifest = {
        "schema_version": "task-transfer-work-v1",
        "replication_root": str(replication_root.resolve()),
        "model": provenance["model"],
        "tokenizer": provenance["tokenizer"],
        "endpoint_id": config["endpoint_id"],
        "metric_id": config["metric_id"],
        "behavioral_theory_status": survivor_manifest["behavioral_theory_status"],
        "best_model": survivor_manifest["best_model"],
        "behavioral_survivor_set": expected_theories,
        "behavioral_survivor_rule_sha256": survivor_manifest["rule_sha256"],
        "behavioral_survivor_set_sha256": survivor_manifest["membership_sha256"],
        "pair_manifest_sha256": _sha(pair_output),
        "source_pair_manifest_sha256": _sha(pairs_path),
        "prediction_manifest_sha256": _sha(predictions_path),
        "transfer_split_sha256": split_hash,
        "config_sha256": _sha(effective_config),
        "work_units": work,
        "work_manifest_hash": canonical_hash(work),
        "matrix_scope": scope,
        "diagonal_required_before_off_diagonal": True,
    }
    (output / "work_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repository = Path(__file__).resolve().parents[3]
    lockfile = repository / "uv.lock"
    transfer_provenance = {
        "schema_version": "task-transfer-provenance-v1",
        **git_identity(repository),
        "model": provenance["model"],
        "tokenizer": provenance["tokenizer"],
        "environment_lock_sha256": file_hash(lockfile),
        "specification_hashes": validate_specifications(repository),
        "pair_manifest_sha256": manifest["pair_manifest_sha256"],
        "transfer_split_sha256": split_hash,
        "endpoint_id": config["endpoint_id"],
        "metric_id": config["metric_id"],
        "seeds": config["seeds"],
        "output_root": str(output.resolve()),
        "config_sha256": manifest["config_sha256"],
        "work_manifest_hash": manifest["work_manifest_hash"],
        "behavioral_theory_status": manifest["behavioral_theory_status"],
        "behavioral_survivor_set": manifest["behavioral_survivor_set"],
        "behavioral_survivor_rule_sha256": manifest[
            "behavioral_survivor_rule_sha256"
        ],
        "behavioral_survivor_set_sha256": manifest[
            "behavioral_survivor_set_sha256"
        ],
        "evidence_mode": "final" if final else "development",
    }
    write_provenance(output / "provenance.json", transfer_provenance, final=final)
    return manifest


def _merge_job_metric_components(job_root: Path) -> Path:
    paths = [job_root / "diagonal_metrics.csv", job_root / "off_diagonal_metrics.csv"]
    frames = [pd.read_csv(path) for path in paths if path.is_file()]
    if not frames:
        raise RuntimeError(f"no transfer metric components for {job_root.name}")
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["model", "theory", "source_task", "target_task"], keep="last"
    )
    path = job_root / "metrics.csv"
    frame.to_csv(path, index=False)
    return path


def gate_diagonal_transfer(
    output: str | Path, *, active_indices: list[int] | None = None
) -> dict:
    """Freeze Gate-D availability before any off-diagonal model execution."""

    output = Path(output)
    manifest = json.loads((output / "work_manifest.json").read_text(encoding="utf-8"))
    eligible, unavailable, missing = [], [], []
    active = (
        set(active_indices)
        if active_indices is not None
        else set(range(len(manifest["work_units"])))
    )
    for index, work in enumerate(manifest["work_units"]):
        if index not in active:
            continue
        if work.get("design") != "single":
            continue
        job_root = output / "jobs" / work["work_id"]
        gate_path = job_root / "diagonal_gate.json"
        if not gate_path.is_file():
            missing.append(work["work_id"])
            continue
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("source_available") is True:
            eligible.append(index)
            continue
        unavailable.append(index)
        selected = json.loads(
            (job_root / "selected_controller.json").read_text(encoding="utf-8")
        )
        rows = []
        for target in work["target_tasks"]:
            if target in set(work["source_tasks"]):
                continue
            rows.append(
                {
                    "model": manifest["model"]["id"],
                    "theory": work["theory"],
                    "source_task": work["source_id"],
                    "target_task": target,
                    "layer": selected["layer"],
                    "rank": selected["rank"],
                    "n_train": selected["n_train"],
                    "n_test": 0,
                    "global_cfr": np.nan,
                    "correlation": np.nan,
                    "slope": np.nan,
                    "bootstrap_low": np.nan,
                    "bootstrap_high": np.nan,
                    "random_p": np.nan,
                    "endpoint_id": manifest["endpoint_id"],
                    "metric_id": manifest["metric_id"],
                    "status": "unavailable",
                }
            )
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
            job_root / "off_diagonal_metrics.csv", index=False
        )
        _merge_job_metric_components(job_root)
    if missing:
        raise RuntimeError(
            f"diagonal transfer gates are incomplete: {missing[:5]} ({len(missing)} total)"
        )
    record = {
        "schema_version": "task-transfer-diagonal-dispatch-v1",
        "eligible_array_indices": eligible,
        "unavailable_array_indices": unavailable,
        "eligible_work_ids": [manifest["work_units"][index]["work_id"] for index in eligible],
        "unavailable_work_ids": [
            manifest["work_units"][index]["work_id"] for index in unavailable
        ],
        "endpoint_id": manifest["endpoint_id"],
        "metric_id": manifest["metric_id"],
        "active_array_indices": sorted(active),
    }
    (output / "diagonal_gate.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def project_pilot_resources(
    *,
    train_seconds: float,
    diagonal_seconds: float,
    off_diagonal_seconds: float,
    pilot_storage_bytes: int,
    source_controllers: int,
    train_scale: float = 1.0,
    evaluation_scale: float = 1.0,
    storage_scale: float = 1.0,
) -> dict:
    measured_seconds = (
        float(train_seconds)
        + float(diagonal_seconds)
        + float(off_diagonal_seconds)
    )
    projected_seconds_per_controller = (
        float(train_seconds) * float(train_scale)
        + (float(diagonal_seconds) + float(off_diagonal_seconds))
        * float(evaluation_scale)
    )
    pilot_hours = measured_seconds / 3600.0
    projected_hours_per_controller = projected_seconds_per_controller / 3600.0
    pilot_storage_gb = float(pilot_storage_bytes) / 1_000_000_000
    return {
        "schema_version": "task-transfer-pilot-projection-v1",
        "pilot_gpu_hours": pilot_hours,
        "pilot_storage_gb": pilot_storage_gb,
        "source_controllers": int(source_controllers),
        "train_scale": float(train_scale),
        "evaluation_scale": float(evaluation_scale),
        "storage_scale": float(storage_scale),
        "projected_gpu_hours_per_source_controller": projected_hours_per_controller,
        "projected_full_gpu_hours": (
            projected_hours_per_controller * int(source_controllers)
        ),
        "projected_full_storage_gb": (
            pilot_storage_gb * float(storage_scale) * int(source_controllers)
        ),
        "projection_rule": (
            "measured_one_source_smoke_pilot_scaled_by_frozen_search_grid_"
            "epochs_random_controls_and_source_controllers"
        ),
        "approval_required": True,
    }


def _write_pilot_projection(output: Path, manifest: dict) -> dict | None:
    completed = []
    for work in manifest["work_units"]:
        job_root = output / "jobs" / work["work_id"]
        resources = {}
        for name in ("train", "evaluate_diagonal", "evaluate_off_diagonal"):
            path = job_root / f"resource_{name}.json"
            if path.is_file():
                resources[name] = json.loads(path.read_text(encoding="utf-8"))
        if {"train", "evaluate_diagonal", "evaluate_off_diagonal"} <= set(resources):
            completed.append((job_root, resources))
    if not completed:
        return None
    job_root, resources = completed[0]
    config = yaml.safe_load(
        (output / "effective_config.yaml").read_text(encoding="utf-8")
    )
    search = config["search"]
    full_candidates = len(search["relative_depths"]) * len(search["ranks"])
    full_epochs = int(search["epochs"])
    full_random = int(config["random_controls"]["subspaces"])
    pilot_random = max(
        int(resources["evaluate_diagonal"].get("random_count", 1)), 1
    )
    train_scale = full_candidates * full_epochs
    evaluation_scale = max(float(full_random) / pilot_random, 1.0)
    storage_scale = max(float(full_candidates), evaluation_scale, 1.0)
    projection = project_pilot_resources(
        train_seconds=resources["train"]["elapsed_seconds"],
        diagonal_seconds=resources["evaluate_diagonal"]["elapsed_seconds"],
        off_diagonal_seconds=resources["evaluate_off_diagonal"]["elapsed_seconds"],
        pilot_storage_bytes=int(resources["evaluate_off_diagonal"]["output_bytes"]),
        source_controllers=len(manifest["work_units"]),
        train_scale=train_scale,
        evaluation_scale=evaluation_scale,
        storage_scale=storage_scale,
    )
    projection.update(
        {
            "pilot_work_id": job_root.name,
            "model": manifest["model"],
            "behavioral_survivor_set": manifest["behavioral_survivor_set"],
            "config_sha256": manifest["config_sha256"],
            "work_manifest_hash": manifest["work_manifest_hash"],
        }
    )
    (output / "resource_projection.json").write_text(
        json.dumps(projection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "PILOT_RESOURCE_REPORT.md").write_text(
        "# Task-transfer pilot resource report\n\n"
        f"Pilot source controller: `{job_root.name}`  \n"
        f"Measured pilot GPU-hours: {projection['pilot_gpu_hours']:.4f}  \n"
        f"Measured output storage: {projection['pilot_storage_gb']:.4f} GB  \n"
        f"Training scale (grid × epochs): {projection['train_scale']:.2f}×  \n"
        f"Evaluation scale (random controls): {projection['evaluation_scale']:.2f}×  \n"
        f"Projected full GPU-hours: {projection['projected_full_gpu_hours']:.4f}  \n"
        f"Projected full storage: {projection['projected_full_storage_gb']:.4f} GB\n\n"
        "The full matrix remains blocked until both projected budgets are explicitly approved.\n",
        encoding="utf-8",
    )
    return projection


def aggregate_transfer(output: str | Path, *, require_complete: bool = True) -> dict:
    output = Path(output)
    manifest = json.loads((output / "work_manifest.json").read_text(encoding="utf-8"))
    expected = {unit["work_id"] for unit in manifest["work_units"]}
    frames, missing = [], []
    for work_id in sorted(expected):
        path = output / "jobs" / work_id / "metrics.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            if set(OUTPUT_COLUMNS) - set(frame.columns):
                raise ValueError(f"transfer job {work_id} has an invalid metric schema")
            frames.append(frame)
        else:
            missing.append(work_id)
    if require_complete and missing:
        raise RuntimeError(f"transfer jobs are incomplete: {missing[:5]} ({len(missing)} total)")
    all_metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    all_metrics.to_csv(output / "all_transfer_metrics.csv", index=False)
    singles = all_metrics[all_metrics.source_task.astype(str).str.fullmatch("[a-z_]+")]
    if not singles.empty:
        singles = gate_source_controllers(singles)
    singles.to_csv(output / "single_task_transfer.csv", index=False)
    all_metrics[all_metrics.source_task.astype(str).str.contains("\\+")].to_csv(output / "pair_task_transfer.csv", index=False)
    all_metrics[all_metrics.source_task.astype(str).str.startswith("without-")].to_csv(output / "leave_one_task_out.csv", index=False)
    all_metrics[all_metrics.source_task.astype(str).str.startswith("to-")].to_csv(output / "training_diversity.csv", index=False)
    geometry_rows = []
    single_units = [unit for unit in manifest["work_units"] if unit["design"] == "single"]
    for left_index, left in enumerate(single_units):
        left_meta = output / "jobs" / left["work_id"] / "selected_controller.json"
        if not left_meta.is_file():
            continue
        left_record = json.loads(left_meta.read_text(encoding="utf-8"))
        left_basis = load_alignment(left_meta.parent / left_record["controller_path"])
        for right in single_units[left_index + 1:]:
            if right["theory"] != left["theory"]:
                continue
            right_meta = output / "jobs" / right["work_id"] / "selected_controller.json"
            if not right_meta.is_file():
                continue
            right_record = json.loads(right_meta.read_text(encoding="utf-8"))
            right_basis = load_alignment(right_meta.parent / right_record["controller_path"])
            geometry_rows.append({
                "model": manifest["model"]["id"], "theory": left["theory"],
                "source_task_i": left["source_id"], "source_task_j": right["source_id"],
                "layer_i": left_record["layer"], "layer_j": right_record["layer"],
                "rank_i": left_record["rank"], "rank_j": right_record["rank"],
                **controller_geometry(left_basis, right_basis),
            })
    pd.DataFrame(geometry_rows).to_csv(output / "controller_geometry.csv", index=False)
    status = {"complete": not missing, "completed_jobs": len(expected) - len(missing), "expected_jobs": len(expected), "missing_jobs": missing, "metric_rows": len(all_metrics)}
    projection = _write_pilot_projection(output, manifest)
    if projection is not None:
        status["resource_projection"] = projection
    (output / "aggregate_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "TRANSFER_REPORT.md").write_text(
        "# Task-transfer report\n\n"
        f"Status: **{'complete' if not missing else 'development/incomplete'}**  \n"
        f"Endpoint: `cognitive_counterfactual_recovery`  \nMetric: `global_cfr_v1`  \n"
        f"Completed work units: {len(expected) - len(missing)}/{len(expected)}.\n\n"
        f"Behavioral theory status: `{manifest['behavioral_theory_status']}`.  \n"
        f"Frozen survivor set: `{manifest['behavioral_survivor_set']}`.  \n"
        "All transfer cells are indexed by survivor theory; transfer results do not "
        "retroactively select a behavioral winner.\n\n"
        "Off-diagonal cells from a source that failed within-task Gate D are recorded as "
        "`unavailable`, not zero. Transfer-predictor inference is a separate Gate-E step and "
        "must not be fitted until this matrix and its predictor definitions are frozen.\n",
        encoding="utf-8",
    )
    return status


def gate_source_controllers(metrics: pd.DataFrame) -> pd.DataFrame:
    """Gate D: unavailable source controllers are never encoded as zero transfer."""

    diagonal = metrics[metrics.source_task == metrics.target_task].copy()
    diagonal["source_available"] = (
        np.isfinite(diagonal.global_cfr)
        & (diagonal.global_cfr > 0)
        & (diagonal.bootstrap_low > 0)
        & (diagonal.correlation > 0)
    )
    availability = diagonal[["model", "theory", "source_task", "source_available"]]
    result = metrics.merge(availability, on=["model", "theory", "source_task"], how="left", validate="many_to_one")
    invalid = ~result.source_available.fillna(False)
    off_diagonal = result.source_task != result.target_task
    result.loc[invalid & off_diagonal, ["global_cfr", "correlation", "slope", "bootstrap_low", "bootstrap_high", "random_p"]] = np.nan
    result.loc[invalid & off_diagonal, "status"] = "unavailable"
    result.loc[~(invalid & off_diagonal), "status"] = "available"
    return result


def explain_transfer(output: str | Path, predictor_table: str | Path, *, alpha: float = 10.0) -> dict:
    """Run Gate E only from a complete frozen matrix and preregistered columns."""

    output = Path(output)
    status = json.loads((output / "aggregate_status.json").read_text(encoding="utf-8"))
    if not status["complete"]:
        raise RuntimeError("transfer-prediction models require a complete frozen matrix")
    matrix_path = output / "single_task_transfer.csv"
    matrix = pd.read_csv(matrix_path)
    matrix = matrix[(matrix.source_task != matrix.target_task) & (matrix.status == "available")]
    features = pd.read_csv(predictor_table)
    keys = ["model", "theory", "source_task", "target_task"]
    combined = matrix.merge(features, on=keys, how="left", validate="one_to_one")
    predictions, coefficients = fit_heldout_target_predictor(combined, alpha=alpha)
    predictions.to_csv(output / "transfer_predictors.csv", index=False)
    coefficients.to_csv(output / "transfer_predictor_coefficients.csv", index=False)
    record = {
        "schema_version": "transfer-prediction-v1",
        "matrix_sha256": _sha(matrix_path),
        "predictor_table_sha256": _sha(Path(predictor_table)),
        "validation": "leave_one_target_task_out",
        "alpha": float(alpha),
        "rows": len(predictions),
    }
    (output / "transfer_prediction_provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record

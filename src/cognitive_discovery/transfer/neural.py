"""GPU task-specific DAS fitting/evaluation using the tested replication runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cognitive_discovery.causal_mechanistic import pipeline as causal_pipeline
from cognitive_discovery.causal_mechanistic.das import DASAlignment, load_alignment, save_alignment
from cognitive_discovery.causal_specificity.controls import orthonormal_random_subspaces
from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.replication.adapters import resolve_relative_layers
from cognitive_discovery.replication.neural import (
    THEORY_TARGET, _baseline_frame, _prediction_series, _record_map, _runner,
)
from cognitive_discovery.replication.workflow import _load_run, _sha256

from .config import load_transfer_config
from .metrics import summarize_transfer
from .splits import audit_transfer_leakage


def transfer_target_tasks(work: dict, scope: str, source_gate: bool | None) -> list[str]:
    """Resolve targets without allowing off-diagonal evaluation before Gate D."""

    targets = list(map(str, work["target_tasks"]))
    sources = set(map(str, work["source_tasks"]))
    if scope == "all":
        return targets
    if work.get("design") != "single" or len(sources) != 1:
        raise ValueError("diagonal/off-diagonal scopes require a single-task controller")
    if scope == "diagonal":
        return [target for target in targets if target in sources]
    if scope == "off_diagonal":
        if source_gate is None:
            raise RuntimeError("off-diagonal evaluation requires a diagonal source gate")
        return [target for target in targets if target not in sources] if source_gate else []
    raise ValueError(f"unknown transfer evaluation scope: {scope}")


def _merge_metric_components(job_root: Path) -> Path:
    paths = [
        job_root / "diagonal_metrics.csv",
        job_root / "off_diagonal_metrics.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.is_file()]
    if not frames:
        raise RuntimeError("no transfer metric components are available")
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(
        ["model", "theory", "source_task", "target_task"], keep="last"
    )
    path = job_root / "metrics.csv"
    result.to_csv(path, index=False)
    return path


def _context(replication_root: Path, transfer_root: Path, work_id: str):
    manifest = json.loads((transfer_root / "work_manifest.json").read_text(encoding="utf-8"))
    matches = [unit for unit in manifest["work_units"] if unit["work_id"] == work_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate transfer work id: {work_id}")
    config, _, _ = _load_run(replication_root)
    return manifest, matches[0], config


def train_controller(replication_root, transfer_root, work_id, transfer_config, *, online=False, smoke=False):
    """Fit/select only on source-task train/selection pairs; never target test."""

    import torch

    replication_root, transfer_root = Path(replication_root), Path(transfer_root)
    manifest, work, replication_config = _context(replication_root, transfer_root, work_id)
    settings = load_transfer_config(transfer_config)
    runner = _runner(replication_config, replication_root, online=online)
    layers = resolve_relative_layers(runner.layer_count, settings["search"]["relative_depths"])
    ranks = list(map(int, settings["search"]["ranks"]))
    if smoke:
        layers, ranks = layers[:1], ranks[:1]
    pairs = read_records(transfer_root / "pair_manifest.parquet")
    predictions = read_records(replication_root / "mechanism/counterfactual_predictions.parquet")
    _, records_by_id = _record_map(replication_root)
    theory = str(work["theory"])
    target_variable = THEORY_TARGET[theory]
    source_tasks = set(work["source_tasks"])
    eligible = pairs[(pairs.target_variable == target_variable) & pairs.task_family.astype(str).isin(source_tasks)]
    train = eligible[eligible.transfer_split == "transfer_train"]
    selection = eligible[eligible.transfer_split == "transfer_selection"]
    if train.empty or selection.empty:
        raise RuntimeError(f"{work_id} has no source-task train or selection pairs")
    # This audit supplies an empty target set at fitting time; target tests are
    # loaded only by evaluate_controller after the controller is immutable.
    audit = audit_transfer_leakage(
        pairs, source_tasks=source_tasks, train_ids=train.pair_id,
        selection_ids=selection.pair_id, test_ids=[],
    )
    needed = set(train.base_condition_id) | set(train.source_condition_id) | set(selection.base_condition_id) | set(selection.source_condition_id)
    cache = {condition_id: causal_pipeline._forward(runner, records_by_id[condition_id], capture_layers=layers) for condition_id in sorted(needed)}
    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)
    target = _prediction_series(predictions, theory=theory)
    job_root = transfer_root / "jobs" / work_id
    job_root.mkdir(parents=True, exist_ok=True)
    candidates = []
    for layer in layers:
        states = {condition_id: result.states[layer] for condition_id, result in cache.items()}
        hidden = len(next(iter(states.values())))
        for rank in ranks:
            alignment = DASAlignment(hidden, rank, seed=int(settings["seeds"]["controller_search"]), device=next(runner.model.parameters()).device)
            optimizer = torch.optim.Adam(alignment.parameters(), lr=float(settings["search"]["learning_rate"]))
            losses = []
            epochs = 1 if smoke else int(settings["search"]["epochs"])
            for epoch in range(epochs):
                order = train.sample(frac=1, random_state=int(settings["seeds"]["controller_search"]) + epoch)
                for pair in order.itertuples():
                    optimizer.zero_grad(set_to_none=True)
                    captured = {}
                    def editor(state, source=states[pair.source_condition_id]):
                        edited = alignment.edit(state, source)
                        captured["penalty"] = (edited - state).float().square().mean()
                        return edited
                    record = records_by_id[pair.base_condition_id]
                    trial = causal_pipeline._trial(record)
                    observed = runner.differentiable_persistence_logit(
                        list(trial.messages), record.condition.response_mapping.labels,
                        positive_label=record.condition.response_mapping.continue_label,
                        editors={layer: editor},
                    )
                    desired = cache[pair.base_condition_id].persistence_logit + float(target.loc[pair.pair_id])
                    loss = (observed.float() - desired) ** 2 + float(settings["search"]["activation_change_penalty"]) * captured["penalty"]
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            basis = alignment.numpy_basis()
            rows = causal_pipeline._evaluate_alignment(
                runner, basis, selection, records_by_id, states,
                _baseline_frame(selection, cache), target, layer=layer,
                split_label="transfer_selection", intervention_type=work_id,
            )
            frame = pd.DataFrame(rows)
            metric = summarize_transfer(
                frame.assign(contrast_id=frame.pair_id.map(selection.set_index("pair_id").contrast_id)),
                model=manifest["model"]["id"], theory=theory, source_task=work["source_id"],
                target_task="source_selection", layer=layer, rank=rank,
                bootstrap_samples=max(100, int(settings["bootstrap"]["development_samples"])),
                seed=int(settings["seeds"]["bootstrap"]),
            )
            candidate = job_root / "candidates" / f"L{layer}__rank{rank}.safetensors"
            save_alignment(candidate, basis, metadata={
                "work_id": work_id, "theory": theory, "source_tasks": sorted(source_tasks),
                "layer": layer, "rank": rank, "train_pair_ids": sorted(train.pair_id.astype(str)),
                "selection_pair_ids": sorted(selection.pair_id.astype(str)),
                "endpoint_id": settings["endpoint_id"], "metric_id": settings["metric_id"],
            })
            candidates.append({**metric, "candidate_path": candidate.relative_to(job_root).as_posix(), "candidate_sha256": _sha256(candidate), "mean_training_loss": float(np.mean(losses))})
    table = pd.DataFrame(candidates).sort_values(["global_cfr", "rank", "layer"], ascending=[False, True, True])
    table.to_csv(job_root / "selection_results.csv", index=False)
    winner = table.iloc[0].to_dict()
    basis = load_alignment(job_root / winner["candidate_path"])
    controller = save_alignment(job_root / "controller.safetensors", basis, metadata={
        "work_id": work_id, "theory": theory, "source_tasks": sorted(source_tasks),
        "layer": int(winner["layer"]), "rank": int(winner["rank"]),
        "selection_metric": "global_cfr_v1", "target_tests_touched": False,
    })
    selected = {
        "schema_version": "task-controller-v1", "work": work,
        "model": manifest["model"], "tokenizer": manifest["tokenizer"],
        "target_definition": f"{theory}:{target_variable}",
        "endpoint_id": settings["endpoint_id"], "metric_id": settings["metric_id"],
        "pair_manifest_sha256": manifest["pair_manifest_sha256"],
        "transfer_split_sha256": manifest["transfer_split_sha256"],
        "layer": int(winner["layer"]), "rank": int(winner["rank"]),
        "controller_path": controller.name, "controller_sha256": _sha256(controller),
        "n_train": len(train), "n_selection": len(selection),
        "train_pair_ids": sorted(train.pair_id.astype(str)),
        "selection_pair_ids": sorted(selection.pair_id.astype(str)),
        "seeds": settings["seeds"], "leakage_audit": audit,
        "target_tests_touched": False,
    }
    (job_root / "selected_controller.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selected


def evaluate_controller(
    replication_root,
    transfer_root,
    work_id,
    transfer_config,
    *,
    online=False,
    smoke=False,
    random_count=None,
    evaluation_scope="all",
):
    replication_root, transfer_root = Path(replication_root), Path(transfer_root)
    manifest, work, replication_config = _context(replication_root, transfer_root, work_id)
    settings = load_transfer_config(transfer_config)
    job_root = transfer_root / "jobs" / work_id
    selected = json.loads((job_root / "selected_controller.json").read_text(encoding="utf-8"))
    if selected.get("target_tests_touched") is not False:
        raise RuntimeError("controller selection did not preserve untouched target tests")
    runner = _runner(replication_config, replication_root, online=online)
    pairs = read_records(transfer_root / "pair_manifest.parquet")
    predictions = read_records(replication_root / "mechanism/counterfactual_predictions.parquet")
    _, records_by_id = _record_map(replication_root)
    theory, layer, rank = work["theory"], int(selected["layer"]), int(selected["rank"])
    target_variable = THEORY_TARGET[theory]
    diagonal_gate_path = job_root / "diagonal_gate.json"
    source_gate = None
    if diagonal_gate_path.is_file():
        source_gate = bool(
            json.loads(diagonal_gate_path.read_text(encoding="utf-8"))["source_available"]
        )
    target_tasks = transfer_target_tasks(work, evaluation_scope, source_gate)
    if not target_tasks:
        return {
            "work_id": work_id,
            "evaluation_scope": evaluation_scope,
            "status": "source_unavailable",
            "targets": 0,
            "pairs": 0,
            "random_subspaces": 0,
        }
    test = pairs[(pairs.target_variable == target_variable) & (pairs.transfer_split == "transfer_test") & pairs.task_family.astype(str).isin(set(target_tasks))]
    if test.empty:
        raise RuntimeError(f"{work_id} has no target-task test pairs")
    audit_transfer_leakage(
        pairs, source_tasks=work["source_tasks"],
        train_ids=pairs[(pairs.target_variable == target_variable) & pairs.task_family.astype(str).isin(set(work["source_tasks"])) & (pairs.transfer_split == "transfer_train")].pair_id,
        selection_ids=pairs[(pairs.target_variable == target_variable) & pairs.task_family.astype(str).isin(set(work["source_tasks"])) & (pairs.transfer_split == "transfer_selection")].pair_id,
        test_ids=test.pair_id,
    )
    needed = set(test.base_condition_id) | set(test.source_condition_id)
    cache = {condition_id: causal_pipeline._forward(runner, records_by_id[condition_id], capture_layers=[layer]) for condition_id in sorted(needed)}
    states = {condition_id: result.states[layer] for condition_id, result in cache.items()}
    target = _prediction_series(predictions, theory=theory)
    basis = load_alignment(job_root / selected["controller_path"])
    all_rows, metrics, null_rows = [], [], []
    count = int(random_count if random_count is not None else settings["random_controls"]["subspaces"])
    if smoke: count = min(count, 2)
    random_bases = orthonormal_random_subspaces(basis.shape[0], rank, count, seed=int(settings["seeds"]["random_controls"]))
    for target_task, part in test.groupby("task_family"):
        rows = causal_pipeline._evaluate_alignment(runner, basis, part, records_by_id, states, _baseline_frame(part, cache), target, layer=layer, split_label="transfer_test", intervention_type=work_id)
        frame = pd.DataFrame(rows).merge(part[["pair_id", "contrast_id"]], on="pair_id", validate="one_to_one")
        all_rows.extend(frame.to_dict("records"))
        random_cfrs = []
        for index, random_basis in enumerate(random_bases):
            random_frame = pd.DataFrame(causal_pipeline._evaluate_alignment(runner, random_basis, part, records_by_id, states, _baseline_frame(part, cache), target, layer=layer, split_label="transfer_test", intervention_type=f"random_{index}"))
            value = summarize_transfer(random_frame.assign(contrast_id=random_frame.pair_id.map(part.set_index("pair_id").contrast_id)), model=manifest["model"]["id"], theory=theory, source_task=work["source_id"], target_task=str(target_task), layer=layer, rank=rank, bootstrap_samples=max(100, int(settings["bootstrap"]["development_samples"])), seed=int(settings["seeds"]["bootstrap"]) + index)["global_cfr"]
            random_cfrs.append(value)
            null_rows.append({"work_id": work_id, "target_task": target_task, "random_index": index, "global_cfr": value})
        frame.attrs["n_train"] = selected["n_train"]
        metrics.append(summarize_transfer(frame, model=manifest["model"]["id"], theory=theory, source_task=work["source_id"], target_task=str(target_task), layer=layer, rank=rank, bootstrap_samples=int(settings["bootstrap"]["samples"]), seed=int(settings["seeds"]["bootstrap"]), random_cfrs=random_cfrs))
    artifact_scope = (
        evaluation_scope if evaluation_scope in {"diagonal", "off_diagonal"} else "all"
    )
    Path(
        write_records(
            all_rows, job_root / f"interventions_{artifact_scope}.parquet"
        )
    )
    pd.DataFrame(null_rows).to_csv(
        job_root / f"random_subspace_null_{artifact_scope}.csv", index=False
    )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame["status"] = "available"
    component = (
        job_root / "diagonal_metrics.csv"
        if evaluation_scope == "diagonal"
        else job_root / "off_diagonal_metrics.csv"
        if evaluation_scope == "off_diagonal"
        else job_root / "off_diagonal_metrics.csv"
    )
    metrics_frame.to_csv(component, index=False)
    _merge_metric_components(job_root)
    if evaluation_scope == "diagonal":
        if len(metrics_frame) != 1:
            raise RuntimeError("diagonal evaluation must emit exactly one source-task row")
        row = metrics_frame.iloc[0]
        source_available = bool(
            np.isfinite(row.global_cfr)
            and row.global_cfr > 0
            and row.bootstrap_low > 0
            and row.correlation > 0
        )
        (job_root / "diagonal_gate.json").write_text(
            json.dumps(
                {
                    "schema_version": "task-transfer-diagonal-gate-v1",
                    "work_id": work_id,
                    "source_task": work["source_id"],
                    "source_available": source_available,
                    "criteria": {
                        "global_cfr_positive": bool(row.global_cfr > 0),
                        "bootstrap_lower_bound_positive": bool(row.bootstrap_low > 0),
                        "correlation_positive": bool(row.correlation > 0),
                    },
                    "endpoint_id": settings["endpoint_id"],
                    "metric_id": settings["metric_id"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    evaluation_record = {
        "schema_version": "task-controller-evaluation-v1",
        "work_id": work_id,
        "evaluation_scope": artifact_scope,
        "controller_sha256": selected["controller_sha256"],
        "selection_record_remained_immutable": True,
        "evaluated_target_tasks": sorted(map(str, test.task_family.unique())),
        "endpoint_id": settings["endpoint_id"],
        "metric_id": settings["metric_id"],
        "pairs": len(all_rows),
        "random_subspaces": count,
    }
    (job_root / f"evaluation_{artifact_scope}.json").write_text(
        json.dumps(evaluation_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "work_id": work_id,
        "evaluation_scope": evaluation_scope,
        "targets": len(metrics),
        "pairs": len(all_rows),
        "random_subspaces": count,
    }

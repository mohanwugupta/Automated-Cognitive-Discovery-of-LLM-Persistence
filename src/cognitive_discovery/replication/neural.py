"""Model-agnostic counterfactual generation, DAS search, and specificity tests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from cognitive_discovery.causal_mechanistic import pipeline as causal_pipeline
from cognitive_discovery.causal_mechanistic.counterfactuals import (
    FrozenTheoryBank,
    build_counterfactual_pairs,
)
from cognitive_discovery.causal_mechanistic.das import (
    DASAlignment,
    load_alignment,
    save_alignment,
)
from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics
from cognitive_discovery.causal_specificity.controls import (
    orthonormal_random_subspaces,
    shuffle_sources,
    shuffle_targets,
)
from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.design.counterbalance import counterbalanced_mappings
from cognitive_discovery.mechanistic.dataset.matched_conditions import (
    MatchedMechanisticCondition,
    _context,
    _history,
    load_mechanistic_manifest,
    validate_matched_conditions,
)
from cognitive_discovery.mechanistic.targets.behavioral_targets import (
    compute_condition_targets,
)
from cognitive_discovery.pipeline import generate_design
from cognitive_discovery.reproducibility.metrics import global_cfr_v1

from .adapters import adapter_registry, resolve_relative_layers
from .controls import audit_informative_target_shuffle, require_cognitive_endpoint
from .provenance import validate_replication_provenance, write_replication_provenance
from .splits import assign_neural_splits, split_manifest_hash
from .state import ReplicationRunState
from .workflow import _interface_messages, _json, _load_run, _ontology_config, _sha256


THEORY_TARGET = {
    "outcome_history": "outcome_history",
    "dual_history": "outcome_history",
    "latent_context": "contextual_outcome_history",
}


def _records_for_counterfactuals(root: Path, output: Path, config: dict):
    ontology = _ontology_config(root, config)
    seed = int(config["seeds"]["counterfactual_design"])
    design, _ = generate_design(
        ontology,
        output=output / "mechanism/design_source",
        conditions=int(config["mechanism"].get("background_conditions", 490)),
        seed=seed,
        design_id="model_agnostic_replication_counterfactuals",
    )
    patterns = [
        ((-1, -1, -1), (1, -1, -1)),
        ((1, 1, 1), (-1, -1, 1)),
        ((-1, -1, -1, -1, -1), (1, 1, -1, -1, -1)),
        ((1, 1, 1, 1, 1), (-1, -1, -1, 1, 1)),
    ]
    holdout = set(config["mechanism"]["holdout_tasks"])
    records = []
    for task in sorted({condition.task_family for condition in design}):
        candidates = sorted(
            [
                condition
                for condition in design
                if condition.task_family == task
                and condition.response_mapping.mapping_id == "continue_x"
            ],
            key=lambda condition: condition.condition_id,
        )
        backgrounds, seen = [], set()
        for condition in candidates:
            key = json.dumps(condition.semantic_factors, sort_keys=True)
            if key not in seen:
                backgrounds.append(condition)
                seen.add(key)
            if len(backgrounds) == int(config["mechanism"].get("backgrounds_per_task", 4)):
                break
        if len(backgrounds) < 3:
            raise RuntimeError(f"insufficient distinct mechanistic backgrounds for {task}")
        for background_index, template in enumerate(backgrounds):
            for pattern_index, (left, right) in enumerate(patterns):
                actions = tuple(
                    ("continue", "disengage", "continue", "continue", "disengage")[
                        : len(left)
                    ]
                )
                families = ["outcome_history"]
                if task in {"bandit", "foraging", "debugging"}:
                    families.append("contextual_history")
                for family in families:
                    contrast_id = (
                        f"replication:{task}:bg{background_index}:pattern{pattern_index}:{family}"
                    )
                    for member, outcomes in ((-1, left), (1, right)):
                        if family == "outcome_history":
                            history = _history(outcomes, actions)
                            context = None
                            target = "outcome_history"
                        else:
                            history = _history(left, actions)
                            context = _context(
                                contrast_id=contrast_id,
                                a_history=_history(right, actions),
                                b_history=history,
                                context_return="A" if member == 1 else "B",
                            )
                            target = "contextual_outcome_history"
                        pair_member_id = f"{contrast_id}:member{member}"
                        for mapping_index, mapping in enumerate(
                            counterbalanced_mappings(("X", "Y"))
                        ):
                            condition = dataclasses.replace(
                                template,
                                design_id="model_agnostic_replication_counterfactuals",
                                condition_id=f"{pair_member_id}-m{mapping_index}",
                                paired_condition_id=pair_member_id,
                                response_mapping=mapping,
                                history=history,
                                contextual_history=context,
                                split="transfer" if task in holdout else "test",
                                sampling_strategy=family,
                            )
                            records.append(
                                MatchedMechanisticCondition(
                                    condition,
                                    contrast_id,
                                    family,
                                    member,
                                    target,
                                    compute_condition_targets(condition),
                                )
                            )
    validate_matched_conditions(records)
    return records


def execute_counterfactual_stage(root: Path, output: Path) -> dict:
    config, state, provenance = _load_run(output)
    require_cognitive_endpoint(config["endpoint_id"])
    validate_replication_provenance(provenance, required_for_stage="counterfactuals")
    state.start("counterfactuals")
    state.write(output / "run_state.json")
    records = _records_for_counterfactuals(root, output, config)
    mechanism_root = output / "mechanism"
    manifest = mechanism_root / "condition_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    frozen_root = output / "behavior/models/frozen_models"
    bank = FrozenTheoryBank(frozen_root)
    pairs, predictions = build_counterfactual_pairs(records, bank)
    compatible = {
        theory: THEORY_TARGET[theory]
        for theory in bank.architectures
        if theory in THEORY_TARGET
    }
    if len(compatible) != len(bank.architectures):
        missing = sorted(set(bank.architectures) - set(compatible))
        raise RuntimeError(f"frozen theories lack well-defined counterfactual targets: {missing}")
    allowed_pairs = set(
        pairs.loc[pairs.target_variable.isin(set(compatible.values())), "pair_id"]
    )
    pairs = pairs[pairs.pair_id.isin(allowed_pairs)].copy()
    predictions = predictions[
        predictions.apply(
            lambda row: compatible.get(row.theory) == row.target_variable, axis=1
        )
    ].copy()
    pairs["neural_group_id"] = pairs.target_variable.astype(str) + ":" + pairs.contrast_id.astype(str)
    pairs = assign_neural_splits(
        pairs,
        group_column="neural_group_id",
        task_column="task_family",
        fitting_tasks=set(config["mechanism"]["fitting_tasks"]),
        holdout_tasks=set(config["mechanism"]["holdout_tasks"]),
        seed=int(config["seeds"]["neural_split"]),
        fractions=tuple(config["mechanism"]["neural_split_fractions"]),
    )
    pairs["pair_split"] = pairs["neural_split"]
    split_lookup = dict(zip(pairs.pair_id, pairs.pair_split))
    predictions["pair_split"] = predictions.pair_id.map(split_lookup)
    if predictions.pair_split.isna().any():
        raise RuntimeError("counterfactual predictions and pair manifest are misaligned")
    pair_path = write_records(pairs.to_dict("records"), mechanism_root / "pair_manifest.parquet")
    prediction_path = write_records(
        predictions.to_dict("records"), mechanism_root / "counterfactual_predictions.parquet"
    )
    pair_hash = _sha256(Path(pair_path))
    neural_hash = split_manifest_hash(pairs, "neural_group_id", "neural_split")
    split_record = {
        "schema_version": "neural-split-v1",
        "seed": config["seeds"]["neural_split"],
        "sha256": neural_hash,
        "counts": pairs.groupby("neural_split").size().to_dict(),
        "fitting_tasks": config["mechanism"]["fitting_tasks"],
        "holdout_tasks": config["mechanism"]["holdout_tasks"],
        "frozen_before_das": True,
    }
    _json(mechanism_root / "neural_split_manifest.json", split_record)
    provenance["counterfactual_pair_manifest_hash"] = pair_hash
    provenance["neural_split_hash"] = neural_hash
    provenance["counterfactual_prediction_hash"] = _sha256(Path(prediction_path))
    write_replication_provenance(output / "provenance.json", provenance)
    state.complete(
        "counterfactuals",
        outcome="pass",
        artifacts={
            "condition_manifest": _sha256(manifest),
            "pair_manifest": pair_hash,
            "predictions": _sha256(Path(prediction_path)),
            "neural_split": _sha256(mechanism_root / "neural_split_manifest.json"),
        },
    )
    state.write(output / "run_state.json")
    return {
        "pairs": len(pairs),
        "predictions": len(predictions),
        "pair_manifest_hash": pair_hash,
        "neural_split_hash": neural_hash,
    }


class InterfaceMechanisticRunner:
    """Translate the selected interface while preserving the shared runner API."""

    def __init__(self, runner, interface):
        self.runner = runner
        self.interface = interface
        self.model = runner.model

    @property
    def layer_count(self):
        return self.runner.layer_count

    def _arguments(self, messages, labels, positive_label):
        actual, candidate_labels, mapping = _interface_messages(messages, labels, self.interface)
        return actual, candidate_labels, mapping[positive_label]

    def forward(self, messages, labels, *, positive_label, **kwargs):
        actual, candidate_labels, candidate_positive = self._arguments(
            messages, labels, positive_label
        )
        return self.runner.forward(
            actual, candidate_labels, positive_label=candidate_positive, **kwargs
        )

    def differentiable_persistence_logit(
        self, messages, labels, *, positive_label, **kwargs
    ):
        actual, candidate_labels, candidate_positive = self._arguments(
            messages, labels, positive_label
        )
        return self.runner.differentiable_persistence_logit(
            actual, candidate_labels, positive_label=candidate_positive, **kwargs
        )

    def choice_output_direction(self, messages, labels, *, positive_label):
        actual, candidate_labels, candidate_positive = self._arguments(
            messages, labels, positive_label
        )
        return self.runner.choice_output_direction(
            actual, candidate_labels, positive_label=candidate_positive
        )


def _runner(config, output: Path, *, online: bool):
    model = config["model"]
    adapter = adapter_registry.create(
        model["adapter"],
        model_id=model["id"],
        revision=model["revision"],
        tokenizer_id=model["tokenizer_id"],
        tokenizer_revision=model["tokenizer_revision"],
        local_files_only=not online,
    ).load()
    selected = json.loads((output / "interface/selected_interface.json").read_text())
    return InterfaceMechanisticRunner(adapter._runner(), selected)


def _record_map(output: Path):
    records = load_mechanistic_manifest(output / "mechanism/condition_manifest.jsonl")
    return records, {record.condition.condition_id: record for record in records}


def _prediction_series(predictions, *, theory: str):
    value = predictions[predictions.theory == theory].set_index("pair_id")
    if value.index.duplicated().any():
        raise RuntimeError(f"duplicate counterfactual prediction for theory {theory}")
    return value.predicted_counterfactual_effect


def _baseline_frame(pairs, cache):
    return pd.DataFrame(
        [
            {
                "pair_id": row.pair_id,
                "base_persistence_logit": cache[row.base_condition_id].persistence_logit,
            }
            for row in pairs.itertuples()
        ]
    ).set_index("pair_id")


def execute_mechanism_stage(
    root: Path, output: Path, *, online: bool = False
) -> dict:
    del root
    import torch

    config, state, provenance = _load_run(output)
    require_cognitive_endpoint(config["endpoint_id"])
    validate_replication_provenance(provenance, required_for_stage="mechanism")
    state.start("mechanism")
    state.write(output / "run_state.json")
    runner = _runner(config, output, online=online)
    layers = resolve_relative_layers(runner.layer_count, config["mechanism"]["relative_depths"])
    ranks = list(map(int, config["mechanism"]["ranks"]))
    search_grid = {
        "model_layer_count": runner.layer_count,
        "relative_depths": config["mechanism"]["relative_depths"],
        "resolved_layers": layers,
        "ranks": ranks,
        "selection_split": "neural_selection",
        "untouched_splits": ["neural_test", "neural_task_holdout"],
    }
    _json(output / "mechanism/search_grid.json", search_grid)
    pairs = read_records(output / "mechanism/pair_manifest.parquet")
    predictions = read_records(output / "mechanism/counterfactual_predictions.parquet")
    _, records_by_id = _record_map(output)
    learning_pairs = pairs[pairs.neural_split.isin(["neural_train", "neural_selection"])]
    needed = set(learning_pairs.base_condition_id) | set(learning_pairs.source_condition_id)
    cache = {}
    for condition_id in sorted(needed):
        cache[condition_id] = causal_pipeline._forward(
            runner, records_by_id[condition_id], capture_layers=layers
        )
    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)
    selection_rows = []
    train_rows = []
    controller_candidates = {}
    selected_theories = json.loads(
        (output / "behavior/models/selected_models.json").read_text()
    )["selected_models"]
    for theory in selected_theories:
        target_variable = THEORY_TARGET[theory]
        train = pairs[
            (pairs.target_variable == target_variable)
            & (pairs.neural_split == "neural_train")
        ]
        selection = pairs[
            (pairs.target_variable == target_variable)
            & (pairs.neural_split == "neural_selection")
        ]
        if train.empty or selection.empty:
            raise RuntimeError(f"theory {theory} lacks train or selection counterfactual pairs")
        target = _prediction_series(predictions, theory=theory)
        for layer in layers:
            states = {condition_id: result.states[layer] for condition_id, result in cache.items()}
            hidden_size = len(next(iter(states.values())))
            for rank in ranks:
                if rank > hidden_size:
                    raise RuntimeError(f"rank {rank} exceeds hidden size {hidden_size}")
                alignment = DASAlignment(
                    hidden_size,
                    rank,
                    seed=int(config["seeds"]["mechanism_search"]),
                    device=next(runner.model.parameters()).device,
                )
                optimizer = torch.optim.Adam(
                    alignment.parameters(), lr=float(config["mechanism"]["learning_rate"])
                )
                losses = []
                for epoch in range(int(config["mechanism"]["epochs"])):
                    order = train.sample(
                        frac=1,
                        random_state=int(config["seeds"]["mechanism_search"]) + epoch,
                    )
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
                            list(trial.messages),
                            record.condition.response_mapping.labels,
                            positive_label=record.condition.response_mapping.continue_label,
                            editors={layer: editor},
                        )
                        desired = (
                            cache[pair.base_condition_id].persistence_logit
                            + float(target.loc[pair.pair_id])
                        )
                        loss = (observed.float() - desired) ** 2 + float(
                            config["mechanism"]["activation_change_penalty"]
                        ) * captured["penalty"]
                        loss.backward()
                        if not torch.isfinite(alignment.parameter.grad).all():
                            raise RuntimeError("non-finite DAS gradient")
                        optimizer.step()
                        losses.append(float(loss.detach().cpu()))
                basis = alignment.numpy_basis()
                candidate_id = f"{theory}__L{layer}__rank{rank}"
                basis_path = output / "mechanism/candidates" / f"{candidate_id}.safetensors"
                save_alignment(
                    basis_path,
                    basis,
                    metadata={
                        "theory": theory,
                        "layer": layer,
                        "rank": rank,
                        "train_pair_ids": train.pair_id.tolist(),
                        "endpoint_id": config["endpoint_id"],
                    },
                )
                rows = causal_pipeline._evaluate_alignment(
                    runner,
                    basis,
                    selection,
                    records_by_id,
                    states,
                    _baseline_frame(selection, cache),
                    target,
                    layer=layer,
                    split_label="neural_selection",
                    intervention_type=candidate_id,
                )
                frame = pd.DataFrame(rows)
                metric = counterfactual_metrics(
                    frame.predicted_counterfactual_effect,
                    frame.neural_counterfactual_effect,
                )
                metric["global_cfr"] = global_cfr_v1(
                    frame.predicted_counterfactual_effect,
                    frame.neural_counterfactual_effect,
                )
                selection_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "theory": theory,
                        "target_variable": target_variable,
                        "layer": layer,
                        "relative_depth": config["mechanism"]["relative_depths"][
                            layers.index(layer)
                        ],
                        "rank": rank,
                        "training_loss": float(np.mean(losses)),
                        "basis_path": basis_path.relative_to(output).as_posix(),
                        "basis_sha256": _sha256(basis_path),
                        **metric,
                    }
                )
                train_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "theory": theory,
                        "target_variable": target_variable,
                        "layer": layer,
                        "rank": rank,
                        "training_split": "neural_train",
                        "training_pairs": len(train),
                        "epochs": int(config["mechanism"]["epochs"]),
                        "mean_training_loss": float(np.mean(losses)),
                        "final_training_loss": float(losses[-1]),
                        "test_splits_touched": False,
                    }
                )
                controller_candidates[candidate_id] = basis_path
                pd.DataFrame(selection_rows).to_csv(
                    output / "mechanism/selection_results.csv", index=False
                )
    table = pd.DataFrame(selection_rows)
    train_path = Path(
        write_records(train_rows, output / "mechanism/train_results.parquet")
    )
    selected = []
    for theory, group in table.groupby("theory"):
        winner = group.sort_values(
            ["global_cfr", "rank", "layer"], ascending=[False, True, True]
        ).iloc[0].to_dict()
        source = controller_candidates[winner["candidate_id"]]
        destination = output / "mechanism/controllers" / f"{theory}.safetensors"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        winner["controller_path"] = destination.relative_to(output).as_posix()
        winner["controller_sha256"] = _sha256(destination)
        selected.append(winner)
    registry = {
        "schema_version": "selected-controller-v1",
        "endpoint_id": config["endpoint_id"],
        "metric_id": config["metric_id"],
        "controllers": selected,
        "test_splits_touched": False,
    }
    registry_path = _json(output / "mechanism/selected_controller.json", registry)
    provenance["selected_controller_hash"] = _sha256(registry_path)
    provenance["selected_controller_artifact_hashes"] = {
        row["theory"]: row["controller_sha256"] for row in selected
    }
    provenance["resolved_layers"] = layers
    write_replication_provenance(output / "provenance.json", provenance)
    state.complete(
        "mechanism",
        outcome="selected",
        artifacts={
            "search_grid": _sha256(output / "mechanism/search_grid.json"),
            "selection_results": _sha256(output / "mechanism/selection_results.csv"),
            "train_results": _sha256(train_path),
            "selected_controller": _sha256(registry_path),
        },
    )
    state.write(output / "run_state.json")
    return registry


def _bootstrap_interval(target, observed, *, groups, samples: int, seed: int):
    target = np.asarray(target, dtype=float)
    observed = np.asarray(observed, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if not (target.shape == observed.shape == groups.shape):
        raise ValueError("bootstrap target, observation, and group arrays must align")
    rng = np.random.default_rng(seed)
    values = []
    unique = np.unique(groups)
    for _ in range(int(samples)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == value) for value in sampled])
        values.append(global_cfr_v1(target[indices], observed[indices]))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def execute_generalization_stage(
    root: Path, output: Path, *, online: bool = False
) -> dict:
    del root
    config, state, provenance = _load_run(output)
    require_cognitive_endpoint(config["endpoint_id"])
    validate_replication_provenance(provenance, required_for_stage="generalization")
    state.start("generalization")
    state.write(output / "run_state.json")
    runner = _runner(config, output, online=online)
    pairs = read_records(output / "mechanism/pair_manifest.parquet")
    predictions = read_records(output / "mechanism/counterfactual_predictions.parquet")
    _, records_by_id = _record_map(output)
    registry = json.loads((output / "mechanism/selected_controller.json").read_text())
    layers = sorted({int(controller["layer"]) for controller in registry["controllers"]})
    evaluation_pairs = pairs[pairs.neural_split.isin(["neural_test", "neural_task_holdout"])]
    needed = set(evaluation_pairs.base_condition_id) | set(evaluation_pairs.source_condition_id)
    cache = {
        condition_id: causal_pipeline._forward(
            runner, records_by_id[condition_id], capture_layers=layers
        )
        for condition_id in sorted(needed)
    }
    all_rows = []
    metric_rows = []
    for controller in registry["controllers"]:
        theory = controller["theory"]
        layer = int(controller["layer"])
        basis = load_alignment(output / controller["controller_path"])
        selected_pairs = evaluation_pairs[
            evaluation_pairs.target_variable == controller["target_variable"]
        ]
        target = _prediction_series(predictions, theory=theory)
        states = {condition_id: value.states[layer] for condition_id, value in cache.items()}
        for split, part in selected_pairs.groupby("neural_split"):
            rows = causal_pipeline._evaluate_alignment(
                runner,
                basis,
                part,
                records_by_id,
                states,
                _baseline_frame(part, cache),
                target,
                layer=layer,
                split_label=split,
                intervention_type="selected",
            )
            frame = pd.DataFrame(rows)
            frame = frame.merge(
                part[["pair_id", "contrast_id"]],
                on="pair_id",
                how="left",
                validate="one_to_one",
            )
            frame["theory"] = theory
            frame["method"] = "selected"
            all_rows.extend(frame.to_dict("records"))
            metric = counterfactual_metrics(
                frame.predicted_counterfactual_effect,
                frame.neural_counterfactual_effect,
            )
            metric["global_cfr"] = global_cfr_v1(
                frame.predicted_counterfactual_effect,
                frame.neural_counterfactual_effect,
            )
            interval = _bootstrap_interval(
                frame.predicted_counterfactual_effect,
                frame.neural_counterfactual_effect,
                groups=frame.contrast_id,
                samples=int(config["mechanism"].get("bootstrap_samples", 2000)),
                seed=int(config["seeds"]["bootstrap"]),
            )
            metric_rows.append(
                {
                    "theory": theory,
                    "split": split,
                    "endpoint_id": config["endpoint_id"],
                    "metric_id": config["metric_id"],
                    "cfr_ci_low": interval[0],
                    "cfr_ci_high": interval[1],
                    **metric,
                }
            )
            for task, task_frame in frame.groupby("task_family"):
                task_metric = counterfactual_metrics(
                    task_frame.predicted_counterfactual_effect,
                    task_frame.neural_counterfactual_effect,
                )
                task_metric["global_cfr"] = global_cfr_v1(
                    task_frame.predicted_counterfactual_effect,
                    task_frame.neural_counterfactual_effect,
                )
                metric_rows.append(
                    {
                        "theory": theory,
                        "split": split,
                        "task_family": task,
                        "endpoint_id": config["endpoint_id"],
                        "metric_id": config["metric_id"],
                        **task_metric,
                    }
                )
    intervention_path = write_records(all_rows, output / "mechanism/interventions.parquet")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "mechanism/generalization_metrics.csv", index=False)
    aggregate = metrics[metrics.task_family.isna()] if "task_family" in metrics else metrics
    gate_rows = []
    for row in aggregate.itertuples():
        gate_rows.append(
            {
                "theory": row.theory,
                "split": row.split,
                "positive_global_cfr": bool(row.global_cfr > 0),
                "bootstrap_lower_bound_positive": bool(row.cfr_ci_low > 0),
                "positive_target_correlation": bool(row.correlation > 0),
            }
        )
    gates = {"endpoint_id": config["endpoint_id"], "splits": gate_rows}
    _json(output / "mechanism/generalization_gates.json", gates)
    expected = {
        (controller["theory"], split)
        for controller in registry["controllers"]
        for split in ("neural_test", "neural_task_holdout")
    }
    observed = {(row["theory"], row["split"]) for row in gate_rows}
    passed = bool(gate_rows) and observed == expected and all(
        row["positive_global_cfr"]
        and row["bootstrap_lower_bound_positive"]
        and row["positive_target_correlation"]
        for row in gate_rows
    )
    state.complete(
        "generalization",
        outcome="pass" if passed else "partial_or_fail",
        artifacts={
            "interventions": _sha256(Path(intervention_path)),
            "metrics": _sha256(output / "mechanism/generalization_metrics.csv"),
            "gates": _sha256(output / "mechanism/generalization_gates.json"),
        },
    )
    state.write(output / "run_state.json")
    return gates


def execute_specificity_stage(
    root: Path, output: Path, *, online: bool = False
) -> dict:
    del root
    config, state, provenance = _load_run(output)
    require_cognitive_endpoint(config["specificity"]["endpoint_id"])
    validate_replication_provenance(provenance, required_for_stage="specificity")
    state.start("specificity")
    state.write(output / "run_state.json")
    runner = _runner(config, output, online=online)
    pairs = read_records(output / "mechanism/pair_manifest.parquet")
    predictions = read_records(output / "mechanism/counterfactual_predictions.parquet")
    interventions = read_records(output / "mechanism/interventions.parquet")
    _, records_by_id = _record_map(output)
    registry = json.loads((output / "mechanism/selected_controller.json").read_text())
    seed = int(config["seeds"]["random_controls"])
    random_count = int(config["specificity"]["random_subspaces_final"])
    summary_rows, effect_rows, random_rows, shuffle_audits = [], [], [], []

    def completed_basis(columns, dimension, rank, *, local_seed):
        values = np.asarray(columns, dtype=float).reshape(dimension, -1)
        rng = np.random.default_rng(local_seed)
        values = np.column_stack((values, rng.normal(size=(dimension, rank))))
        basis = np.linalg.qr(values, mode="reduced")[0][:, :rank]
        if basis.shape != (dimension, rank):
            raise RuntimeError("control direction could not be completed to matched rank")
        return basis

    for controller_index, controller in enumerate(registry["controllers"]):
        theory = str(controller["theory"])
        layer, rank = int(controller["layer"]), int(controller["rank"])
        target_variable = str(controller["target_variable"])
        theory_target = _prediction_series(predictions, theory=theory)
        train_pairs = pairs[
            (pairs.target_variable == target_variable)
            & (pairs.neural_split == "neural_train")
        ].copy()
        evaluation_pairs = pairs[
            (pairs.target_variable == target_variable)
            & pairs.neural_split.isin(["neural_test", "neural_task_holdout"])
        ].copy()
        condition_ids = (
            set(train_pairs.base_condition_id)
            | set(train_pairs.source_condition_id)
            | set(evaluation_pairs.base_condition_id)
            | set(evaluation_pairs.source_condition_id)
        )
        cache = {
            condition_id: causal_pipeline._forward(
                runner, records_by_id[condition_id], capture_layers=(layer,)
            )
            for condition_id in sorted(condition_ids)
        }
        states = {condition_id: result.states[layer] for condition_id, result in cache.items()}
        hidden_size = len(next(iter(states.values())))
        baseline = _baseline_frame(evaluation_pairs, cache)
        selected = interventions[
            (interventions.theory == theory)
            & interventions.pair_id.isin(evaluation_pairs.pair_id)
            & (interventions.method == "selected")
        ].copy()
        matched_norms = selected.set_index("pair_id").intervention_norm.to_dict()
        if set(matched_norms) != set(evaluation_pairs.pair_id):
            raise RuntimeError("specificity rows do not match selected-controller evaluation rows")

        train_matrix = np.vstack([states[value] for value in train_pairs.base_condition_id])
        centered = train_matrix - train_matrix.mean(axis=0, keepdims=True)
        baseline_logits = np.asarray(
            [cache[value].persistence_logit for value in train_pairs.base_condition_id],
            dtype=float,
        )
        ridge = Ridge(alpha=1.0).fit(centered, baseline_logits)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        first = records_by_id[train_pairs.iloc[0].base_condition_id]
        trial = causal_pipeline._trial(first)
        output_direction = runner.choice_output_direction(
            list(trial.messages),
            first.condition.response_mapping.labels,
            positive_label=first.condition.response_mapping.continue_label,
        )
        named_bases = {
            "output_readout_direction": completed_basis(
                output_direction, hidden_size, rank, local_seed=seed + 101 * controller_index
            ),
            "predictive_ridge_direction": completed_basis(
                ridge.coef_, hidden_size, rank, local_seed=seed + 103 * controller_index
            ),
            "pca_variance_subspace": completed_basis(
                right[:rank].T, hidden_size, rank, local_seed=seed + 107 * controller_index
            ),
        }

        control_frames = {}
        for name, basis in named_bases.items():
            rows = causal_pipeline._evaluate_alignment(
                runner,
                basis,
                evaluation_pairs,
                records_by_id,
                states,
                baseline,
                theory_target,
                layer=layer,
                split_label="prospective_heldout",
                intervention_type=name,
                matched_norms=matched_norms,
            )
            control_frames[name] = pd.DataFrame(rows)
            effect_rows.extend({**row, "theory": theory} for row in rows)

        shuffled_input = evaluation_pairs.copy()
        shuffled_input["current_state_score"] = [
            cache[value].persistence_logit for value in shuffled_input.base_condition_id
        ]
        try:
            shuffled_pairs = shuffle_sources(
                shuffled_input, seed=seed + 1009 * (controller_index + 1)
            )
            donor_targets = theory_target.reindex(
                shuffled_pairs.shuffled_source_pair_id
            ).to_numpy(dtype=float)
            original_targets = theory_target.reindex(shuffled_pairs.pair_id).to_numpy(dtype=float)
            source_changed = np.abs(donor_targets - original_targets) > 1e-12
            source_status = (
                "informative_control" if source_changed.any() else "uninformative_control"
            )
            if source_status == "informative_control":
                rows = causal_pipeline._evaluate_alignment(
                    runner,
                    load_alignment(output / controller["controller_path"]),
                    shuffled_pairs,
                    records_by_id,
                    states,
                    baseline,
                    theory_target,
                    layer=layer,
                    split_label="prospective_heldout",
                    intervention_type="shuffled_source_base",
                    matched_norms=matched_norms,
                )
                control_frames["shuffled_source_base"] = pd.DataFrame(rows)
                effect_rows.extend({**row, "theory": theory} for row in rows)
        except ValueError as error:
            source_status = "uninformative_control"
            source_changed = np.zeros(len(evaluation_pairs), dtype=bool)
            shuffle_audits.append(
                {"theory": theory, "control": "shuffled_source_base", "reason": str(error)}
            )

        target_frame = evaluation_pairs[
            ["pair_id", "contrast_id", "task_family"]
        ].copy()
        target_frame["predicted_counterfactual_effect"] = target_frame.pair_id.map(
            theory_target
        )
        try:
            shuffled_target = shuffle_targets(
                target_frame,
                seed=seed + 2003 * (controller_index + 1),
            )
            audit = audit_informative_target_shuffle(
                shuffled_target.rename(
                    columns={
                        "original_predicted_counterfactual_effect": "target",
                        "predicted_counterfactual_effect": "shuffled_target",
                    }
                )[["target", "shuffled_target"]]
            )
        except ValueError as error:
            audit = audit_informative_target_shuffle(
                pd.DataFrame({"target": [0.0], "shuffled_target": [0.0]})
            )
            shuffle_audits.append(
                {"theory": theory, "control": "shuffled_target", "reason": str(error)}
            )
        shuffle_audits.extend(
            [
                {
                    "theory": theory,
                    "control": "shuffled_source_base",
                    "status": source_status,
                    "changed_fraction": float(np.mean(source_changed)),
                },
                {"theory": theory, "control": "shuffled_target", **dataclasses.asdict(audit)},
            ]
        )

        random_bases = orthonormal_random_subspaces(
            hidden_size,
            rank,
            random_count,
            seed=seed + 10007 * (controller_index + 1),
        )
        random_by_split = {
            split: [] for split in sorted(evaluation_pairs.neural_split.unique())
        }
        for random_index, basis in enumerate(random_bases):
            rows = causal_pipeline._evaluate_alignment(
                runner,
                basis,
                evaluation_pairs,
                records_by_id,
                states,
                baseline,
                theory_target,
                layer=layer,
                split_label="prospective_heldout",
                intervention_type="matched_random_subspace",
                matched_norms=matched_norms,
            )
            frame = pd.DataFrame(rows).merge(
                evaluation_pairs[["pair_id", "neural_split"]], on="pair_id", validate="one_to_one"
            )
            for split, part in frame.groupby("neural_split"):
                value = global_cfr_v1(
                    part.predicted_counterfactual_effect,
                    part.neural_counterfactual_effect,
                )
                random_by_split[split].append(value)
                random_rows.append(
                    {
                        "theory": theory,
                        "split": split,
                        "random_index": random_index,
                        "global_cfr": value,
                        "rank": rank,
                        "layer": layer,
                    }
                )

        for split, selected_part in selected.groupby("pair_split"):
            selected_cfr = global_cfr_v1(
                selected_part.predicted_counterfactual_effect,
                selected_part.neural_counterfactual_effect,
            )
            null = random_by_split[split]
            named = {}
            for name, frame in control_frames.items():
                ids = set(evaluation_pairs.loc[evaluation_pairs.neural_split == split, "pair_id"])
                part = frame[frame.pair_id.isin(ids)]
                if not part.empty:
                    named[name] = global_cfr_v1(
                        part.predicted_counterfactual_effect,
                        part.neural_counterfactual_effect,
                    )
            mapping_values = {
                str(mapping): global_cfr_v1(
                    part.predicted_counterfactual_effect,
                    part.neural_counterfactual_effect,
                )
                for mapping, part in selected_part.groupby("response_mapping")
            }
            mapping_gap = (
                max(mapping_values.values()) - min(mapping_values.values())
                if len(mapping_values) == 2
                else float("inf")
            )
            p_value = (1 + sum(value >= selected_cfr for value in null)) / (
                len(null) + 1
            )
            summary_rows.append(
                {
                    "theory": theory,
                    "split": split,
                    "endpoint_id": config["endpoint_id"],
                    "metric_id": config["metric_id"],
                    "selected_global_cfr": selected_cfr,
                    "random_mean_global_cfr": float(np.mean(null)),
                    "random_max_global_cfr": float(np.max(null)),
                    "finite_sample_random_p": p_value,
                    "named_control_global_cfr": json.dumps(named, sort_keys=True),
                    "response_mapping_cfr": json.dumps(mapping_values, sort_keys=True),
                    "response_mapping_cfr_gap": mapping_gap,
                    "shuffled_source_status": source_status,
                    "shuffled_target_status": audit.status,
                    "passes_random_control": bool(p_value <= 0.05),
                    "passes_named_controls": bool(
                        named and selected_cfr > max(named.values())
                    ),
                    "passes_response_mapping_control": bool(
                        len(mapping_values) == 2
                        and all(value > 0 for value in mapping_values.values())
                        and mapping_gap
                        <= float(
                            config["specificity"][
                                "maximum_response_mapping_cfr_gap"
                            ]
                        )
                    ),
                }
            )

    table = pd.DataFrame(summary_rows)
    table.to_csv(output / "mechanism/specificity_controls.csv", index=False)
    effect_path = Path(
        write_records(effect_rows, output / "mechanism/specificity_effects.parquet")
    )
    pd.DataFrame(random_rows).to_csv(
        output / "mechanism/random_subspace_null.csv", index=False
    )
    _json(output / "mechanism/target_shuffle_audit.json", shuffle_audits)
    controls_complete = bool(len(table)) and bool(
        (table.shuffled_source_status == "informative_control").all()
    )
    passed = bool(
        len(table)
        and controls_complete
        and table.passes_random_control.all()
        and table.passes_named_controls.all()
        and table.passes_response_mapping_control.all()
    )
    gates = {
        "endpoint_id": config["endpoint_id"],
        "passed": passed,
        "status": (
            "pass" if passed else "incomplete_controls" if not controls_complete else "fail"
        ),
        "required_controls_complete": controls_complete,
        "random_subspaces": random_count,
        "controls_use_endpoint": config["endpoint_id"],
        "target_shuffles": shuffle_audits,
    }
    _json(output / "mechanism/specificity_gates.json", gates)
    state.complete(
        "specificity",
        outcome=gates["status"],
        artifacts={
            "controls": _sha256(output / "mechanism/specificity_controls.csv"),
            "effects": _sha256(effect_path),
            "random_null": _sha256(output / "mechanism/random_subspace_null.csv"),
            "shuffle_audit": _sha256(output / "mechanism/target_shuffle_audit.json"),
            "gates": _sha256(output / "mechanism/specificity_gates.json"),
        },
    )
    state.write(output / "run_state.json")
    return gates

"""Artifact-oriented action-history direction disambiguation pipeline.

The original mechanistic output is a read-only input. This module stores only
directions, scalar projections, intervention outcomes, and aggregate statistics.
"""

from __future__ import annotations

from collections import defaultdict
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
from cognitive_discovery.mechanistic.activations.streaming_stats import (
    StreamingMean,
    StreamingRidge,
)
from cognitive_discovery.mechanistic.calibration.projection_to_computation import (
    DirectionCalibration,
)
from cognitive_discovery.mechanistic.dataset.matched_conditions import (
    load_mechanistic_manifest,
)
from cognitive_discovery.mechanistic.representations.mean_difference import (
    MeanDifferenceDirections,
    load_directions,
    save_directions,
)
from cognitive_discovery.mechanistic.representations.ridge_probe import (
    RidgeProbe,
    probe_metrics,
)
from cognitive_discovery.mechanistic.steering.controls import random_directions

from .directions import (
    cosine_similarity,
    normalize,
    orthogonalize_to_subspace,
    pairwise_similarity,
)
from .matching import MatchConfig, build_matched_pairs
from .residualization import ActionResidualizer


PRIMARY_LAYERS = (8, 30)
PRIMARY_DIRECTIONS = (
    "action_raw",
    "action_orthogonal",
    "action_residualized",
    "persistence",
    "gradient",
)
ACTION_DIRECTIONS = (
    "action_raw",
    "action_orthogonal",
    "action_residualized",
)


def _root(config: dict, output=None) -> Path:
    return Path(
        output
        or config.get("output_root", "artifacts/action_history_disambiguation_v1")
    )


def _source_root(config: dict, source=None) -> Path:
    return Path(
        source or config.get("source_mechanistic_root", "artifacts/mechanistic_v1")
    )


def _json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector_sha256(vector) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(vector, dtype=np.float32).tobytes()
    ).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _records_path(root: Path, relative: str) -> Path:
    path = root / relative
    if path.exists():
        return path
    alternate = path.with_suffix(".csv.gz")
    if alternate.exists():
        return alternate
    return path


def _direction_path(root: Path, layer: int, name: str) -> Path:
    return root / "directions" / f"L{int(layer)}_{name}.safetensors"


def _save_vector(root: Path, layer: int, name: str, vector, **metadata) -> Path:
    return save_directions(
        _direction_path(root, layer, name),
        {name: {int(layer): normalize(vector).astype(np.float32)}},
        metadata={
            "protocol": "action_history_disambiguation_v1",
            "layer": int(layer),
            "direction": name,
            "activation_position": "final_prompt_token",
            "layer_index_convention": "zero_based_block_output",
            **metadata,
        },
    )


def _load_vector(root: Path, layer: int, name: str) -> np.ndarray:
    return load_directions(_direction_path(root, layer, name))[name][int(layer)]


def _load_primary_directions(root: Path) -> dict[int, dict[str, np.ndarray]]:
    return {
        layer: {name: _load_vector(root, layer, name) for name in PRIMARY_DIRECTIONS}
        for layer in PRIMARY_LAYERS
    }


def _make_runner(config, *, model_path=None, revision=None, online=False):
    return MechanisticQwenRunner.from_pretrained(
        model_path or config["model"],
        revision=revision or config.get("model_revision"),
        local_files_only=not online,
    )


def _render(record):
    return get_renderer(record.condition.task_family).render(record.condition)


def _forward_record(runner, record, *, layer=None, editor=None):
    trial = _render(record)
    editors = {int(layer): editor} if layer is not None and editor is not None else None
    capture = PRIMARY_LAYERS if layer is None else (int(layer),)
    return runner.forward(
        list(trial.messages),
        record.condition.response_mapping.labels,
        positive_label=record.condition.response_mapping.continue_label,
        editors=editors,
        capture_layers=capture,
    )


def _gradient_record(runner, record, layer: int):
    trial = _render(record)
    return runner.state_and_persistence_gradient(
        list(trial.messages),
        record.condition.response_mapping.labels,
        positive_label=record.condition.response_mapping.continue_label,
        layer=layer,
    )


def _source_projection_frame(source: Path) -> pd.DataFrame:
    path = _records_path(source, "representation/projections.parquet")
    if not path.exists():
        raise FileNotFoundError(f"source scalar projections are missing: {path}")
    frame = read_records(path)
    selected = frame[(frame.direction == "action_history") & (frame.layer == 30)].copy()
    selected = selected.drop_duplicates("condition_id")
    if selected.condition_id.duplicated().any() or selected.empty:
        raise ValueError("source projections do not define one row per condition")
    return selected.reset_index(drop=True)


def _match_config(config: dict) -> MatchConfig:
    settings = config.get("matching", {})
    return MatchConfig(
        decision_caliper=float(settings.get("decision_caliper", 0.15)),
        history_min_delta=float(settings.get("history_min_delta", 1.0)),
        history_caliper=float(settings.get("history_caliper", 1e-8)),
        decision_min_delta=float(settings.get("decision_min_delta", 0.75)),
        nuisance_max_distance=float(settings.get("nuisance_max_distance", 2.5)),
    )


def prepare_action_history_run(
    config: dict,
    *,
    source: str | Path | None = None,
    output: str | Path | None = None,
):
    """Freeze original vectors, behavior coefficients, targets, and matched pairs."""

    root = _root(config, output)
    source_root = _source_root(config, source)
    root.mkdir(parents=True, exist_ok=True)
    source_metadata_path = source_root / "run_metadata.json"
    source_manifest = source_root / "manifests/mechanistic_conditions.jsonl"
    source_direction = source_root / "directions/action_history.safetensors"
    source_coefficients = source_root / "calibration/frozen_behavioral_coefficients.csv"
    for path in (
        source_metadata_path,
        source_manifest,
        source_direction,
        source_coefficients,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"required frozen source artifact is missing: {path}"
            )
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("activation_position") != "final_prompt_token":
        raise ValueError("source direction was not measured at the final prompt token")
    if source_metadata.get("layer_index_convention") != "zero_based_block_output":
        raise ValueError("source directions do not use zero-based block outputs")
    if int(source_metadata.get("layer_count", 0)) <= max(PRIMARY_LAYERS):
        raise ValueError(
            "source model does not expose both preregistered primary layers"
        )

    originals = load_directions(source_direction)["action_history"]
    for layer in PRIMARY_LAYERS:
        _save_vector(
            root,
            layer,
            "action_raw",
            originals[layer],
            estimator="frozen_original_paired_mean_difference",
            source_file=str(source_direction),
            source_file_sha256=_sha256(source_direction),
            source_vector_sha256=_vector_sha256(originals[layer]),
            immutable=True,
        )
    # Give the critical frozen vector an unambiguous named artifact as requested.
    save_directions(
        root / "directions/original_action_history_L30.safetensors",
        {"original_action_history_L30": {30: originals[30]}},
        metadata={
            "protocol": "action_history_disambiguation_v1",
            "immutable": True,
            "source_file_sha256": _sha256(source_direction),
        },
    )

    manifest_target = root / "manifests/mechanistic_conditions.jsonl"
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest, manifest_target)
    records = load_mechanistic_manifest(manifest_target)
    mapping_by_id = {
        record.condition.condition_id: record.condition.response_mapping.mapping_id
        for record in records
    }
    frame = _source_projection_frame(source_root)
    frame["response_mapping"] = frame.condition_id.map(mapping_by_id)
    if frame.response_mapping.isna().any():
        raise ValueError("source manifest and projection condition IDs disagree")

    training = frame[frame.split == "train"].copy()
    residualizer = ActionResidualizer().fit(training)
    frame["action_residual"] = residualizer.residualize(frame)
    frame["residualizer_fit_row"] = frame.condition_id.isin(
        residualizer.fit_condition_ids_
    )
    targets_path = write_records(
        frame.to_dict("records"), root / "targets/residualized_targets.parquet"
    )
    residualizer_path = root / "targets/action_residualizer.json"
    residualizer_path.parent.mkdir(parents=True, exist_ok=True)
    residualizer_path.write_text(residualizer.dumps(), encoding="utf-8")

    matched = build_matched_pairs(frame, _match_config(config))
    matched_path = write_records(
        matched.to_dict("records"), root / "representation/matched_pairs.parquet"
    )
    coefficients = pd.read_csv(source_coefficients)
    action_coefficients = coefficients[coefficients.target == "action_history"].copy()
    if action_coefficients.task_family.nunique() != 7:
        raise ValueError(
            "frozen action-history coefficients must cover all seven tasks"
        )
    calibration_root = root / "calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    action_coefficients.to_csv(
        calibration_root / "frozen_action_coefficients.csv", index=False
    )

    train_design = residualizer._design(training, fitting=False)
    train_residual = residualizer.residualize(training)
    correlations = []
    for index, name in enumerate(residualizer.feature_names_[1:], start=1):
        predictor = train_design[:, index]
        correlation = (
            float(np.corrcoef(predictor, train_residual)[0, 1])
            if np.std(predictor) > 1e-12 and np.std(train_residual) > 1e-12
            else 0.0
        )
        correlations.append(
            {"predictor": name, "training_residual_correlation": correlation}
        )
    pd.DataFrame(correlations).to_csv(
        root / "targets/residualization_diagnostics.csv", index=False
    )
    metadata = {
        "protocol_version": config.get(
            "protocol_version", "action_history_disambiguation_v1"
        ),
        "source_mechanistic_root": str(source_root),
        "source_run_metadata_sha256": _sha256(source_metadata_path),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_action_direction_sha256": _sha256(source_direction),
        "original_action_history_L30_vector_sha256": _vector_sha256(originals[30]),
        "model_id": source_metadata.get("model_id", config.get("model")),
        "model_revision": source_metadata.get("model_revision"),
        "git_commit": _git_commit(),
        "activation_position": "final_prompt_token",
        "layer_index_convention": "zero_based_block_output",
        "primary_layers": list(PRIMARY_LAYERS),
        "optional_diagnostic_layers": list(
            config.get("layers", {}).get("diagnostic", [4, 12, 20, 28, 31])
        ),
        "direction_fitting_split": "train",
        "residualizer_fit_row_hash": residualizer.fit_row_hash_,
        "behavioral_coefficients_frozen": True,
        "source_artifacts_immutable": True,
        "full_activations_saved": False,
        "steering_doses": list(
            map(
                float,
                config.get("steering", {}).get("doses", [-2, -1, -0.5, 0, 0.5, 1, 2]),
            )
        ),
        "random_direction_count": int(config.get("random_null", {}).get("count", 100)),
    }
    _json(root / "run_metadata.json", metadata)
    return {
        "output": root,
        "targets": targets_path,
        "matched_pairs": matched_path,
        "conditions": len(frame),
        "training_conditions": len(training),
        "matched_pair_count": len(matched),
    }


def fit_action_history_directions(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Fit train-only probes, reproduce the original contrast, and average gradients."""

    root = _root(config, output)
    records = load_mechanistic_manifest(root / "manifests/mechanistic_conditions.jsonl")
    targets = read_records(_records_path(root, "targets/residualized_targets.parquet"))
    target_by_id = targets.set_index("condition_id")
    training_records = [
        record for record in records if record.condition.split == "train"
    ]
    if limit is not None:
        training_records = training_records[: int(limit)]
    if not training_records:
        raise ValueError("no training records were selected")
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    if runner.layer_count <= max(PRIMARY_LAYERS):
        raise ValueError("loaded model is incompatible with the frozen layers")
    alpha = float(config.get("directions", {}).get("ridge_alpha", 1.0))
    ridge = {
        layer: {
            target: StreamingRidge()
            for target in ("action_history", "persistence_logit", "action_residual")
        }
        for layer in PRIMARY_LAYERS
    }
    gradient_mean = {layer: StreamingMean() for layer in PRIMARY_LAYERS}
    pair_states = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in training_records:
        row = target_by_id.loc[record.condition.condition_id]
        for layer in PRIMARY_LAYERS:
            result = _gradient_record(runner, record, layer)
            state = result.state[None, :]
            for target in ridge[layer]:
                ridge[layer][target].update(state, [float(row[target])])
            gradient_mean[layer].update(result.gradient)
            if record.contrast_family == "action_history":
                pair_states[record.contrast_id][record.contrast_member][layer].append(
                    result.state
                )

    reproduction = {layer: MeanDifferenceDirections() for layer in PRIMARY_LAYERS}
    complete_pairs = 0
    for members in pair_states.values():
        if 1 not in members or -1 not in members:
            continue
        complete_pairs += 1
        for layer in PRIMARY_LAYERS:
            plus = np.mean(members[1][layer], axis=0)
            minus = np.mean(members[-1][layer], axis=0)
            reproduction[layer].update("action_reproduction", layer, plus - minus)

    metadata_rows = []
    threshold = float(
        config.get("directions", {}).get("reproduction_min_cosine", 0.999)
    )
    for layer in PRIMARY_LAYERS:
        raw = _load_vector(root, layer, "action_raw")
        action_probe = normalize(
            ridge[layer]["action_history"].solve(alpha).coefficient
        )
        persistence = normalize(
            ridge[layer]["persistence_logit"].solve(alpha).coefficient
        )
        residual = normalize(ridge[layer]["action_residual"].solve(alpha).coefficient)
        gradient = normalize(gradient_mean[layer].mean)
        orthogonal, basis = orthogonalize_to_subspace(raw, (persistence, gradient))
        reproduced, reproduction_rows = reproduction[layer].directions()
        reproduced = reproduced["action_reproduction"][layer]
        reproduction_cosine = cosine_similarity(raw, reproduced)
        # A limited developer run may omit or truncate contrasts. It is diagnostic,
        # never allowed to overwrite the frozen vector.
        engineering_checked = limit is None
        engineering_passed = (
            bool(reproduction_cosine >= threshold) if engineering_checked else None
        )
        if engineering_checked and not engineering_passed:
            raise RuntimeError(
                f"layer-{layer} clean reproduction cosine {reproduction_cosine:.6f} "
                f"is below {threshold:.6f}; frozen artifacts/model do not agree"
            )
        _save_vector(
            root, layer, "action_probe", action_probe, estimator="streaming_ridge"
        )
        _save_vector(
            root,
            layer,
            "action_reproduction",
            reproduced,
            estimator="clean_paired_mean_difference",
        )
        _save_vector(
            root, layer, "persistence", persistence, estimator="streaming_ridge"
        )
        _save_vector(
            root,
            layer,
            "gradient",
            gradient,
            estimator="mean_per_example_semantic_persistence_logit_gradient",
        )
        _save_vector(
            root,
            layer,
            "action_orthogonal",
            orthogonal,
            controls=["persistence", "gradient"],
            control_rank=int(basis.shape[1]),
        )
        _save_vector(
            root,
            layer,
            "action_residualized",
            residual,
            estimator="streaming_ridge_on_train_only_action_residual",
        )
        metadata_rows.append(
            {
                "layer": layer,
                "training_examples": ridge[layer]["action_history"].count,
                "gradient_examples": gradient_mean[layer].count,
                "action_contrast_pairs": int(reproduction_rows[0]["pairs"]),
                "reproduction_cosine": reproduction_cosine,
                "reproduction_threshold": threshold,
                "engineering_reproduction_checked": engineering_checked,
                "engineering_reproduction_passed": engineering_passed,
                "raw_dot_persistence": float(normalize(raw) @ persistence),
                "raw_dot_gradient": float(normalize(raw) @ gradient),
                "orthogonal_dot_persistence": float(orthogonal @ persistence),
                "orthogonal_dot_gradient": float(orthogonal @ gradient),
            }
        )
    _json(
        root / "directions/metadata.json",
        {
            "primary_layers": list(PRIMARY_LAYERS),
            "training_records": len(training_records),
            "complete_action_contrast_pairs": complete_pairs,
            "direction_rows": metadata_rows,
            "full_activations_saved": False,
        },
    )
    return {"training_records": len(training_records), "layers": list(PRIMARY_LAYERS)}


def _random_decoding_rows(buffers: dict, *, alpha: float) -> list[dict]:
    rows = []
    for layer in PRIMARY_LAYERS:
        if not buffers[(layer, "train")][0]:
            continue
        train_x = np.vstack(buffers[(layer, "train")][0])
        train_y = np.asarray(buffers[(layer, "train")][1], dtype=float)
        x_mean, y_mean = train_x.mean(axis=0), train_y.mean()
        centered_x = train_x - x_mean
        slope = (centered_x * (train_y - y_mean)[:, None]).sum(axis=0) / (
            np.square(centered_x).sum(axis=0) + alpha
        )
        intercept = y_mean - slope * x_mean
        for split in ("validation", "test", "transfer"):
            if not buffers[(layer, split)][0]:
                continue
            x = np.vstack(buffers[(layer, split)][0])
            y = np.asarray(buffers[(layer, split)][1], dtype=float)
            prediction = intercept + x * slope
            for index in range(x.shape[1]):
                rows.append(
                    {
                        "layer": layer,
                        "random_index": index,
                        "split": split,
                        **probe_metrics(y, prediction[:, index]),
                    }
                )
    return rows


def project_action_history_directions(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Project all conditions onto compact directions and random controls."""

    root = _root(config, output)
    records = load_mechanistic_manifest(root / "manifests/mechanistic_conditions.jsonl")
    if limit is not None:
        records = records[: int(limit)]
    targets = read_records(_records_path(root, "targets/residualized_targets.parquet"))
    target_by_id = targets.set_index("condition_id")
    directions = _load_primary_directions(root)
    random_count = int(config.get("random_null", {}).get("count", 100))
    seed = int(config.get("seed", 73001))
    randoms = {
        layer: random_directions(
            len(next(iter(directions[layer].values()))),
            random_count,
            seed=seed + 7919 * layer,
        )
        for layer in PRIMARY_LAYERS
    }
    random_buffers = {
        (layer, split): ([], [])
        for layer in PRIMARY_LAYERS
        for split in ("train", "validation", "test", "transfer")
    }
    activation_moments = {layer: StreamingMean() for layer in PRIMARY_LAYERS}
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    rows = []
    for record in records:
        result = _forward_record(runner, record)
        target = target_by_id.loc[record.condition.condition_id]
        common = {
            "condition_id": record.condition.condition_id,
            "paired_condition_id": record.condition.paired_condition_id,
            "task_family": record.condition.task_family,
            "response_mapping": record.condition.response_mapping.mapping_id,
            "split": record.condition.split,
            "action_history": float(target.action_history),
            "action_residual": float(target.action_residual),
            "persistence_logit": float(result.persistence_logit),
        }
        for layer in PRIMARY_LAYERS:
            state = result.states[layer]
            if record.condition.split == "train":
                activation_moments[layer].update(state)
            for name, direction in directions[layer].items():
                rows.append(
                    {
                        **common,
                        "layer": layer,
                        "direction": name,
                        "projection": float(state @ direction),
                    }
                )
            projection = randoms[layer] @ state
            buffer_x, buffer_y = random_buffers[(layer, record.condition.split)]
            buffer_x.append(projection[None, :])
            buffer_y.append(float(target.action_residual))
    path = write_records(rows, root / "representation/projections.parquet")
    random_rows = _random_decoding_rows(
        random_buffers,
        alpha=float(config.get("directions", {}).get("ridge_alpha", 1.0)),
    )
    pd.DataFrame(random_rows).to_csv(
        root / "representation/random_decoding.csv", index=False
    )
    scale_rows = []
    for layer, statistic in activation_moments.items():
        if statistic.mean is None:
            continue
        activation_scale = float(np.sqrt(np.mean(statistic.variance)))
        scale_rows.append(
            {
                "layer": layer,
                "activation_scale": activation_scale,
                "definition": "sqrt(mean per-coordinate train activation variance)",
                "training_examples": statistic.count,
            }
        )
    pd.DataFrame(scale_rows).to_csv(
        root / "calibration/layer_activation_scales.csv", index=False
    )
    return {"projection_rows": len(rows), "path": path, "conditions": len(records)}


def _probe_rows(frame: pd.DataFrame, *, alpha: float) -> list[dict]:
    rows = []
    for (layer, direction), part in frame.groupby(["layer", "direction"]):
        train = part[part.split == "train"]
        if len(train) < 2:
            continue
        for target in ("action_history", "action_residual", "persistence_logit"):
            probe = RidgeProbe(alpha=alpha).fit(train[["projection"]], train[target])
            for split in ("validation", "test", "transfer"):
                test = part[part.split == split]
                if len(test) < 2:
                    continue
                rows.append(
                    {
                        "layer": int(layer),
                        "direction": direction,
                        "target": target,
                        "split": split,
                        **probe_metrics(
                            test[target], probe.predict(test[["projection"]])
                        ),
                    }
                )
    return rows


def analyze_action_history_representations(
    config: dict, *, output: str | Path | None = None
):
    """Quantify direction overlap, residual decoding, matched contrasts, and gates 1/2."""

    root = _root(config, output)
    projections = read_records(
        _records_path(root, "representation/projections.parquet")
    )
    alpha = float(config.get("directions", {}).get("ridge_alpha", 1.0))
    decoding = pd.DataFrame(_probe_rows(projections, alpha=alpha))
    decoding.to_csv(root / "representation/residual_decoding.csv", index=False)
    similarity_rows = []
    for layer, directions in _load_primary_directions(root).items():
        for row in pairwise_similarity(directions):
            left = (
                projections[
                    (projections.layer == layer)
                    & (projections.direction == row["direction_a"])
                    & (projections.split == "test")
                ]
                .set_index("condition_id")
                .projection
            )
            right = (
                projections[
                    (projections.layer == layer)
                    & (projections.direction == row["direction_b"])
                    & (projections.split == "test")
                ]
                .set_index("condition_id")
                .projection
            )
            overlap = left.to_frame("left").join(right.to_frame("right"), how="inner")
            projection_r = (
                float(np.corrcoef(overlap.left, overlap.right)[0, 1])
                if len(overlap) > 1
                and overlap.left.std() > 1e-12
                and overlap.right.std() > 1e-12
                else np.nan
            )
            similarity_rows.append(
                {
                    "layer": layer,
                    **row,
                    "heldout_split": "test",
                    "heldout_projection_correlation": projection_r,
                    "heldout_projection_shared_r2": projection_r**2,
                    "heldout_examples": len(overlap),
                }
            )
    similarity = pd.DataFrame(similarity_rows)
    similarity.to_csv(root / "representation/direction_similarity.csv", index=False)

    pairs = read_records(_records_path(root, "representation/matched_pairs.parquet"))
    lookup = projections.set_index(["condition_id", "layer", "direction"]).projection
    detailed = []
    for pair in pairs.itertuples():
        for layer in PRIMARY_LAYERS:
            for direction in PRIMARY_DIRECTIONS:
                high = float(lookup.loc[(pair.high_condition_id, layer, direction)])
                low = float(lookup.loc[(pair.low_condition_id, layer, direction)])
                target_delta = (
                    float(pair.action_delta)
                    if pair.contrast_kind == "history_decision_matched"
                    else float(pair.decision_delta)
                )
                detailed.append(
                    {
                        "contrast_kind": pair.contrast_kind,
                        "task_family": pair.task_family,
                        "response_mapping": pair.response_mapping,
                        "split": getattr(pair, "split", "all"),
                        "high_condition_id": pair.high_condition_id,
                        "low_condition_id": pair.low_condition_id,
                        "layer": layer,
                        "direction": direction,
                        "projection_delta": high - low,
                        "target_delta": target_delta,
                        "correct_sign": bool((high - low) * target_delta > 0),
                    }
                )
    detail_frame = pd.DataFrame(detailed)
    write_records(
        detail_frame.to_dict("records"),
        root / "representation/matched_state_detail.parquet",
    )
    summary = detail_frame.groupby(
        ["contrast_kind", "layer", "direction"], as_index=False
    ).agg(
        pairs=("projection_delta", "size"),
        mean_projection_difference=("projection_delta", "mean"),
        mean_absolute_projection_difference=(
            "projection_delta",
            lambda x: float(np.abs(x).mean()),
        ),
        sign_accuracy=("correct_sign", "mean"),
    )
    rds_rows = []
    for (layer, direction), part in summary.groupby(["layer", "direction"]):
        history = part[part.contrast_kind == "history_decision_matched"]
        decision = part[part.contrast_kind == "decision_history_matched"]
        if history.empty or decision.empty:
            continue
        numerator = abs(float(history.mean_projection_difference.iloc[0]))
        denominator = abs(float(decision.mean_projection_difference.iloc[0]))
        rds_rows.append(
            {
                "layer": int(layer),
                "direction": direction,
                "representational_dissociation_score": numerator / (denominator + 1e-8),
            }
        )
    summary = summary.merge(
        pd.DataFrame(rds_rows), on=["layer", "direction"], how="left"
    )
    summary.to_csv(root / "representation/matched_state_results.csv", index=False)

    random_decoding = pd.read_csv(root / "representation/random_decoding.csv")
    gate_settings = config.get("gates", {})
    residual_rows = decoding[
        (decoding.direction == "action_residualized")
        & (decoding.target == "action_residual")
        & (decoding.split == "validation")
    ]
    gate1_layers = []
    for row in residual_rows.itertuples():
        null = random_decoding[
            (random_decoding.layer == row.layer)
            & (random_decoding.split == "validation")
        ]
        random95 = float(null.r2.quantile(0.95)) if len(null) else np.nan
        gate1_layers.append(
            {
                "layer": int(row.layer),
                "residual_validation_r2": float(row.r2),
                "random_r2_95": random95,
                "passed": bool(row.r2 > random95),
            }
        )
    history_results = summary[
        (summary.contrast_kind == "history_decision_matched")
        & (summary.direction.isin(("action_raw", "action_residualized")))
    ]
    minimum_sign = float(gate_settings.get("minimum_matched_sign_accuracy", 0.6))
    gate2_layers = [
        {
            "layer": int(layer),
            "best_sign_accuracy": float(part.sign_accuracy.max()),
            "passed": bool(part.sign_accuracy.max() >= minimum_sign),
        }
        for layer, part in history_results.groupby("layer")
    ]
    gates = {
        "gate_1_representational_independence": {
            "passed": bool(gate1_layers and any(row["passed"] for row in gate1_layers)),
            "layers": gate1_layers,
        },
        "gate_2_matched_state_specificity": {
            "passed": bool(gate2_layers and any(row["passed"] for row in gate2_layers)),
            "layers": gate2_layers,
        },
        "gate_3_causal_specificity": {"passed": None, "status": "awaiting_steering"},
        "gate_4_computational_correspondence": {
            "passed": None,
            "status": "awaiting_steering",
        },
    }
    _json(root / "gates.json", gates)

    calibrations = []
    frozen = pd.read_csv(root / "calibration/frozen_action_coefficients.csv").iloc[0]
    for (layer, direction), part in projections[
        projections.direction.isin(ACTION_DIRECTIONS)
    ].groupby(["layer", "direction"]):
        train = part[part.split == "train"]
        calibration = DirectionCalibration.fit(
            train.projection,
            train.action_history,
            target_mean=float(frozen.computational_mean),
            target_scale=float(frozen.computational_scale),
        )
        calibrations.append(
            {
                "layer": int(layer),
                "direction": direction,
                **calibration.to_dict(),
                "fitting_split": "train",
                "units": "frozen_action_history_standard_deviations",
            }
        )
    pd.DataFrame(calibrations).to_csv(
        root / "calibration/action_direction_calibration.csv", index=False
    )
    return {
        "decoding_rows": len(decoding),
        "matched_result_rows": len(summary),
        "gates": gates,
    }


def _balanced_steering_records(records, examples_per_task: int):
    selected = []
    by_task = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_task[record.condition.task_family][
            record.condition.response_mapping.mapping_id
        ].append(record)
    for task in sorted(by_task):
        mappings = by_task[task]
        levels = sorted(mappings)
        if len(levels) != 2:
            raise ValueError(f"steering task {task} lacks both response mappings")
        per_mapping = max(1, int(examples_per_task) // 2)
        chosen = []
        for mapping in levels:
            chosen.extend(
                sorted(mappings[mapping], key=lambda r: r.condition.condition_id)[
                    :per_mapping
                ]
            )
        selected.extend(chosen[: int(examples_per_task)])
    if len({record.condition.task_family for record in selected}) != 7:
        raise ValueError("steering subset must cover all seven tasks")
    return selected


def _editor(direction, alpha):
    from cognitive_discovery.mechanistic.steering.intervene import direction_editor

    return direction_editor(direction, float(alpha))


def run_action_history_steering(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Run matched-neural-norm controls and secondary action-unit interventions."""

    root = _root(config, output)
    records = load_mechanistic_manifest(root / "manifests/mechanistic_conditions.jsonl")
    examples = int(config.get("steering", {}).get("examples_per_task", 8))
    selected = _balanced_steering_records(records, examples)
    if limit is not None:
        selected = selected[: int(limit)]
    doses = tuple(
        map(
            float, config.get("steering", {}).get("doses", [-2, -1, -0.5, 0, 0.5, 1, 2])
        )
    )
    if doses != (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        raise ValueError("the primary steering grid must be [-2,-1,-.5,0,.5,1,2]")
    scales = pd.read_csv(root / "calibration/layer_activation_scales.csv").set_index(
        "layer"
    )
    calibrations = pd.read_csv(
        root / "calibration/action_direction_calibration.csv"
    ).set_index(["layer", "direction"])
    directions = _load_primary_directions(root)
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    baselines = {
        record.condition.condition_id: _forward_record(runner, record).persistence_logit
        for record in selected
    }
    rows = []
    for record in selected:
        condition_id = record.condition.condition_id
        baseline = float(baselines[condition_id])
        common = {
            "condition_id": condition_id,
            "task_family": record.condition.task_family,
            "response_mapping": record.condition.response_mapping.mapping_id,
            "baseline_persistence_logit": baseline,
        }
        for layer in PRIMARY_LAYERS:
            activation_scale = float(scales.loc[layer, "activation_scale"])
            for direction, vector in directions[layer].items():
                modes = (
                    ("neural_norm", "computational_units")
                    if direction in ACTION_DIRECTIONS
                    else ("neural_norm",)
                )
                for mode in modes:
                    calibration = None
                    if mode == "computational_units":
                        calibration = DirectionCalibration.from_dict(
                            calibrations.loc[(layer, direction)]
                        )
                    for dose in doses:
                        if mode == "neural_norm":
                            alpha = dose * activation_scale
                            delta_action = np.nan
                        else:
                            alpha = float(calibration.alpha_for_delta(dose))
                            delta_action = dose * calibration.target_scale
                        if dose == 0:
                            observed = baseline
                        else:
                            observed = _forward_record(
                                runner,
                                record,
                                layer=layer,
                                editor=_editor(vector, alpha),
                            ).persistence_logit
                        rows.append(
                            {
                                **common,
                                "layer": layer,
                                "direction": direction,
                                "steering_mode": mode,
                                "dose": dose,
                                "neural_alpha": alpha,
                                "delta_action_history": delta_action,
                                "observed_persistence_logit": observed,
                                "persistence_logit_change": float(observed - baseline),
                            }
                        )
    path = write_records(rows, root / "steering/detailed_dose_response.parquet")
    return {"rows": len(rows), "conditions": len(selected), "path": path}


def run_random_direction_null(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    shard_index: int = 0,
    shard_count: int = 1,
    limit: int | None = None,
):
    """Run a resumable shard of >=100 full seven-dose random interventions."""

    root = _root(config, output)
    total = int(config.get("random_null", {}).get("count", 100))
    if total < 100:
        raise ValueError("the random-direction null requires at least 100 directions")
    shard_index, shard_count = int(shard_index), int(shard_count)
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid random-null shard")
    indices = np.array_split(np.arange(total), shard_count)[shard_index].tolist()
    if limit is not None:
        indices = indices[: int(limit)]
    records = load_mechanistic_manifest(root / "manifests/mechanistic_conditions.jsonl")
    examples = int(config.get("random_null", {}).get("examples_per_task", 2))
    selected = _balanced_steering_records(records, examples)
    doses = tuple(map(float, config.get("steering", {}).get("doses", [])))
    if len(doses) != 7 or 0.0 not in doses:
        raise ValueError("random null must use the full seven-dose grid")
    scales = pd.read_csv(root / "calibration/layer_activation_scales.csv").set_index(
        "layer"
    )
    primary = _load_primary_directions(root)
    seed = int(config.get("seed", 73001))
    randoms = {
        layer: random_directions(
            len(next(iter(primary[layer].values()))), total, seed=seed + 7919 * layer
        )
        for layer in PRIMARY_LAYERS
    }
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    baselines = {
        record.condition.condition_id: _forward_record(runner, record).persistence_logit
        for record in selected
    }
    rows = []
    for record in selected:
        baseline = float(baselines[record.condition.condition_id])
        for layer in PRIMARY_LAYERS:
            scale = float(scales.loc[layer, "activation_scale"])
            for random_index in indices:
                vector = randoms[layer][random_index]
                for dose in doses:
                    alpha = dose * scale
                    observed = (
                        baseline
                        if dose == 0
                        else _forward_record(
                            runner,
                            record,
                            layer=layer,
                            editor=_editor(vector, alpha),
                        ).persistence_logit
                    )
                    rows.append(
                        {
                            "condition_id": record.condition.condition_id,
                            "task_family": record.condition.task_family,
                            "response_mapping": record.condition.response_mapping.mapping_id,
                            "layer": layer,
                            "random_index": random_index,
                            "dose": dose,
                            "neural_alpha": alpha,
                            "baseline_persistence_logit": baseline,
                            "observed_persistence_logit": observed,
                            "persistence_logit_change": float(observed - baseline),
                        }
                    )
    path = write_records(
        rows,
        root
        / "steering/random_shards"
        / f"random_{shard_index:03d}_of_{shard_count:03d}.parquet",
    )
    return {"path": path, "rows": len(rows), "random_indices": indices}


def run_matched_projection_patching(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Optional projection-only patching on decision-matched history pairs."""

    from cognitive_discovery.mechanistic.patching.projection_patch import (
        projection_patch,
    )

    root = _root(config, output)
    steering_path = _records_path(root, "steering/detailed_dose_response.parquet")
    if not steering_path.exists():
        raise RuntimeError("patching is sequenced after the primary steering run")
    pairs = read_records(_records_path(root, "representation/matched_pairs.parquet"))
    pairs = pairs[pairs.contrast_kind == "history_decision_matched"]
    pair_limit = int(config.get("patching", {}).get("pairs", 16))
    if limit is not None:
        pair_limit = min(pair_limit, int(limit))
    pairs = (
        pairs.sort_values(["task_family", "score"])
        .groupby("task_family")
        .head(max(1, pair_limit // 7))
    )
    records = load_mechanistic_manifest(root / "manifests/mechanistic_conditions.jsonl")
    by_id = {record.condition.condition_id: record for record in records}
    directions = _load_primary_directions(root)
    seed = int(config.get("seed", 73001))
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    rows = []
    for pair in pairs.itertuples():
        source_record = by_id[pair.high_condition_id]
        target_record = by_id[pair.low_condition_id]
        baseline = _forward_record(runner, target_record).persistence_logit
        for layer in PRIMARY_LAYERS:
            source_state = _forward_record(runner, source_record, layer=layer).states[
                layer
            ]
            controls = {
                "action_raw": directions[layer]["action_raw"],
                "action_residualized": directions[layer]["action_residualized"],
                "persistence": directions[layer]["persistence"],
                "random": random_directions(len(source_state), 1, seed=seed + layer)[0],
            }
            for direction, vector in controls.items():
                editor = lambda state, source=source_state, d=vector: projection_patch(
                    state, source, d
                )
                observed = _forward_record(
                    runner, target_record, layer=layer, editor=editor
                ).persistence_logit
                rows.append(
                    {
                        "source_condition_id": pair.high_condition_id,
                        "target_condition_id": pair.low_condition_id,
                        "task_family": pair.task_family,
                        "layer": layer,
                        "direction": direction,
                        "baseline_persistence_logit": baseline,
                        "patched_persistence_logit": observed,
                        "persistence_logit_change": float(observed - baseline),
                    }
                )
    path = root / "patching/matched_projection_patch.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"rows": len(rows), "path": path}


def _dose_metrics(dose, change) -> dict[str, float]:
    from scipy.stats import spearmanr

    x = np.asarray(dose, dtype=float)
    y = np.asarray(change, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    denominator = float(np.square(y - y.mean()).sum())
    r2 = (
        1.0 - float(np.square(y - prediction).sum()) / denominator
        if denominator > 0
        else np.nan
    )
    rho = float(spearmanr(x, y).statistic) if len(np.unique(x)) > 1 else np.nan
    return {
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "dose_r2": r2,
        "monotonic_rho": rho,
        "mean_absolute_effect": float(np.abs(y).mean()),
    }


def _summarize_primary_steering(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping = ["layer", "direction", "steering_mode", "task_family"]
    for keys, part in frame.groupby(grouping):
        curve = part.groupby("dose", as_index=False).persistence_logit_change.mean()
        rows.append(
            dict(zip(grouping, keys))
            | _dose_metrics(curve.dose, curve.persistence_logit_change)
        )
    task_rows = pd.DataFrame(rows)
    aggregate = []
    for keys, part in frame.groupby(["layer", "direction", "steering_mode"]):
        curve = part.groupby("dose", as_index=False).persistence_logit_change.mean()
        task_slopes = task_rows[
            (task_rows.layer == keys[0])
            & (task_rows.direction == keys[1])
            & (task_rows.steering_mode == keys[2])
        ].linear_slope
        nonzero = task_slopes[np.abs(task_slopes) > 1e-12]
        sign_consistency = (
            float(max((nonzero > 0).mean(), (nonzero < 0).mean()))
            if len(nonzero)
            else 0.0
        )
        aggregate.append(
            {
                "layer": int(keys[0]),
                "direction": keys[1],
                "steering_mode": keys[2],
                "task_family": "ALL",
                **_dose_metrics(curve.dose, curve.persistence_logit_change),
                "task_sign_consistency": sign_consistency,
            }
        )
    task_rows["task_sign_consistency"] = np.nan
    return pd.concat([task_rows, pd.DataFrame(aggregate)], ignore_index=True)


def _load_random_shards(root: Path, expected: int) -> pd.DataFrame:
    frames = []
    for parquet in sorted((root / "steering/random_shards").glob("*.parquet")):
        frames.append(pd.read_parquet(parquet))
    for csv in sorted((root / "steering/random_shards").glob("*.csv.gz")):
        frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError("no random-direction steering shards were found")
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["condition_id", "layer", "random_index", "dose"]
    )
    observed = set(map(int, frame.random_index.unique()))
    if observed != set(range(expected)):
        missing = sorted(set(range(expected)) - observed)
        raise RuntimeError(
            f"random null is incomplete; missing indices: {missing[:20]}"
        )
    return frame


def _summarize_random_null(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layer, index), part in frame.groupby(["layer", "random_index"]):
        curve = part.groupby("dose", as_index=False).persistence_logit_change.mean()
        rows.append(
            {
                "layer": int(layer),
                "random_index": int(index),
                **_dose_metrics(curve.dose, curve.persistence_logit_change),
            }
        )
    return pd.DataFrame(rows)


def _behavioral_correspondence(
    steering: pd.DataFrame, coefficients: pd.DataFrame, scales: pd.DataFrame
) -> pd.DataFrame:
    action = steering[steering.steering_mode == "computational_units"].copy()
    beta = coefficients.set_index("task_family").behavioral_coefficient
    scale = float(coefficients.computational_scale.iloc[0])
    rows = []
    for (layer, direction, task), part in action.groupby(
        ["layer", "direction", "task_family"]
    ):
        curve = part.groupby("dose", as_index=False).persistence_logit_change.mean()
        metrics = _dose_metrics(curve.dose, curve.persistence_logit_change)
        task_beta = float(beta.loc[task])
        rows.append(
            {
                "layer": int(layer),
                "direction": direction,
                "task_family": task,
                "behavioral_beta_action": task_beta,
                "expected_slope_per_standardized_dose": task_beta * scale,
                "observed_slope_per_standardized_dose": metrics["linear_slope"],
                "slope_error": metrics["linear_slope"] - task_beta * scale,
            }
        )
    result = pd.DataFrame(rows)
    summaries = []
    for (layer, direction), part in result.groupby(["layer", "direction"]):
        slope_r = (
            float(
                np.corrcoef(
                    part.behavioral_beta_action,
                    part.observed_slope_per_standardized_dose,
                )[0, 1]
            )
            if len(part) > 2 and part.observed_slope_per_standardized_dose.std() > 0
            else np.nan
        )
        detailed = action[
            (action.layer == layer) & (action.direction == direction)
        ].copy()
        detailed["predicted_change"] = (
            detailed.task_family.map(beta) * detailed.delta_action_history
        )
        prediction_r = (
            float(
                np.corrcoef(
                    detailed.predicted_change, detailed.persistence_logit_change
                )[0, 1]
            )
            if detailed.predicted_change.std() > 0
            and detailed.persistence_logit_change.std() > 0
            else np.nan
        )
        summaries.append(
            {
                "layer": int(layer),
                "direction": direction,
                "task_family": "ALL",
                "behavioral_beta_action": np.nan,
                "expected_slope_per_standardized_dose": np.nan,
                "observed_slope_per_standardized_dose": np.nan,
                "slope_error": np.nan,
                "task_slope_beta_correlation": slope_r,
                "predicted_observed_correlation": prediction_r,
            }
        )
    result["task_slope_beta_correlation"] = np.nan
    result["predicted_observed_correlation"] = np.nan
    return pd.concat([result, pd.DataFrame(summaries)], ignore_index=True)


def _make_figures(
    root: Path,
    decoding: pd.DataFrame,
    similarity: pd.DataFrame,
    matched: pd.DataFrame,
    steering: pd.DataFrame,
    random_detail: pd.DataFrame,
    correspondence: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    shown = decoding[
        (decoding.split == "validation")
        & (
            (
                (decoding.direction == "action_raw")
                & (decoding.target == "action_history")
            )
            | (
                (decoding.direction == "persistence")
                & (decoding.target == "persistence_logit")
            )
            | (
                (decoding.direction == "action_residualized")
                & (decoding.target == "action_residual")
            )
        )
    ].copy()
    shown["series"] = shown.direction.str.replace("_", " ")
    width = 0.23
    for index, (name, part) in enumerate(shown.groupby("series")):
        values = [
            (
                float(part[part.layer == layer].r2.iloc[0])
                if (part.layer == layer).any()
                else np.nan
            )
            for layer in PRIMARY_LAYERS
        ]
        ax.bar(np.arange(2) + (index - 1) * width, values, width, label=name)
    ax.set_xticks(np.arange(2), ["L8", "L30"])
    ax.set_ylabel("Held-out $R^2$")
    ax.set_title("Action history and current persistence decoding")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths.append(figure_root / "figure_1_representation_across_depth.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    geometry_pairs = {
        frozenset(("action_raw", "persistence")): "action–persistence",
        frozenset(("action_raw", "gradient")): "action–gradient",
        frozenset(("persistence", "gradient")): "persistence–gradient",
    }
    for ax, layer in zip(axes, PRIMARY_LAYERS):
        values = []
        for pair, label in geometry_pairs.items():
            row = similarity[
                (similarity.layer == layer)
                & similarity.apply(
                    lambda item: frozenset((item.direction_a, item.direction_b))
                    == pair,
                    axis=1,
                )
            ]
            values.append((label, float(row.cosine.iloc[0]) if len(row) else np.nan))
        ax.bar(range(len(values)), [value for _, value in values])
        ax.set_xticks(
            range(len(values)), [name for name, _ in values], rotation=30, ha="right"
        )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"Layer {layer}")
    axes[0].set_ylabel("Cosine similarity")
    fig.tight_layout()
    paths.append(figure_root / "figure_2_direction_geometry.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, layer in zip(axes, PRIMARY_LAYERS):
        part = matched[matched.layer == layer]
        labels, values = [], []
        for direction in PRIMARY_DIRECTIONS:
            for kind, short in (
                ("history_decision_matched", "H|D"),
                ("decision_history_matched", "D|H"),
            ):
                row = part[(part.direction == direction) & (part.contrast_kind == kind)]
                labels.append(f"{direction.replace('action_', 'A-')}\n{short}")
                values.append(
                    float(row.mean_absolute_projection_difference.iloc[0])
                    if len(row)
                    else np.nan
                )
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(values)), labels, rotation=65, ha="right", fontsize=7)
        ax.set_title(f"Layer {layer}")
    axes[0].set_ylabel("Mean |projection difference|")
    fig.tight_layout()
    paths.append(figure_root / "figure_3_matched_dissociation.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    neural = steering[steering.steering_mode == "neural_norm"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, layer in zip(axes, PRIMARY_LAYERS):
        for direction in ACTION_DIRECTIONS:
            part = neural[(neural.layer == layer) & (neural.direction == direction)]
            curve = part.groupby("dose").persistence_logit_change.mean()
            ax.plot(
                curve.index,
                curve.values,
                marker="o",
                label=direction.replace("action_", ""),
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Matched neural-norm dose")
    axes[0].set_ylabel("Persistence-logit change")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths.append(figure_root / "figure_4_critical_action_steering.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, layer in zip(axes, PRIMARY_LAYERS):
        for direction in ("action_raw", "persistence", "gradient"):
            part = neural[(neural.layer == layer) & (neural.direction == direction)]
            curve = part.groupby("dose").persistence_logit_change.mean()
            ax.plot(curve.index, curve.values, marker="o", label=direction)
        random_curves = (
            random_detail[random_detail.layer == layer]
            .groupby(["random_index", "dose"])
            .persistence_logit_change.mean()
            .unstack(0)
        )
        if not random_curves.empty:
            ax.fill_between(
                random_curves.index,
                random_curves.quantile(0.05, axis=1),
                random_curves.quantile(0.95, axis=1),
                color="grey",
                alpha=0.25,
                label="random 5–95%",
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Matched neural-norm dose")
    axes[0].set_ylabel("Persistence-logit change")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths.append(figure_root / "figure_5_control_dose_responses.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    task_correspondence = correspondence[correspondence.task_family != "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, layer in zip(axes, PRIMARY_LAYERS):
        for direction, marker in zip(ACTION_DIRECTIONS, ("o", "s", "^")):
            part = task_correspondence[
                (task_correspondence.layer == layer)
                & (task_correspondence.direction == direction)
            ]
            ax.scatter(
                part.behavioral_beta_action,
                part.observed_slope_per_standardized_dose,
                label=direction.replace("action_", ""),
                marker=marker,
            )
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Frozen behavioral action coefficient")
    axes[0].set_ylabel("Observed task steering slope")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths.append(figure_root / "figure_6_behavioral_correspondence.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    beta = coefficients.set_index("task_family").behavioral_coefficient
    computational = steering[
        (steering.steering_mode == "computational_units")
        & (steering.direction == "action_residualized")
    ].copy()
    computational["predicted_change"] = (
        computational.task_family.map(beta) * computational.delta_action_history
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, layer in zip(axes, PRIMARY_LAYERS):
        part = computational[computational.layer == layer]
        ax.scatter(
            part.predicted_change, part.persistence_logit_change, s=12, alpha=0.5
        )
        if len(part):
            extent = float(
                max(
                    abs(part.predicted_change).max(),
                    abs(part.persistence_logit_change).max(),
                    1e-3,
                )
            )
            ax.plot([-extent, extent], [-extent, extent], linestyle="--", color="grey")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Behavioral-model predicted change")
    axes[0].set_ylabel("Observed persistence-logit change")
    fig.tight_layout()
    paths.append(figure_root / "figure_7_predicted_vs_observed.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)
    return paths


def _lookup_metric(frame, *, layer, direction, target=None, split=None, column="r2"):
    part = frame[(frame.layer == layer) & (frame.direction == direction)]
    if target is not None:
        part = part[part.target == target]
    if split is not None:
        part = part[part.split == split]
    return float(part.iloc[0][column]) if len(part) else np.nan


def finalize_action_history_run(config: dict, *, output: str | Path | None = None):
    """Aggregate interventions, evaluate all gates, classify the result, and report."""

    root = _root(config, output)
    steering = read_records(
        _records_path(root, "steering/detailed_dose_response.parquet")
    )
    expected_random = int(config.get("random_null", {}).get("count", 100))
    random_detail = _load_random_shards(root, expected_random)
    dose_summary = _summarize_primary_steering(steering)
    dose_summary.to_csv(root / "steering/dose_response.csv", index=False)
    random_summary = _summarize_random_null(random_detail)
    random_summary.to_csv(root / "steering/random_null.csv", index=False)
    coefficients = pd.read_csv(root / "calibration/frozen_action_coefficients.csv")
    scales = pd.read_csv(root / "calibration/layer_activation_scales.csv")
    correspondence = _behavioral_correspondence(steering, coefficients, scales)
    correspondence.to_csv(root / "steering/behavioral_correspondence.csv", index=False)

    gates_path = root / "gates.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    candidates = dose_summary[
        (dose_summary.task_family == "ALL")
        & (dose_summary.steering_mode == "neural_norm")
        & dose_summary.direction.isin(("action_orthogonal", "action_residualized"))
    ].copy()
    causal_rows = []
    for row in candidates.itertuples():
        null = random_summary[random_summary.layer == row.layer]
        null95 = float(null.linear_slope.abs().quantile(0.95))
        causal_rows.append(
            {
                "layer": int(row.layer),
                "direction": row.direction,
                "absolute_slope": abs(float(row.linear_slope)),
                "random_absolute_slope_95": null95,
                "passed": bool(abs(row.linear_slope) > null95),
            }
        )
    gates["gate_3_causal_specificity"] = {
        "passed": bool(causal_rows and any(row["passed"] for row in causal_rows)),
        "comparisons": causal_rows,
    }
    threshold = float(
        config.get("gates", {}).get("minimum_behavioral_correlation", 0.3)
    )
    correspondence_summary = correspondence[
        (correspondence.task_family == "ALL")
        & correspondence.direction.isin(("action_orthogonal", "action_residualized"))
    ]
    computational_rows = []
    for row in correspondence_summary.itertuples():
        task_r = float(row.task_slope_beta_correlation)
        predicted_r = float(row.predicted_observed_correlation)
        computational_rows.append(
            {
                "layer": int(row.layer),
                "direction": row.direction,
                "task_slope_beta_correlation": task_r,
                "predicted_observed_correlation": predicted_r,
                "passed": bool(task_r >= threshold and predicted_r >= threshold),
            }
        )
    gates["gate_4_computational_correspondence"] = {
        "passed": bool(
            computational_rows and any(row["passed"] for row in computational_rows)
        ),
        "minimum_correlation": threshold,
        "comparisons": computational_rows,
    }
    _json(gates_path, gates)

    decoding = pd.read_csv(root / "representation/residual_decoding.csv")
    similarity = pd.read_csv(root / "representation/direction_similarity.csv")
    matched = pd.read_csv(root / "representation/matched_state_results.csv")
    gate_values = [
        bool(gates[f"gate_{index}_{name}"]["passed"])
        for index, name in (
            (1, "representational_independence"),
            (2, "matched_state_specificity"),
            (3, "causal_specificity"),
            (4, "computational_correspondence"),
        )
    ]
    raw_gradient_l30 = similarity[
        (similarity.layer == 30)
        & similarity.apply(
            lambda row: frozenset((row.direction_a, row.direction_b))
            == frozenset(("action_raw", "gradient")),
            axis=1,
        )
    ]
    gradient_overlap = (
        abs(float(raw_gradient_l30.cosine.iloc[0])) if len(raw_gradient_l30) else np.nan
    )
    raw_l30 = dose_summary[
        (dose_summary.layer == 30)
        & (dose_summary.direction == "action_raw")
        & (dose_summary.steering_mode == "neural_norm")
        & (dose_summary.task_family == "ALL")
    ]
    random_l30 = random_summary[random_summary.layer == 30]
    raw_unusual = bool(
        len(raw_l30)
        and abs(float(raw_l30.linear_slope.iloc[0]))
        > float(random_l30.linear_slope.abs().quantile(0.95))
    )
    if all(gate_values):
        outcome = "H1_upstream_action_history_implementation"
        claim = (
            "Action history is represented independently of the current decision and "
            "causally contributes to persistence with behavioral-model correspondence."
        )
    elif (
        gates["gate_3_causal_specificity"]["passed"]
        and not gates["gate_4_computational_correspondence"]["passed"]
    ):
        outcome = "H2_shared_late_stay_switch_state"
        claim = (
            "The separable direction is causal, but its task pattern does not justify "
            "identifying it with the behavioral action-history computation."
        )
    elif not raw_unusual and np.isfinite(gradient_overlap) and gradient_overlap >= 0.7:
        outcome = "H3_persistence_readout_artifact"
        claim = (
            "The clean late steering effect is best explained as manipulation of an "
            "already-formed persistence/readout state."
        )
    else:
        outcome = "H4_mixed_representation"
        claim = (
            "The selected late direction mixes action-history information with downstream "
            "decision geometry; only the surviving controlled fraction is supported."
        )

    figure_paths = _make_figures(
        root,
        decoding,
        similarity,
        matched,
        steering,
        random_detail,
        correspondence,
        coefficients,
    )
    repro = json.loads((root / "directions/metadata.json").read_text(encoding="utf-8"))
    repro_l30 = next(row for row in repro["direction_rows"] if row["layer"] == 30)
    raw_persistence_l30 = similarity[
        (similarity.layer == 30)
        & similarity.apply(
            lambda row: frozenset((row.direction_a, row.direction_b))
            == frozenset(("action_raw", "persistence")),
            axis=1,
        )
    ]
    persistence_overlap = (
        float(raw_persistence_l30.cosine.iloc[0])
        if len(raw_persistence_l30)
        else np.nan
    )
    best_behavior = (
        float(correspondence_summary.task_slope_beta_correlation.max())
        if len(correspondence_summary)
        else np.nan
    )
    random_rho95 = float(random_l30.monotonic_rho.abs().quantile(0.95))
    raw_rho = float(raw_l30.monotonic_rho.iloc[0]) if len(raw_l30) else np.nan
    raw_l8 = dose_summary[
        (dose_summary.layer == 8)
        & (dose_summary.direction == "action_raw")
        & (dose_summary.steering_mode == "neural_norm")
        & (dose_summary.task_family == "ALL")
    ]
    raw_l8_slope = float(raw_l8.linear_slope.iloc[0]) if len(raw_l8) else np.nan
    causal_by_key = {(row["layer"], row["direction"]): row for row in causal_rows}
    l30_orthogonal_pass = causal_by_key.get(
        (30, "action_orthogonal"), {"passed": False}
    )["passed"]
    l30_residual_pass = causal_by_key.get(
        (30, "action_residualized"), {"passed": False}
    )["passed"]
    answers = [
        f"L8 action-history validation R² is {_lookup_metric(decoding, layer=8, direction='action_raw', target='action_history', split='validation'):.3f}.",
        f"L30 action-history validation R² is {_lookup_metric(decoding, layer=30, direction='action_raw', target='action_history', split='validation'):.3f}.",
        f"Current-persistence overlap is quantified in direction_similarity.csv; L30 raw-versus-persistence cosine is {persistence_overlap:.3f}.",
        f"Train-only residual action history remains above its random control: Gate 1 is {gates['gate_1_representational_independence']['passed']}.",
        f"Decision-matched history separation passes Gate 2: {gates['gate_2_matched_state_specificity']['passed']}.",
        f"The L30 action-history/persistence-probe cosine is {persistence_overlap:.3f}.",
        f"The L30 action-history/direct-gradient cosine is {gradient_overlap:.3f}.",
        f"L30 raw monotonicity |ρ|={abs(raw_rho):.3f}; random-direction 95th percentile |ρ|={random_rho95:.3f}.",
        f"Raw L8 steering has an all-task matched-norm slope of {raw_l8_slope:.3f}.",
        f"Residualized L8 causal specificity is included in Gate 3, which is {gates['gate_3_causal_specificity']['passed']}.",
        f"Raw L30 clean-vector reproduction cosine is {repro_l30['reproduction_cosine']:.6f}; its steering curve is reported in Figure 4.",
        f"L30 output-subspace-orthogonal steering exceeds the matched random null: {l30_orthogonal_pass}.",
        f"L30 statistically residualized steering exceeds the matched random null: {l30_residual_pass}.",
        f"The best task-slope/frozen-beta correlation is {best_behavior:.3f}; Gate 4 is {gates['gate_4_computational_correspondence']['passed']}.",
        f"The preregistered evidence rules select {outcome}.",
        claim,
    ]
    report_lines = [
        "# Action-History Direction Disambiguation",
        "",
        f"**Outcome:** `{outcome}`",
        "",
        f"**Justified claim:** {claim}",
        "",
        "## Primary gates",
        "",
        *[
            f"- Gate {index}: **{'PASS' if passed else 'FAIL'}**"
            for index, passed in enumerate(gate_values, start=1)
        ],
        "",
        "## Required questions",
        "",
        *[f"{index}. {answer}" for index, answer in enumerate(answers, start=1)],
        "",
        "## Artifact and interpretation guardrails",
        "",
        "- The exact original L30 vector and frozen behavioral coefficients were not refit.",
        "- All residualization and direction fitting used training rows only.",
        "- Primary causal comparisons use equal residual-stream displacement norms.",
        f"- The random null contains {expected_random} directions at each primary layer and all seven doses.",
        "- No full activation matrices were written.",
    ]
    report_path = root / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "report_generated": True,
            "outcome_classification": outcome,
            "gate_results": {
                f"gate_{i + 1}": value for i, value in enumerate(gate_values)
            },
            "figure_count": len(figure_paths),
            "random_direction_count_observed": int(
                random_summary.random_index.nunique()
            ),
            "full_activations_saved": False,
        }
    )
    _json(metadata_path, metadata)
    return {
        "report": report_path,
        "outcome": outcome,
        "figures": figure_paths,
        "gates": gates,
    }

"""Artifact-oriented orchestration for the mechanistic follow-up."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.experiments.registry import get_renderer

from .activations.qwen_runner import MechanisticQwenRunner
from .calibration.projection_to_computation import DirectionCalibration
from .dataset.matched_conditions import (
    compile_mechanistic_conditions,
    load_mechanistic_manifest,
    write_mechanistic_manifest,
)
from .patching.mediation import mediation_summary
from .patching.projection_patch import projection_patch
from .representations.mean_difference import (
    MeanDifferenceDirections,
    load_directions,
    preliminary_candidate_layers,
    save_directions,
)
from .representations.ridge_probe import (
    RidgeProbe,
    contrast_sign_accuracy,
    probe_metrics,
)
from .representations.specificity import specificity_table
from .steering.controls import matched_orthogonal_direction, random_directions
from .steering.dose_response import dose_response_metrics
from .steering.intervene import direction_editor
from .steering.quantitative_predictions import causal_prediction_metrics
from .targets.behavioral_targets import (
    TARGET_COLUMNS,
    load_behavioral_coefficients,
    load_behavioral_normalization,
    load_frozen_handoff,
)


DIRECTION_FILENAMES = {
    "outcome_history": "outcome_history.safetensors",
    "contextual_outcome_history": "contextual_history.safetensors",
    "action_history": "action_history.safetensors",
    "generic_value": "generic_value_control.safetensors",
}
DIRECTION_FAMILIES = {
    "outcome_history": "outcome_history",
    "contextual_history": "contextual_outcome_history",
    "action_history": "action_history",
    "current_value_control": "generic_value",
}


def _root(config: dict, output=None) -> Path:
    return Path(output or config.get("output_root", "artifacts/mechanistic_v1"))


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


def _manifest_path(root: Path) -> Path:
    return root / "manifests/mechanistic_conditions.jsonl"


def _projection_path(root: Path) -> Path:
    parquet = root / "representation/projections.parquet"
    return parquet if parquet.exists() else parquet.with_suffix(".csv.gz")


def _direction_path(root: Path, target: str) -> Path:
    return root / "directions" / DIRECTION_FILENAMES[target]


def _load_all_directions(root: Path) -> dict[str, dict[int, np.ndarray]]:
    directions = {}
    for target, filename in DIRECTION_FILENAMES.items():
        path = root / "directions" / filename
        if path.exists():
            directions.update(load_directions(path))
    if not directions:
        raise FileNotFoundError("no fitted mechanistic directions were found")
    return directions


def prepare_mechanistic_run(
    config: dict,
    *,
    theory_output: str | Path,
    output: str | Path | None = None,
    smoke: bool = False,
):
    """Freeze targets, prompt hashes, paired conditions, and behavioral coefficients."""

    root = _root(config, output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "calibration").mkdir(parents=True, exist_ok=True)
    handoff, behavioral_hash = load_frozen_handoff(theory_output)
    conditions = compile_mechanistic_conditions(config, smoke=smoke)
    manifest = write_mechanistic_manifest(conditions, root / "manifests")
    prompt_rows = []
    for record in conditions:
        trial = get_renderer(record.condition.task_family).render(record.condition)
        prompt_rows.append(
            {
                "condition_id": record.condition.condition_id,
                "prompt_hash": trial.prompt_hash,
                "response_mapping": record.condition.response_mapping.mapping_id,
            }
        )
    pd.DataFrame(prompt_rows).to_csv(root / "manifests/prompt_hashes.csv", index=False)
    coefficients = []
    normalizations = {}
    for target in TARGET_COLUMNS:
        frozen, source = load_behavioral_coefficients(
            theory_output, handoff, target=target
        )
        normalization, normalization_source = load_behavioral_normalization(
            theory_output, handoff, target=target
        )
        normalizations[target] = normalization
        for task, value in frozen.items():
            coefficients.append(
                {
                    "target": target,
                    "task_family": task,
                    "behavioral_coefficient": value,
                    "source": source,
                    "computational_mean": normalization["mean"],
                    "computational_scale": normalization["scale"],
                    "normalization_source": normalization_source,
                }
            )
    pd.DataFrame(coefficients).to_csv(
        root / "calibration/frozen_behavioral_coefficients.csv", index=False
    )
    split_manifest = {
        split: sorted(
            {
                record.contrast_id
                for record in conditions
                if record.condition.split == split
            }
        )
        for split in ("train", "validation", "test", "transfer")
    }
    _json(root / "manifests/split_manifest.json", split_manifest)
    manifest_path = Path(manifest["path"])
    metadata = {
        "protocol_version": config.get("protocol_version", "mechanistic_v1"),
        "model_id": config.get("model", "Qwen/Qwen3.5-4B"),
        "model_revision": config.get("model_revision"),
        "git_commit": _git_commit(),
        "behavioral_model_hash": behavioral_hash,
        "mechanistic_condition_hash": _sha256(manifest_path),
        "activation_position": "final_prompt_token",
        "layer_index_convention": "zero_based_block_output",
        "direction_fitting_split": "train",
        "random_seed": int(config.get("seed", 73001)),
        "full_activations_saved": False,
        "behavioral_theory_outcome": handoff.get("theory_outcome"),
        "behavioral_handoff_ready_flag": bool(
            handoff.get("mechanistic_analysis_ready", False)
        ),
        "computational_target_normalization": normalizations,
        "smoke": bool(smoke),
    }
    _json(root / "run_metadata.json", metadata)
    return {
        "manifest": manifest_path,
        "metadata": root / "run_metadata.json",
        **manifest,
    }


def _make_runner(config, *, model_path=None, revision=None, online=False):
    return MechanisticQwenRunner.from_pretrained(
        model_path or config["model"],
        revision=revision or config.get("model_revision"),
        local_files_only=not online,
    )


def _forward_record(runner, record, *, editors=None, capture_layers=None):
    trial = get_renderer(record.condition.task_family).render(record.condition)
    return runner.forward(
        list(trial.messages),
        record.condition.response_mapping.labels,
        positive_label=record.condition.response_mapping.continue_label,
        editors=editors,
        capture_layers=capture_layers,
    )


def scan_directions(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """First GPU pass: stream paired differences into one mean per layer/target."""

    root = _root(config, output)
    records = load_mechanistic_manifest(_manifest_path(root))
    groups = defaultdict(list)
    for record in records:
        if (
            record.condition.split == "train"
            and record.contrast_family in DIRECTION_FAMILIES
        ):
            groups[record.contrast_id].append(record)
    selected_groups = sorted(groups.items())[:limit]
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    scanner = MeanDifferenceDirections()
    processed = 0
    for contrast_id, group in selected_groups:
        member_states = {1: defaultdict(list), -1: defaultdict(list)}
        for record in group:
            result = _forward_record(runner, record)
            for layer, state in result.states.items():
                member_states[record.contrast_member][layer].append(state)
        target = DIRECTION_FAMILIES[group[0].contrast_family]
        for layer in range(runner.layer_count):
            plus = np.mean(member_states[1][layer], axis=0)
            minus = np.mean(member_states[-1][layer], axis=0)
            scanner.update(target, layer, plus - minus)
        processed += 1
    directions, rows = scanner.directions()
    direction_root = root / "directions"
    direction_root.mkdir(parents=True, exist_ok=True)
    for target, layer_vectors in directions.items():
        save_directions(
            _direction_path(root, target),
            {target: layer_vectors},
            metadata={"pairs": processed, "fitting_split": "train"},
        )
    preliminary = preliminary_candidate_layers(
        rows,
        count=int(
            config.get("representations", {}).get("preliminary_layers_per_target", 4)
        ),
    )
    _json(
        direction_root / "metadata.json",
        {
            "layer_count": runner.layer_count,
            "layer_index_convention": runner.layer_convention,
            "activation_position": runner.activation_position,
            "full_activations_saved": False,
            "direction_statistics": rows,
            "preliminary_candidate_layers": preliminary,
        },
    )
    run_metadata_path = root / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text())
    participant = getattr(runner, "participant", None)
    run_metadata.update(
        {
            "observed_model_id": getattr(participant, "model_id", config.get("model")),
            "observed_model_revision": getattr(
                participant, "revision", config.get("model_revision")
            ),
            "layer_count": runner.layer_count,
        }
    )
    _json(run_metadata_path, run_metadata)
    return {
        "directions": sorted(directions),
        "pairs": processed,
        "layers": runner.layer_count,
    }


class _VectorMoments:
    def __init__(self, count: int):
        self.n = 0
        self.sx = np.zeros(count)
        self.sx2 = np.zeros(count)
        self.sy = 0.0
        self.sy2 = 0.0
        self.sxy = np.zeros(count)

    def update(self, projection, target):
        x = np.asarray(projection, dtype=float)
        y = float(target)
        self.n += 1
        self.sx += x
        self.sx2 += np.square(x)
        self.sy += y
        self.sy2 += y * y
        self.sxy += x * y


def _random_control_rows(moments, *, alpha: float = 1.0):
    rows = []
    keys = sorted({key[:2] for key in moments})
    for target, layer in keys:
        train = moments.get((target, layer, "train"))
        if train is None or train.n < 2:
            continue
        centered_xx = train.sx2 - np.square(train.sx) / train.n
        centered_xy = train.sxy - train.sx * train.sy / train.n
        slope = centered_xy / (centered_xx + alpha)
        intercept = train.sy / train.n - slope * train.sx / train.n
        for split in ("validation", "test", "transfer"):
            test = moments.get((target, layer, split))
            if test is None or test.n < 2:
                continue
            sse = (
                test.sy2
                - 2 * intercept * test.sy
                - 2 * slope * test.sxy
                + test.n * np.square(intercept)
                + 2 * intercept * slope * test.sx
                + np.square(slope) * test.sx2
            )
            sst = test.sy2 - test.sy * test.sy / test.n
            centered_xy_test = test.sxy - test.sx * test.sy / test.n
            centered_xx_test = test.sx2 - np.square(test.sx) / test.n
            centered_yy_test = sst
            correlation = centered_xy_test / np.sqrt(
                np.maximum(centered_xx_test * centered_yy_test, 1e-30)
            )
            correlation = np.sign(slope) * correlation
            for index in range(len(slope)):
                rows.append(
                    {
                        "target": target,
                        "layer": int(layer),
                        "random_index": index,
                        "split": split,
                        "r2": float(1 - sse[index] / sst) if sst > 0 else float("nan"),
                        "pearson_r": float(correlation[index]),
                        "mse": float(sse[index] / test.n),
                    }
                )
    return rows


def scan_projections(
    config: dict,
    *,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Second GPU pass: save scalar projections, never hidden vectors."""

    root = _root(config, output)
    records = load_mechanistic_manifest(_manifest_path(root))
    records = records[:limit] if limit is not None else records
    directions = _load_all_directions(root)
    metadata = json.loads((root / "directions/metadata.json").read_text())
    preliminary = metadata["preliminary_candidate_layers"]
    random_count = int(config.get("representations", {}).get("random_controls", 100))
    seed = int(config.get("seed", 73001))
    random_vectors = {
        (target, int(layer)): random_directions(
            len(directions[target][int(layer)]),
            random_count,
            seed=seed + 1009 * int(layer) + sum(map(ord, target)),
        )
        for target, layers in preliminary.items()
        if target in directions and target in TARGET_COLUMNS
        for layer in layers
    }
    moments = {
        (target, layer, split): _VectorMoments(random_count)
        for target, layer in random_vectors
        for split in ("train", "validation", "test", "transfer")
    }
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    rows = []
    for record in records:
        result = _forward_record(runner, record)
        mapping_sign = (
            1.0
            if record.condition.response_mapping.continue_label
            < record.condition.response_mapping.disengage_label
            else -1.0
        )
        common = {
            "condition_id": record.condition.condition_id,
            "paired_condition_id": record.condition.paired_condition_id,
            "contrast_id": record.contrast_id,
            "contrast_family": record.contrast_family,
            "contrast_member": record.contrast_member,
            "task_family": record.condition.task_family,
            "split": record.condition.split,
            "response_mapping_sign": mapping_sign,
            "persistence_logit": result.persistence_logit,
            **record.targets,
        }
        for target, layer_vectors in directions.items():
            if target not in TARGET_COLUMNS:
                continue
            for layer, direction in layer_vectors.items():
                state = result.states[layer]
                rows.append(
                    {
                        **common,
                        "direction": target,
                        "layer": layer,
                        "projection": float(state @ direction),
                    }
                )
                key = (target, layer)
                if key in random_vectors:
                    moments[(target, layer, record.condition.split)].update(
                        random_vectors[key] @ state, record.targets[target]
                    )
    representation_root = root / "representation"
    representation_root.mkdir(parents=True, exist_ok=True)
    projection_path = write_records(rows, representation_root / "projections.parquet")
    random_rows = _random_control_rows(
        moments, alpha=float(config.get("representations", {}).get("ridge_alpha", 1.0))
    )
    pd.DataFrame(random_rows).to_csv(
        representation_root / "random_controls.csv", index=False
    )
    return {
        "rows": len(rows),
        "path": projection_path,
        "random_controls": len(random_rows),
    }


def _collapse_mappings(frame: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "paired_condition_id",
        "contrast_id",
        "contrast_family",
        "contrast_member",
        "task_family",
        "split",
        "direction",
        "layer",
    ]
    values = [
        "projection",
        "persistence_logit",
        *TARGET_COLUMNS,
        "success_evidence",
        "progress_evidence",
        "continuation_cost",
        "disengagement_value",
        "continuation_value",
    ]
    aggregations = {name: "mean" for name in values if name in frame}
    return frame.groupby(identity, as_index=False).agg(aggregations)


def _score_probe(train, test, *, target: str, alpha: float):
    probe = RidgeProbe(alpha=alpha).fit(train[["projection"]], train[target])
    prediction = probe.predict(test[["projection"]])
    metrics = probe_metrics(test[target], prediction)
    family = {
        "outcome_history": "outcome_history",
        "contextual_outcome_history": "contextual_history",
        "action_history": "action_history",
    }[target]
    relevant = test.contrast_family.astype(str) == family
    metrics["contrast_sign_accuracy"] = (
        contrast_sign_accuracy(test[relevant], prediction[relevant])
        if relevant.any()
        else float("nan")
    )
    return probe, metrics


def analyze_representations(config: dict, *, output: str | Path | None = None):
    """Fit frozen probes, LOTO tests, controls, contextual indices, and gates."""

    root = _root(config, output)
    raw = read_records(_projection_path(root))
    frame = _collapse_mappings(raw)
    alpha = float(config.get("representations", {}).get("ridge_alpha", 1.0))
    metric_rows, loto_rows, specificity_rows = [], [], []
    for (target, layer), part in frame.groupby(["direction", "layer"]):
        train = part[part.split == "train"]
        if len(train) < 2:
            continue
        for split in ("validation", "test", "transfer"):
            test = part[part.split == split]
            if len(test) < 2:
                continue
            _, metrics = _score_probe(train, test, target=target, alpha=alpha)
            task_metrics = []
            probe = RidgeProbe(alpha=alpha).fit(train[["projection"]], train[target])
            for _, task_frame in test.groupby("task_family"):
                if len(task_frame) > 1:
                    task_metrics.append(
                        probe_metrics(
                            task_frame[target],
                            probe.predict(task_frame[["projection"]]),
                        )["r2"]
                    )
            metric_rows.append(
                {
                    "target": target,
                    "layer": int(layer),
                    "split": split,
                    **metrics,
                    "task_macro_r2": (
                        float(np.mean(task_metrics)) if task_metrics else float("nan")
                    ),
                }
            )
        discovery_tasks = sorted(part[part.split != "transfer"].task_family.unique())
        for task in discovery_tasks:
            source = part[(part.split == "train") & (part.task_family != task)]
            test = part[(part.split == "test") & (part.task_family == task)]
            if len(source) < 2 or len(test) < 2:
                continue
            _, metrics = _score_probe(source, test, target=target, alpha=alpha)
            loto_rows.append(
                {
                    "target": target,
                    "layer": int(layer),
                    "heldout_task": task,
                    "target_normalization_used": False,
                    **metrics,
                }
            )
        selected = raw[(raw.direction == target) & (raw.layer == layer)]
        table = specificity_table(selected)
        for row in table.to_dict("records"):
            specificity_rows.append({"target": target, "layer": int(layer), **row})
    metrics = pd.DataFrame(metric_rows)
    loto = pd.DataFrame(loto_rows)
    specificity = pd.DataFrame(specificity_rows)
    representation_root = root / "representation"
    metrics.to_csv(representation_root / "layer_metrics.csv", index=False)
    loto.to_csv(representation_root / "loto_metrics.csv", index=False)
    specificity.to_csv(representation_root / "specificity.csv", index=False)

    contextual_rows = []
    for layer in sorted(set(frame.layer)):
        contextual = frame[
            (frame.direction == "contextual_outcome_history") & (frame.layer == layer)
        ]
        train, test = (
            contextual[contextual.split == "train"],
            contextual[contextual.split == "test"],
        )
        if len(train) < 3 or len(test) < 3:
            continue
        baseline = RidgeProbe(alpha=alpha, rank=1).fit(
            train[["outcome_history"]], train.contextual_outcome_history
        )
        full = RidgeProbe(alpha=alpha, rank=2).fit(
            train[["outcome_history", "projection"]], train.contextual_outcome_history
        )
        baseline_r2 = probe_metrics(
            test.contextual_outcome_history, baseline.predict(test[["outcome_history"]])
        )["r2"]
        full_r2 = probe_metrics(
            test.contextual_outcome_history,
            full.predict(test[["outcome_history", "projection"]]),
        )["r2"]
        raw_r2_rows = metrics[
            (metrics.target == "outcome_history")
            & (metrics.layer == layer)
            & (metrics.split == "test")
        ]
        context_r2_rows = metrics[
            (metrics.target == "contextual_outcome_history")
            & (metrics.layer == layer)
            & (metrics.split == "test")
        ]
        contextual_rows.append(
            {
                "layer": int(layer),
                "raw_r2": float(raw_r2_rows.r2.iloc[0]) if len(raw_r2_rows) else np.nan,
                "contextual_r2": (
                    float(context_r2_rows.r2.iloc[0])
                    if len(context_r2_rows)
                    else np.nan
                ),
                "contextual_transformation_index": (
                    float(context_r2_rows.r2.iloc[0] - raw_r2_rows.r2.iloc[0])
                    if len(raw_r2_rows) and len(context_r2_rows)
                    else np.nan
                ),
                "baseline_contextual_r2": baseline_r2,
                "partial_contextual_delta_r2": full_r2 - baseline_r2,
            }
        )
    pd.DataFrame(contextual_rows).to_csv(
        representation_root / "contextual_transformation.csv", index=False
    )

    preliminary = json.loads((root / "directions/metadata.json").read_text())[
        "preliminary_candidate_layers"
    ]
    random_controls = pd.read_csv(representation_root / "random_controls.csv")
    settings = config.get("representations", {})
    minimum_r2 = float(settings.get("minimum_validation_r2", 0.0))
    minimum_r = float(settings.get("minimum_validation_pearson_r", 0.1))
    minimum_sign = float(settings.get("minimum_contrast_sign_accuracy", 0.6))
    maximum_mapping = float(settings.get("maximum_response_mapping_r2", 0.1))
    maximum = int(settings.get("candidate_layers", 4))
    minimum_loto = float(settings.get("minimum_loto_r2", 0.0))
    validation = metrics[metrics.split == "validation"].copy()
    validation = validation[
        validation.apply(
            lambda row: int(row.layer) in preliminary.get(str(row.target), []), axis=1
        )
    ]
    candidates = []
    for row in validation.sort_values(
        ["r2", "pearson_r"], ascending=False
    ).itertuples():
        controls = random_controls[
            (random_controls.target == row.target)
            & (random_controls.layer == row.layer)
            & (random_controls.split == "validation")
        ]
        random_95 = float(controls.r2.quantile(0.95)) if len(controls) else np.nan
        heldout_loto = loto[(loto.target == row.target) & (loto.layer == row.layer)]
        loto_r2 = float(heldout_loto.r2.mean()) if len(heldout_loto) else np.nan
        mapping_rows = specificity[
            (specificity.target == row.target)
            & (specificity.layer == row.layer)
            & (specificity.control == "response_mapping_sign")
        ]
        mapping_r2 = (
            float(mapping_rows.shared_variance.iloc[0]) if len(mapping_rows) else np.nan
        )
        representation_gate = bool(
            row.r2 >= minimum_r2
            and np.isfinite(row.pearson_r)
            and row.pearson_r >= minimum_r
            and (not np.isfinite(random_95) or row.r2 > random_95)
            and (not np.isfinite(loto_r2) or loto_r2 >= minimum_loto)
            and np.isfinite(row.contrast_sign_accuracy)
            and row.contrast_sign_accuracy >= minimum_sign
            and (not np.isfinite(mapping_r2) or mapping_r2 <= maximum_mapping)
        )
        candidates.append(
            {
                "target": row.target,
                "layer": int(row.layer),
                "validation_r2": float(row.r2),
                "validation_pearson_r": float(row.pearson_r),
                "loto_r2": loto_r2,
                "random_r2_95": random_95,
                "contrast_sign_accuracy": float(row.contrast_sign_accuracy),
                "response_mapping_r2": mapping_r2,
                "representation_gate": representation_gate,
            }
        )
    candidates = pd.DataFrame(candidates)
    if not candidates.empty:
        candidates["selected"] = False
        tolerance = float(settings.get("earlier_layer_r2_tolerance", 0.02))
        proposed = []
        target_order = (
            "contextual_outcome_history",
            "outcome_history",
            "action_history",
        )
        for target in target_order:
            passed = candidates[
                (candidates.target == target) & candidates.representation_gate
            ]
            if passed.empty:
                continue
            peak_index = passed.validation_r2.idxmax()
            peak_r2 = float(candidates.loc[peak_index, "validation_r2"])
            comparable = passed[passed.validation_r2 >= peak_r2 - tolerance]
            early_index = comparable.layer.idxmin()
            proposed.append(early_index)
            if peak_index != early_index:
                proposed.append(peak_index)
        for index in list(dict.fromkeys(proposed))[:maximum]:
            candidates.loc[index, "selected"] = True
    candidates.to_csv(representation_root / "candidate_layers.csv", index=False)

    calibration_rows = []
    frozen_coefficients = pd.read_csv(
        root / "calibration/frozen_behavioral_coefficients.csv"
    )
    if not candidates.empty:
        for row in candidates[candidates.selected].itertuples():
            part = frame[(frame.direction == row.target) & (frame.layer == row.layer)]
            train = part[part.split == "train"]
            validation_part = part[part.split == "validation"]
            normalization = frozen_coefficients[
                frozen_coefficients.target.astype(str) == row.target
            ].iloc[0]
            calibration = DirectionCalibration.fit(
                train.projection,
                train[row.target],
                target_mean=float(normalization.computational_mean),
                target_scale=float(normalization.computational_scale),
            )
            heldout = (
                calibration.evaluate(
                    validation_part.projection, validation_part[row.target]
                )
                if len(validation_part) > 1
                else {"r2": np.nan, "pearson_r": np.nan, "mse": np.nan}
            )
            calibration_rows.append(
                {
                    "target": row.target,
                    "layer": int(row.layer),
                    **calibration.to_dict(),
                    **{f"heldout_{key}": value for key, value in heldout.items()},
                    "fitting_split": "train",
                    "computational_units": "frozen_behavioral_standard_deviations",
                }
            )
    calibration_root = root / "calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calibration_rows).to_csv(
        calibration_root / "direction_calibration.csv", index=False
    )
    gate = {
        "representation_passed": bool(len(candidates) and candidates.selected.any()),
        "selected_layers": (
            candidates[candidates.selected][["target", "layer"]].to_dict("records")
            if len(candidates)
            else []
        ),
        "steering_authorized": bool(len(candidates) and candidates.selected.any()),
    }
    _json(representation_root / "gates.json", gate)
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["selected_direction_layers"] = gate["selected_layers"]
    metadata["steering_calibration"] = "train-only standardized computational units"
    _json(metadata_path, metadata)
    return gate


def _selected_intervention_records(records, per_task: int):
    representatives = [
        record
        for record in records
        if record.condition.condition_id.endswith("-m0")
        and record.condition.split in {"test", "transfer"}
    ]
    output = []
    counts = defaultdict(int)
    for record in representatives:
        task = record.condition.task_family
        if counts[task] < per_task:
            output.append(record)
            counts[task] += 1
    return output


def run_causal_interventions(
    config: dict,
    *,
    theory_output: str | Path,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Third GPU pass: gated bidirectional steering and subspace patching."""

    root = _root(config, output)
    gate = json.loads((root / "representation/gates.json").read_text())
    steering_root, patching_root = root / "steering", root / "patching"
    steering_root.mkdir(parents=True, exist_ok=True)
    patching_root.mkdir(parents=True, exist_ok=True)
    if not gate["steering_authorized"]:
        pd.DataFrame().to_csv(steering_root / "steering_results.csv", index=False)
        pd.DataFrame().to_csv(patching_root / "projection_patching.csv", index=False)
        _json(steering_root / "stopped_by_gate.json", gate)
        return {"skipped": True, "reason": "representation gate failed"}
    directions = _load_all_directions(root)
    calibration_frame = pd.read_csv(root / "calibration/direction_calibration.csv")
    calibration = {
        (str(row.target), int(row.layer)): DirectionCalibration.from_dict(row._asdict())
        for row in calibration_frame.itertuples()
    }
    handoff, _ = load_frozen_handoff(theory_output)
    coefficients_by_target = {
        target: load_behavioral_coefficients(theory_output, handoff, target=target)[0]
        for target in TARGET_COLUMNS
    }
    records = load_mechanistic_manifest(_manifest_path(root))
    settings = config.get("steering", {})
    selected_records = _selected_intervention_records(
        records, int(settings.get("examples_per_task", 8))
    )
    selected_records = (
        selected_records[:limit] if limit is not None else selected_records
    )
    doses = tuple(
        float(value)
        for value in settings.get("computational_doses", (-2, -1, -0.5, 0, 0.5, 1, 2))
    )
    control_doses = tuple(
        float(value) for value in settings.get("control_doses", (-1, 1))
    )
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    steering_rows = []
    for selected in gate["selected_layers"]:
        target, layer = str(selected["target"]), int(selected["layer"])
        direction = directions[target][layer]
        calibrated = calibration[(target, layer)]
        coefficients = coefficients_by_target[target]
        orthogonal = matched_orthogonal_direction(
            direction, seed=int(config.get("seed", 73001)) + layer
        )
        random = random_directions(
            len(direction), 1, seed=int(config.get("seed", 73001)) + 10000 + layer
        )[0]
        generic_value = directions.get("generic_value", {}).get(layer)
        for record in selected_records:
            baseline = _forward_record(runner, record, capture_layers={layer})
            task = record.condition.task_family
            if task not in coefficients:
                continue
            trial = get_renderer(record.condition.task_family).render(record.condition)
            output_direction = runner.choice_output_direction(
                list(trial.messages),
                record.condition.response_mapping.labels,
                positive_label=record.condition.response_mapping.continue_label,
            )
            controls = [
                ("candidate", direction, doses),
                ("orthogonal", orthogonal, control_doses),
                ("random", random, control_doses),
                ("persistence_output", output_direction, control_doses),
            ]
            if generic_value is not None:
                controls.append(("generic_value", generic_value, control_doses))
            for control, vector, used_doses in controls:
                for dose in used_doses:
                    alpha = float(calibrated.alpha_for_delta(dose))
                    steered = _forward_record(
                        runner,
                        record,
                        editors={layer: direction_editor(vector, alpha)},
                        capture_layers={layer},
                    )
                    actual_projection_delta = float(
                        (steered.states[layer] - baseline.states[layer]) @ direction
                    )
                    steering_rows.append(
                        {
                            "condition_id": record.condition.condition_id,
                            "task_family": task,
                            "target": target,
                            "layer": layer,
                            "control": control,
                            "computational_dose": dose,
                            "activation_alpha": alpha,
                            "base_logit": baseline.persistence_logit,
                            "steered_logit": steered.persistence_logit,
                            "observed_logit_change": steered.persistence_logit
                            - baseline.persistence_logit,
                            "predicted_logit_change": coefficients[task] * dose,
                            "actual_candidate_projection_change": actual_projection_delta,
                            "calibrated_projection_change": calibrated.slope
                            * actual_projection_delta,
                        }
                    )
    steering_path = write_records(
        steering_rows, steering_root / "steering_results.parquet"
    )

    # Patching is evaluated on matched signed contrasts, never unrelated prompts.
    representatives = [
        record
        for record in records
        if record.condition.condition_id.endswith("-m0")
        and record.condition.split in {"test", "transfer"}
        and record.contrast_family in {"outcome_history", "contextual_history"}
    ]
    by_contrast = defaultdict(list)
    for record in representatives:
        by_contrast[record.contrast_id].append(record)
    patch_rows = []
    patch_limit = int(config.get("patching", {}).get("contrasts_per_family", 16))
    family_counts = defaultdict(int)
    for contrast_id, pair in sorted(by_contrast.items()):
        if len(pair) != 2:
            continue
        family = pair[0].contrast_family
        if family_counts[family] >= patch_limit:
            continue
        plus = next(item for item in pair if item.contrast_member == 1)
        minus = next(item for item in pair if item.contrast_member == -1)
        for selected in gate["selected_layers"]:
            target, layer = str(selected["target"]), int(selected["layer"])
            direction = directions[target][layer]
            source = _forward_record(runner, plus, capture_layers={layer})
            target_base = _forward_record(runner, minus, capture_layers={layer})

            def editor(state, source_state=source.states[layer], vector=direction):
                return projection_patch(state, source_state, vector)

            patched = _forward_record(
                runner, minus, editors={layer: editor}, capture_layers={layer}
            )
            patch_rows.append(
                {
                    "contrast_id": contrast_id,
                    "contrast_family": family,
                    "task_family": plus.condition.task_family,
                    "target": target,
                    "layer": layer,
                    "control": "candidate",
                    "source_projection": float(source.states[layer] @ direction),
                    "target_projection": float(target_base.states[layer] @ direction),
                    "patched_projection": float(patched.states[layer] @ direction),
                    "total_effect": source.persistence_logit
                    - target_base.persistence_logit,
                    "mediated_effect": patched.persistence_logit
                    - target_base.persistence_logit,
                }
            )
            random_patch_direction = random_directions(
                len(direction),
                1,
                seed=(
                    int(config.get("seed", 73001))
                    + 7919 * layer
                    + int(hashlib.sha256(contrast_id.encode()).hexdigest()[:8], 16)
                ),
            )[0]

            def random_editor(
                state,
                source_state=source.states[layer],
                vector=random_patch_direction,
            ):
                return projection_patch(state, source_state, vector)

            random_patched = _forward_record(
                runner,
                minus,
                editors={layer: random_editor},
                capture_layers={layer},
            )
            patch_rows.append(
                {
                    "contrast_id": contrast_id,
                    "contrast_family": family,
                    "task_family": plus.condition.task_family,
                    "target": target,
                    "layer": layer,
                    "control": "random",
                    "source_projection": float(
                        source.states[layer] @ random_patch_direction
                    ),
                    "target_projection": float(
                        target_base.states[layer] @ random_patch_direction
                    ),
                    "patched_projection": float(
                        random_patched.states[layer] @ random_patch_direction
                    ),
                    "total_effect": source.persistence_logit
                    - target_base.persistence_logit,
                    "mediated_effect": random_patched.persistence_logit
                    - target_base.persistence_logit,
                }
            )
        family_counts[family] += 1
    patch_path = write_records(
        patch_rows, patching_root / "projection_patching.parquet"
    )
    return {"skipped": False, "steering": steering_path, "patching": patch_path}


def finalize_mechanistic_run(config: dict, *, output: str | Path | None = None):
    """Aggregate causal results, render figures/report, and enforce size limits."""

    from .reporting.build_report import (
        build_mechanistic_report,
        generate_mechanistic_figures,
    )

    root = _root(config, output)

    def optional_csv(path: Path):
        try:
            return (
                pd.read_csv(path)
                if path.exists() and path.stat().st_size
                else pd.DataFrame()
            )
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    steering_path = root / "steering/steering_results.parquet"
    if not steering_path.exists():
        steering_path = steering_path.with_suffix(".csv.gz")
    if steering_path.exists():
        steering = read_records(steering_path)
    else:
        csv = root / "steering/steering_results.csv"
        steering = optional_csv(csv)
    causal_metrics = pd.DataFrame()
    dose_metrics = pd.DataFrame()
    control_summary = pd.DataFrame()
    computational_alignment_r = np.nan
    if not steering.empty:
        candidate = steering[steering.control == "candidate"]
        cells = candidate.groupby(
            ["target", "layer", "task_family", "computational_dose"], as_index=False
        )[["predicted_logit_change", "observed_logit_change"]].mean()
        metric_rows = []
        for (target, layer), part in cells.groupby(["target", "layer"]):
            metric_rows.append(
                {
                    "target": target,
                    "layer": layer,
                    **causal_prediction_metrics(
                        part.predicted_logit_change, part.observed_logit_change
                    ),
                }
            )
        causal_metrics = pd.DataFrame(metric_rows)
        causal_metrics.to_csv(root / "steering/predicted_vs_observed.csv", index=False)
        dose_rows = []
        for (target, layer, task), part in cells.groupby(
            ["target", "layer", "task_family"]
        ):
            dose_rows.append(
                {
                    "target": target,
                    "layer": layer,
                    "task_family": task,
                    **dose_response_metrics(
                        part.computational_dose, part.observed_logit_change
                    ),
                }
            )
        dose_metrics = pd.DataFrame(dose_rows)
        dose_metrics.to_csv(root / "steering/dose_response.csv", index=False)
        control_summary = steering.groupby(
            ["target", "layer", "control"], as_index=False
        ).observed_logit_change.agg(
            mean_effect="mean",
            mean_absolute_effect=lambda values: float(np.mean(np.abs(values))),
            observations="size",
        )
        control_summary.to_csv(root / "steering/control_summary.csv", index=False)
        if (
            len(candidate) > 1
            and np.std(candidate.computational_dose) > 0
            and np.std(candidate.calibrated_projection_change) > 0
        ):
            computational_alignment_r = float(
                np.corrcoef(
                    candidate.computational_dose,
                    candidate.calibrated_projection_change,
                )[0, 1]
            )
    patch_path = root / "patching/projection_patching.parquet"
    if not patch_path.exists():
        patch_path = patch_path.with_suffix(".csv.gz")
    patching = read_records(patch_path) if patch_path.exists() else pd.DataFrame()
    mediation = pd.DataFrame()
    if not patching.empty:
        mediation = mediation_summary(patching)
        mediation.to_csv(root / "patching/mediation_summary.csv", index=False)
        from .patching.contextual_patch import contextual_patch_advantage

        contextual_patch_advantage(patching).to_csv(
            root / "patching/contextual_patch_advantage.csv", index=False
        )
    gate_path = root / "representation/gates.json"
    gates = json.loads(gate_path.read_text())
    gate_settings = config.get("mechanistic_gates", {})
    minimum_causal = float(gate_settings.get("minimum_causal_correlation", 0.3))
    minimum_monotonic = float(gate_settings.get("minimum_monotonic_rho", 0.5))
    minimum_alignment = float(
        gate_settings.get("minimum_computational_alignment_correlation", 0.95)
    )
    causal_r = (
        float(causal_metrics.correlation.max()) if len(causal_metrics) else np.nan
    )
    monotonic_rho = (
        float(dose_metrics.monotonic_rho.mean()) if len(dose_metrics) else np.nan
    )
    candidate_control = control_summary[control_summary.control == "candidate"]
    random_control = control_summary[control_summary.control == "random"]
    steering_specific = bool(
        len(candidate_control)
        and len(random_control)
        and candidate_control.mean_absolute_effect.mean()
        > random_control.mean_absolute_effect.mean()
    )
    candidate_patch = (
        mediation[mediation.control == "candidate"] if len(mediation) else mediation
    )
    random_patch = (
        mediation[mediation.control == "random"] if len(mediation) else mediation
    )
    patch_specific = bool(
        len(candidate_patch)
        and len(random_patch)
        and candidate_patch.mediated_effect.abs().mean()
        > random_patch.mediated_effect.abs().mean()
    )
    gates.update(
        {
            "computational_alignment_passed": bool(
                np.isfinite(computational_alignment_r)
                and computational_alignment_r >= minimum_alignment
            ),
            "computational_alignment_correlation": computational_alignment_r,
            "steering_specificity_passed": steering_specific,
            "steering_causality_passed": bool(
                np.isfinite(causal_r)
                and causal_r >= minimum_causal
                and np.isfinite(monotonic_rho)
                and monotonic_rho >= minimum_monotonic
                and steering_specific
            ),
            "causal_prediction_correlation": causal_r,
            "mean_monotonic_rho": monotonic_rho,
            "mediation_passed": patch_specific,
            "projection_patch_specificity_passed": patch_specific,
        }
    )
    _json(gate_path, gates)
    generate_mechanistic_figures(root)
    report = build_mechanistic_report(root)
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["report_generated"] = True
    metadata["full_activation_artifacts"] = []
    _json(metadata_path, metadata)
    return {"report": report}

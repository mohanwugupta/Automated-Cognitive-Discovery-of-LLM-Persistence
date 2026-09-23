"""Frozen source-trained representational analyses (Level 3 only)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def _hash_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _hash_groups(groups) -> str:
    return hashlib.sha256(
        json.dumps(sorted(map(str, groups)), separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class FrozenDecoder:
    model: str
    source_task: str
    variable: str
    layer: int
    alpha: float
    coefficients: np.ndarray
    intercept: float
    training_group_sha256: str
    decoder_sha256: str


def fit_source_decoder(
    features,
    targets,
    *,
    model: str,
    source_task: str,
    variable: str,
    layer: int,
    semantic_group_ids,
    alpha: float,
) -> FrozenDecoder:
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    groups = list(semantic_group_ids)
    if features.ndim != 2 or len(features) != len(targets) or len(groups) != len(targets):
        raise ValueError("decoder features, targets, and semantic groups must align")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("decoder inputs must be finite")
    fit = Ridge(alpha=float(alpha)).fit(features, targets)
    coefficients = np.asarray(fit.coef_, dtype=float).copy()
    payload = {
        "model": model, "source_task": source_task, "variable": variable,
        "layer": int(layer), "alpha": float(alpha),
        "coefficient_sha256": _hash_array(coefficients),
        "intercept": float(fit.intercept_),
        "training_group_sha256": _hash_groups(groups),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    coefficients.setflags(write=False)
    return FrozenDecoder(
        model=model, source_task=source_task, variable=variable, layer=int(layer),
        alpha=float(alpha), coefficients=coefficients,
        intercept=float(fit.intercept_), training_group_sha256=payload["training_group_sha256"],
        decoder_sha256=digest,
    )


def evaluate_frozen_decoder(
    decoder: FrozenDecoder,
    features,
    targets,
    *,
    target_task: str,
    semantic_group_ids,
) -> dict:
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    groups = list(semantic_group_ids)
    if features.ndim != 2 or features.shape[1] != len(decoder.coefficients):
        raise ValueError("target features do not match the frozen source decoder")
    if len(features) != len(targets) or len(groups) != len(targets):
        raise ValueError("target features, labels, and semantic groups must align")
    prediction = features @ decoder.coefficients + decoder.intercept
    correlation = (
        float(np.corrcoef(targets, prediction)[0, 1])
        if len(targets) > 1 and np.std(targets) > 0 and np.std(prediction) > 0
        else np.nan
    )
    return {
        "model": decoder.model, "source_task": decoder.source_task,
        "fit_task": decoder.source_task, "target_task": target_task,
        "variable": decoder.variable, "layer": decoder.layer,
        "metric": "heldout_r2", "value": float(r2_score(targets, prediction)),
        "correlation": correlation, "n": int(len(targets)),
        "decoder_sha256": decoder.decoder_sha256,
        "target_group_sha256": _hash_groups(groups),
        "refit_on_target": False, "evidence_level": 3,
        "endpoint_id": "frozen_source_decoding_v1",
    }


def principal_angles(source_basis, target_basis) -> np.ndarray:
    source, _ = np.linalg.qr(np.asarray(source_basis, dtype=float))
    target, _ = np.linalg.qr(np.asarray(target_basis, dtype=float))
    singular = np.linalg.svd(source.T @ target, compute_uv=False)
    return np.arccos(np.clip(singular, -1.0, 1.0))


def validate_representation_transfer(rows) -> None:
    required = {
        "source_task", "target_task", "variable", "layer", "metric", "value",
        "mapping_robustness", "source_validity", "target_validity",
        "decoder_sha256", "refit_on_target", "endpoint_id",
    }
    missing = required - set(rows)
    if missing:
        raise ValueError(f"representation transfer lacks columns: {sorted(missing)}")
    if rows.refit_on_target.astype(bool).any():
        raise ValueError("source-trained decoder was refit on target")
    if set(rows.endpoint_id) != {"frozen_source_decoding_v1"}:
        raise ValueError("representation endpoint drift")


def run_representational_transfer(
    feature_rows: pd.DataFrame,
    *,
    output,
    alpha: float = 1.0,
) -> dict:
    """Fit on source-task train rows and evaluate untouched target-task test rows."""

    required = {
        "model", "task_family", "variable", "layer", "semantic_group",
        "response_mapping", "analysis_split", "target",
    }
    missing = required - set(feature_rows)
    if missing:
        raise ValueError(f"representation feature table lacks: {sorted(missing)}")
    feature_columns = sorted(
        (name for name in feature_rows if name.startswith("feature_")),
        key=lambda name: int(name.removeprefix("feature_")),
    )
    if not feature_columns:
        raise ValueError("representation table contains no compact feature columns")
    if feature_rows.groupby("semantic_group").analysis_split.nunique().max() != 1:
        raise ValueError("representation semantic group crossed neural splits")
    split_rows = (
        feature_rows[["semantic_group", "analysis_split"]].astype(str)
        .drop_duplicates().sort_values(["semantic_group", "analysis_split"])
        .to_dict("records")
    )
    neural_split_sha256 = hashlib.sha256(
        json.dumps(split_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows, decoders, loto_rows, sharing_rows = [], [], [], []
    keys = ["model", "variable", "layer"]
    for (model, variable, layer), block in feature_rows.groupby(keys, sort=True):
        tasks = sorted(block.task_family.astype(str).unique())
        for source_task in tasks:
            source = block[
                (block.task_family.astype(str) == source_task)
                & (block.analysis_split == "train")
            ]
            if source.semantic_group.nunique() < 3:
                continue
            decoder = fit_source_decoder(
                source[feature_columns], source.target,
                model=str(model), source_task=source_task, variable=str(variable),
                layer=int(layer), semantic_group_ids=source.semantic_group, alpha=alpha,
            )
            decoders.append({
                "model": model, "source_task": source_task, "variable": variable,
                "layer": int(layer), "decoder_sha256": decoder.decoder_sha256,
                "training_group_sha256": decoder.training_group_sha256,
                "coefficients": json.dumps(decoder.coefficients.tolist()),
                "intercept": decoder.intercept,
            })
            for target_task in tasks:
                target = block[
                    (block.task_family.astype(str) == target_task)
                    & (block.analysis_split == "test")
                ]
                if target.empty:
                    continue
                overlap = set(source.semantic_group.astype(str)) & set(target.semantic_group.astype(str))
                if overlap:
                    raise ValueError("representation source training leaked into target test groups")
                result = evaluate_frozen_decoder(
                    decoder, target[feature_columns], target.target,
                    target_task=target_task, semantic_group_ids=target.semantic_group,
                )
                mapping_values = []
                for _, mapping_part in target.groupby("response_mapping"):
                    if len(mapping_part) < 2:
                        continue
                    mapping_values.append(evaluate_frozen_decoder(
                        decoder, mapping_part[feature_columns], mapping_part.target,
                        target_task=target_task,
                        semantic_group_ids=mapping_part.semantic_group,
                    )["value"])
                result["mapping_robustness"] = (
                    float(max(mapping_values) - min(mapping_values))
                    if len(mapping_values) >= 2 else np.nan
                )
                result["source_validity"] = True
                result["target_validity"] = target.semantic_group.nunique() >= 3
                rows.append(result)
        # Leave-one-task-out: no target-task row contributes to the fit.
        for target_task in tasks:
            source = block[
                (block.task_family.astype(str) != target_task)
                & (block.analysis_split == "train")
            ]
            target = block[
                (block.task_family.astype(str) == target_task)
                & (block.analysis_split == "test")
            ]
            if source.semantic_group.nunique() < 3 or target.empty:
                continue
            decoder = fit_source_decoder(
                source[feature_columns], source.target, model=str(model),
                source_task="all_compatible_except_target", variable=str(variable),
                layer=int(layer), semantic_group_ids=source.semantic_group, alpha=alpha,
            )
            result = evaluate_frozen_decoder(
                decoder, target[feature_columns], target.target,
                target_task=target_task, semantic_group_ids=target.semantic_group,
            )
            result["excluded_task"] = target_task
            result["target_outcomes_used_for_fit"] = False
            loto_rows.append(result)
        train = block[block.analysis_split == "train"]
        test = block[block.analysis_split == "test"]
        if not train.empty and not test.empty:
            shared = Ridge(alpha=float(alpha)).fit(train[feature_columns], train.target)
            shared_prediction = shared.predict(test[feature_columns])
            sharing_rows.append({
                "model": model, "variable": variable, "layer": int(layer),
                "sharing_model": "one_shared", "r2": float(r2_score(test.target, shared_prediction)),
                "mse": float(np.mean((test.target.to_numpy() - shared_prediction) ** 2)),
                "heldout_conditions": True,
            })
            private_prediction = np.full(len(test), np.nan)
            shared_private_prediction = np.full(len(test), np.nan)
            aligned_prediction = np.full(len(test), np.nan)
            coefficients = []
            residual = train.target.to_numpy() - shared.predict(train[feature_columns])
            for task in tasks:
                train_mask = train.task_family.astype(str).to_numpy() == task
                test_mask = test.task_family.astype(str).to_numpy() == task
                if not train_mask.any() or not test_mask.any():
                    continue
                private = Ridge(alpha=float(alpha)).fit(
                    train.loc[train_mask, feature_columns], train.loc[train_mask, "target"]
                )
                correction = Ridge(alpha=float(alpha)).fit(
                    train.loc[train_mask, feature_columns], residual[train_mask]
                )
                coefficients.append(private.coef_ / max(np.linalg.norm(private.coef_), 1e-12))
                private_prediction[test_mask] = private.predict(test.loc[test_mask, feature_columns])
                shared_private_prediction[test_mask] = (
                    shared.predict(test.loc[test_mask, feature_columns])
                    + correction.predict(test.loc[test_mask, feature_columns])
                )
            if coefficients:
                aligned_direction = np.mean(coefficients, axis=0)
                aligned_direction /= max(np.linalg.norm(aligned_direction), 1e-12)
                train_projection = train[feature_columns].to_numpy() @ aligned_direction
                test_projection = test[feature_columns].to_numpy() @ aligned_direction
                # Stable task columns follow sorted tasks for both partitions.
                train_onehot = np.column_stack([
                    (train.task_family.astype(str).to_numpy() == task).astype(float) for task in tasks
                ])
                test_onehot = np.column_stack([
                    (test.task_family.astype(str).to_numpy() == task).astype(float) for task in tasks
                ])
                aligned_fit = Ridge(alpha=float(alpha)).fit(
                    np.column_stack((train_projection, train_onehot)), train.target
                )
                aligned_prediction = aligned_fit.predict(
                    np.column_stack((test_projection, test_onehot))
                )
            for name, prediction in (
                ("shared_plus_private", shared_private_prediction),
                ("aligned_task_specific", aligned_prediction),
                ("fully_task_specific", private_prediction),
            ):
                valid = np.isfinite(prediction)
                if not valid.any():
                    continue
                sharing_rows.append({
                    "model": model, "variable": variable, "layer": int(layer),
                    "sharing_model": name,
                    "r2": float(r2_score(test.target.to_numpy()[valid], prediction[valid])),
                    "mse": float(np.mean((test.target.to_numpy()[valid] - prediction[valid]) ** 2)),
                    "heldout_conditions": True,
                })
    transfer = pd.DataFrame(rows)
    if transfer.empty:
        raise RuntimeError("no valid source-trained representation models were fit")
    validate_representation_transfer(transfer)
    root = __import__("pathlib").Path(output)
    root.mkdir(parents=True, exist_ok=True)
    transfer.to_csv(root / "transfer_long.csv", index=False)
    transfer[transfer.source_task == transfer.target_task].to_csv(
        root / "within_task.csv", index=False
    )
    pd.DataFrame(decoders).to_csv(root / "decoder_manifest.csv", index=False)
    pd.DataFrame(loto_rows).to_csv(root / "loto_transfer.csv", index=False)
    pd.DataFrame(sharing_rows).to_csv(root / "shared_private_models.csv", index=False)
    (root / "neural_split_manifest.json").write_text(json.dumps({
        "schema_version": "cross-task-neural-split-v1",
        "sha256": neural_split_sha256, "groups": len(split_rows),
        "frozen_before_representation_fit": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "transfer_rows": len(transfer), "decoder_count": len(decoders),
        "loto_rows": len(loto_rows), "sharing_rows": len(sharing_rows),
        "neural_split_sha256": neural_split_sha256,
    }


def persistence_effect_geometry(
    rows: pd.DataFrame,
    *,
    rank: int,
) -> pd.DataFrame:
    """Compact effect-space overlap, holding manipulation families out of fitting."""

    required = {"task_family", "manipulation_family", "contrast_id", "contrast_member"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"effect geometry lacks columns: {sorted(missing)}")
    features = sorted(
        (name for name in rows if name.startswith("feature_")),
        key=lambda name: int(name.removeprefix("feature_")),
    )
    effects = []
    for keys, part in rows.groupby(["task_family", "manipulation_family", "contrast_id"]):
        low = part[part.contrast_member == -1]
        high = part[part.contrast_member == 1]
        if low.empty or high.empty:
            continue
        effects.append({
            "task_family": keys[0], "manipulation_family": keys[1],
            "contrast_id": keys[2],
            "delta": high[features].mean().to_numpy() - low[features].mean().to_numpy(),
        })
    result = []
    tasks = sorted({row["task_family"] for row in effects})
    for source in tasks:
        matrix = np.vstack([row["delta"] for row in effects if row["task_family"] == source])
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        source_basis = vh[: min(int(rank), len(vh))].T
        for target in tasks:
            target_matrix = np.vstack([row["delta"] for row in effects if row["task_family"] == target])
            _, _, target_vh = np.linalg.svd(target_matrix, full_matrices=False)
            target_basis = target_vh[: min(int(rank), len(target_vh))].T
            angles = principal_angles(source_basis, target_basis)
            result.append({
                "source_task": source, "target_task": target, "rank": int(rank),
                "mean_principal_angle": float(np.mean(angles)),
                "subspace_overlap": float(np.mean(np.cos(angles) ** 2)),
                "evidence_level": 3,
            })
    return pd.DataFrame(result)

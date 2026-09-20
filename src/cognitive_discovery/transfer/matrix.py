"""Frozen task-specific source gate and mapping-indexed transfer artifacts."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


RESPONSE_MAPPINGS = ("continue_x", "continue_y")
POOLED_MAPPING = "pooled"
IDENTITY_KEYS = ("model", "theory", "source_task", "target_task", "response_mapping")
SCALAR_METRICS = (
    "global_cfr", "bootstrap_low", "bootstrap_high", "correlation", "slope",
    "random_mean", "random_max", "random_p", "mapping_gap",
)


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def mapping_diagnostics(rows, expected=RESPONSE_MAPPINGS) -> dict:
    frame = pd.DataFrame(rows)
    values = {}
    if not frame.empty and {"response_mapping", "global_cfr"} <= set(frame):
        values = {
            str(row.response_mapping): float(row.global_cfr)
            for row in frame.itertuples()
            if _finite(row.global_cfr)
        }
    complete = set(values) == set(expected)
    observed = [values[name] for name in expected] if complete else []
    return {
        "mapping_values": values,
        "mapping_complete": complete,
        "mapping_averaged_cfr": float(np.mean(observed)) if observed else np.nan,
        "mapping_gap": float(max(observed) - min(observed)) if observed else np.nan,
        "mapping_sign_consistent": bool(observed and (all(v > 0 for v in observed) or all(v < 0 for v in observed))),
        "mapping_each_positive": bool(observed and all(v > 0 for v in observed)),
    }


def evaluate_source_validity(pooled_row, mapping_rows, rule: dict) -> dict:
    """Apply the preregistered five-part source gate without scientific repair."""

    pooled = dict(pooled_row)
    diagnostics = mapping_diagnostics(mapping_rows, tuple(rule["response_mapping"]["values"]))
    mapping_rule = rule["response_mapping"]
    mapping_pass = bool(
        diagnostics["mapping_complete"]
        and _finite(diagnostics["mapping_gap"])
        and diagnostics["mapping_gap"] <= float(mapping_rule["maximum_global_cfr_gap"])
        and (not mapping_rule.get("require_positive_each_mapping", False) or diagnostics["mapping_each_positive"])
        and (not mapping_rule.get("require_sign_consistency", False) or diagnostics["mapping_sign_consistent"])
    )
    criteria = {
        "heldout_global_cfr_positive": _finite(pooled.get("global_cfr")) and float(pooled["global_cfr"]) > 0,
        "bootstrap_lower_bound_positive": _finite(pooled.get("bootstrap_low")) and float(pooled["bootstrap_low"]) > 0,
        "intervention_target_correlation_positive": _finite(pooled.get("correlation")) and float(pooled["correlation"]) > 0,
        "beats_matched_random_subspace_null": _finite(pooled.get("random_p")) and float(pooled["random_p"]) <= float(rule["random_p_max"]),
        "response_mapping_robustness_passes": mapping_pass,
    }
    source_valid = bool(all(criteria.values()))
    return {
        "source_valid": source_valid,
        "source_available": source_valid,
        "source_status": "valid_source_controller" if source_valid else "invalid_source_controller",
        "criteria": criteria,
        **diagnostics,
        "response_mapping_robustness_passes": mapping_pass,
        "rule_version": rule["rule_version"],
    }


def classify_transfer(row, *, source_valid: bool) -> str:
    if not source_valid:
        return "unavailable"
    row = dict(row)
    if not all(_finite(row.get(name)) for name in ("global_cfr", "bootstrap_low", "bootstrap_high")):
        return "inconclusive"
    if float(row["global_cfr"]) > 0 and float(row["bootstrap_low"]) > 0:
        return "positive_transfer"
    if float(row["global_cfr"]) < 0 and float(row["bootstrap_high"]) < 0:
        return "negative_transfer"
    return "inconclusive"


def unavailable_transfer_rows(
    *, model, theory, source_task, target_tasks, layer, rank, n_train,
    n_selection, controller_hash, split_hash,
) -> pd.DataFrame:
    rows = []
    for target in target_tasks:
        for mapping in (POOLED_MAPPING, *RESPONSE_MAPPINGS):
            row = {
                "model": model, "theory": theory, "source_task": source_task,
                "target_task": target, "response_mapping": mapping,
                "source_valid": False, "layer": int(layer), "rank": int(rank),
                "n_train": int(n_train), "n_selection": int(n_selection), "n_test": 0,
                "controller_hash": controller_hash, "split_hash": split_hash,
                "endpoint_id": "cognitive_counterfactual_recovery",
                "metric_id": "global_cfr_v1", "status": "unavailable",
                "transfer_status": "unavailable",
            }
            row.update({name: np.nan for name in SCALAR_METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def _safe(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_")


def write_matrix_artifacts(output, metrics, controllers, gates, *, task_order) -> dict:
    """Write the exact diagonal/matrix contract from mapping-indexed job rows."""

    output = Path(output)
    diagonal_root, matrix_root = output / "diagonal", output / "matrix"
    diagonal_root.mkdir(parents=True, exist_ok=True)
    matrix_root.mkdir(parents=True, exist_ok=True)
    metrics, controllers, gates = metrics.copy(), controllers.copy(), gates.copy()
    if metrics.duplicated(list(IDENTITY_KEYS)).any():
        raise ValueError("duplicate model/theory/source/target/mapping transfer rows")
    controllers.to_csv(diagonal_root / "controller_manifest.csv", index=False)
    gates.to_csv(diagonal_root / "source_validity.csv", index=False)
    diagonal = metrics[metrics.source_task.astype(str) == metrics.target_task.astype(str)].copy()
    diagonal["task"] = diagonal.source_task
    diagonal[diagonal.response_mapping == POOLED_MAPPING].to_csv(
        diagonal_root / "diagonal_results.csv", index=False
    )
    diagonal[diagonal.response_mapping.isin(RESPONSE_MAPPINGS)].to_csv(
        diagonal_root / "response_mapping_results.csv", index=False
    )
    gate_keys = ["model", "theory", "source_task"]
    gate_columns = gate_keys + [
        column for column in (
            "source_valid", "source_status", "mapping_gap", "mapping_sign_consistent",
            "response_mapping_robustness_passes",
        ) if column in gates
    ]
    merged = metrics.merge(gates[gate_columns], on=gate_keys, how="left", validate="many_to_one", suffixes=("", "_gate"))
    if "source_valid_gate" in merged:
        merged["source_valid"] = merged["source_valid_gate"].fillna(False).astype(bool)
        merged = merged.drop(columns="source_valid_gate")
    else:
        merged["source_valid"] = merged.get("source_valid", False)
    if "mapping_gap_gate" in merged:
        merged["mapping_gap"] = merged["mapping_gap_gate"]
        merged = merged.drop(columns="mapping_gap_gate")
    merged["transfer_status"] = [
        classify_transfer(row, source_valid=bool(row["source_valid"]))
        for row in merged.to_dict("records")
    ]
    scalar = list(SCALAR_METRICS)
    invalid = ~merged.source_valid.fillna(False)
    offdiag = merged.source_task.astype(str) != merged.target_task.astype(str)
    merged.loc[invalid & offdiag, [c for c in scalar if c in merged]] = np.nan
    merged.loc[invalid & offdiag, "transfer_status"] = "unavailable"
    mapping_long = merged[merged.response_mapping.isin(RESPONSE_MAPPINGS)].copy()
    mapping_long.to_csv(matrix_root / "transfer_long.csv", index=False)

    summary_rows = []
    for keys, part in merged.groupby(["model", "theory", "source_task", "target_task"], dropna=False):
        pooled = part[part.response_mapping == POOLED_MAPPING]
        base = pooled.iloc[0].to_dict() if len(pooled) else part.iloc[0].to_dict()
        diagnostics = mapping_diagnostics(part[part.response_mapping.isin(RESPONSE_MAPPINGS)])
        base.update(diagnostics)
        summary_rows.append(base)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(matrix_root / "transfer_summary.csv", index=False)
    for (theory, mapping), part in mapping_long.groupby(["theory", "response_mapping"]):
        matrix = part.pivot(index="source_task", columns="target_task", values="global_cfr")
        matrix = matrix.reindex(index=task_order, columns=task_order)
        matrix.to_csv(matrix_root / f"transfer_matrix_{_safe(theory)}_{_safe(mapping)}.csv")
    return {
        "diagonal_rows": int(len(diagonal)),
        "mapping_rows": int(len(mapping_long)),
        "summary_rows": int(len(summary)),
    }

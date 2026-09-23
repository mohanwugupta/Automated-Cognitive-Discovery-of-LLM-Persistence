"""Prospective causal-hypothesis and transfer guards (Levels 4–5)."""

from __future__ import annotations

import json
import re

import pandas as pd


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def classify_control(*, target_changed: bool, observed_effect: float, null_high: float) -> str:
    if not target_changed:
        return "uninformative_control"
    return "control_failed" if abs(float(observed_effect)) > abs(float(null_high)) else "control_passed"


def validate_causal_transfer_rows(rows: pd.DataFrame) -> None:
    required = {
        "model", "source_task", "target_task", "variable", "basis_sha256",
        "registered_basis_sha256", "fit_tasks", "endpoint_id", "metric_id", "global_cfr",
    }
    missing = required - set(rows)
    if missing:
        raise ValueError(f"causal transfer lacks columns: {sorted(missing)}")
    if set(rows.endpoint_id.astype(str)) != {"cognitive_counterfactual_recovery"}:
        raise ValueError("causal endpoint drift")
    if set(rows.metric_id.astype(str)) != {"global_cfr_v1"}:
        raise ValueError("causal metric drift")
    for row in rows.itertuples(index=False):
        if not SHA256.fullmatch(str(row.basis_sha256)) or row.basis_sha256 != row.registered_basis_sha256:
            raise ValueError("causal intervention basis drifted after hypothesis freeze")
        fit_tasks = set(map(str, json.loads(row.fit_tasks)))
        if str(row.target_task) != str(row.source_task) and str(row.target_task) in fit_tasks:
            raise ValueError("target-task refitting is prohibited for cross-task causal transfer")
        if str(row.source_task) not in fit_tasks:
            raise ValueError("causal source task is absent from basis fit tasks")


def register_causal_hypotheses(rows: pd.DataFrame) -> list[dict]:
    required = {"model", "variable", "source_tasks", "target_tasks", "basis_sha256", "rationale"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"causal hypothesis table lacks columns: {sorted(missing)}")
    result = []
    for index, row in enumerate(rows.to_dict("records")):
        if not SHA256.fullmatch(str(row["basis_sha256"])):
            raise ValueError("causal hypothesis needs a frozen basis hash")
        result.append({
            "hypothesis_id": f"causal-hypothesis-{index:04d}", **row,
            "endpoint_id": "cognitive_counterfactual_recovery",
            "metric_id": "global_cfr_v1", "frozen_before_causal_outcomes": True,
        })
    return result

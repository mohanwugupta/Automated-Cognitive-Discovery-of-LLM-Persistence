"""Phase-A regression adapter over the approved frozen Llama summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from cognitive_discovery.reproducibility.metrics import global_cfr_v1


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_llama_phase_a(root: str | Path) -> dict:
    root = Path(root)
    interface_root = root / "paper/generated/llama_interface_v2"
    behavior_root = root / "paper/generated/llama_behavior_v2"
    mechanism_root = root / "paper/generated/llama_mechanistic_v2"

    interface_protocol = _read_json(interface_root / "protocol.json")
    interface_complete = _read_json(interface_root / "complete.json")
    validation_gates = _read_json(interface_root / "validation_gates.json")
    if interface_complete.get("chosen") != "question":
        raise RuntimeError("frozen Llama interface selection changed")
    interface_passed = bool(
        interface_complete.get("validated")
        and len(validation_gates) == 7
        and all(value.get("approved") is True for value in validation_gates.values())
    )

    behavior_protocol = _read_json(behavior_root / "protocol.json")
    selection = _read_json(behavior_root / "selection.json")
    metrics = pd.read_csv(behavior_root / "metrics.csv")
    test = metrics[(metrics.phase == "test") & (metrics.task == "all")].set_index("model")
    immediate_r2 = float(test.loc["immediate_state", "r2"])
    selected_model = str(selection["chosen"])
    selected_r2 = float(test.loc[selected_model, "r2"])

    mechanism_protocol = _read_json(mechanism_root / "protocol.json")
    controller = _read_json(mechanism_root / "selection.json")
    interventions = pd.read_csv(mechanism_root / "interventions.csv.gz")
    selected = interventions[interventions.method == "selected"]
    cognitive = {
        split: global_cfr_v1(
            group.predicted_counterfactual_effect,
            group.neural_counterfactual_effect,
        )
        for split, group in selected.groupby("pair_split")
    }
    metric_rows = pd.read_csv(mechanism_root / "metrics.csv")
    metric_rows = metric_rows[
        (metric_rows.method == "selected")
        & (metric_rows.endpoint == "predicted_counterfactual_effect")
    ].set_index("split")
    for split, value in cognitive.items():
        if abs(value - float(metric_rows.loc[split, "recovery"])) > 1e-12:
            raise RuntimeError("frozen Llama cognitive recovery rows are inconsistent")

    revisions = {
        interface_protocol["revision"],
        behavior_protocol["revision"],
        mechanism_protocol["revision"],
    }
    if len(revisions) != 1:
        raise RuntimeError("frozen Llama stages use different model revisions")
    input_paths = [
        interface_root / "protocol.json",
        interface_root / "complete.json",
        interface_root / "validation_gates.json",
        behavior_root / "protocol.json",
        behavior_root / "metrics.csv",
        behavior_root / "selection.json",
        mechanism_root / "protocol.json",
        mechanism_root / "selection.json",
        mechanism_root / "interventions.csv.gz",
        mechanism_root / "metrics.csv",
    ]
    return {
        "schema_version": "llama-phase-a-replay-v1",
        "model": {
            "id": interface_protocol["model"],
            "revision": revisions.pop(),
            "adapter": "llama",
        },
        "interface": {
            "selected": "question",
            "passed": interface_passed,
            "approved_task_count": sum(
                value.get("approved") is True for value in validation_gates.values()
            ),
        },
        "behavior": {
            "selected_model": selected_model,
            "theory_status": "resolved" if selection.get("passed") else "unresolved",
            "passed": bool(
                selection.get("passed")
                and selected_r2 >= 0.6
                and selected_r2 >= immediate_r2 + 0.01
            ),
            "immediate_state_test_r2": immediate_r2,
            "dual_history_test_r2": float(test.loc["dual_history", "r2"]),
        },
        "mechanism": {
            "layer": int(controller["layer"]),
            "rank": int(controller["rank"]),
            "endpoint_id": "cognitive_counterfactual_recovery",
            "metric_id": "global_cfr_v1",
            "neural_test_global_cfr": cognitive["mech_pair_test"],
            "neural_test_ci": [
                float(metric_rows.loc["mech_pair_test", "low"]),
                float(metric_rows.loc["mech_pair_test", "high"]),
            ],
            "neural_task_holdout_global_cfr": cognitive["mech_task_holdout"],
            "neural_task_holdout_ci": [
                float(metric_rows.loc["mech_task_holdout", "low"]),
                float(metric_rows.loc["mech_task_holdout", "high"]),
            ],
        },
        "specificity": {
            "random_control_passed": False,
            "status": "not_run",
            "limitation": "Phase A freezes behavioral and cognitive-recovery regression only.",
        },
        "optional": {"abstraction": "not_run", "ood": "not_run"},
        "input_hashes": {
            path.relative_to(root).as_posix(): _sha256(path) for path in input_paths
        },
    }

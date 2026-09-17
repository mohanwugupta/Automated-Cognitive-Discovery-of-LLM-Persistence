"""Assemble compact fresh-run results and the component-level report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .provenance import validate_replication_provenance
from .report import write_replication_report
from .state import ReplicationRunState
from .workflow import _json, _load_run, _sha256


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _aggregate_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if "task_family" in frame:
        frame = frame[frame.task_family.isna()]
    return json.loads(frame.to_json(orient="records"))


def execute_report_stage(root: Path, output: Path) -> dict:
    del root
    config, state, provenance = _load_run(output)
    required = "interface" if state.replication_status == "measurement_failure" else "report"
    validate_replication_provenance(provenance, required_for_stage=required)
    state.start("report")
    state.write(output / "run_state.json")

    interface = _read_json(output / "interface/selected_interface.json", {})
    selection = _read_json(output / "behavior/models/selected_models.json", {})
    controller = _read_json(output / "mechanism/selected_controller.json", {})
    generalization = _aggregate_metrics(output / "mechanism/generalization_metrics.csv")
    specificity = _read_json(output / "mechanism/specificity_gates.json", {})
    comparison_path = output / "behavior/models/model_comparison.csv"
    comparison = (
        json.loads(pd.read_csv(comparison_path).to_json(orient="records"))
        if comparison_path.exists()
        else []
    )
    results = {
        "schema_version": "replication-results-v1",
        "model": config["model"],
        "interface": interface,
        "behavior": {
            "passed": selection.get("behavioral_replication") == "pass",
            "replication_status": selection.get("behavioral_replication", "not_run"),
            "theory_status": selection.get("behavioral_theory_status", "not_run"),
            "selected_models": selection.get("selected_models", []),
            "model_comparison": comparison,
        },
        "mechanism": {
            "endpoint_id": config["endpoint_id"],
            "metric_id": config["metric_id"],
            "relative_depths": config["mechanism"]["relative_depths"],
            "ranks": config["mechanism"]["ranks"],
            "controllers": controller.get("controllers", []),
            "generalization": generalization,
        },
        "specificity": {
            "random_control_passed": bool(specificity.get("passed", False)),
            **specificity,
        },
        "optional": {
            "abstraction": "not_run" if not config["optional"]["abstraction"]["enabled"] else "pending",
            "ood": "not_run" if not config["optional"]["ood"]["enabled"] else "pending",
        },
    }
    result_path = _json(output / "replication_results.json", results)
    report_path = write_replication_report(output / "REPLICATION_REPORT.md", results, provenance)
    state.complete(
        "report",
        outcome="complete",
        artifacts={
            "results": _sha256(result_path),
            "report": _sha256(report_path),
        },
    )
    state.write(output / "run_state.json")
    return {"results": str(result_path), "report": str(report_path)}

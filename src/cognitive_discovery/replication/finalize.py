"""Assemble compact fresh-run results and the component-level report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

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
    return json.loads(frame.to_json(orient="records"))


def _specificity_by_theory(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    required = {
        "theory",
        "passes_random_control",
        "passes_named_controls",
        "passes_response_mapping_control",
    }
    if required - set(frame):
        return {}
    result = {}
    for theory, group in frame.groupby("theory"):
        result[str(theory)] = {
            "passed": bool(
                group.passes_random_control.all()
                and group.passes_named_controls.all()
                and group.passes_response_mapping_control.all()
            ),
            "splits": sorted(map(str, group.split.unique())) if "split" in group else [],
            "random_comparison": json.loads(
                group[
                    [
                        "split",
                        "selected_global_cfr",
                        "random_mean_global_cfr",
                        "random_max_global_cfr",
                        "finite_sample_random_p",
                        "passes_random_control",
                    ]
                ].to_json(orient="records")
            ),
            "response_mapping_robustness": json.loads(
                group[
                    [
                        "split",
                        "response_mapping_cfr",
                        "response_mapping_cfr_gap",
                        "passes_response_mapping_control",
                    ]
                ].to_json(orient="records")
            ),
            "informative_specificity_controls": json.loads(
                group[
                    [
                        "split",
                        "shuffled_source_status",
                        "shuffled_target_status",
                        "named_control_global_cfr",
                        "passes_named_controls",
                    ]
                ].to_json(orient="records")
            ),
        }
    return result


def execute_report_stage(root: Path, output: Path) -> dict:
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
    specificity_by_theory = _specificity_by_theory(
        output / "mechanism/specificity_controls.csv"
    )
    comparison_path = output / "behavior/models/model_comparison.csv"
    comparison = (
        json.loads(pd.read_csv(comparison_path).to_json(orient="records"))
        if comparison_path.exists()
        else []
    )
    heldout_by_theory = {
        str(row["model"]): {"r2": float(row["r2"]), "mse": float(row["mse"])}
        for row in comparison
        if row.get("phase") == "test" and row.get("task") == "all"
    }
    historical_comparison = {}
    if provenance.get("run_kind") == "pipeline_self_replication":
        run_spec = yaml.safe_load(
            (output / "prospective_run_spec.yaml").read_text(encoding="utf-8")
        )
        historical_path = root / run_spec["historical_comparison"]["source"]
        historical_comparison = yaml.safe_load(
            historical_path.read_text(encoding="utf-8")
        )
    results = {
        "schema_version": "replication-results-v1",
        "model": config["model"],
        "interface": interface,
        "behavior": {
            "passed": selection.get("behavioral_replication") == "pass",
            "replication_status": selection.get("behavioral_replication", "not_run"),
            "theory_status": selection.get("behavioral_theory_status", "not_run"),
            "best_model": selection.get("best_model", selection.get("selected_model")),
            "behavioral_survivor_set": selection.get(
                "behavioral_survivor_set", selection.get("selected_models", [])
            ),
            "selected_model": selection.get("selected_model", selection.get("best_model")),
            "selected_models": selection.get(
                "selected_models", selection.get("behavioral_survivor_set", [])
            ),
            "survivor_rule": selection.get("survivor_rule", {}),
            "model_comparison": comparison,
            "heldout_by_theory": heldout_by_theory,
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
            "by_theory": specificity_by_theory,
            **specificity,
        },
        "optional": {
            "abstraction": "not_run" if not config["optional"]["abstraction"]["enabled"] else "pending",
            "ood": "not_run" if not config["optional"]["ood"]["enabled"] else "pending",
        },
        "historical_comparison": historical_comparison,
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

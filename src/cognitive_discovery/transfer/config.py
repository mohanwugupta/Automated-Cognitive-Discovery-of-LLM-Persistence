"""Frozen configuration contract for task-transfer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml


TASKS = (
    "bandit", "foraging", "solvability", "information_sampling",
    "waiting", "effort", "debugging",
)
RESPONSE_MAPPINGS = ("continue_x", "continue_y")


class TransferConfigError(ValueError):
    pass


def validate_transfer_config(config: Mapping) -> None:
    required = {"schema_version", "endpoint_id", "metric_id", "response_mappings", "tasks", "splits", "search", "bootstrap", "random_controls", "source_validity", "predictors", "seeds", "artifact_policy", "resource_projection"}
    missing = required - set(config)
    if missing:
        raise TransferConfigError(f"transfer config missing keys: {sorted(missing)}")
    if config["schema_version"] != "task-transfer-v1":
        raise TransferConfigError("unsupported transfer schema")
    if config["endpoint_id"] != "cognitive_counterfactual_recovery":
        raise TransferConfigError("task transfer must use cognitive_counterfactual_recovery")
    if config["metric_id"] != "global_cfr_v1":
        raise TransferConfigError("task transfer must use global_cfr_v1")
    if tuple(config["tasks"]) != TASKS:
        raise TransferConfigError("tasks and ordering must equal the frozen seven-task battery")
    if tuple(config["response_mappings"]) != RESPONSE_MAPPINGS:
        raise TransferConfigError("response mappings must remain continue_x and continue_y")
    fractions = tuple(map(float, config["splits"]["fractions"]))
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-12 or any(value <= 0 for value in fractions):
        raise TransferConfigError("transfer split fractions must be three positive values summing to one")
    if not config["search"].get("relative_depths") or not config["search"].get("ranks"):
        raise TransferConfigError("relative-depth/rank search grid is required")
    if int(config["bootstrap"].get("samples", 0)) < 100:
        raise TransferConfigError("final transfer bootstrap requires at least 100 samples")
    if int(config["random_controls"].get("subspaces", 0)) < 100:
        raise TransferConfigError("final transfer inference requires at least 100 random subspaces")
    gate = config["source_validity"]
    if gate.get("rule_version") != "task_source_validity_v1":
        raise TransferConfigError("source-validity rule version changed")
    if not 0 < float(gate.get("random_p_max", 0)) <= .05:
        raise TransferConfigError("matched-random source gate must use p <= .05")
    mapping = gate.get("response_mapping", {})
    if tuple(mapping.get("values", ())) != RESPONSE_MAPPINGS:
        raise TransferConfigError("source gate must evaluate both frozen response mappings")
    if float(mapping.get("maximum_global_cfr_gap", -1)) != .25:
        raise TransferConfigError("v1 response-mapping CFR-gap threshold is frozen at .25")
    if mapping.get("require_positive_each_mapping") is not True or mapping.get("require_sign_consistency") is not True:
        raise TransferConfigError("v1 response-mapping source gate requires positive, sign-consistent mappings")
    if config["predictors"].get("frozen_before_matrix") is not True or not config["predictors"].get("spec_path"):
        raise TransferConfigError("a frozen transfer-predictor specification is required")
    projection = config["resource_projection"]
    if projection.get("multiplier_basis") != "parameter_count_ratio_planning_only":
        raise TransferConfigError("cross-model pilot projection basis changed")
    if config["artifact_policy"].get("save_full_activations") is not False:
        raise TransferConfigError("full activation banks are prohibited")


def load_transfer_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TransferConfigError("transfer config must be a mapping")
    validate_transfer_config(value)
    return value

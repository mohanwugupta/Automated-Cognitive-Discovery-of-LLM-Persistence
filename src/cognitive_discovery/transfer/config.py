"""Frozen configuration contract for task-transfer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml


TASKS = (
    "bandit", "foraging", "solvability", "information_sampling",
    "waiting", "effort", "debugging",
)


class TransferConfigError(ValueError):
    pass


def validate_transfer_config(config: Mapping) -> None:
    required = {"schema_version", "endpoint_id", "metric_id", "tasks", "splits", "search", "bootstrap", "random_controls", "predictors", "seeds", "artifact_policy"}
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
    fractions = tuple(map(float, config["splits"]["fractions"]))
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-12 or any(value <= 0 for value in fractions):
        raise TransferConfigError("transfer split fractions must be three positive values summing to one")
    if not config["search"].get("relative_depths") or not config["search"].get("ranks"):
        raise TransferConfigError("relative-depth/rank search grid is required")
    if int(config["bootstrap"].get("samples", 0)) < 100:
        raise TransferConfigError("final transfer bootstrap requires at least 100 samples")
    if int(config["random_controls"].get("subspaces", 0)) < 100:
        raise TransferConfigError("final transfer inference requires at least 100 random subspaces")
    if config["artifact_policy"].get("save_full_activations") is not False:
        raise TransferConfigError("full activation banks are prohibited")


def load_transfer_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TransferConfigError("transfer config must be a mapping")
    validate_transfer_config(value)
    return value

"""Fail-closed configuration for a fresh persistence replication."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from .adapters import adapter_registry


REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
TASKS = frozenset(
    {
        "bandit",
        "foraging",
        "solvability",
        "information_sampling",
        "waiting",
        "effort",
        "debugging",
    }
)


class ReplicationConfigError(ValueError):
    pass


def _merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_replication_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "model",
        "endpoint_id",
        "metric_id",
        "task_families",
        "interface",
        "behavior",
        "model_comparison",
        "mechanism",
        "specificity",
        "optional",
        "seeds",
        "artifact_policy",
    }
    missing = required - set(config)
    if missing:
        raise ReplicationConfigError(f"replication config missing keys: {sorted(missing)}")
    if config["schema_version"] != "model-agnostic-replication-v1":
        raise ReplicationConfigError("unsupported replication config schema_version")
    model = config["model"]
    if not model.get("id"):
        raise ReplicationConfigError("model.id is required")
    revision = model.get("revision")
    if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
        raise ReplicationConfigError("new replication model revision must be an immutable 40-hex commit")
    tokenizer_revision = model.get("tokenizer_revision")
    if not isinstance(tokenizer_revision, str) or not REVISION_PATTERN.fullmatch(tokenizer_revision):
        raise ReplicationConfigError("tokenizer revision must be an immutable 40-hex commit")
    if model.get("adapter") not in adapter_registry.names():
        raise ReplicationConfigError("model.adapter is not registered")
    if config["endpoint_id"] != "cognitive_counterfactual_recovery":
        raise ReplicationConfigError(
            "natural-effect and other endpoints are excluded; the core endpoint is "
            "cognitive_counterfactual_recovery"
        )
    if config["metric_id"] != "global_cfr_v1":
        raise ReplicationConfigError("the core replication metric must be global_cfr_v1")
    if set(config["task_families"]) != TASKS:
        raise ReplicationConfigError("task_families must equal the canonical seven-task battery")
    candidates = config["interface"].get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ReplicationConfigError("at least one preregistered response interface is required")
    splits = config["behavior"].get("split_fractions", {})
    contextual_conditions = int(config["behavior"].get("contextual_conditions", 0))
    if contextual_conditions < 6 or contextual_conditions % 2:
        raise ReplicationConfigError(
            "behavior.contextual_conditions must be an even design that identifies "
            "the latent-context model"
        )
    if set(splits) != {"behavior_train", "behavior_selection", "behavior_test"}:
        raise ReplicationConfigError("behavior split names are not canonical")
    if abs(sum(map(float, splits.values())) - 1.0) > 1e-12:
        raise ReplicationConfigError("behavior split fractions must sum to one")
    mechanism = config["mechanism"]
    depths = list(map(float, mechanism.get("relative_depths", [])))
    ranks = list(map(int, mechanism.get("ranks", [])))
    if not depths or any(not 0 < value < 1 for value in depths):
        raise ReplicationConfigError("mechanism relative depths must lie in (0, 1)")
    if not ranks or any(value < 1 for value in ranks):
        raise ReplicationConfigError("mechanism ranks must be positive")
    fitting = set(mechanism.get("fitting_tasks", []))
    holdout = set(mechanism.get("holdout_tasks", []))
    if fitting & holdout or fitting | holdout != TASKS:
        raise ReplicationConfigError("fitting and holdout tasks must partition the task battery")
    if int(config["specificity"].get("random_subspaces_final", 0)) < 100:
        raise ReplicationConfigError("final specificity requires at least 100 random subspaces")
    mapping_gap = float(
        config["specificity"].get("maximum_response_mapping_cfr_gap", -1)
    )
    if mapping_gap < 0:
        raise ReplicationConfigError(
            "specificity maximum_response_mapping_cfr_gap must be non-negative"
        )
    for stage in ("abstraction", "ood"):
        optional = config["optional"].get(stage, {})
        if optional.get("enabled") and optional.get("requires_stage") != "specificity":
            raise ReplicationConfigError(f"optional {stage} must be gated on specificity")
    if not isinstance(config["seeds"], dict) or not config["seeds"]:
        raise ReplicationConfigError("all replication seeds must be explicit")
    if config["artifact_policy"].get("save_full_activations") is not False:
        raise ReplicationConfigError("full activation banks are prohibited by default")


def load_replication_config(
    path: str | Path,
    *,
    model_id: str,
    revision: str,
    adapter: str,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplicationConfigError("replication config must be a mapping")
    value = _merge(
        value,
        {
            "model": {
                "id": model_id,
                "revision": revision,
                "adapter": adapter,
                "tokenizer_id": tokenizer_id or model_id,
                "tokenizer_revision": tokenizer_revision or revision,
            }
        },
    )
    if overrides:
        value = _merge(value, overrides)
    validate_replication_config(value)
    return value

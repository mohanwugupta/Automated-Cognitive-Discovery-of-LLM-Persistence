"""Fail-closed contract for the three-model cross-task discovery program."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

import yaml


TASKS = (
    "bandit", "foraging", "solvability", "information_sampling",
    "waiting", "effort", "debugging",
)
MODEL_KEYS = ("qwen", "gemma", "llama")
REVISION = re.compile(r"^[0-9a-f]{40}$")
EVIDENCE_LEVELS = {
    "behavior": 1, "computational": 2, "representation": 3,
    "causal": 4, "cross_task_causal": 5,
}


class CrossTaskConfigError(ValueError):
    pass


def validate_cross_task_config(config: Mapping) -> None:
    required = {
        "schema_version", "analysis_id", "models", "tasks", "compatibility_matrix",
        "interfaces", "design", "splits", "computational", "active_discovery",
        "representation", "causal", "endpoints", "seeds", "artifact_policy",
        "evidence_levels",
    }
    missing = required - set(config)
    if missing:
        raise CrossTaskConfigError(f"cross-task config missing keys: {sorted(missing)}")
    if config["schema_version"] != "cross-task-discovery-v1":
        raise CrossTaskConfigError("unsupported cross-task discovery schema")
    if config["analysis_id"] != "cross_task_discovery_three_llms_v1":
        raise CrossTaskConfigError("analysis identity changed")
    if tuple(config["tasks"]) != TASKS:
        raise CrossTaskConfigError("task order must equal the canonical seven-task battery")
    models = list(config["models"])
    if tuple(model.get("key") for model in models) != MODEL_KEYS:
        raise CrossTaskConfigError("model order must be qwen, gemma, llama")
    for model in models:
        for field in ("id", "tokenizer_id", "adapter"):
            if not model.get(field):
                raise CrossTaskConfigError(f"model {model.get('key')} lacks {field}")
        for field in ("revision", "tokenizer_revision"):
            if not REVISION.fullmatch(str(model.get(field, ""))):
                raise CrossTaskConfigError(
                    f"model {model.get('key')} requires immutable {field}"
                )
    if len({model["revision"] for model in models}) != 3:
        raise CrossTaskConfigError("each architecture must have an explicit pinned revision")
    if not config["compatibility_matrix"].get("path"):
        raise CrossTaskConfigError("task-variable compatibility path is required")
    if int(config["interfaces"].get("calibration_conditions_per_task", 0)) < 2:
        raise CrossTaskConfigError("interface calibration requires independent task-level conditions")
    if int(config["interfaces"].get("validation_conditions_per_task", 0)) < 2:
        raise CrossTaskConfigError("interface validation requires independent task-level conditions")
    splits = config["splits"]
    fractions = tuple(float(value) for value in splits.get("fractions", ()))
    if len(fractions) != 3 or any(value <= 0 for value in fractions) or abs(sum(fractions) - 1) > 1e-12:
        raise CrossTaskConfigError("three positive split fractions must sum to one")
    if tuple(splits.get("names", ())) != ("train", "selection", "test"):
        raise CrossTaskConfigError("split names are frozen as train/selection/test")
    if splits.get("group_response_mappings") is not True:
        raise CrossTaskConfigError("response mappings must remain grouped")
    bank = set(config["computational"].get("model_bank", ()))
    required_bank = {
        "immediate_state", "dynamic_reevaluation", "choice_perseveration",
        "outcome_history", "dual_history", "latent_context", "latent_motivation",
        "option_termination", "task_set_reinstatement", "meta_control",
    }
    if not required_bank <= bank:
        raise CrossTaskConfigError(
            f"computational bank omitted: {sorted(required_bank - bank)}"
        )
    if set(config["computational"].get("flexible_baselines", ())) != {
        "linear_interactions", "mlp", "gru"
    }:
        raise CrossTaskConfigError("frozen flexible baseline bank changed")
    if tuple(config["computational"].get("sharing_structures", ())) != (
        "fully_shared", "task_specific", "hierarchical", "ontology_conditioned"
    ):
        raise CrossTaskConfigError("M1–M4 parameter-sharing structures changed")
    if config["active_discovery"].get("uses_neural_data") is not False:
        raise CrossTaskConfigError("behavioral acquisition may not use neural data")
    if config["active_discovery"].get("untouched_validation") is not True:
        raise CrossTaskConfigError("untouched behavioral validation is required")
    representation = config["representation"]
    if representation.get("target_refit_allowed") is not False:
        raise CrossTaskConfigError("cross-task representation target refitting is prohibited")
    if representation.get("activation_position") != "final_prompt_token":
        raise CrossTaskConfigError("representation activation position changed")
    causal = config["causal"]
    if causal.get("requires_frozen_hypotheses") is not True:
        raise CrossTaskConfigError("causal tests require prospectively frozen hypotheses")
    endpoints = config["endpoints"]
    if endpoints != {
        "behavior": "semantic_persistence_logit_v1",
        "representation": "frozen_source_decoding_v1",
        "causal": "cognitive_counterfactual_recovery",
        "causal_metric": "global_cfr_v1",
    }:
        raise CrossTaskConfigError("scientific endpoint identity drift")
    if dict(config["evidence_levels"]) != EVIDENCE_LEVELS:
        raise CrossTaskConfigError("evidence hierarchy changed")
    if config["artifact_policy"].get("save_full_activations") is not False:
        raise CrossTaskConfigError("permanent full activation banks are prohibited")
    if config["artifact_policy"].get("delete_temporary_activations") is not True:
        raise CrossTaskConfigError("temporary activation cleanup is required")
    if not config.get("seeds") or any(not isinstance(value, int) for value in config["seeds"].values()):
        raise CrossTaskConfigError("all seeds must be explicit integers")


def load_cross_task_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CrossTaskConfigError("cross-task config must be a mapping")
    validate_cross_task_config(value)
    return value

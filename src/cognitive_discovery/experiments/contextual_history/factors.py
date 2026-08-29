"""Frozen factor definitions and legality checks for contextual histories."""

from __future__ import annotations

import json


CONTEXTUAL_TASKS = ("bandit", "foraging", "debugging")
HISTORY_VALENCES = ("positive", "negative", "mixed")
CONTEXT_RETURNS = ("A", "B", "novel_C")
CUE_RELIABILITY = {"low": 0.60, "medium": 0.80, "high": 0.95}
CHANGE_POINTS = ("same_environment", "change_point")
PHASE_ORDER = ("A1", "B", "A2")


def _sequence(value) -> tuple:
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(value)


def validate_contextual_history(context: dict[str, object]) -> None:
    required = {
        "phase_order",
        "a_history_valence",
        "b_history_valence",
        "a_history_actions",
        "a_history_outcomes",
        "b_history_actions",
        "b_history_outcomes",
        "context_return",
        "cue_reliability",
        "cue_probability",
        "change_point",
        "cue_rule",
        "a_environment_rate",
        "b_environment_rate",
        "critical_contrast_id",
    }
    missing = required - set(context)
    if missing:
        raise ValueError(f"contextual history is missing: {sorted(missing)}")
    if _sequence(context["phase_order"]) != PHASE_ORDER:
        raise ValueError("contextual episode must have A1→B→A2 ordering")
    if context["a_history_valence"] not in HISTORY_VALENCES:
        raise ValueError("illegal A-history valence")
    if context["b_history_valence"] not in HISTORY_VALENCES:
        raise ValueError("illegal B-history valence")
    if context["context_return"] not in CONTEXT_RETURNS:
        raise ValueError("illegal context return")
    if context["cue_reliability"] not in CUE_RELIABILITY:
        raise ValueError("illegal cue reliability")
    expected = CUE_RELIABILITY[str(context["cue_reliability"])]
    if abs(float(context["cue_probability"]) - expected) > 1e-12:
        raise ValueError("cue probability disagrees with registered reliability")
    if context["change_point"] not in CHANGE_POINTS:
        raise ValueError("illegal change-point level")
    for prefix in ("a", "b"):
        actions = _sequence(context[f"{prefix}_history_actions"])
        outcomes = _sequence(context[f"{prefix}_history_outcomes"])
        if len(actions) != len(outcomes) or not actions:
            raise ValueError(f"{prefix.upper()} history must be non-empty and aligned")
    if not str(context["cue_rule"]).strip():
        raise ValueError("context cue must encode an operational environmental rule")
    for name in ("a_environment_rate", "b_environment_rate"):
        value = float(context[name])
        if not 0 < value < 1:
            raise ValueError(f"{name} must be a probability strictly between zero and one")
    if abs(float(context["a_environment_rate"]) - float(context["b_environment_rate"])) < 0.05:
        raise ValueError("A and B must predict meaningfully different environment statistics")


def contextual_domain(task_family: str, contextual: bool = True) -> str:
    return f"{task_family}:{'contextual' if contextual else 'original'}"


def contextual_declarative_spec(config: dict) -> dict:
    """SweetPea-facing frozen grammar used by the native compiler as well."""

    tasks = config.get("contextual_history", {}).get("task_families", CONTEXTUAL_TASKS)
    return {
        "factors": {
            "task_family": list(tasks),
            "a_history_valence": list(HISTORY_VALENCES),
            "b_history_valence": list(HISTORY_VALENCES),
            "context_return": list(CONTEXT_RETURNS),
            "cue_reliability": list(CUE_RELIABILITY),
            "change_point": list(CHANGE_POINTS),
            "latent_payoff_regime": ["continuous calibrated A/B rates"],
            "current_prospective_evidence": ["weak", "medium", "strong"],
            "continuation_cost": ["low", "medium", "high"],
            "disengagement_value": ["low", "medium", "high"],
            "response_mapping": ["continue_x", "continue_y"],
        },
        "crossing": [
            "task_family",
            "a_history_valence",
            "b_history_valence",
            "context_return",
            "cue_reliability",
            "current_prospective_evidence",
        ],
        "constraints": [
            "phase order is A1→B→A2",
            "critical contrasts swap A/B histories and match the current state",
            "response mappings share semantic state and environment seed",
            "no A2 outcome is available at the critical decision",
        ],
    }


def build_contextual_sweetpea_block(config: dict):
    try:
        from sweetpea import CrossBlock, Factor
    except ImportError as error:
        raise RuntimeError(
            "SweetPea backend requested but optional dependency is not installed"
        ) from error
    spec = contextual_declarative_spec(config)
    factors = {name: Factor(name, levels) for name, levels in spec["factors"].items()}
    crossing = [factors[name] for name in spec["crossing"]]
    return CrossBlock(list(factors.values()), crossing, [])

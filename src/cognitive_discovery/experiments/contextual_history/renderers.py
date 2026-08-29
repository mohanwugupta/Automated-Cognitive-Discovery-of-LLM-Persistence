"""Domain-grounded rendering for contextual reinstatement trials."""

from __future__ import annotations

import json

from .change_point import change_point_text
from .factors import validate_contextual_history


DOMAIN_RULES = {
    "bandit": (
        "Sources with the same circuit marker share one payoff generator; different "
        "markers use independently calibrated reward probabilities."
    ),
    "foraging": (
        "Patches with the same soil-and-season signature share one replenishment process; "
        "different signatures follow independently measured renewal rates."
    ),
    "debugging": (
        "Failures with the same subsystem signature share one causal bug process; different "
        "signatures arise from independently diagnosed fault generators."
    ),
}


def _decode(value):
    return tuple(json.loads(value)) if isinstance(value, str) else tuple(value)


def _history(actions, outcomes) -> str:
    paired = [f"{action} ({int(outcome):+d})" for action, outcome in zip(actions, outcomes)]
    return ", ".join(paired)


def contextual_history_text(condition) -> str:
    context = dict(condition.contextual_history or {})
    validate_contextual_history(context)
    task_rule = DOMAIN_RULES[condition.task_family]
    a = _history(
        _decode(context["a_history_actions"]),
        _decode(context["a_history_outcomes"]),
    )
    b = _history(
        _decode(context["b_history_actions"]),
        _decode(context["b_history_outcomes"]),
    )
    returned = {
        "A": "the diagnostic signature previously observed in A",
        "B": "the diagnostic signature most recently observed in B",
        "novel_C": "a third, previously unobserved diagnostic signature C",
    }[str(context["context_return"])]
    probability = round(100 * float(context["cue_probability"]))
    return (
        f"Environmental rule: {task_rule} {context['cue_rule']}\n"
        f"Calibration identifies environment A with an event rate of "
        f"{100 * float(context['a_environment_rate']):.1f}% and environment B with an "
        f"event rate of {100 * float(context['b_environment_rate']):.1f}%.\n"
        f"Phase A1 — environment A: {a}.\n"
        f"Phase B — environment B: {b}.\n"
        f"Phase A2 — critical decision: the current diagnostic cue matches {returned}. "
        f"Calibration data show this cue identifies its generator with {probability}% reliability. "
        f"{change_point_text(str(context['change_point']))}"
    )

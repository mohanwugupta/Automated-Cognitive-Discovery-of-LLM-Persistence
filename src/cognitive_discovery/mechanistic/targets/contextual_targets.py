"""Frozen latent-context target; it is never fit to neural measurements."""

from __future__ import annotations

import json

from .history_targets import action_history, outcome_history


def _context(condition_or_mapping) -> dict | None:
    if isinstance(condition_or_mapping, dict):
        value = condition_or_mapping.get("contextual_history", condition_or_mapping)
    else:
        value = getattr(condition_or_mapping, "contextual_history", None)
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value)


def context_relevant_history(
    condition_or_mapping,
    *,
    kind: str = "outcome",
    recent_value: float | None = None,
) -> float:
    """Reproduce ``FeatureEncoder``'s frozen contextual-history equation.

    The current cue selects A, B, or no stored context, weighted by the
    registered cue reliability. A declared change point suppresses the matched
    trace. The residual reliability mass falls back to the recent B trace.
    """

    if kind not in {"outcome", "action"}:
        raise ValueError("kind must be 'outcome' or 'action'")
    context = _context(condition_or_mapping)
    if recent_value is None:
        if isinstance(condition_or_mapping, dict):
            history = condition_or_mapping.get(
                "history_outcomes" if kind == "outcome" else "history_actions", ()
            )
        else:
            history_spec = getattr(condition_or_mapping, "history")
            history = (
                history_spec.outcomes if kind == "outcome" else history_spec.actions
            )
        recent_value = (
            outcome_history(history) if kind == "outcome" else action_history(history)
        )
    if context is None:
        return float(recent_value)
    kernel = outcome_history if kind == "outcome" else action_history
    context_return = str(context["context_return"])
    if str(context["change_point"]) == "change_point":
        matched = 0.0
    elif context_return == "A":
        matched = kernel(context[f"a_history_{kind}s"])
    elif context_return == "B":
        matched = kernel(context[f"b_history_{kind}s"])
    elif context_return == "novel_C":
        matched = 0.0
    else:
        raise ValueError(f"unknown context return: {context_return}")
    reliability = float(context["cue_probability"])
    return float(reliability * matched + (1.0 - reliability) * recent_value)


def contextual_outcome_history(condition_or_mapping) -> float:
    return context_relevant_history(condition_or_mapping, kind="outcome")


def contextual_action_history(condition_or_mapping) -> float:
    return context_relevant_history(condition_or_mapping, kind="action")

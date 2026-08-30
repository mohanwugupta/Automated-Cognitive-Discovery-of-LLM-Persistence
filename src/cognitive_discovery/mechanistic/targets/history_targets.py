"""Frozen history kernels shared with the behavioral model."""

from __future__ import annotations

import json


HISTORY_DECAY = 0.7


def _sequence(values) -> tuple:
    if isinstance(values, str):
        values = json.loads(values)
    return tuple(values or ())


def discounted_kernel(values, *, decay: float = HISTORY_DECAY) -> float:
    """Recency-weighted sum with lag zero assigned to the latest observation."""

    sequence = tuple(float(value) for value in _sequence(values))
    return float(
        sum(float(decay) ** lag * value for lag, value in enumerate(reversed(sequence)))
    )


def outcome_history(outcomes, *, decay: float = HISTORY_DECAY) -> float:
    return discounted_kernel(outcomes, decay=decay)


def _semantic_action(action: object) -> float:
    value = str(action).lower()
    if value == "continue":
        return 1.0
    if value == "disengage":
        return -1.0
    raise ValueError(f"unknown semantic action: {action}")


def action_history(actions, *, decay: float = HISTORY_DECAY) -> float:
    return discounted_kernel(
        tuple(_semantic_action(action) for action in _sequence(actions)), decay=decay
    )

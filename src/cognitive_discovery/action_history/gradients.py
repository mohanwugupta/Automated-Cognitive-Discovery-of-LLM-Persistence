"""Small utilities for validating semantic persistence-logit gradients."""

from __future__ import annotations

import numpy as np


def semantic_persistence_logit(logits, token_ids: dict[str, int], positive_label: str):
    labels = tuple(token_ids)
    if positive_label not in token_ids or len(labels) != 2:
        raise ValueError("semantic persistence logit requires two mapped labels")
    negative_label = next(label for label in labels if label != positive_label)
    return (
        logits[..., token_ids[positive_label]] - logits[..., token_ids[negative_label]]
    )


def finite_difference_directional_derivative(
    function, point, direction, *, epsilon=1e-4
):
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if point.shape != direction.shape:
        raise ValueError("point and direction must have the same shape")
    return float(
        (function(point + epsilon * direction) - function(point - epsilon * direction))
        / (2.0 * epsilon)
    )

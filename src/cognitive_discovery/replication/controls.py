"""Endpoint firewall and control-integrity checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class EndpointMismatchError(ValueError):
    pass


def require_cognitive_endpoint(endpoint_id: str) -> None:
    if endpoint_id != "cognitive_counterfactual_recovery":
        raise EndpointMismatchError(
            "all core controls and generalization tests must use the same cognitive "
            "counterfactual recovery endpoint"
        )


@dataclass(frozen=True)
class TargetShuffleAudit:
    rows: int
    changed_rows: int
    maximum_absolute_change: float
    status: str


def audit_informative_target_shuffle(
    frame,
    *,
    original_column: str = "target",
    shuffled_column: str = "shuffled_target",
    tolerance: float = 1e-12,
) -> TargetShuffleAudit:
    missing = {original_column, shuffled_column} - set(frame.columns)
    if missing:
        raise ValueError(f"target shuffle audit missing columns: {sorted(missing)}")
    original = np.asarray(frame[original_column], dtype=float)
    shuffled = np.asarray(frame[shuffled_column], dtype=float)
    if len(original) == 0 or not (np.isfinite(original).all() and np.isfinite(shuffled).all()):
        raise ValueError("target shuffle audit requires finite non-empty targets")
    difference = np.abs(original - shuffled)
    changed = int(np.sum(difference > tolerance))
    return TargetShuffleAudit(
        rows=len(original),
        changed_rows=changed,
        maximum_absolute_change=float(difference.max()),
        status="informative_control" if changed else "uninformative_control",
    )

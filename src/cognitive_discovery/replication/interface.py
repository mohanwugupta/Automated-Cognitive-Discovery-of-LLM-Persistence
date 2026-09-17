"""Measurement-interface selection without scientific-outcome peeking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class InterfaceSelection:
    selected: str | None
    passed: bool
    replication_status: str
    approved_tasks: int


def select_valid_interface(
    candidate_gates: Mapping[str, Mapping[str, Mapping]],
    *,
    required_tasks,
) -> InterfaceSelection:
    """Select among preregistered candidates using validity gates only.

    Candidate keys are sorted for deterministic tie breaking after minimizing
    the worst task-level polarity/mapping gap.
    """

    required_tasks = set(map(str, required_tasks))
    eligible = []
    for candidate, task_gates in candidate_gates.items():
        if set(task_gates) != required_tasks:
            continue
        if not all(value.get("approved") is True for value in task_gates.values()):
            continue
        worst_gap = max(
            float(
                value.get(
                    "mean_absolute_mapping_gap",
                    value.get("wording_order_effect", float("inf")),
                )
            )
            for value in task_gates.values()
        )
        eligible.append((worst_gap, str(candidate)))
    if not eligible:
        return InterfaceSelection(None, False, "measurement_failure", 0)
    _, selected = min(eligible)
    return InterfaceSelection(selected, True, "interface_valid", len(required_tasks))

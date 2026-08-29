"""Declarative experimental ontology shared by every task renderer."""

from __future__ import annotations

from types import MappingProxyType


FACTOR_LEVELS = MappingProxyType(
    {
        "continuation_value": ("low", "medium", "high"),
        "disengagement_value": ("low", "medium", "high"),
        "continuation_cost": ("low", "medium", "high"),
        "progress_evidence": ("negative", "neutral", "positive"),
        "success_evidence": ("low", "medium", "high"),
        "uncertainty": ("low", "high"),
        "prior_investment": ("low", "high"),
        "controllability": ("low", "high"),
        "environmental_stability": ("stable", "changing"),
        "goal_continuity": ("same", "new"),
    }
)

ORDINAL_VALUE = {
    "negative": -1.0,
    "low": -1.0,
    "neutral": 0.0,
    "medium": 0.0,
    "positive": 1.0,
    "high": 1.0,
    "changing": -1.0,
    "stable": 1.0,
    "new": -1.0,
    "same": 1.0,
}

_UNIVERSAL = {
    "continuation_value",
    "disengagement_value",
    "continuation_cost",
    "success_evidence",
    "uncertainty",
    "environmental_stability",
    "goal_continuity",
}

TASK_FACTOR_AVAILABILITY = MappingProxyType(
    {
        "bandit": frozenset(_UNIVERSAL | {"prior_investment"}),
        "foraging": frozenset(
            _UNIVERSAL | {"progress_evidence", "prior_investment", "controllability"}
        ),
        "solvability": frozenset(
            _UNIVERSAL | {"progress_evidence", "prior_investment", "controllability"}
        ),
        "information_sampling": frozenset(
            _UNIVERSAL | {"progress_evidence", "prior_investment"}
        ),
        "waiting": frozenset(_UNIVERSAL | {"prior_investment"}),
        "effort": frozenset(
            _UNIVERSAL | {"progress_evidence", "prior_investment", "controllability"}
        ),
        "debugging": frozenset(
            _UNIVERSAL | {"progress_evidence", "prior_investment", "controllability"}
        ),
    }
)


def continuation_advantage(factors: dict[str, str | None]) -> float:
    """Ontology-level derived factor, not an instruction shown to participants."""

    return (
        ORDINAL_VALUE[str(factors["continuation_value"])]
        - ORDINAL_VALUE[str(factors["disengagement_value"])]
        - ORDINAL_VALUE[str(factors["continuation_cost"])]
    )


def validate_factor_assignment(task_family: str, factors: dict[str, str | None]) -> None:
    if task_family not in TASK_FACTOR_AVAILABILITY:
        raise ValueError(f"unknown task family: {task_family}")
    if set(factors) != set(FACTOR_LEVELS):
        raise ValueError("factor assignment must contain the complete ontology")
    available = TASK_FACTOR_AVAILABILITY[task_family]
    for name, levels in FACTOR_LEVELS.items():
        value = factors[name]
        if name not in available and value is not None:
            raise ValueError(f"{name} is unavailable for {task_family}")
        if name in available and value not in levels:
            raise ValueError(f"illegal {name} level for {task_family}: {value}")


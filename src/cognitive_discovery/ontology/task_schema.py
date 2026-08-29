"""Immutable abstract conditions passed from design compiler to renderers."""

from __future__ import annotations

from dataclasses import dataclass
import json

from .factors import continuation_advantage, validate_factor_assignment
from .histories import HistorySpec


@dataclass(frozen=True)
class ResponseMapping:
    continue_label: str
    disengage_label: str

    def __post_init__(self) -> None:
        if self.continue_label == self.disengage_label:
            raise ValueError("response labels must be distinct")

    @property
    def labels(self) -> tuple[str, str]:
        return tuple(sorted((self.continue_label, self.disengage_label)))

    @property
    def mapping_id(self) -> str:
        return f"continue_{self.continue_label.lower()}"

    def to_dict(self) -> dict[str, str]:
        return {"continue": self.continue_label, "disengage": self.disengage_label}


@dataclass(frozen=True)
class ConditionSpec:
    design_id: str
    condition_id: str
    paired_condition_id: str
    task_family: str
    semantic_factors: dict[str, str | None]
    factor_available: dict[str, bool]
    history: HistorySpec
    response_mapping: ResponseMapping
    environment_seed: int
    sampling_strategy: str = "coverage"
    split: str = "unassigned"
    contextual_history: dict[str, object] | None = None

    def __post_init__(self) -> None:
        validate_factor_assignment(self.task_family, self.semantic_factors)
        expected = {name: value is not None for name, value in self.semantic_factors.items()}
        if self.factor_available != expected:
            raise ValueError("factor availability disagrees with explicit missingness")

    @property
    def continuation_advantage(self) -> float:
        return continuation_advantage(self.semantic_factors)

    def semantic_payload(self) -> dict:
        payload = {
            "task_family": self.task_family,
            "factors": self.semantic_factors,
            "factor_available": self.factor_available,
            "history": {
                "length": self.history.length,
                "valence": self.history.valence,
                "actions": self.history.actions,
                "outcomes": self.history.outcomes,
            },
            "environment_seed": self.environment_seed,
        }
        if self.contextual_history is not None:
            payload["contextual_history"] = self.contextual_history
        return payload

    def to_dict(self) -> dict:
        return {
            "design_id": self.design_id,
            "condition_id": self.condition_id,
            "paired_condition_id": self.paired_condition_id,
            **self.semantic_payload(),
            "response_mapping": json.dumps(self.response_mapping.to_dict(), sort_keys=True),
            "continuation_advantage": self.continuation_advantage,
            "sampling_strategy": self.sampling_strategy,
            "split": self.split,
        }

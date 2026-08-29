"""Semantically normalized action and outcome histories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HistorySpec:
    length: int
    valence: str
    actions: tuple[str, ...]
    outcomes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.length <= 5:
            raise ValueError("history length must lie between zero and five")
        if self.valence not in {"negative", "neutral", "mixed", "positive"}:
            raise ValueError(f"unknown history valence: {self.valence}")
        if len(self.actions) != self.length or len(self.outcomes) != self.length:
            raise ValueError("history arrays must match history length")
        if set(self.actions) - {"continue", "disengage"}:
            raise ValueError("history actions must use semantic labels")
        if set(self.outcomes) - {-1, 0, 1}:
            raise ValueError("history outcomes must be normalized to -1/0/+1")

    @property
    def action_text(self) -> str:
        return ", ".join(self.actions) if self.actions else "none"

    @property
    def outcome_text(self) -> str:
        symbols = {-1: "failure", 0: "neutral", 1: "success"}
        return ", ".join(symbols[value] for value in self.outcomes) if self.outcomes else "none"


def make_history(length: int, valence: str, variant: int = 0) -> HistorySpec:
    length = int(length)
    if length == 0:
        return HistorySpec(0, "neutral", (), ())
    outcome_patterns = {
        "negative": (-1,),
        "neutral": (0,),
        "mixed": (1, -1, 0),
        "positive": (1,),
    }
    action_patterns = (
        ("continue",),
        ("continue", "disengage", "continue"),
        ("disengage", "continue"),
    )
    outcomes = tuple(
        outcome_patterns[valence][index % len(outcome_patterns[valence])]
        for index in range(length)
    )
    pattern = action_patterns[int(variant) % len(action_patterns)]
    actions = tuple(pattern[index % len(pattern)] for index in range(length))
    return HistorySpec(length, valence, actions, outcomes)

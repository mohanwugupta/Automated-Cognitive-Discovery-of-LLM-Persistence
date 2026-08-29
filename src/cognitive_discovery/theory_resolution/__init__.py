"""Frozen prediction, discrimination, and mechanistic handoff for Round 3."""

from .equivalence import resolve_theory_outcome
from .frozen_predictions import freeze_surviving_models, predict_frozen_models
from .paired_model_test import compare_frozen_theories

__all__ = [
    "compare_frozen_theories",
    "freeze_surviving_models",
    "predict_frozen_models",
    "resolve_theory_outcome",
]

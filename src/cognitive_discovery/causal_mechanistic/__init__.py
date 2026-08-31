"""Counterfactual-first causal mechanistic discovery."""

from .counterfactuals import FrozenTheoryBank, build_counterfactual_pairs
from .interventions import interchange_subspace, remove_subspace, replace_whole_state
from .metrics import counterfactual_metrics, counterfactual_recovery

__all__ = [
    "FrozenTheoryBank",
    "build_counterfactual_pairs",
    "counterfactual_metrics",
    "counterfactual_recovery",
    "interchange_subspace",
    "remove_subspace",
    "replace_whole_state",
]

"""Action-history versus persistence/readout disambiguation workflow."""

from .directions import cosine_similarity, normalize, orthogonalize_to_subspace
from .matching import MatchConfig, build_matched_pairs, validate_matched_pairs
from .residualization import ActionResidualizer

__all__ = [
    "ActionResidualizer",
    "MatchConfig",
    "build_matched_pairs",
    "cosine_similarity",
    "normalize",
    "orthogonalize_to_subspace",
    "validate_matched_pairs",
]

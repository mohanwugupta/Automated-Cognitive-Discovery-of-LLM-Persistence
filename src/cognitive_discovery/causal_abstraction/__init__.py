"""Causal-abstraction discovery for the frozen persistence controller."""

from .analysis import identify_synthetic_abstraction
from .design import CONTRAST_FAMILIES, validate_contrast_manifest
from .variables import ABSTRACTIONS, FrozenAbstractionBank

__all__ = [
    "ABSTRACTIONS",
    "CONTRAST_FAMILIES",
    "FrozenAbstractionBank",
    "identify_synthetic_abstraction",
    "validate_contrast_manifest",
]

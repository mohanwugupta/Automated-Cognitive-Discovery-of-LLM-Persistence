"""Task-structured cognitive model hierarchy for Discovery Round 2."""

from .few_shot import evaluate_few_shot_adaptation
from .random_effects import HierarchicalFit, fit_hierarchical_model
from .task_descriptors import DESCRIPTOR_NAMES, TASK_DESCRIPTORS, descriptor_frame
from .variance_decomposition import parameter_sign_consistency, variance_decomposition

__all__ = [
    "DESCRIPTOR_NAMES",
    "TASK_DESCRIPTORS",
    "HierarchicalFit",
    "descriptor_frame",
    "evaluate_few_shot_adaptation",
    "fit_hierarchical_model",
    "parameter_sign_consistency",
    "variance_decomposition",
]

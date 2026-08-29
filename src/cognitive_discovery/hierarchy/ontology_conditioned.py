"""Public ontology-conditioned model API."""

from .random_effects import HierarchicalFit, fit_hierarchical_model


def fit_ontology_conditioned(frame, architecture, **kwargs) -> HierarchicalFit:
    return fit_hierarchical_model(frame, architecture, variant="M4", **kwargs)


__all__ = ["HierarchicalFit", "fit_ontology_conditioned"]

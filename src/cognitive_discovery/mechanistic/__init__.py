"""Mechanistic follow-up for frozen persistence computations.

This namespace is deliberately separate from :mod:`experiments.collection`:
behavioral collection remains activation-free, while this stage exposes only
streamed decision-position summaries and scalar projections.
"""

from .pipeline import (
    analyze_representations,
    finalize_mechanistic_run,
    prepare_mechanistic_run,
    run_causal_interventions,
    scan_directions,
    scan_projections,
)

__all__ = [
    "analyze_representations",
    "finalize_mechanistic_run",
    "prepare_mechanistic_run",
    "run_causal_interventions",
    "scan_directions",
    "scan_projections",
]

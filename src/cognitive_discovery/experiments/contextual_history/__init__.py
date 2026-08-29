"""A→B→A contextual-history experiments for Round-3 theory resolution."""

from .factors import (
    CONTEXTUAL_TASKS,
    CUE_RELIABILITY,
    build_contextual_sweetpea_block,
    contextual_declarative_spec,
    contextual_domain,
    validate_contextual_history,
)
from .reinstatement import compile_contextual_history_design

__all__ = [
    "CONTEXTUAL_TASKS",
    "CUE_RELIABILITY",
    "build_contextual_sweetpea_block",
    "compile_contextual_history_design",
    "contextual_domain",
    "contextual_declarative_spec",
    "validate_contextual_history",
]

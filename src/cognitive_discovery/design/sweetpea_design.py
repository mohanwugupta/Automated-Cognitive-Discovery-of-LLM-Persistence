"""SweetPea-facing declarative design description.

The built-in compiler is the reproducible offline backend. ``build_sweetpea_block``
provides a real SweetPea block when that optional package is installed; scientific
factor definitions and exclusions remain identical across both backends.
"""

from __future__ import annotations

from cognitive_discovery.ontology.factors import FACTOR_LEVELS


def declarative_spec(config: dict) -> dict:
    return {
        "factors": {
            "task_family": list(config["design"]["task_families"]),
            **{name: list(levels) for name, levels in FACTOR_LEVELS.items()},
            "history_length": list(config["design"]["history_lengths"]),
            "history_valence": list(config["design"]["history_valences"]),
            "response_mapping": ["continue_x", "continue_y"],
        },
        "derived_factors": {
            "continuation_advantage": "continuation_value - disengagement_value - continuation_cost"
        },
        "crossing": list(config["design"]["core_crossing"]),
        "constraints": [
            "unavailable task constructs are missing",
            "zero-length history has neutral valence",
            "paired response mappings share semantic condition and environment seed",
        ],
    }


def build_sweetpea_block(config: dict):
    try:
        from sweetpea import CrossBlock, Factor
    except ImportError as error:
        raise RuntimeError(
            "SweetPea backend requested but the optional 'sweetpea' package is not installed"
        ) from error
    spec = declarative_spec(config)
    factors = {name: Factor(name, levels) for name, levels in spec["factors"].items()}
    crossing = [factors[name] for name in spec["crossing"]]
    return CrossBlock(list(factors.values()), crossing, [])

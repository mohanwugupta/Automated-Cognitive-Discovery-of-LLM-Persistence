"""Central legal-design constraints; renderers contain no sampling rules."""

from cognitive_discovery.ontology.factors import validate_factor_assignment
from cognitive_discovery.ontology.histories import HistorySpec


def validate_condition(task_family: str, factors: dict, history: HistorySpec) -> None:
    validate_factor_assignment(task_family, factors)
    if history.length == 0 and history.valence != "neutral":
        raise ValueError("zero-length history must be neutral")
    if factors.get("goal_continuity") == "new" and history.length == 0:
        # This is legal: the absence of history is itself an informative cell.
        return


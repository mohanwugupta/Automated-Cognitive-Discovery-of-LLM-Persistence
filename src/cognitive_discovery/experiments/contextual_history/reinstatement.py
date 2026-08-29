"""Deterministic SweetPea-compatible compiler for A→B→A contrasts."""

from __future__ import annotations

import hashlib
import json

from cognitive_discovery.design.counterbalance import counterbalanced_mappings
from cognitive_discovery.design.sampling import balanced_factor_assignment
from cognitive_discovery.ontology.histories import make_history
from cognitive_discovery.ontology.task_schema import ConditionSpec

from .factors import (
    CHANGE_POINTS,
    CONTEXTUAL_TASKS,
    CONTEXT_RETURNS,
    CUE_RELIABILITY,
    PHASE_ORDER,
    validate_contextual_history,
)


_VALENCE_PAIRS = (
    ("positive", "negative"),
    ("positive", "mixed"),
    ("negative", "mixed"),
)


def _pick(values, index: int):
    values = tuple(values)
    return values[index % len(values)]


def _cue_rule(task: str, group: int) -> str:
    rules = {
        "bandit": (
            "The blue circuit uses a high-volatility generator and the amber circuit a "
            "low-volatility generator.",
            "Circuit families are distinguished by audited reward-frequency bands.",
        ),
        "foraging": (
            "Wet-season loam replenishes after harvest while dry-season gravel depletes.",
            "Soil signatures identify independently tracked renewal processes.",
        ),
        "debugging": (
            "Memory faults respond to allocation diagnostics while protocol faults respond "
            "to packet-trace diagnostics.",
            "Subsystem signatures identify independently reproduced causal failures.",
        ),
    }
    return _pick(rules[task], group)


def _environment_rates(group: int) -> tuple[float, float]:
    # Dense deterministic calibration grid. It is a scientific environmental
    # variable, not a uniqueness nonce, and is fixed within each critical pair.
    a = 0.10 + ((37 * int(group)) % 801) / 1000.0
    b = 0.10 + ((53 * int(group) + 211) % 801) / 1000.0
    if abs(a - b) < 0.05:
        b = b + 0.20 if b <= 0.70 else b - 0.20
    return round(a, 3), round(b, 3)


def compile_contextual_history_design(
    config: dict,
    *,
    n_conditions: int,
    seed: int,
    design_id: str = "theory_resolution_contextual_v1",
    task_families=None,
) -> list[ConditionSpec]:
    """Compile semantic conditions plus both response mappings.

    Adjacent semantic conditions form critical contrasts: their current state,
    cue, environment seed, and labels are matched while A/B histories are swapped.
    """

    if int(n_conditions) < 1:
        raise ValueError("n_conditions must be positive")
    tasks = tuple(task_families or config.get("contextual_history", {}).get(
        "task_families", CONTEXTUAL_TASKS
    ))
    unknown = set(tasks) - set(CONTEXTUAL_TASKS)
    if unknown:
        raise ValueError(f"contextual history is not implemented for: {sorted(unknown)}")
    labels = tuple(config.get("response_labels", ("X", "Y")))
    mappings = counterbalanced_mappings(labels)
    factor_levels = config["design"].get("factor_levels", {})
    conditions: list[ConditionSpec] = []
    for semantic_index in range(int(n_conditions)):
        contrast_group = semantic_index // 2
        orientation = semantic_index % 2
        task = tasks[contrast_group % len(tasks)]
        task_index = contrast_group // len(tasks)
        factors = balanced_factor_assignment(task, task_index, int(seed), factor_levels)
        first, second = _pick(_VALENCE_PAIRS, task_index)
        first_history = make_history(3, first, int(seed) + 2 * contrast_group)
        second_history = make_history(3, second, int(seed) + 2 * contrast_group + 1)
        a_history, b_history = (
            (first_history, second_history),
            (second_history, first_history),
        )[orientation]
        a_valence, b_valence = a_history.valence, b_history.valence
        reliability = _pick(tuple(CUE_RELIABILITY), task_index // 3)
        context_return = _pick(CONTEXT_RETURNS, task_index // 9)
        change_point = _pick(CHANGE_POINTS, task_index // 27)
        critical_id = f"critical-{task}-{contrast_group:08d}"
        a_rate, b_rate = _environment_rates(contrast_group)
        context = {
            "phase_order": PHASE_ORDER,
            "a_history_valence": a_valence,
            "b_history_valence": b_valence,
            "a_history_actions": a_history.actions,
            "a_history_outcomes": a_history.outcomes,
            "b_history_actions": b_history.actions,
            "b_history_outcomes": b_history.outcomes,
            "context_return": context_return,
            "cue_reliability": reliability,
            "cue_probability": CUE_RELIABILITY[reliability],
            "change_point": change_point,
            "cue_rule": _cue_rule(task, task_index),
            "a_environment_rate": a_rate,
            "b_environment_rate": b_rate,
            "critical_contrast_id": critical_id,
        }
        validate_contextual_history(context)
        payload = {
            "task": task,
            "factors": factors,
            "contextual_history": {
                key: value
                for key, value in context.items()
                if key != "critical_contrast_id"
            },
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=list).encode()
        ).hexdigest()
        pair_id = f"{design_id}-{digest[:16]}"
        availability = {name: value is not None for name, value in factors.items()}
        environment_seed = int(seed) * 1_000_003 + contrast_group
        for mapping_index, mapping in enumerate(mappings):
            conditions.append(
                ConditionSpec(
                    design_id=design_id,
                    condition_id=f"{pair_id}-m{mapping_index}",
                    paired_condition_id=pair_id,
                    task_family=task,
                    semantic_factors=dict(factors),
                    factor_available=availability,
                    # B is the immediately preceding history for direct-history models.
                    history=b_history,
                    response_mapping=mapping,
                    environment_seed=environment_seed,
                    contextual_history=context,
                )
            )
    return conditions

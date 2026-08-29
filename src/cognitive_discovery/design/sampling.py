"""Balanced coverage-oriented compiler for legal semantic conditions."""

from __future__ import annotations

import hashlib
import json
import random

from cognitive_discovery.ontology.factors import FACTOR_LEVELS, TASK_FACTOR_AVAILABILITY
from cognitive_discovery.ontology.histories import make_history
from cognitive_discovery.ontology.task_schema import ConditionSpec

from .constraints import validate_condition
from .counterbalance import counterbalanced_mappings


def _balanced_pick(levels, index: int, seed: int, salt: str):
    levels = list(levels)
    epoch, position = divmod(int(index), len(levels))
    salt_value = sum((offset + 1) * ord(character) for offset, character in enumerate(salt))
    random.Random(int(seed) + 104_729 * epoch + salt_value).shuffle(levels)
    return levels[position]


def _level_assignment(task: str, task_index: int, seed: int, configured: dict) -> dict:
    available = TASK_FACTOR_AVAILABILITY[task]
    factors: dict[str, str | None] = {}
    for factor_index, (name, defaults) in enumerate(FACTOR_LEVELS.items()):
        levels = tuple(configured.get(name, defaults))
        if name not in available:
            factors[name] = None
            continue
        # Independently shuffled epochs give exact marginal balance without
        # locking equally sized factors into the same repeated combination.
        factors[name] = str(
            _balanced_pick(levels, task_index, seed, f"{task}:{factor_index}:{name}")
        )
    return factors


def compile_design(
    config: dict,
    *,
    n_conditions: int | None = None,
    seed: int | None = None,
    design_id: str = "discovery_v1",
    exclude_semantic_hashes: set[str] | None = None,
) -> list[ConditionSpec]:
    """Compile unique semantic conditions and paired response mappings.

    ``n_conditions`` counts semantic conditions; the returned rendered design
    has twice as many rows because every condition has two label mappings.
    """

    n_conditions = int(
        n_conditions
        if n_conditions is not None
        else config["design"]["discovery_conditions"]
    )
    seed = int(seed if seed is not None else config["design_seed"])
    if n_conditions < 1:
        raise ValueError("n_conditions must be positive")
    tasks = tuple(config["design"]["task_families"])
    unknown = set(tasks) - set(TASK_FACTOR_AVAILABILITY)
    if unknown:
        raise ValueError(f"unknown task families: {sorted(unknown)}")
    labels = tuple(config.get("response_labels", ("X", "Y")))
    mappings = counterbalanced_mappings(labels)
    factor_levels = config["design"].get("factor_levels", {})
    history_lengths = tuple(config["design"]["history_lengths"])
    history_valences = tuple(config["design"]["history_valences"])
    excluded = set(exclude_semantic_hashes or ())
    task_attempts = {task: 0 for task in tasks}
    compiled: list[ConditionSpec] = []
    attempts = 0
    maximum_attempts = max(10_000, n_conditions * 100)
    while len(compiled) // 2 < n_conditions:
        if attempts >= maximum_attempts:
            raise RuntimeError("could not generate enough unique legal conditions")
        semantic_index = len(compiled) // 2
        task = tasks[semantic_index % len(tasks)]
        candidate_index = task_attempts[task]
        task_attempts[task] += 1
        factors = _level_assignment(task, candidate_index, seed, factor_levels)
        length = int(
            _balanced_pick(history_lengths, candidate_index, seed, f"{task}:history_length")
        )
        valence = str(
            _balanced_pick(history_valences, candidate_index, seed, f"{task}:history_valence")
        )
        if length == 0:
            valence = "neutral"
        history = make_history(length, valence, candidate_index + seed)
        validate_condition(task, factors, history)
        environment_seed = seed * 1_000_003 + semantic_index
        semantic_payload = {
            "task": task,
            "factors": factors,
            "history": {
                "length": history.length,
                "valence": history.valence,
                "actions": history.actions,
                "outcomes": history.outcomes,
            },
        }
        semantic_hash = hashlib.sha256(
            json.dumps(semantic_payload, sort_keys=True).encode()
        ).hexdigest()
        attempts += 1
        if semantic_hash in excluded:
            continue
        excluded.add(semantic_hash)
        pair_id = f"{design_id}-{semantic_hash[:16]}"
        availability = {name: value is not None for name, value in factors.items()}
        for mapping_index, mapping in enumerate(mappings):
            compiled.append(
                ConditionSpec(
                    design_id=design_id,
                    condition_id=f"{pair_id}-m{mapping_index}",
                    paired_condition_id=pair_id,
                    task_family=task,
                    semantic_factors=dict(factors),
                    factor_available=availability,
                    history=history,
                    response_mapping=mapping,
                    environment_seed=environment_seed,
                )
            )
    return compiled

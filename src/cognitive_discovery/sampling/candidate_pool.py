"""Legal SweetPea-spec candidate pools and stable design hashes."""

from __future__ import annotations

import json
import re

import pandas as pd

from cognitive_discovery.design.manifests import semantic_hash, semantic_hash_from_row
from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.experiments.contextual_history import (
    compile_contextual_history_design,
    contextual_domain,
)


def observed_semantic_hashes(records: pd.DataFrame) -> set[str]:
    """Hash endpoint observations only, avoiding prefix hashes as duplicates."""

    frame = records
    if "terminated" in frame and frame.terminated.astype(bool).any():
        frame = frame[frame.terminated.astype(bool)]
    if "response_mapping" in frame:
        subset = (
            ["design_id", "episode_id"]
            if "episode_id" in frame
            else ["paired_condition_id"]
        )
        frame = frame.drop_duplicates(subset=subset)
    return {semantic_hash_from_row(row) for _, row in frame.iterrows()}


def condition_frame(conditions) -> pd.DataFrame:
    """One model-ready row per semantic condition (not per label mapping)."""

    seen, rows = set(), []
    for condition in conditions:
        if condition.paired_condition_id in seen:
            continue
        seen.add(condition.paired_condition_id)
        row = {
            "design_id": condition.design_id,
            "condition_id": condition.condition_id,
            "paired_condition_id": condition.paired_condition_id,
            "task_family": condition.task_family,
            "episode_id": condition.paired_condition_id,
            "step": condition.history.length,
            "history_actions": json.dumps(condition.history.actions),
            "history_outcomes": json.dumps(condition.history.outcomes),
            "history_length": condition.history.length,
            "history_valence": condition.history.valence,
            "response_mapping": json.dumps(
                condition.response_mapping.to_dict(), sort_keys=True
            ),
            "continuation_advantage": condition.continuation_advantage,
            "semantic_hash": semantic_hash(condition),
        }
        row.update(
            {
                f"factor_{name}": value
                for name, value in condition.semantic_factors.items()
            }
        )
        row.update(
            {
                f"factor_available_{name}": bool(value)
                for name, value in condition.factor_available.items()
            }
        )
        if condition.contextual_history is not None:
            for name, value in condition.contextual_history.items():
                row[f"context_{name}"] = (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (tuple, list, dict))
                    else value
                )
        rows.append(row)
    return pd.DataFrame(rows)


def generate_candidate_pool(
    config: dict,
    observed_records: pd.DataFrame,
    *,
    budget: int | None = None,
    multiplier: int | None = None,
    seed: int | None = None,
):
    settings = config.get("active_sampling", {})
    budget = int(budget or settings.get("budget", 1400))
    multiplier = int(multiplier or settings.get("candidate_multiplier", 10))
    if multiplier < 10:
        raise ValueError("active candidate pool must be at least 10x the sample budget")
    excluded = observed_semantic_hashes(observed_records)
    observed_prefixes = {value[:16] for value in excluded}
    if "paired_condition_id" in observed_records:
        for value in observed_records.paired_condition_id.astype(str).unique():
            match = re.search(r"-([0-9a-f]{16})(?::s\d+)?$", value)
            if match:
                observed_prefixes.add(match.group(1))
    target = budget * multiplier
    oversized = target + max(budget, len(observed_prefixes))
    generated = compile_design(
        config,
        n_conditions=oversized,
        seed=int(
            seed
            if seed is not None
            else settings.get("candidate_seed", config["design_seed"] + 20_000_000)
        ),
        design_id="active_round2_candidates",
        exclude_semantic_hashes=excluded,
    )
    selected_pairs = []
    for condition in generated[::2]:
        if semantic_hash(condition)[:16] in observed_prefixes:
            continue
        selected_pairs.append(condition.paired_condition_id)
        if len(selected_pairs) == target:
            break
    selected_pairs = set(selected_pairs)
    conditions = [
        condition
        for condition in generated
        if condition.paired_condition_id in selected_pairs
    ]
    if len(conditions) != 2 * target:
        raise RuntimeError(
            "candidate oversampling did not produce enough unseen conditions"
        )
    frame = condition_frame(conditions)
    if set(frame.semantic_hash) & excluded:
        raise RuntimeError("candidate pool overlaps previously observed conditions")
    return conditions, frame


def generate_theory_candidate_pool(
    config: dict,
    observed_records: pd.DataFrame,
    *,
    candidate_count: int | None = None,
    contextual_fraction: float | None = None,
    seed: int | None = None,
):
    """Combine unseen original-grammar and contextual-history candidates."""

    settings = config.get("theory_resolution", {})
    total = int(candidate_count or settings.get("candidate_count", 20_000))
    fraction = float(
        contextual_fraction
        if contextual_fraction is not None
        else settings.get("contextual_fraction", 0.5)
    )
    if total < 20_000 and not bool(settings.get("allow_small_candidate_pool", False)):
        raise ValueError("Round-3 candidate pool must contain at least 20,000 conditions")
    if not 0 < fraction < 1:
        raise ValueError("contextual candidate fraction must lie between zero and one")
    base_seed = int(seed if seed is not None else settings.get("candidate_seed", 303_001))
    contextual_count = int(round(total * fraction))
    original_count = total - contextual_count
    excluded = observed_semantic_hashes(observed_records)
    original = compile_design(
        config,
        n_conditions=original_count,
        seed=base_seed,
        design_id="theory_resolution_original_v1",
        exclude_semantic_hashes=excluded,
    )
    contextual = compile_contextual_history_design(
        config,
        n_conditions=contextual_count,
        seed=base_seed + 1_000_000,
    )
    conditions = [*original, *contextual]
    frame = condition_frame(conditions)
    frame["domain"] = [
        contextual_domain(row.task_family, pd.notna(row.get("context_phase_order")))
        for _, row in frame.iterrows()
    ]
    if len(frame) != total:
        raise RuntimeError(f"candidate compiler produced {len(frame)} rather than {total}")
    if frame.semantic_hash.duplicated().any():
        duplicates = int(frame.semantic_hash.duplicated().sum())
        raise RuntimeError(f"candidate pool contains {duplicates} semantic duplicates")
    if set(frame.semantic_hash) & excluded:
        raise RuntimeError("Round-3 candidate pool overlaps prior observations")
    return conditions, frame

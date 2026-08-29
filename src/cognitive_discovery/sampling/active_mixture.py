"""Quota-controlled 40/40/20 active mixture with pair-safe selection."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd


@dataclass
class ActiveSelection:
    conditions: list
    scores: pd.DataFrame
    selected: pd.DataFrame


def _integer_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {name: total * float(weight) for name, weight in weights.items()}
    quotas = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda name: (raw[name] - quotas[name], weights[name]),
        reverse=True,
    )
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def select_active_mixture(
    conditions,
    candidate_scores: pd.DataFrame,
    *,
    budget: int,
    weights=None,
    excluded_hashes: set[str] | None = None,
) -> ActiveSelection:
    weights = dict(
        weights or {"coverage": 0.4, "information": 0.4, "disagreement": 0.2}
    )
    if set(weights) != {"coverage", "information", "disagreement"}:
        raise ValueError(
            "active weights must define coverage, information, and disagreement"
        )
    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError("active mixture weights must sum to one")
    required = {
        "paired_condition_id",
        "semantic_hash",
        "task_family",
        "coverage_score",
        "information_score",
        "disagreement_score",
    }
    missing = required - set(candidate_scores)
    if missing:
        raise ValueError(f"candidate scores are missing: {sorted(missing)}")
    scores = candidate_scores.copy()
    excluded = set(excluded_hashes or ())
    scores = scores[~scores.semantic_hash.astype(str).isin(excluded)].copy()
    scores["active_score"] = sum(
        float(weight) * scores[f"{name}_score"] for name, weight in weights.items()
    )
    tasks = tuple(sorted(scores.task_family.astype(str).unique()))
    task_quotas = _integer_quotas(
        int(budget), {task: 1.0 / len(tasks) for task in tasks}
    )
    selected_rows, selected_ids = [], set()
    for task in tasks:
        part = scores[scores.task_family.astype(str) == task]
        strategy_quotas = _integer_quotas(task_quotas[task], weights)
        for strategy in ("coverage", "information", "disagreement"):
            available = part[~part.paired_condition_id.isin(selected_ids)].sort_values(
                [f"{strategy}_score", "active_score", "semantic_hash"],
                ascending=[False, False, True],
            )
            chosen = available.head(strategy_quotas[strategy]).copy()
            chosen["sampling_strategy"] = strategy
            selected_rows.extend(chosen.to_dict("records"))
            selected_ids.update(chosen.paired_condition_id.astype(str))
    if len(selected_rows) != int(budget):
        raise RuntimeError(
            f"selected {len(selected_rows)} conditions; expected {budget}"
        )
    selected = pd.DataFrame(selected_rows)
    if (
        selected.semantic_hash.duplicated().any()
        or set(selected.semantic_hash) & excluded
    ):
        raise RuntimeError(
            "active selection contains a duplicate or previously observed condition"
        )
    strategy = dict(
        zip(selected.paired_condition_id.astype(str), selected.sampling_strategy)
    )
    selected_conditions = [
        replace(condition, sampling_strategy=strategy[condition.paired_condition_id])
        for condition in conditions
        if condition.paired_condition_id in strategy
    ]
    if len(selected_conditions) != 2 * int(budget):
        raise RuntimeError("active selection lost response-mapping counterbalances")
    return ActiveSelection(selected_conditions, scores, selected)

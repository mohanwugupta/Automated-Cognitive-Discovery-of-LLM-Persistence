"""Quota-balanced 60/20/20 discriminating/coverage/random sampler."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class DiscriminationSelection:
    conditions: list
    scores: pd.DataFrame
    selected: pd.DataFrame


def _integer_quotas(total: int, groups: tuple[str, ...]) -> dict[str, int]:
    base, remainder = divmod(int(total), len(groups))
    return {
        group: base + (index < remainder) for index, group in enumerate(groups)
    }


def _balanced_take(
    scores: pd.DataFrame,
    *,
    count: int,
    order_column: str,
    ascending: bool,
    excluded: set[str],
) -> pd.DataFrame:
    available = scores[~scores.selection_unit.astype(str).isin(excluded)].copy()
    domains = tuple(sorted(available.domain.astype(str).unique()))
    quotas = _integer_quotas(int(count), domains)
    pieces, chosen_units = [], set()

    def take_units(part: pd.DataFrame, target: int):
        chosen = []
        ordered = part.sort_values(
            [order_column, "semantic_hash"], ascending=[ascending, True]
        )
        for unit in ordered.selection_unit.astype(str).drop_duplicates():
            rows = ordered[ordered.selection_unit.astype(str) == unit]
            if len(chosen) + len(rows) <= int(target):
                chosen.extend(rows.to_dict("records"))
                chosen_units.add(unit)
            if len(chosen) == int(target):
                break
        return pd.DataFrame(chosen, columns=ordered.columns)

    for domain in domains:
        part = available[
            (available.domain.astype(str) == domain)
            & ~available.selection_unit.astype(str).isin(chosen_units)
        ]
        pieces.append(take_units(part, quotas[domain]))
    pieces = [piece for piece in pieces if not piece.empty]
    selected = pd.concat(pieces, ignore_index=True) if pieces else available.head(0)
    if len(selected) < int(count):
        remainder = available[
            ~available.selection_unit.astype(str).isin(chosen_units)
        ]
        fill = take_units(remainder, int(count) - len(selected))
        selected = (
            fill.reset_index(drop=True)
            if selected.empty
            else pd.concat([selected, fill], ignore_index=True)
        )
    if len(selected) != int(count):
        raise RuntimeError(
            f"could select only {len(selected)} of {count} conditions without splitting a critical contrast"
        )
    return selected


def _strategy_counts(budget: int, allocation: dict[str, float]) -> dict[str, int]:
    raw = {name: int(budget) * float(weight) for name, weight in allocation.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = int(budget) - sum(counts.values())
    order = sorted(allocation, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def select_discrimination_mixture(
    conditions,
    scores: pd.DataFrame,
    *,
    budget: int,
    score_weights=None,
    allocation=None,
    seed: int = 0,
) -> DiscriminationSelection:
    score_weights = dict(
        score_weights or {"disagreement": 0.50, "uncertainty": 0.25, "coverage": 0.25}
    )
    allocation = dict(
        allocation or {"discriminating": 0.60, "coverage": 0.20, "random": 0.20}
    )
    if abs(sum(score_weights.values()) - 1.0) > 1e-9:
        raise ValueError("theory score weights must sum to one")
    if abs(sum(allocation.values()) - 1.0) > 1e-9:
        raise ValueError("sampling allocation must sum to one")
    required = {
        "paired_condition_id",
        "semantic_hash",
        "task_family",
        "domain",
        "disagreement_score",
        "uncertainty_score",
        "coverage_score",
    }
    missing = required - set(scores)
    if missing:
        raise ValueError(f"sampling scores are missing: {sorted(missing)}")
    table = scores.copy()
    if table.semantic_hash.duplicated().any():
        raise ValueError("candidate scores contain duplicate semantic conditions")
    table["active_discrimination_score"] = sum(
        float(weight) * table[f"{name}_score"]
        for name, weight in score_weights.items()
    )
    critical = table.get(
        "context_critical_contrast_id", pd.Series([np.nan] * len(table), index=table.index)
    )
    table["selection_unit"] = [
        f"context:{value}" if pd.notna(value) else f"semantic:{pair_id}"
        for value, pair_id in zip(critical, table.paired_condition_id)
    ]
    rng = np.random.default_rng(int(seed))
    table["random_priority"] = rng.random(len(table))
    counts = _strategy_counts(int(budget), allocation)
    selected_parts = []
    selected_units: set[str] = set()
    # Draw the reference sample before consulting any model- or coverage-based score.
    for strategy, column, ascending in (
        ("random", "random_priority", True),
        ("coverage", "coverage_score", False),
        ("discriminating", "active_discrimination_score", False),
    ):
        chosen = _balanced_take(
            table,
            count=counts[strategy],
            order_column=column,
            ascending=ascending,
            excluded=selected_units,
        ).copy()
        chosen["sampling_strategy"] = strategy
        chosen["selection_seed"] = int(seed)
        selected_parts.append(chosen)
        selected_units.update(chosen.selection_unit.astype(str))
    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) != int(budget) or selected.semantic_hash.duplicated().any():
        raise RuntimeError("discrimination mixture lost quota or duplicate guarantees")
    strategy_by_pair = dict(
        zip(selected.paired_condition_id.astype(str), selected.sampling_strategy)
    )
    selected_conditions = [
        replace(condition, sampling_strategy=strategy_by_pair[condition.paired_condition_id])
        for condition in conditions
        if condition.paired_condition_id in strategy_by_pair
    ]
    if len(selected_conditions) != 2 * int(budget):
        raise RuntimeError("response-mapping counterbalances were not retained")
    return DiscriminationSelection(selected_conditions, table, selected)


def freeze_discrimination_split(
    conditions,
    *,
    fraction: float = 0.25,
    seed: int = 0,
) -> list:
    """Reserve semantic pairs, stratified by task/domain/strategy, before collection."""

    if not 0 < float(fraction) < 1:
        raise ValueError("discrimination fraction must lie between zero and one")
    representatives = {}
    for condition in conditions:
        representatives.setdefault(condition.paired_condition_id, condition)
    groups: dict[tuple[str, str, str], dict[str, list[str]]] = {}
    for pair_id, condition in representatives.items():
        domain = (
            "contextual" if condition.contextual_history is not None else "original"
        )
        key = (condition.task_family, domain, condition.sampling_strategy)
        critical = (
            str(condition.contextual_history["critical_contrast_id"])
            if condition.contextual_history is not None
            else str(pair_id)
        )
        groups.setdefault(key, {}).setdefault(critical, []).append(pair_id)
    rng = np.random.default_rng(int(seed))
    heldout = set()
    for units in groups.values():
        values = np.asarray(sorted(units), dtype=object)
        rng.shuffle(values)
        count = max(1, int(round(float(fraction) * len(values))))
        for unit in values[:count]:
            heldout.update(units[str(unit)])
    return [
        replace(
            condition,
            split=(
                "model_discrimination"
                if condition.paired_condition_id in heldout
                else "round3_train"
            ),
        )
        for condition in conditions
    ]

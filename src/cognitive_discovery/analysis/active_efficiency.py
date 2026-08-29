"""Equal-budget coverage-only versus active-mixture efficiency curves."""

from __future__ import annotations

import re

import pandas as pd

from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model
from cognitive_discovery.models.fitting import regression_metrics, task_macro_metrics


def _root_pair(value):
    return re.sub(r":s\d+$", "", str(value))


def _balanced_head(scores, budget, strategy):
    tasks = sorted(scores.task_family.astype(str).unique())
    per_task, remainder = divmod(int(budget), len(tasks))
    pieces = []
    for index, task in enumerate(tasks):
        count = per_task + int(index < remainder)
        part = scores[scores.task_family.astype(str) == task]
        if strategy == "coverage_only":
            part = part.sort_values(
                ["coverage_score", "semantic_hash"], ascending=[False, True]
            )
        else:
            # Preserve the registered mixture while preferring the objective
            # score within each assigned component.
            quotas = {
                "coverage": round(count * 0.4),
                "information": round(count * 0.4),
            }
            quotas["disagreement"] = count - sum(quotas.values())
            selected = []
            used = set()
            for component in ("coverage", "information", "disagreement"):
                available = part[
                    (part.sampling_strategy == component)
                    & ~part.paired_condition_id.isin(used)
                ].sort_values(f"{component}_score", ascending=False)
                chosen = available.head(quotas[component])
                selected.append(chosen)
                used.update(chosen.paired_condition_id)
            part = pd.concat(selected, ignore_index=True)
        pieces.append(part.head(count))
    return pd.concat(pieces, ignore_index=True).head(int(budget))


def compare_active_efficiency(
    round1: pd.DataFrame,
    active: pd.DataFrame,
    final_test: pd.DataFrame,
    sampling_scores: pd.DataFrame,
    *,
    architecture: str,
    budgets=(100, 250, 500, 1000),
) -> pd.DataFrame:
    scores = sampling_scores.copy()
    active = active.copy()
    active["_semantic_root"] = active.paired_condition_id.map(_root_pair)
    available = set(active._semantic_root)
    scores = scores[scores.paired_condition_id.astype(str).isin(available)]
    base = round1[
        (
            (round1.split == "discovery")
            if "split" in round1
            else pd.Series(True, index=round1.index)
        )
    ].copy()
    rows = []
    for budget in budgets:
        for policy in ("coverage_only", "active_mixture"):
            selected = _balanced_head(scores, budget, policy)
            roots = set(selected.paired_condition_id.astype(str))
            observations = active[active._semantic_root.isin(roots)].drop(
                columns="_semantic_root"
            )
            status = (
                "evaluated"
                if len(roots) == min(int(budget), len(scores))
                else "insufficient"
            )
            fit = fit_hierarchical_model(
                pd.concat([base, observations], ignore_index=True),
                architecture,
                variant="M4",
            )
            prediction = fit.predict(final_test, include_random=True)
            rows.append(
                {
                    "policy": policy,
                    "budget": int(budget),
                    "semantic_conditions": len(roots),
                    "observations": len(observations),
                    "comparison_scope": "queried_active_support",
                    "status": status,
                    **regression_metrics(final_test.persistence_logit, prediction),
                    **task_macro_metrics(final_test, prediction),
                }
            )
    return pd.DataFrame(rows)

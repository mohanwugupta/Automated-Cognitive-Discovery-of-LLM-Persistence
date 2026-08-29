"""Behavior-only collection with exact semantic action replay across mappings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
import math
import random

from cognitive_discovery.data.schema import Observation
from cognitive_discovery.ontology.histories import HistorySpec

from .registry import get_renderer


def _score(condition, participant):
    trial = get_renderer(condition.task_family).render(condition)
    result = participant.binary_decision(
        list(trial.messages),
        condition.response_mapping.labels,
        positive_label=condition.response_mapping.continue_label,
    )
    required = {"p_positive", "p_negative"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"binary decision missing fields: {sorted(missing)}")
    if "hidden_states" in result:
        raise ValueError("behavior-only collection cannot retain hidden states")
    positive = max(float(result["p_positive"]), 1e-12)
    negative = max(float(result["p_negative"]), 1e-12)
    total = positive + negative
    positive, negative = positive / total, negative / total
    return trial, result, positive, negative


def collect_conditions(
    conditions,
    participant,
    *,
    model_revision: str | None = None,
    sample_actions: bool = True,
    expand_history_prefixes: bool = False,
):
    groups = defaultdict(list)
    for condition in conditions:
        groups[condition.paired_condition_id].append(condition)
    if any(len(group) != 2 for group in groups.values()):
        raise ValueError("collection requires both mappings for every semantic condition")
    observations = []
    for pair_id in sorted(groups):
        pair = sorted(groups[pair_id], key=lambda item: item.response_mapping.mapping_id)
        maximum_step = pair[0].history.length if expand_history_prefixes else 0
        for step in range(maximum_step + 1):
            if expand_history_prefixes:
                valence = pair[0].history.valence if step else "neutral"
                prefixes = [
                    replace(
                        condition,
                        history=HistorySpec(
                            step,
                            valence,
                            condition.history.actions[:step],
                            condition.history.outcomes[:step],
                        ),
                    )
                    for condition in pair
                ]
            else:
                prefixes = pair
                step = 0
            scored = [_score(condition, participant) for condition in prefixes]
            average_probability = sum(item[2] for item in scored) / len(scored)
            rng = random.Random(pair[0].environment_seed + 1_000_000 + step)
            semantic_action = (
                "continue"
                if sample_actions and rng.random() < average_probability
                else "disengage"
            )
            if not sample_actions:
                semantic_action = "continue" if average_probability >= 0.5 else "disengage"
            for condition, (trial, result, p_continue, p_disengage) in zip(prefixes, scored):
                condition_id = (
                    f"{condition.condition_id}:s{step}"
                    if expand_history_prefixes
                    else condition.condition_id
                )
                paired_condition_id = (
                    f"{condition.paired_condition_id}:s{step}"
                    if expand_history_prefixes
                    else condition.paired_condition_id
                )
                observations.append(
                    Observation(
                        design_id=condition.design_id,
                        condition_id=condition_id,
                        paired_condition_id=paired_condition_id,
                        task_family=condition.task_family,
                        episode_id=condition.condition_id,
                        step=step,
                        factors=dict(condition.semantic_factors),
                        factor_available=dict(condition.factor_available),
                        history_actions=condition.history.actions,
                        history_outcomes=condition.history.outcomes,
                        semantic_continue_token=condition.response_mapping.continue_label,
                        semantic_disengage_token=condition.response_mapping.disengage_label,
                        response_mapping=json.dumps(
                            condition.response_mapping.to_dict(), sort_keys=True
                        ),
                        p_continue=p_continue,
                        p_disengage=p_disengage,
                        persistence_logit=math.log(p_continue / p_disengage),
                        sampled_action=semantic_action,
                        # Prefixes are exogenously prescribed policy probes. The
                        # sampled action is secondary and does not absorb the
                        # sequence; the endpoint closes the rendered episode.
                        terminated=step == maximum_step,
                        prompt_hash=trial.prompt_hash,
                        environment_seed=condition.environment_seed,
                        model=getattr(participant, "model_id", "unknown"),
                        model_revision=(
                            model_revision or getattr(participant, "revision", None)
                        ),
                        split=condition.split,
                        sampling_strategy=condition.sampling_strategy,
                        p_action_mass_raw=(
                            float(result["p_action_mass_raw"])
                            if result.get("p_action_mass_raw") is not None
                            else None
                        ),
                        top_token_is_action=(
                            bool(result["top_token_is_action"])
                            if result.get("top_token_is_action") is not None
                            else None
                        ),
                    )
                )
    return observations


def expanded_render_conditions(conditions):
    """Yield every label-mapped history prefix used by expanded collection."""

    for condition in conditions:
        for step in range(condition.history.length + 1):
            yield replace(
                condition,
                condition_id=f"{condition.condition_id}:s{step}",
                paired_condition_id=f"{condition.paired_condition_id}:s{step}",
                history=HistorySpec(
                    step,
                    condition.history.valence if step else "neutral",
                    condition.history.actions[:step],
                    condition.history.outcomes[:step],
                ),
            )

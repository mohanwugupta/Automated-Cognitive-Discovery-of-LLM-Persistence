"""Compact semantic contrasts for identifying history computations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import pandas as pd

from cognitive_discovery.design.counterbalance import counterbalanced_mappings
from cognitive_discovery.design.manifests import condition_from_dict
from cognitive_discovery.design.sampling import balanced_factor_assignment
from cognitive_discovery.experiments.contextual_history.factors import PHASE_ORDER
from cognitive_discovery.ontology.histories import HistorySpec
from cognitive_discovery.ontology.task_schema import ConditionSpec

from ..targets.behavioral_targets import compute_condition_targets
from .split import mechanistic_split


CONTRAST_FAMILIES = (
    "outcome_history",
    "contextual_history",
    "raw_history_control",
    "action_history",
    "current_value_control",
)
CONTEXTUAL_TASKS = ("bandit", "foraging", "debugging")


def _valence(outcomes) -> str:
    values = tuple(int(value) for value in outcomes)
    if all(value > 0 for value in values):
        return "positive"
    if all(value < 0 for value in values):
        return "negative"
    if not values or all(value == 0 for value in values):
        return "neutral"
    return "mixed"


def _history(outcomes, actions=("continue", "disengage", "continue")) -> HistorySpec:
    outcomes = tuple(int(value) for value in outcomes)
    actions = tuple(str(value) for value in actions)
    return HistorySpec(len(outcomes), _valence(outcomes), actions, outcomes)


def _context(
    *,
    contrast_id: str,
    a_history: HistorySpec,
    b_history: HistorySpec,
    context_return: str,
) -> dict[str, object]:
    return {
        "phase_order": PHASE_ORDER,
        "a_history_valence": a_history.valence,
        "b_history_valence": b_history.valence,
        "a_history_actions": a_history.actions,
        "a_history_outcomes": a_history.outcomes,
        "b_history_actions": b_history.actions,
        "b_history_outcomes": b_history.outcomes,
        "context_return": context_return,
        "cue_reliability": "high",
        "cue_probability": 0.95,
        "change_point": "same_environment",
        "cue_rule": (
            "An independently calibrated diagnostic marker identifies whether the current "
            "generator matches phase A or phase B."
        ),
        "a_environment_rate": 0.75,
        "b_environment_rate": 0.25,
        "critical_contrast_id": contrast_id,
    }


@dataclass(frozen=True)
class MatchedMechanisticCondition:
    condition: ConditionSpec
    contrast_id: str
    contrast_family: str
    contrast_member: int
    target_name: str
    targets: dict[str, float]

    def to_dict(self) -> dict:
        return {
            **self.condition.to_dict(),
            "contrast_id": self.contrast_id,
            "contrast_family": self.contrast_family,
            "contrast_member": int(self.contrast_member),
            "target_name": self.target_name,
            **{name: float(value) for name, value in self.targets.items()},
        }


def _pair_semantics(
    family: str,
    *,
    task: str,
    factors: dict,
    contrast_id: str,
) -> tuple[
    tuple[HistorySpec, dict | None, dict], tuple[HistorySpec, dict | None, dict], str
]:
    actions = ("continue", "disengage", "continue")
    positive = _history((1, 1, 1), actions)
    negative = _history((-1, -1, -1), actions)
    neutral = _history((0, 0, 0), actions)
    left_factors, right_factors = dict(factors), dict(factors)
    if family == "outcome_history":
        return (
            (positive, None, left_factors),
            (negative, None, right_factors),
            "outcome_history",
        )
    if family == "action_history":
        plus = _history((1, -1, 1), ("continue", "continue", "continue"))
        minus = _history((1, -1, 1), ("disengage", "disengage", "disengage"))
        return (
            (plus, None, left_factors),
            (minus, None, right_factors),
            "action_history",
        )
    if family == "current_value_control":
        left_factors["success_evidence"] = "high"
        right_factors["success_evidence"] = "low"
        return (
            (neutral, None, left_factors),
            (neutral, None, right_factors),
            "success_evidence",
        )
    if task not in CONTEXTUAL_TASKS:
        raise ValueError(f"contextual contrast cannot be rendered for {task}")
    if family == "contextual_history":
        # The raw/recent B history is identical. Only the cue selecting A versus B changes.
        plus_context = _context(
            contrast_id=contrast_id,
            a_history=positive,
            b_history=negative,
            context_return="A",
        )
        minus_context = _context(
            contrast_id=contrast_id,
            a_history=positive,
            b_history=negative,
            context_return="B",
        )
        return (
            (negative, plus_context, left_factors),
            (negative, minus_context, right_factors),
            "contextual_outcome_history",
        )
    if family == "raw_history_control":
        # A is selected with 95% reliability while recent B changes: O differs,
        # but O* changes by only the 5% fallback mass.
        plus_context = _context(
            contrast_id=contrast_id,
            a_history=positive,
            b_history=positive,
            context_return="A",
        )
        minus_context = _context(
            contrast_id=contrast_id,
            a_history=positive,
            b_history=negative,
            context_return="A",
        )
        return (
            (positive, plus_context, left_factors),
            (negative, minus_context, right_factors),
            "outcome_history",
        )
    raise ValueError(f"unknown mechanistic contrast family: {family}")


def compile_mechanistic_conditions(config: dict, *, smoke: bool = False):
    settings = config.get("dataset", {})
    count = int(
        settings.get("smoke_contrasts", 20)
        if smoke
        else settings.get("contrasts", 1200)
    )
    if count < len(CONTRAST_FAMILIES):
        raise ValueError(f"at least {len(CONTRAST_FAMILIES)} contrasts are required")
    discovery = tuple(
        config.get(
            "discovery_tasks", ("bandit", "foraging", "solvability", "debugging")
        )
    )
    heldout = tuple(
        config.get("heldout_tasks", ("waiting", "effort", "information_sampling"))
    )
    all_tasks = discovery + heldout
    seed = int(config.get("seed", 73001))
    labels = tuple(config.get("response_labels", ("X", "Y")))
    mappings = counterbalanced_mappings(labels)
    factor_levels = config.get("factor_levels", {})
    output: list[MatchedMechanisticCondition] = []
    for index in range(count):
        family = CONTRAST_FAMILIES[index % len(CONTRAST_FAMILIES)]
        task = all_tasks[(index // len(CONTRAST_FAMILIES)) % len(all_tasks)]
        if (
            family in {"contextual_history", "raw_history_control"}
            and task not in CONTEXTUAL_TASKS
        ):
            task = CONTEXTUAL_TASKS[
                (index // len(CONTRAST_FAMILIES)) % len(CONTEXTUAL_TASKS)
            ]
        contrast_id = f"mech-{family}-{index:06d}"
        factors = balanced_factor_assignment(task, index, seed, factor_levels)
        plus, minus, target_name = _pair_semantics(
            family, task=task, factors=factors, contrast_id=contrast_id
        )
        split = mechanistic_split(
            contrast_id,
            task,
            discovery_tasks=discovery,
            heldout_tasks=heldout,
            seed=seed,
        )
        environment_seed = seed * 1_000_003 + index
        for member, semantic in ((1, plus), (-1, minus)):
            history, context, member_factors = semantic
            availability = {
                name: value is not None for name, value in member_factors.items()
            }
            pair_id = f"{contrast_id}-{'plus' if member > 0 else 'minus'}"
            for mapping_index, mapping in enumerate(mappings):
                condition = ConditionSpec(
                    design_id="mechanistic_v1",
                    condition_id=f"{pair_id}-m{mapping_index}",
                    paired_condition_id=pair_id,
                    task_family=task,
                    semantic_factors=member_factors,
                    factor_available=availability,
                    history=history,
                    response_mapping=mapping,
                    environment_seed=environment_seed,
                    sampling_strategy=family,
                    split=split,
                    contextual_history=context,
                )
                output.append(
                    MatchedMechanisticCondition(
                        condition=condition,
                        contrast_id=contrast_id,
                        contrast_family=family,
                        contrast_member=member,
                        target_name=target_name,
                        targets=compute_condition_targets(condition),
                    )
                )
    validate_matched_conditions(output)
    return output


def _representatives(records):
    return [
        record for record in records if record.condition.condition_id.endswith("-m0")
    ]


def validate_matched_conditions(records) -> None:
    def targets_equal(left, right):
        if set(left) != set(right):
            return False
        return all(
            left[name] == right[name]
            or (
                isinstance(left[name], float)
                and isinstance(right[name], float)
                and math.isnan(left[name])
                and math.isnan(right[name])
            )
            for name in left
        )

    mappings: dict[str, list[MatchedMechanisticCondition]] = {}
    for record in records:
        mappings.setdefault(record.condition.paired_condition_id, []).append(record)
    if set(map(len, mappings.values())) != {2}:
        raise ValueError("every semantic condition must have both response mappings")
    for pair_id, mapped in mappings.items():
        mapping_ids = {item.condition.response_mapping.mapping_id for item in mapped}
        if mapping_ids != {"continue_x", "continue_y"}:
            raise ValueError(f"response-label counterbalancing failed for {pair_id}")
        left, right = mapped
        if (
            left.condition.semantic_payload() != right.condition.semantic_payload()
            or not targets_equal(left.targets, right.targets)
        ):
            raise ValueError(f"response mappings changed semantic state for {pair_id}")
    groups: dict[str, list[MatchedMechanisticCondition]] = {}
    for record in _representatives(records):
        groups.setdefault(record.contrast_id, []).append(record)
    if not groups or set(map(len, groups.values())) != {2}:
        raise ValueError(
            "every mechanistic contrast must have exactly two semantic members"
        )
    for contrast_id, pair in groups.items():
        plus = next((item for item in pair if item.contrast_member == 1), None)
        minus = next((item for item in pair if item.contrast_member == -1), None)
        if plus is None or minus is None:
            raise ValueError(f"contrast lacks signed members: {contrast_id}")
        a, b = plus.condition, minus.condition
        if a.task_family != b.task_family or a.environment_seed != b.environment_seed:
            raise ValueError(f"task/current environment mismatch in {contrast_id}")
        family = plus.contrast_family
        if (
            family != "current_value_control"
            and a.semantic_factors != b.semantic_factors
        ):
            raise ValueError(f"current-state factors changed in {contrast_id}")
        if family == "outcome_history":
            if (
                a.history.actions != b.history.actions
                or a.history.outcomes == b.history.outcomes
            ):
                raise ValueError(f"outcome contrast is not isolated: {contrast_id}")
        elif family == "action_history":
            if (
                a.history.outcomes != b.history.outcomes
                or a.history.actions == b.history.actions
            ):
                raise ValueError(f"action contrast is not isolated: {contrast_id}")
        elif family == "contextual_history":
            if a.history != b.history:
                raise ValueError(
                    f"contextual contrast changed raw history: {contrast_id}"
                )
            left, right = dict(a.contextual_history), dict(b.contextual_history)
            left.pop("context_return")
            right.pop("context_return")
            if left != right:
                raise ValueError(
                    f"contextual contrast changed a nuisance field: {contrast_id}"
                )
        elif family == "raw_history_control":
            if (
                a.contextual_history["a_history_outcomes"]
                != b.contextual_history["a_history_outcomes"]
            ):
                raise ValueError(
                    f"raw control changed selected A history: {contrast_id}"
                )
        elif family == "current_value_control":
            differing = {
                key
                for key in a.semantic_factors
                if a.semantic_factors[key] != b.semantic_factors[key]
            }
            if differing != {"success_evidence"} or a.history != b.history:
                raise ValueError(
                    f"current-value contrast is not isolated: {contrast_id}"
                )
        if not plus.targets[plus.target_name] > minus.targets[minus.target_name]:
            raise ValueError(f"signed target does not increase in {contrast_id}")


def write_mechanistic_manifest(records, directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    jsonl = root / "mechanistic_conditions.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True, default=list) + "\n" for row in rows),
        encoding="utf-8",
    )
    parquet = root / "mechanistic_conditions.parquet"
    parquet_error = None
    try:
        pd.DataFrame(rows).to_parquet(parquet, index=False)
    except (ImportError, ModuleNotFoundError, ValueError, TypeError) as error:
        parquet = None
        parquet_error = str(error)
    return {
        "path": str(jsonl),
        "parquet_path": str(parquet) if parquet else None,
        "rows": len(rows),
        "contrasts": len({record.contrast_id for record in records}),
        "parquet_error": parquet_error,
    }


def load_mechanistic_manifest(path: str | Path) -> list[MatchedMechanisticCondition]:
    path = Path(path)
    if path.suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict("records")
    else:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    output = []
    for row in rows:
        condition = condition_from_dict(row)
        output.append(
            MatchedMechanisticCondition(
                condition=condition,
                contrast_id=str(row["contrast_id"]),
                contrast_family=str(row["contrast_family"]),
                contrast_member=int(row["contrast_member"]),
                target_name=str(row["target_name"]),
                targets={
                    name: float(row[name])
                    for name in (
                        "outcome_history",
                        "contextual_outcome_history",
                        "action_history",
                        "success_evidence",
                        "progress_evidence",
                        "continuation_cost",
                        "disengagement_value",
                        "continuation_value",
                    )
                },
            )
        )
    validate_matched_conditions(output)
    return output

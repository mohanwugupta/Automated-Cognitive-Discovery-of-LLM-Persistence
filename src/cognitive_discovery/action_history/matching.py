"""Caliper matching for history-versus-current-decision dissociation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_NUISANCE_COLUMNS = (
    "success_evidence",
    "progress_evidence",
    "continuation_cost",
    "disengagement_value",
    "continuation_value",
    "outcome_history",
    "contextual_outcome_history",
)


@dataclass(frozen=True)
class MatchConfig:
    decision_caliper: float = 0.15
    history_min_delta: float = 1.0
    history_caliper: float = 1e-8
    decision_min_delta: float = 0.75
    nuisance_max_distance: float = 2.5
    nuisance_columns: tuple[str, ...] = DEFAULT_NUISANCE_COLUMNS


def _mapping_column(frame: pd.DataFrame) -> str:
    for name in ("response_mapping", "response_mapping_sign"):
        if name in frame:
            return name
    raise ValueError("matching requires response_mapping or response_mapping_sign")


def _scaled_nuisance(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    columns = []
    for name in names:
        if name not in frame:
            continue
        value = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(value)
        mean = float(np.nanmean(value)) if (~missing).any() else 0.0
        filled = np.where(missing, mean, value)
        scale = float(np.std(filled))
        scale = scale if scale > 1e-8 else 1.0
        columns.extend(((filled - mean) / scale, missing.astype(float)))
    return np.column_stack(columns) if columns else np.zeros((len(frame), 0))


def _candidate_rows(frame: pd.DataFrame, kind: str, config: MatchConfig) -> list[dict]:
    nuisance = _scaled_nuisance(frame, config.nuisance_columns)
    action = frame.action_history.to_numpy(dtype=float)
    decision = frame.persistence_logit.to_numpy(dtype=float)
    identifiers = frame.condition_id.astype(str).to_numpy()
    candidates = []
    for left in range(len(frame)):
        for right in range(left + 1, len(frame)):
            delta_a = abs(action[left] - action[right])
            delta_d = abs(decision[left] - decision[right])
            if kind == "history_decision_matched":
                eligible = (
                    delta_d <= config.decision_caliper
                    and delta_a >= config.history_min_delta
                )
                contrast_strength = delta_a
                caliper_distance = delta_d
                high = left if action[left] >= action[right] else right
            else:
                eligible = (
                    delta_a <= config.history_caliper
                    and delta_d >= config.decision_min_delta
                )
                contrast_strength = delta_d
                caliper_distance = delta_a
                high = left if decision[left] >= decision[right] else right
            if not eligible:
                continue
            nuisance_distance = float(np.linalg.norm(nuisance[left] - nuisance[right]))
            if nuisance_distance > config.nuisance_max_distance:
                continue
            low = right if high == left else left
            # Caliper fit is primary; nuisance fit is secondary; large contrast
            # resolves remaining ties deterministically.
            score = (
                caliper_distance + 0.05 * nuisance_distance - 1e-6 * contrast_strength
            )
            candidates.append(
                {
                    "score": score,
                    "high_index": high,
                    "low_index": low,
                    "high_condition_id": identifiers[high],
                    "low_condition_id": identifiers[low],
                    "action_delta": float(action[high] - action[low]),
                    "decision_delta": float(decision[high] - decision[low]),
                    "abs_action_delta": float(delta_a),
                    "abs_decision_delta": float(delta_d),
                    "nuisance_distance": nuisance_distance,
                }
            )
    return sorted(
        candidates,
        key=lambda row: (
            row["score"],
            row["high_condition_id"],
            row["low_condition_id"],
        ),
    )


def build_matched_pairs(
    frame: pd.DataFrame, config: MatchConfig | None = None
) -> pd.DataFrame:
    """Build non-overlapping within-task, within-mapping pairs for both contrasts."""

    config = config or MatchConfig()
    required = {"condition_id", "task_family", "action_history", "persistence_logit"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"matching columns are missing: {missing}")
    mapping = _mapping_column(frame)
    grouping = ["task_family", mapping]
    if "split" in frame:
        grouping.append("split")
    rows = []
    for keys, part in frame.groupby(grouping, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        common = dict(zip(grouping, keys))
        part = part.reset_index(drop=True)
        for kind in ("history_decision_matched", "decision_history_matched"):
            used: set[str] = set()
            for candidate in _candidate_rows(part, kind, config):
                pair_ids = {
                    candidate["high_condition_id"],
                    candidate["low_condition_id"],
                }
                if used & pair_ids:
                    continue
                used.update(pair_ids)
                rows.append({"contrast_kind": kind, **common, **candidate})
    output = pd.DataFrame(rows)
    # Counterbalanced prompts must contribute equally. Trim deterministically
    # within task/split/contrast strata so a mapping cannot dominate the result.
    mapping_levels = sorted(output[mapping].astype(str).unique()) if len(output) else []
    if len(mapping_levels) == 2:
        strata = ["contrast_kind", "task_family"]
        if "split" in output:
            strata.append("split")
        balanced = []
        for _, part in output.groupby(strata, dropna=False, sort=True):
            counts = part.groupby(mapping, dropna=False).size()
            if len(counts) != 2:
                continue
            count = int(counts.min())
            for _, mapped in part.groupby(mapping, dropna=False, sort=True):
                balanced.append(
                    mapped.sort_values(["score", "high_condition_id"]).head(count)
                )
        if balanced:
            output = pd.concat(balanced, ignore_index=True)
    validate_matched_pairs(output, config)
    return output


def validate_matched_pairs(
    pairs: pd.DataFrame, config: MatchConfig | None = None
) -> None:
    config = config or MatchConfig()
    if pairs.empty:
        raise ValueError("matching produced no eligible pairs")
    history = pairs.contrast_kind.eq("history_decision_matched")
    decision = pairs.contrast_kind.eq("decision_history_matched")
    if not history.any() or not decision.any():
        raise ValueError("both matched contrast types are required")
    if not (
        (
            pairs.loc[history, "abs_decision_delta"] <= config.decision_caliper + 1e-12
        ).all()
        and (
            pairs.loc[history, "abs_action_delta"] >= config.history_min_delta - 1e-12
        ).all()
        and (
            pairs.loc[decision, "abs_action_delta"] <= config.history_caliper + 1e-12
        ).all()
        and (
            pairs.loc[decision, "abs_decision_delta"]
            >= config.decision_min_delta - 1e-12
        ).all()
    ):
        raise ValueError("a matched pair violates its preregistered caliper")
    if (pairs.nuisance_distance > config.nuisance_max_distance + 1e-12).any():
        raise ValueError("a matched pair violates the nuisance caliper")

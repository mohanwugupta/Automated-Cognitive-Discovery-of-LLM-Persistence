from __future__ import annotations

import math

import numpy as np
import pandas as pd


FORBIDDEN_FUTURE_FIELDS = {
    "subsequent_reward",
    "subsequent_outcome",
    "subsequent_effort",
    "subsequent_success",
    "termination_reason",
    "episode_duration",
    "realized_future_reward",
}


def assert_no_future_leakage(features) -> None:
    forbidden = sorted(set(features) & FORBIDDEN_FUTURE_FIELDS)
    if forbidden:
        raise ValueError(f"future leakage fields requested: {forbidden}")


def validate_records(records) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(
            [record.to_dict() if hasattr(record, "to_dict") else record for record in records]
        )
    required = {
        "condition_id",
        "paired_condition_id",
        "episode_id",
        "task_family",
        "step",
        "terminated",
        "p_continue",
        "p_disengage",
        "persistence_logit",
    }
    missing = required - set(records)
    if missing:
        raise ValueError(f"records missing required fields: {sorted(missing)}")
    if records.condition_id.duplicated().any():
        raise ValueError("duplicate condition identifiers")
    probabilities = records[["p_continue", "p_disengage"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("semantic probabilities must be finite and strictly positive")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("semantic probabilities must sum to one")
    expected = np.log(probabilities[:, 0] / probabilities[:, 1])
    if not np.allclose(expected, records.persistence_logit.to_numpy(dtype=float), atol=1e-7):
        raise ValueError("persistence logit is inconsistent with semantic probabilities")
    for episode_id, episode in records.groupby("episode_id", sort=False):
        absorbed = False
        for row in episode.sort_values("step").itertuples():
            if absorbed:
                raise ValueError(f"post-termination state in episode {episode_id}")
            absorbed = bool(row.terminated)
    return records


def pilot_gate(records, config: dict) -> dict:
    frame = validate_records(records)
    lower, upper = config["collection"]["pilot_probability_bounds"]
    parse_rate = float(frame.p_continue.notna().mean())
    mean_probability = float(frame.p_continue.mean())
    mapping_gap = float(
        frame.groupby("paired_condition_id").p_continue.agg(lambda values: abs(np.diff(values)[0])).mean()
    )
    top_token_rate = (
        float(frame.top_token_is_action.fillna(False).astype(bool).mean())
        if "top_token_is_action" in frame
        else float("nan")
    )
    checks = {
        "parse_rate": parse_rate >= float(config["collection"]["minimum_parse_rate"]),
        "behavioral_range": lower < mean_probability < upper,
        "logit_variance": float(frame.persistence_logit.std())
        >= float(config["collection"]["minimum_logit_sd"]),
        "counterbalance": mapping_gap <= float(config["collection"]["maximum_mapping_gap"]),
        "valid_semantic_tokens": (
            top_token_rate
            >= float(config["collection"].get("minimum_top_token_action_rate", 0.90))
            if math.isfinite(top_token_rate)
            else True
        ),
    }
    return {
        "approved": all(checks.values()),
        "checks": checks,
        "parse_rate": parse_rate,
        "mean_p_continue": mean_probability,
        "persistence_logit_sd": float(frame.persistence_logit.std()),
        "mean_absolute_mapping_gap": mapping_gap,
        "top_token_action_rate": top_token_rate,
    }

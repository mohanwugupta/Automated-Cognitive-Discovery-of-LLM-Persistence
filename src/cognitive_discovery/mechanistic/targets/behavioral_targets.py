"""Immutable computational targets and behavioral intervention coefficients."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cognitive_discovery.ontology.factors import ORDINAL_VALUE

from .contextual_targets import contextual_outcome_history
from .history_targets import action_history, outcome_history


TARGET_COLUMNS = (
    "outcome_history",
    "contextual_outcome_history",
    "action_history",
)
TARGET_PARAMETERS = {
    "outcome_history": "history_outcome_kernel",
    "contextual_outcome_history": "context_relevant_outcome_kernel",
    "action_history": "history_action_kernel",
}
CONTROL_FACTORS = (
    "success_evidence",
    "progress_evidence",
    "continuation_cost",
    "disengagement_value",
    "continuation_value",
)


def _number(value) -> float:
    if value is None:
        return float("nan")
    if value in ORDINAL_VALUE:
        return float(ORDINAL_VALUE[value])
    return float(value)


def compute_condition_targets(condition) -> dict[str, float]:
    """Compute all targets solely from the frozen semantic condition."""

    result = {
        "outcome_history": outcome_history(condition.history.outcomes),
        "contextual_outcome_history": contextual_outcome_history(condition),
        "action_history": action_history(condition.history.actions),
    }
    for factor in CONTROL_FACTORS:
        result[factor] = _number(condition.semantic_factors.get(factor))
    return result


def attach_behavioral_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach targets to a flattened condition frame without fitting anything."""

    output = frame.copy()
    raw_outcome = output["history_outcomes"].map(outcome_history)
    output["outcome_history"] = raw_outcome
    output["action_history"] = output["history_actions"].map(action_history)
    contextual = []
    for row, recent in zip(output.to_dict("records"), raw_outcome):

        def present(value):
            if value is None:
                return False
            try:
                missing = pd.isna(value)
            except (TypeError, ValueError):
                return True
            return not bool(missing) if isinstance(missing, (bool, np.bool_)) else True

        context = {
            key.removeprefix("context_"): value
            for key, value in row.items()
            if key.startswith("context_") and present(value)
        }
        payload = {
            "contextual_history": context or None,
            "history_outcomes": row["history_outcomes"],
        }
        contextual.append(
            contextual_outcome_history(payload) if context else float(recent)
        )
    output["contextual_outcome_history"] = contextual
    return output


def load_frozen_handoff(theory_root: str | Path) -> tuple[dict, str]:
    path = Path(theory_root) / "theory/mechanistic_targets.json"
    if not path.exists():
        raise FileNotFoundError(
            f"frozen behavioral handoff is absent: {path}; complete Round 3 first"
        )
    raw = path.read_bytes()
    handoff = json.loads(raw)
    required = {"parameterization", "task_specific_interventions"}
    missing = required - set(handoff)
    if missing:
        raise ValueError(f"mechanistic handoff is missing: {sorted(missing)}")
    return handoff, hashlib.sha256(raw).hexdigest()


def behavioral_coefficients(handoff: dict, *, target: str) -> dict[str, float]:
    parameter = TARGET_PARAMETERS[target]
    rows = [
        row
        for row in handoff.get("task_specific_interventions", ())
        if str(row.get("parameter")) == parameter
    ]
    return {
        str(row["task_family"]): float(row["intervention_derivative"]) for row in rows
    }


def load_behavioral_coefficients(
    theory_root: str | Path, handoff: dict, *, target: str
) -> tuple[dict[str, float], str]:
    """Load an independently frozen coefficient table for a mechanistic target.

    The post-comparison handoff is primary. If that winning architecture does
    not contain a secondary control target, use its explicitly frozen Round-3
    comparator rather than estimating anything from mechanistic outcomes.
    """

    primary = behavioral_coefficients(handoff, target=target)
    if primary:
        return primary, "theory/mechanistic_targets.json"
    parameter = TARGET_PARAMETERS[target]
    architectures = {
        "outcome_history": ("outcome_history", "dual_history"),
        "contextual_outcome_history": ("latent_context",),
        "action_history": ("dual_history",),
    }[target]
    root = Path(theory_root)
    for architecture in architectures:
        path = root / "frozen_models" / architecture / "parameters.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        selected = table[table.parameter.astype(str) == parameter]
        if len(selected):
            return (
                {
                    str(row.task_family): float(row.estimate)
                    for row in selected.itertuples()
                },
                str(path.relative_to(root)),
            )
    return {}, "unavailable"


def load_behavioral_normalization(
    theory_root: str | Path, handoff: dict, *, target: str
) -> tuple[dict[str, float], str]:
    """Return the behavioral encoder's immutable mean/scale for a target."""

    parameter = TARGET_PARAMETERS[target]
    registered = handoff.get("computational_target_normalization", {}).get(parameter)
    if registered is not None:
        return (
            {"mean": float(registered["mean"]), "scale": float(registered["scale"])},
            "theory/mechanistic_targets.json",
        )
    architectures = {
        "outcome_history": ("outcome_history", "dual_history"),
        "contextual_outcome_history": ("latent_context",),
        "action_history": ("dual_history",),
    }[target]
    root = Path(theory_root)
    for architecture in architectures:
        path = root / "frozen_models" / architecture / "model.pkl"
        if not path.exists():
            continue
        with path.open("rb") as handle:
            fit = pickle.load(handle)
        features = tuple(fit.encoder.features)
        if parameter not in features:
            continue
        index = features.index(parameter)
        scale = float(fit.encoder.scales[2 * index])
        if scale <= 0:
            raise ValueError(f"frozen behavioral scale is nonpositive for {target}")
        return (
            {
                "mean": float(fit.encoder.means[2 * index]),
                "scale": scale,
            },
            str(path.relative_to(root)),
        )
    raise FileNotFoundError(
        f"no frozen behavioral normalization is available for {target}"
    )


def predicted_logit_change(
    tasks, computational_deltas, coefficients: dict[str, float]
) -> np.ndarray:
    tasks = np.asarray(tasks, dtype=str)
    deltas = np.asarray(computational_deltas, dtype=float)
    if tasks.shape != deltas.shape:
        raise ValueError("tasks and computational deltas must have the same shape")
    missing = sorted(set(tasks) - set(coefficients))
    if missing:
        raise ValueError(f"frozen behavioral coefficients are missing tasks: {missing}")
    return np.asarray(
        [coefficients[task] * delta for task, delta in zip(tasks, deltas)], dtype=float
    )

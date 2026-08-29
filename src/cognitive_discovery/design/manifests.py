from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from cognitive_discovery.ontology.histories import HistorySpec
from cognitive_discovery.ontology.task_schema import ConditionSpec, ResponseMapping

from .sweetpea_design import declarative_spec


def semantic_hash(condition) -> str:
    contextual = (
        {
            key: value
            for key, value in condition.contextual_history.items()
            if key != "critical_contrast_id"
        }
        if condition.contextual_history is not None
        else None
    )
    payload = {
        "task": condition.task_family,
        "factors": condition.semantic_factors,
        "history": {
            "length": condition.history.length,
            "valence": condition.history.valence,
            "actions": condition.history.actions,
            "outcomes": condition.history.outcomes,
        },
        "contextual_history": contextual,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list).encode()
    ).hexdigest()


def semantic_hash_from_row(row) -> str:
    """Hash a standardized observation using its semantic endpoint state."""

    get = (
        row.get
        if hasattr(row, "get")
        else lambda name, default=None: getattr(row, name, default)
    )
    actions = get("history_actions", ())
    outcomes = get("history_outcomes", ())
    if isinstance(actions, str):
        actions = json.loads(actions)
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    factors = {
        name.removeprefix("factor_"): get(name)
        for name in row.index
        if name.startswith("factor_") and not name.startswith("factor_available_")
    }
    payload = {
        "task": str(get("task_family")),
        "factors": factors,
        "history": {
            "length": len(actions),
            "valence": get("history_valence", _history_valence(outcomes)),
            "actions": tuple(actions),
            "outcomes": tuple(int(value) for value in outcomes),
        },
    }
    context = {}
    for name in row.index:
        if not name.startswith("context_"):
            continue
        value = get(name)
        if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
            continue
        context[name.removeprefix("context_")] = value
    if context:
        context.pop("critical_contrast_id", None)
        payload["contextual_history"] = context
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list).encode()
    ).hexdigest()


def _history_valence(outcomes) -> str:
    values = tuple(int(value) for value in outcomes)
    if not values or all(value == 0 for value in values):
        return "neutral"
    if all(value > 0 for value in values):
        return "positive"
    if all(value < 0 for value in values):
        return "negative"
    return "mixed"


def condition_from_dict(row: dict) -> ConditionSpec:
    history = row["history"]
    mapping = row["response_mapping"]
    contextual = row.get("contextual_history")
    if contextual is not None and not isinstance(contextual, dict):
        contextual = None if pd.isna(contextual) else dict(contextual)
    if isinstance(mapping, str):
        mapping = json.loads(mapping)
    return ConditionSpec(
        design_id=str(row["design_id"]),
        condition_id=str(row["condition_id"]),
        paired_condition_id=str(row["paired_condition_id"]),
        task_family=str(row["task_family"]),
        semantic_factors=dict(row.get("factors", row.get("semantic_factors", {}))),
        factor_available={
            name: bool(value) for name, value in row["factor_available"].items()
        },
        history=HistorySpec(
            int(history["length"]),
            str(history["valence"]),
            tuple(history["actions"]),
            tuple(int(value) for value in history["outcomes"]),
        ),
        response_mapping=ResponseMapping(
            continue_label=str(mapping["continue"]),
            disengage_label=str(mapping["disengage"]),
        ),
        environment_seed=int(row["environment_seed"]),
        sampling_strategy=str(row.get("sampling_strategy", "coverage")),
        split=str(row.get("split", "unassigned")),
        contextual_history=(dict(contextual) if contextual is not None else None),
    )


def load_condition_manifest(path: str | Path) -> list[ConditionSpec]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(path).to_dict("records")
    else:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    return [condition_from_dict(row) for row in rows]


def write_design_artifacts(conditions, config: dict, directory: str | Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ontology = declarative_spec(config)
    (directory / "ontology.json").write_text(
        json.dumps(ontology["factors"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "sweetpea_spec.json").write_text(
        json.dumps(ontology, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [condition.to_dict() for condition in conditions]
    path = directory / "condition_manifest.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=list) + "\n" for row in rows),
        encoding="utf-8",
    )
    parquet_path = directory / "condition_manifest.parquet"
    parquet_error = None
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        parquet_error = str(error)
        parquet_path = None
    return {
        "conditions": len(rows),
        "semantic_conditions": len(rows) // 2,
        "path": str(path),
        "parquet_path": str(parquet_path) if parquet_path else None,
        "parquet_error": parquet_error,
    }

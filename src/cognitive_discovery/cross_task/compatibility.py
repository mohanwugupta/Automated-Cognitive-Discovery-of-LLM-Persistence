"""Frozen task × variable compatibility expansion and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

from .config import TASKS


REQUIRED_CELL_FIELDS = {
    "task", "variable", "manipulable", "levels", "semantic_definition",
    "nuisance_variables_to_hold_fixed", "history_requirements", "notes",
}


def _expanded_cells(value: Mapping) -> list[dict]:
    variables = value.get("variables", {})
    tasks = value.get("tasks", {})
    rows = []
    for task in TASKS:
        task_record = tasks.get(task)
        if not isinstance(task_record, Mapping):
            raise ValueError(f"compatibility matrix lacks task: {task}")
        manipulable = set(task_record.get("manipulable", ()))
        overrides = task_record.get("overrides", {})
        for variable, definition in variables.items():
            override = overrides.get(variable, {})
            available = variable in manipulable
            row = {
                "task": task,
                "variable": variable,
                "manipulable": available,
                "levels": list(override.get("levels", definition["levels"])) if available else [],
                "semantic_definition": override.get(
                    "semantic_definition", definition["semantic_definition"]
                ),
                "nuisance_variables_to_hold_fixed": list(override.get(
                    "nuisance_variables_to_hold_fixed",
                    definition.get("nuisance_variables_to_hold_fixed", []),
                )),
                "history_requirements": override.get(
                    "history_requirements", definition.get("history_requirements", "none")
                ),
                "notes": override.get(
                    "notes",
                    task_record.get("notes", "validated by the canonical task renderer")
                    if available else
                    f"not implemented as an independent manipulation in the canonical {task} renderer",
                ),
            }
            rows.append(row)
    return rows


def validate_compatibility_matrix(value: Mapping) -> pd.DataFrame:
    if value.get("schema_version") != "task-variable-compatibility-v1":
        raise ValueError("unsupported task-variable compatibility schema")
    variables = value.get("variables", {})
    if len(variables) != 13:
        raise ValueError("v1 compatibility ontology must contain exactly 13 variables")
    rows = _expanded_cells(value)
    frame = pd.DataFrame(rows)
    if len(frame) != len(TASKS) * len(variables):
        raise ValueError("compatibility matrix is incomplete")
    if frame.groupby(["task", "variable"]).size().ne(1).any():
        raise ValueError("task-variable cells must be unique")
    for row in rows:
        if set(row) != REQUIRED_CELL_FIELDS:
            raise ValueError("expanded compatibility cell has an invalid schema")
        if row["manipulable"] and len(row["levels"]) < 2:
            raise ValueError(f"manipulable cell lacks levels: {row['task']} × {row['variable']}")
        if not row["manipulable"] and row["levels"]:
            raise ValueError("unavailable task-variable cells cannot declare levels")
        if not row["semantic_definition"] or not row["notes"]:
            raise ValueError("compatibility cells require definitions and notes")
    return frame


def load_compatibility_matrix(path: str | Path) -> pd.DataFrame:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("compatibility matrix must be a mapping")
    return validate_compatibility_matrix(value)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .sweetpea_design import declarative_spec


def semantic_hash(condition) -> str:
    payload = {
        "task": condition.task_family,
        "factors": condition.semantic_factors,
        "history": {
            "length": condition.history.length,
            "valence": condition.history.valence,
            "actions": condition.history.actions,
            "outcomes": condition.history.outcomes,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list).encode()
    ).hexdigest()


def write_design_artifacts(conditions, config: dict, directory: str | Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ontology = declarative_spec(config)
    (directory / "ontology.json").write_text(
        json.dumps(ontology["factors"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
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

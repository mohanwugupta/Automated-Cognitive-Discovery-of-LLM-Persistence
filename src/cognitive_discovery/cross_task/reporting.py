"""Reports generated only from frozen cross-task result tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_model_report(model_root: str | Path) -> Path:
    root = Path(model_root)
    provenance = json.loads((root.parent / "provenance.json").read_text(encoding="utf-8"))
    model_key = root.name
    model = next(item for item in provenance["models"] if item["key"] == model_key)
    paths = {
        "behavior": root / "behavior/variable_effects.csv",
        "computational": root / "computational/model_comparison.csv",
        "survivors": root / "computational/survivor_set.json",
        "representation": root / "representation/transfer_long.csv",
        "causal": root / "causal/transfer_long.csv",
    }
    absent = [name for name, path in paths.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"report inputs are absent: {absent}")
    behavior = pd.read_csv(paths["behavior"])
    representation = pd.read_csv(paths["representation"])
    causal = pd.read_csv(paths["causal"])
    survivors = json.loads(paths["survivors"].read_text(encoding="utf-8"))
    text = f"""# Cross-task discovery report: {model_key}

Model: `{model['id']}`  
Revision: `{model['revision']}`  
Behavior endpoint: `semantic_persistence_logit_v1`  
Representation endpoint: `frozen_source_decoding_v1`  
Causal endpoint: `cognitive_counterfactual_recovery` / `global_cfr_v1`

## Behavioral and computational discovery

- Behavioral task-variable effects estimated: {len(behavior)}
- Behavior-only surviving theories: {', '.join(survivors['behavioral_survivor_set'])}
- Theory status: {survivors['behavioral_theory_status']}
- Neural outcomes used to choose survivors: no

## Representation (Level 3)

- Frozen source-to-target evaluations: {len(representation)}
- This is representation evidence, not causal evidence.

## Causality (Levels 4–5)

- Registered causal evaluations: {len(causal)}
- Cross-task rows: {(causal.source_task != causal.target_task).sum()}
- A null or architecture-specific result is retained as a scientific outcome.

## Interpretation guard

Behavioral similarity, computational fit, neural representation, and causal neural
transfer are separate evidence levels. No Level-3 decoder result is described as a
causal mechanism. Historical replication outputs were not scientific inputs.
"""
    path = root / "DISCOVERY_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_cross_model_report(output: str | Path) -> Path:
    root = Path(output)
    tables = {
        name: pd.read_csv(root / filename)
        for name, filename in {
            "behavioral": "behavioral_comparison.csv",
            "computational": "computational_comparison.csv",
            "representational": "representational_comparison.csv",
            "causal": "causal_comparison.csv",
        }.items()
    }
    text = """# Cross-model discovery report

This report compares structural patterns across Qwen, Gemma, and Llama. It does
not require identical neural coordinates and does not turn missing transfer into
a failed experiment.

""" + "\n".join(
        f"- {name.title()} comparison rows: {len(frame)}"
        for name, frame in tables.items()
    ) + """

Interpretation space includes universal structure, shared computation in different
coordinates, mechanistic families, behavioral-only commonality, and broad fracture.
The frozen tables—not manually entered values—are the source of all summaries.
"""
    path = root / "CROSS_MODEL_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path

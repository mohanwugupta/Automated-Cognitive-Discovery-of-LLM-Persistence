"""Relate behavioral, computational, representational, and causal maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _pairwise_behavior(effect: pd.DataFrame) -> pd.DataFrame:
    pivot = effect.pivot_table(index="task", columns="variable", values="effect")
    rows = []
    for source in pivot.index:
        for target in pivot.index:
            left, right = pivot.loc[source], pivot.loc[target]
            valid = left.notna() & right.notna()
            similarity = (
                float(np.corrcoef(left[valid], right[valid])[0, 1])
                if valid.sum() >= 2 and left[valid].std() > 0 and right[valid].std() > 0
                else np.nan
            )
            rows.append({
                "source_task": source, "target_task": target,
                "behavioral_effect_similarity": similarity,
                "shared_variables": int(valid.sum()),
            })
    return pd.DataFrame(rows)


def synthesize_levels(
    behavior: pd.DataFrame,
    representation: pd.DataFrame,
    causal: pd.DataFrame,
    *,
    output: str | Path,
) -> dict:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    similarity = _pairwise_behavior(behavior)
    representation_summary = representation.groupby(
        ["source_task", "target_task"], as_index=False
    ).value.mean().rename(columns={"value": "representational_transfer"})
    causal_summary = causal.groupby(
        ["source_task", "target_task"], as_index=False
    ).global_cfr.mean().rename(columns={"global_cfr": "causal_transfer"})
    relationships = similarity.merge(
        representation_summary, on=["source_task", "target_task"], how="outer"
    ).merge(causal_summary, on=["source_task", "target_task"], how="outer")
    relationships.to_csv(root / "level_relationships.csv", index=False)
    similarity.to_csv(root / "task_similarity.csv", index=False)
    # Clusters are deliberately descriptive and outcome-transparent: connected
    # components under positive finite similarity, never a causal claim.
    similarity.assign(
        same_descriptive_family=lambda x: x.behavioral_effect_similarity > 0
    ).to_csv(root / "discovered_task_clusters.csv", index=False)
    return {"task_pairs": int(len(relationships)), "evidence_level_max": 5}


def cross_model_tables(model_roots: dict[str, Path], output: str | Path) -> dict:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    specs = {
        "behavioral_comparison.csv": "behavior/variable_effects.csv",
        "computational_comparison.csv": "computational/model_comparison.csv",
        "representational_comparison.csv": "representation/transfer_long.csv",
        "causal_comparison.csv": "causal/transfer_long.csv",
    }
    counts = {}
    for filename, relative in specs.items():
        frames = []
        for model, model_root in model_roots.items():
            path = model_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"cross-model input is absent: {path}")
            frame = pd.read_csv(path)
            frame.insert(0, "architecture", model)
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
        result.to_csv(root / filename, index=False)
        counts[filename] = len(result)
    return counts

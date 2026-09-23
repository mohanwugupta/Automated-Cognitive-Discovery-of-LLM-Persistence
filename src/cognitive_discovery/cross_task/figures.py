"""Figures generated exclusively from frozen cross-task CSV tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _heatmap(frame, index, columns, values, path: Path, title: str):
    pivot = frame.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    figure, axis = plt.subplots(figsize=(max(6, .8 * len(pivot.columns)), max(4, .5 * len(pivot))))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _scatter(frame, x, y, path: Path, title: str):
    clean = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(clean[x], clean[y], alpha=.7)
    axis.axhline(0, color="black", linewidth=.5)
    axis.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "), title=title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_model_figures(model_root: str | Path) -> list[Path]:
    root = Path(model_root)
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    behavior = pd.read_csv(root / "behavior/variable_effects.csv")
    computational = pd.read_csv(root / "computational/model_comparison.csv")
    representation = pd.read_csv(root / "representation/transfer_long.csv")
    sharing = pd.read_csv(root / "representation/shared_private_models.csv")
    causal = pd.read_csv(root / "causal/transfer_long.csv")
    relationships = pd.read_csv(root / "synthesis/level_relationships.csv")
    paths = [destination / name for name in (
        "behavioral_effects.png", "computational_models.png",
        "representational_transfer.png", "shared_private.png",
        "causal_transfer.png", "behavior_to_representation.png",
        "representation_to_causal.png",
    )]
    _heatmap(behavior, "task", "variable", "effect", paths[0], "Behavioral causal effects")
    _heatmap(
        computational[computational.split == "test"], "model", "sharing", "r2",
        paths[1], "Computational model and sharing performance",
    )
    _heatmap(
        representation, "source_task", "target_task", "value", paths[2],
        "Frozen source-to-target representation transfer",
    )
    _heatmap(sharing, "sharing_model", "variable", "r2", paths[3], "Shared/private representation models")
    _heatmap(causal, "source_task", "target_task", "global_cfr", paths[4], "Variable-specific causal transfer")
    _scatter(
        relationships, "behavioral_effect_similarity", "representational_transfer",
        paths[5], "Behavioral similarity vs representation transfer",
    )
    _scatter(
        relationships, "representational_transfer", "causal_transfer",
        paths[6], "Representation transfer vs causal transfer",
    )
    return paths


def generate_cross_model_figure(cross_model_root: str | Path) -> Path:
    root = Path(cross_model_root)
    behavior = pd.read_csv(root / "behavioral_comparison.csv")
    summary = behavior.groupby(["architecture", "task"], as_index=False).effect.mean()
    path = root / "cross_model_structure.png"
    _heatmap(summary, "architecture", "task", "effect", path, "Cross-model behavioral structure")
    return path

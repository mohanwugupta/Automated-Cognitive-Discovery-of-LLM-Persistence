"""Registered paper-figure drafts generated from analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def generate_figures(run_directory: str | Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []
    root = Path(run_directory)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    written = []

    figure, axis = plt.subplots(figsize=(11, 2.2))
    stages = ["Ontology", "Design", "Render", "LLM", "Models", "Discovery"]
    positions = np.linspace(0.08, 0.92, len(stages))
    for position, stage in zip(positions, stages):
        axis.text(
            position,
            0.5,
            stage,
            ha="center",
            va="center",
            transform=axis.transAxes,
            bbox={"boxstyle": "round,pad=.45", "facecolor": "#e8f0fe"},
        )
    for left, right in zip(positions[:-1], positions[1:]):
        axis.annotate(
            "",
            xy=(right - 0.045, 0.5),
            xytext=(left + 0.045, 0.5),
            xycoords=axis.transAxes,
            arrowprops={"arrowstyle": "->", "color": "#444444"},
        )
    axis.axis("off")
    path = output / "figure1_framework.png"
    _save(figure, path)
    written.append(path)

    marginal_path = root / "analysis/response_surface/marginal_effects.csv"
    if marginal_path.exists():
        data = pd.read_csv(marginal_path).head(18).sort_values("task_macro_effect")
        figure, axis = plt.subplots(figsize=(8, 6))
        labels = data.factor.astype(str) + " / " + data.level.astype(str)
        axis.barh(labels, data.task_macro_effect, color="#4c78a8")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Task-macro centered persistence-logit effect")
        axis.set_title("Persistence response surface")
        path = output / "figure2_response_surface.png"
        _save(figure, path)
        written.append(path)

    comparison_path = root / "analysis/model_comparison/model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        data = comparison[comparison.split == "interpolation_test"].copy()
        data = data.sort_values("macro_r2").tail(18)
        figure, axis = plt.subplots(figsize=(8, 6))
        labels = data.model.astype(str) + " / " + data.sharing.astype(str)
        axis.barh(labels, data.macro_r2, color="#f58518")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Held-out task-macro R²")
        axis.set_title("Cognitive model tournament and flexible predictors")
        path = output / "figure3_model_tournament.png"
        _save(figure, path)
        written.append(path)

        explainable = data.dropna(subset=["explainable_variance_fraction"])
        if not explainable.empty:
            figure, axis = plt.subplots(figsize=(8, 5))
            labels = explainable.model.astype(str) + " / " + explainable.sharing.astype(str)
            axis.barh(labels, explainable.explainable_variance_fraction, color="#54a24b")
            axis.axvline(1, color="black", linestyle="--", linewidth=0.8)
            axis.set_xlabel("Explainable-variance fraction")
            axis.set_title("Fraction of flexible-predictor performance explained")
            path = output / "figure5_explainable_variance.png"
            _save(figure, path)
            written.append(path)

    loto_path = root / "analysis/loto/loto.csv"
    if loto_path.exists():
        loto = pd.read_csv(loto_path)
        matrix = loto.pivot_table(index="model", columns="heldout_task", values="r2")
        figure, axis = plt.subplots(figsize=(10, 6))
        image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_title("Strict zero-shot leave-one-task-out R²")
        figure.colorbar(image, ax=axis, label="R²")
        path = output / "figure4_loto.png"
        _save(figure, path)
        written.append(path)

    residual_path = root / "analysis/residual_discovery/interactions.csv"
    if residual_path.exists():
        residuals = pd.read_csv(residual_path).head(10).sort_values("heldout_r2_gain")
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.barh(residuals.interaction, residuals.heldout_r2_gain, color="#e45756")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Held-out residual R² gain")
        axis.set_title("Discovered residual structure")
        path = output / "figure6_residual_discovery.png"
        _save(figure, path)
        written.append(path)

    predictions_path = root / "validation/predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        figure, axis = plt.subplots(figsize=(5.5, 5.5))
        axis.scatter(
            predictions.persistence_logit,
            predictions.predicted_persistence_logit,
            s=10,
            alpha=0.45,
        )
        bounds = [
            min(predictions.persistence_logit.min(), predictions.predicted_persistence_logit.min()),
            max(predictions.persistence_logit.max(), predictions.predicted_persistence_logit.max()),
        ]
        axis.plot(bounds, bounds, color="black", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Observed persistence logit")
        axis.set_ylabel("Frozen-model prediction")
        axis.set_title("Independent validation")
        path = output / "figure7_independent_validation.png"
        _save(figure, path)
        written.append(path)
    return written


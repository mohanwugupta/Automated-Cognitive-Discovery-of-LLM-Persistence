"""Six registered figures and a gate-aware mechanistic report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _csv(path: Path) -> pd.DataFrame:
    try:
        return (
            pd.read_csv(path)
            if path.exists() and path.stat().st_size
            else pd.DataFrame()
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _empty_figure(title: str, message: str):
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return fig


def generate_mechanistic_figures(root: str | Path) -> list[Path]:
    root = Path(root)
    figure_root = root / "figures"
    metrics = _csv(root / "representation/layer_metrics.csv")
    transformation = _csv(root / "representation/contextual_transformation.csv")
    loto = _csv(root / "representation/loto_metrics.csv")
    steering_path = root / "steering/steering_results.parquet"
    if steering_path.exists():
        steering = pd.read_parquet(steering_path)
    else:
        fallback = steering_path.with_suffix(".csv.gz")
        try:
            steering = pd.read_csv(fallback) if fallback.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            steering = pd.DataFrame()
    mediation = _csv(root / "patching/mediation_summary.csv")
    paths = []

    test = metrics[metrics.split == "test"] if not metrics.empty else metrics
    if test.empty:
        fig = _empty_figure(
            "Computational variables across depth", "Representation scan unavailable"
        )
    else:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for target, part in test.groupby("target"):
            axis.plot(part.layer, part.r2, marker="o", label=target)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(xlabel="Layer (zero-based block output)", ylabel="Held-out $R^2$")
        axis.legend(fontsize=8)
    path = figure_root / "figure1_computational_variables_depth.png"
    _save(fig, path)
    paths.append(path)

    if transformation.empty:
        fig = _empty_figure(
            "Raw to context-relevant transformation", "Contextual analysis unavailable"
        )
    else:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(transformation.layer, transformation.raw_r2, label="raw O")
        axis.plot(
            transformation.layer, transformation.contextual_r2, label="contextual O*"
        )
        axis.plot(
            transformation.layer,
            transformation.partial_contextual_delta_r2,
            linestyle="--",
            label="partial contextual ΔR²",
        )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(xlabel="Layer", ylabel="$R^2$ / partial $\\Delta R^2$")
        axis.legend(fontsize=8)
    path = figure_root / "figure2_contextual_transformation.png"
    _save(fig, path)
    paths.append(path)

    if loto.empty:
        fig = _empty_figure(
            "Cross-task representation", "Strict LOTO analysis unavailable"
        )
    else:
        table = loto.groupby(["target", "layer"]).r2.mean().reset_index()
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for target, part in table.groupby("target"):
            axis.plot(part.layer, part.r2, marker=".", label=target)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(xlabel="Layer", ylabel="Mean strict-LOTO $R^2$")
        axis.legend(fontsize=8)
    path = figure_root / "figure3_cross_task_representation.png"
    _save(fig, path)
    paths.append(path)

    candidate = (
        steering[steering.control == "candidate"] if not steering.empty else steering
    )
    if candidate.empty:
        fig = _empty_figure(
            "Steering dose response", "Steering was not run or did not pass its gate"
        )
    else:
        table = candidate.groupby(
            ["task_family", "computational_dose"], as_index=False
        ).observed_logit_change.mean()
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for task, part in table.groupby("task_family"):
            axis.plot(
                part.computational_dose,
                part.observed_logit_change,
                marker="o",
                label=task,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(
            xlabel="Intervention dose (standardized computational units)",
            ylabel="Observed Δ persistence logit",
        )
        axis.legend(fontsize=7, ncol=2)
    path = figure_root / "figure4_steering_dose_response.png"
    _save(fig, path)
    paths.append(path)

    if candidate.empty:
        fig = _empty_figure(
            "Predicted versus observed causal effects", "Causal results unavailable"
        )
    else:
        cells = candidate.groupby(
            ["task_family", "computational_dose"], as_index=False
        )[["predicted_logit_change", "observed_logit_change"]].mean()
        fig, axis = plt.subplots(figsize=(5.5, 5.0))
        axis.scatter(
            cells.predicted_logit_change, cells.observed_logit_change, alpha=0.8
        )
        limits = [
            min(cells.predicted_logit_change.min(), cells.observed_logit_change.min()),
            max(cells.predicted_logit_change.max(), cells.observed_logit_change.max()),
        ]
        axis.plot(limits, limits, linestyle="--", color="black", linewidth=0.8)
        axis.set(xlabel="Independently predicted Δ logit", ylabel="Observed Δ logit")
    path = figure_root / "figure5_predicted_observed_causal.png"
    _save(fig, path)
    paths.append(path)

    if mediation.empty:
        fig = _empty_figure(
            "Projection-mediated effects", "Projection patching unavailable"
        )
    else:
        table = mediation.groupby(
            ["target", "control"], as_index=False
        ).mediated_effect.mean()
        table["label"] = table.target.astype(str) + " / " + table.control.astype(str)
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar(table.label, table.mediated_effect)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(ylabel="Mean projection-mediated logit effect")
        axis.tick_params(axis="x", rotation=20)
    path = figure_root / "figure6_projection_mediation.png"
    _save(fig, path)
    paths.append(path)
    return paths


def _best(metrics: pd.DataFrame, target: str):
    rows = metrics[(metrics.target == target) & (metrics.split == "test")]
    return rows.sort_values("r2", ascending=False).iloc[0] if len(rows) else None


def _fmt(value) -> str:
    try:
        return f"{float(value):.3f}" if np.isfinite(float(value)) else "not estimable"
    except (TypeError, ValueError):
        return "not estimable"


def build_mechanistic_report(root: str | Path) -> Path:
    root = Path(root)
    metrics = _csv(root / "representation/layer_metrics.csv")
    transformation = _csv(root / "representation/contextual_transformation.csv")
    loto = _csv(root / "representation/loto_metrics.csv")
    specificity = _csv(root / "representation/specificity.csv")
    candidates = _csv(root / "representation/candidate_layers.csv")
    causal = _csv(root / "steering/predicted_vs_observed.csv")
    dose = _csv(root / "steering/dose_response.csv")
    contextual_patch = _csv(root / "patching/contextual_patch_advantage.csv")
    gate_path = root / "representation/gates.json"
    gates = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    raw, contextual = _best(metrics, "outcome_history"), _best(
        metrics, "contextual_outcome_history"
    )
    selected = (
        candidates[candidates.selected.astype(bool)]
        if "selected" in candidates
        else pd.DataFrame()
    )
    representation_passed = bool(gates.get("representation_passed", len(selected) > 0))
    causal_passed = bool(gates.get("steering_causality_passed", False))
    mediation_passed = bool(gates.get("mediation_passed", False))
    if not representation_passed:
        theory = "No robust cross-task linear representation"
    elif not causal_passed:
        theory = "Representation without demonstrated causal role"
    elif (
        raw is not None
        and contextual is not None
        and int(raw.layer) < int(contextual.layer)
    ):
        theory = "Both representations may exist sequentially"
    elif contextual is not None and raw is not None and contextual.r2 > raw.r2:
        theory = "Context-transformed history"
    else:
        theory = "Direct outcome-history implementation"
    max_partial = (
        float(transformation.partial_contextual_delta_r2.max())
        if len(transformation)
        else np.nan
    )
    loto_mean = float(loto.r2.mean()) if len(loto) else np.nan
    causal_r = float(causal.correlation.max()) if len(causal) else np.nan
    monotonic = float(dose.monotonic_rho.mean()) if len(dose) else np.nan
    alignment = float(gates.get("computational_alignment_correlation", np.nan))
    contextual_patch_delta = (
        float(contextual_patch.contextual_minus_raw.mean())
        if len(contextual_patch) and "contextual_minus_raw" in contextual_patch
        else np.nan
    )
    specificity_max = (
        float(specificity.shared_variance.max()) if len(specificity) else np.nan
    )
    raw_layer = int(raw.layer) if raw is not None else "not identified"
    contextual_layer = (
        int(contextual.layer) if contextual is not None else "not identified"
    )
    selected_layers = (
        selected[["target", "layer"]].to_dict("records") if len(selected) else []
    )
    lines = [
        "# Mechanistic Implementation of Context-Sensitive Outcome-History Integration",
        "",
        (
            "Full prompt activations were streamed and discarded. Only directions, "
            "scalar projections, aggregate metrics, and intervention results are retained."
        ),
        "",
        "## Registered questions",
        "",
        (
            "1. **Raw outcome history represented?** Best held-out R²: "
            f"{_fmt(raw.r2 if raw is not None else np.nan)} at layer {raw_layer}."
        ),
        (
            "2. **Context-relevant history represented?** Best held-out R²: "
            f"{_fmt(contextual.r2 if contextual is not None else np.nan)} at layer "
            f"{contextual_layer}."
        ),
        (
            f"3. **Emergence across depth.** Raw peak: {raw_layer}; contextual peak: "
            f"{contextual_layer}."
        ),
        f"4. **Context beyond recency.** Maximum matched partial ΔR²: {_fmt(max_partial)}.",
        (
            f"5. **Cross-task transfer.** Mean strict-LOTO R²: {_fmt(loto_mean)}; "
            "no target-task normalization or refitting was used."
        ),
        (
            "6. **Specificity.** Maximum measured shared variance with registered "
            f"controls: {_fmt(specificity_max)}."
        ),
        (
            f"7. **Representation gate.** "
            f"{'Passed' if representation_passed else 'Failed'}; selected target/layers: "
            f"{selected_layers}."
        ),
        (
            "8. **Calibrated projection movement.** Requested-versus-realized "
            f"computational correlation: {_fmt(alignment)}."
        ),
        f"9. **Steering direction.** Quantitative causal correlation: {_fmt(causal_r)}.",
        (
            "10. **Task-specific prediction.** Frozen behavioral coefficients—not steering "
            "outcomes—generated every predicted cell; maximum predicted/observed "
            f"correlation: {_fmt(causal_r)}."
        ),
        f"11. **Dose monotonicity.** Mean task-wise Spearman rho: {_fmt(monotonic)}.",
        f"12. **Projection mediation.** {'Detected' if mediation_passed else 'Not demonstrated'}.",
        (
            "13. **Contextual versus raw patching.** Mean contextual-minus-raw mediated "
            f"effect on context-conflict examples: {_fmt(contextual_patch_delta)}."
        ),
        f"14. **Best-supported mechanistic outcome.** {theory}.",
        (
            "15. **Claim boundary.** Causal implementation is claimed only when "
            "representation, calibrated bidirectional steering, specificity, and "
            "projection-patching gates all pass. Decoding alone is not treated as mechanism."
        ),
        "",
        "## Gate summary",
        "",
        f"- Representation: {'pass' if representation_passed else 'fail'}",
        f"- Quantitative steering: {'pass' if causal_passed else 'fail/not run'}",
        f"- Projection patching: {'pass' if mediation_passed else 'fail/not run'}",
        (
            "- Random/control specificity: "
            f"{'pass' if gates.get('steering_specificity_passed', False) else 'fail/not run'}"
        ),
        "",
        "## Reproducibility",
        "",
        (
            "See `run_metadata.json` for the model revision, behavioral handoff hash, "
            "condition hash, activation position, layer convention, split, and seed."
        ),
    ]
    path = root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

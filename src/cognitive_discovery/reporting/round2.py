"""Paper-oriented Round-2 report and five registered figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _csv(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def generate_round2_report(root: str | Path) -> Path:
    root = Path(root)
    comparison = _csv(root / "hierarchy/model_comparison.csv")
    loto = _csv(root / "hierarchy/loto.csv")
    bootstrap = _csv(root / "hierarchy/architecture_bootstrap.csv")
    residuals = _csv(root / "residuals/heldout_gains.csv")
    recovery = _csv(root / "audits/teacher_recovery.csv")
    metrics_path = root / "final_validation/metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    best = (
        comparison[comparison.split == "interpolation_test"]
        .sort_values(["macro_r2", "r2"], ascending=False)
        .iloc[0]
        if not comparison.empty
        else None
    )
    m4 = (
        comparison[
            (comparison.split == "interpolation_test") & (comparison.variant == "M4")
        ]
        .sort_values("macro_r2", ascending=False)
        .iloc[0]
        if not comparison.empty
        else None
    )
    confirmed = (
        residuals[residuals.discovered.astype(bool)]
        if not residuals.empty
        else residuals
    )
    equivalent = (
        bool(
            bootstrap[
                bootstrap.task_family == "__macro__"
            ].observationally_equivalent.iloc[0]
        )
        if not bootstrap.empty and (bootstrap.task_family == "__macro__").any()
        else None
    )
    lines = [
        "# Active Hierarchical Discovery of Persistence Computations",
        "",
        "## Primary result",
        "",
        (
            f"Best interpolation result: **{best.architecture} / {best.variant}** "
            f"(task-macro R²={best.macro_r2:.3f})."
            if best is not None
            else "Hierarchy fitting is pending."
        ),
        (
            f"Best ontology-conditioned architecture: **{m4.architecture}** "
            f"(task-macro R²={m4.macro_r2:.3f})."
            if m4 is not None
            else "Ontology-conditioned comparison is pending."
        ),
        "",
        "## Generalization tests",
        "",
        f"Strict zero-shot rows: {len(loto)}. Held-out target outcomes are excluded from population fits.",
        (
            (
                "Dual history and latent context remain observationally equivalent under the paired bootstrap."
                if equivalent
                else "The paired dual-history versus latent-context bootstrap does not currently support equivalence."
            )
            if equivalent is not None
            else "Paired architecture bootstrap is pending."
        ),
        "",
        "## Validity audits",
        "",
        (
            f"Matched flexible teacher checks passed: {int(recovery.passed.sum())}/{len(recovery)}."
            if not recovery.empty
            else "Matched flexible teacher recovery is pending."
        ),
        (
            f"Confirmed nested residual interactions: {len(confirmed)}."
            if not residuals.empty
            else "Nested residual discovery is pending."
        ),
        "",
        "## Final untouched validation",
        "",
        (
            f"Frozen-model R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}, "
            f"calibration slope={metrics['calibration_slope']:.3f}."
            if metrics
            else "Final coverage-random collection is pending."
        ),
        "",
        "## Interpretation boundary",
        "",
        "M4 support is evidence for a shared computational form with task-structured parameters; "
        "it is not evidence for invariant coefficients. Fractions above one are reported raw as "
        "generalization advantages and are never silently capped.",
    ]
    path = root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_round2_figures(root: str | Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []
    root = Path(root)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    written = []

    figure, axis = plt.subplots(figsize=(9, 3))
    axis.axis("off")
    axis.text(
        0.08,
        0.5,
        "Task ontology\n$z_t$",
        ha="center",
        va="center",
        bbox={"boxstyle": "round", "facecolor": "#dbeafe"},
    )
    axis.text(
        0.5,
        0.5,
        "$\\theta_t=\\mu+Bz_t+u_t$",
        ha="center",
        va="center",
        bbox={"boxstyle": "round", "facecolor": "#fef3c7"},
    )
    axis.text(
        0.9,
        0.5,
        "$f(X,H;\\theta_t)$",
        ha="center",
        va="center",
        bbox={"boxstyle": "round", "facecolor": "#dcfce7"},
    )
    axis.annotate(
        "",
        xy=(0.39, 0.5),
        xytext=(0.18, 0.5),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->"},
    )
    axis.annotate(
        "",
        xy=(0.81, 0.5),
        xytext=(0.65, 0.5),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->"},
    )
    path = output / "figure1_hierarchical_architecture.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    intervals = _csv(root / "hierarchy/task_parameter_intervals.csv")
    if not intervals.empty:
        wanted = [
            "history_action_kernel",
            "history_outcome_kernel",
            "factor_success_evidence",
            "factor_progress_evidence",
            "factor_continuation_cost",
            "factor_disengagement_value",
        ]
        data = intervals[intervals.parameter.isin(wanted)].copy()
        figure, axis = plt.subplots(figsize=(10, max(4, 0.25 * len(data))))
        labels = data.task_family.astype(str) + " / " + data.parameter.astype(str)
        positions = np.arange(len(data))
        axis.errorbar(
            data.estimate_mean,
            positions,
            xerr=[
                data.estimate_mean - data.interval_low,
                data.interval_high - data.estimate_mean,
            ],
            fmt="o",
        )
        axis.set_yticks(positions, labels)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title("Shared form, partially pooled task parameters")
        path = output / "figure2_task_parameters.png"
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(path)

    comparison = _csv(root / "hierarchy/model_comparison.csv")
    if not comparison.empty:
        data = (
            comparison[comparison.split == "interpolation_test"]
            .sort_values("macro_r2")
            .tail(28)
        )
        figure, axis = plt.subplots(figsize=(9, 8))
        axis.barh(data.architecture + " / " + data.variant, data.macro_r2)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Task-macro R²")
        path = output / "figure3_hierarchy_comparison.png"
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(path)

    few = _csv(root / "hierarchy/few_shot.csv")
    if not few.empty:
        summary = few.groupby(
            ["architecture", "requested_semantic_conditions"], as_index=False
        ).r2.mean()
        figure, axis = plt.subplots(figsize=(7, 5))
        for architecture, part in summary.groupby("architecture"):
            axis.plot(
                part.requested_semantic_conditions,
                part.r2,
                marker="o",
                label=architecture,
            )
        axis.set_xlabel("Target semantic conditions")
        axis.set_ylabel("Mean held-out-task R²")
        axis.legend()
        path = output / "figure4_few_shot.png"
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(path)

    efficiency = _csv(root / "active_sampling/budget_curves.csv")
    if not efficiency.empty:
        figure, axis = plt.subplots(figsize=(7, 5))
        for policy, part in efficiency.groupby("policy"):
            axis.plot(part.budget, part.macro_r2, marker="o", label=policy)
        axis.set_xlabel("Round-2 semantic-condition budget")
        axis.set_ylabel("Final-test macro R²")
        axis.legend()
        path = output / "figure5_active_efficiency.png"
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(path)
    return written

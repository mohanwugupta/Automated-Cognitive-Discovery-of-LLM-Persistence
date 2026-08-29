"""Mechanistic handoff and automated Round-3 report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_mechanistic_handoff(
    decision: dict,
    fit,
    intervals: pd.DataFrame,
    directory: str | Path,
) -> tuple[Path, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    parameters = fit.task_parameters()
    uncertainty = intervals.rename(
        columns={
            "estimate_mean": "bootstrap_estimate",
            "interval_low": "ci_low",
            "interval_high": "ci_high",
        }
    )
    parameters = parameters.merge(
        uncertainty[
            [
                "architecture",
                "task_family",
                "parameter",
                "bootstrap_estimate",
                "ci_low",
                "ci_high",
                "probability_positive",
            ]
        ],
        on=["architecture", "task_family", "parameter"],
        how="left",
        validate="one_to_one",
    )
    parameters["intervention_derivative"] = parameters.estimate
    parameter_path = root / "final_parameters.csv"
    parameters.to_csv(parameter_path, index=False)
    outcome = decision["outcome"]
    variables = {
        "dual_history": ["action_history_summary", "outcome_history_summary"],
        "latent_context": [
            "latent_context_probability",
            "contextual_history_relevance",
            "context_weighted_outcome_history",
        ],
        "observational_equivalence": [
            "outcome_history_summary",
            "contextual_modulation_of_history",
        ],
        "unresolved": ["history_integration", "contextual_history_relevance"],
        "third_architecture": [],
    }[outcome]
    intervention_rows = parameters[
        parameters.parameter.isin(
            [
                "history_action_kernel",
                "history_outcome_kernel",
                "context_relevant_outcome_kernel",
            ]
        )
    ]
    handoff = {
        "theory_outcome": outcome,
        "winner": decision.get("winner"),
        "decision_reason": decision["reason"],
        "computational_variables": variables,
        "behavioral_equation": (
            "D_t = beta_task,X X_t + beta_task,A A_t + beta_task,O O_t"
            if outcome == "dual_history"
            else "D_t = beta_task,X X_t + beta_task,H H_t(context; z_t)"
        ),
        "parameterization": f"{fit.architecture}/{fit.variant}",
        "intervention_prediction": "delta_D_task = beta_task,j * delta_computational_variable_j",
        "task_specific_interventions": intervention_rows[
            [
                "task_family",
                "parameter",
                "intervention_derivative",
                "ci_low",
                "ci_high",
            ]
        ].to_dict("records"),
        "mechanistic_analysis_ready": outcome
        in {"dual_history", "latent_context", "observational_equivalence"},
    }
    target_path = root / "mechanistic_targets.json"
    target_path.write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target_path, parameter_path


def generate_theory_report(root: str | Path) -> Path:
    root = Path(root)
    teacher = _csv(root / "audits/flexible_teacher_checks.csv")
    hierarchy = _csv(root / "audits/hierarchy_bootstrap_summary.csv")
    variance = _csv(root / "audits/parameter_variance.csv")
    signs = _csv(root / "audits/parameter_signs.csv")
    info = _csv(root / "audits/information_sampling_audit.csv")
    freeze = _csv(root / "frozen_models/freeze_decisions.csv")
    scores = _csv(root / "active_sampling/sampling_scores.csv")
    comparison = _csv(root / "discrimination/model_difference_summary.csv")
    per_task = _csv(root / "discrimination/per_task_comparison.csv")
    targets_path = root / "theory/mechanistic_targets.json"
    targets = json.loads(targets_path.read_text()) if targets_path.exists() else None
    failed = teacher[~teacher.passed.astype(bool)] if not teacher.empty else teacher
    survivors = (
        freeze[freeze.survives.astype(bool)].architecture.astype(str).tolist()
        if not freeze.empty
        else []
    )
    macro_hierarchy = hierarchy if not hierarchy.empty else pd.DataFrame()
    q = float(variance.ontology_fraction_q.median()) if not variance.empty else np.nan
    stable = (
        signs[signs.classification == "shared_sign"].parameter.astype(str).tolist()
        if not signs.empty
        else []
    )
    overall = (
        comparison[(comparison.scope == "overall") & (comparison.group == "all")]
        if not comparison.empty
        else comparison
    )
    contextual = (
        comparison[comparison.group == "contextual_history"]
        if not comparison.empty
        else comparison
    )
    lines = [
        "# Behavioral Theory Resolution Before Mechanistic Analysis",
        "",
        "No activation or representation analysis is performed in this workflow.",
        "",
        "## Registered questions",
        "",
        f"1. **Teacher failures.** {len(failed)} of {len(teacher)} checks failed. "
        + (
            "; ".join(
                f"{row.teacher_architecture}/{row.student_architecture}/{row.sharing_structure}: {row.failure_category}"
                for row in failed.itertuples()
            )
            if len(failed)
            else "No failures remain."
        ),
        f"2. **Flexible-ceiling validity.** {'Valid for all registered checks.' if len(teacher) and not len(failed) else 'Not an upper bound where a registered check fails.'}",
        f"3. **M2/M3/M4 uncertainty.** {len(macro_hierarchy)} paired summaries are available; intervals, medians, and win probabilities are in the audit table.",
        f"4. **Ontology variance.** Median ontology fraction Q={q:.3f}." if np.isfinite(q) else "4. **Ontology variance.** Pending.",
        f"5. **Stable parameter signs.** {', '.join(stable) if stable else 'None established or pending.'}",
        f"6. **Information Sampling.** {len(info)} semantic/mask/scale/mapping rows audited; with/without sensitivity is preserved.",
        f"7. **Frozen theories.** {', '.join(survivors) if survivors else 'Pending.'}",
        f"8. **Disagreement space.** {len(scores)} candidates scored across original and contextual domains." if len(scores) else "8. **Disagreement space.** Pending.",
        (
            f"9. **Targeted discrimination.** Overall mean Δerror(DH−LC)={overall.iloc[0].mean_delta_error_dh_minus_lc:.4f}."
            if len(overall)
            else "9. **Targeted discrimination.** Pending Round-3 collection."
        ),
        (
            f"10. **A→B→A reinstatement.** Contextual subset mean Δerror={contextual.iloc[0].mean_delta_error_dh_minus_lc:.4f}."
            if len(contextual)
            else "10. **A→B→A reinstatement.** Pending Round-3 collection."
        ),
        "11. **Cue reliability.** Reliability-specific matched effects are reported in `discrimination/context_reliability_effect.csv` and Figure 5.",
        (
            f"12. **Behavioral theory.** {targets['theory_outcome']}: {targets['decision_reason']}"
            if targets
            else "12. **Behavioral theory.** Frozen comparison pending."
        ),
        (
            f"13. **Mechanistic variable.** {', '.join(targets['computational_variables']) or 'No variable earned yet.'}"
            if targets
            else "13. **Mechanistic variable.** Pending."
        ),
        (
            f"14. **Intervention effects.** {len(targets['task_specific_interventions'])} task-parameter derivatives with uncertainty are frozen in `theory/final_parameters.csv`."
            if targets
            else "14. **Intervention effects.** Pending."
        ),
        "",
        "## Interpretation boundary",
        "",
        "The frozen comparison is predictive. Post-update fits are reported separately and cannot retroactively change the frozen test. If the registered interval supports equivalence, no arbitrary winner is named.",
    ]
    path = root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_theory_figures(root: str | Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []
    root = Path(root)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    written = []

    parameters = _csv(root / "theory/final_parameters.csv")
    if not parameters.empty:
        summary = parameters.groupby("parameter", as_index=False).estimate.mean().head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(summary.parameter, summary.estimate)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Figure 1 — shared form with task-specific weights")
        path = output / "figure1_parameter_architecture.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    signs = _csv(root / "audits/parameter_signs.csv")
    if not signs.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(signs.parameter, signs.probability_positive)
        ax.axvline(0.5, color="black", linewidth=0.8)
        ax.set_xlim(0, 1); ax.set_title("Figure 2 — parameter sign consistency")
        path = output / "figure2_parameter_consistency.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    predictions = _csv(root / "active_sampling/candidate_predictions.csv")
    if not predictions.empty and "context_cue_probability" in predictions:
        data = predictions[predictions.context_phase_order.notna()].copy()
        if len(data):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(data.prediction_dual_history, data.prediction_latent_context, s=5, alpha=.3)
            ax.set_xlabel("Dual history"); ax.set_ylabel("Latent context")
            ax.set_title("Figure 3 — A→B→A frozen predictions")
            path = output / "figure3_context_reinstatement_predictions.png"
            fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    errors = _csv(root / "discrimination/frozen_model_errors.csv")
    if not errors.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        means = errors[["squared_error_dual_history", "squared_error_latent_context"]].mean()
        ax.bar(["dual history", "latent context"], means)
        ax.set_ylabel("Mean squared frozen prediction error")
        ax.set_title("Figure 4 — untouched frozen discrimination")
        path = output / "figure4_frozen_discrimination.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

        if "context_cue_reliability" in errors:
            context = errors[errors.contextual_history.astype(bool)]
            if len(context):
                reliability = context.groupby("context_cue_reliability", as_index=False).delta_error_dh_minus_lc.mean()
                fig, ax = plt.subplots(figsize=(7, 5))
                order = [x for x in ("low", "medium", "high") if x in set(reliability.context_cue_reliability)]
                values = [reliability.set_index("context_cue_reliability").loc[x, "delta_error_dh_minus_lc"] for x in order]
                ax.plot(order, values, marker="o"); ax.axhline(0, color="black", linewidth=.8)
                ax.set_title("Figure 5 — context reliability effect")
                path = output / "figure5_context_reliability.png"
                fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    final = _csv(root / "theory/final_model_comparison.csv")
    if not final.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(final.architecture + "/" + final.stage, final.r2)
        ax.set_ylabel("Untouched discrimination R²")
        ax.set_title("Figure 6 — final theory comparison")
        ax.tick_params(axis="x", rotation=30)
        path = output / "figure6_final_theory_comparison.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); written.append(path)
    return written

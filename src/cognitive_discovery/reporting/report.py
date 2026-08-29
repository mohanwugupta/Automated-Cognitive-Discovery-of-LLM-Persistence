"""Automated scientific report covering the PRD's fifteen registered questions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read(path, default=None):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else default


def _best(frame, split="interpolation_test", *, exclude=()):
    if frame is None or frame.empty:
        return None
    part = frame[frame.split == split] if "split" in frame else frame
    if exclude:
        part = part[~part.model.isin(exclude)]
    return part.sort_values("macro_r2", ascending=False).iloc[0] if not part.empty else None


def generate_report(run_directory: str | Path, metadata: dict | None = None) -> Path:
    root = Path(run_directory)
    behavior_path = root / "behavior/standardized.parquet"
    if behavior_path.exists():
        behavior = pd.read_parquet(behavior_path)
    else:
        behavior = pd.read_csv(root / "behavior/standardized.csv.gz")
    marginals = _read(root / "analysis/response_surface/marginal_effects.csv", pd.DataFrame())
    interactions = _read(root / "analysis/response_surface/interactions.csv", pd.DataFrame())
    comparison = _read(root / "analysis/model_comparison/model_comparison.csv", pd.DataFrame())
    loto = _read(root / "analysis/loto/loto.csv", pd.DataFrame())
    residuals = _read(root / "analysis/residual_discovery/interactions.csv", pd.DataFrame())
    validation_metrics_path = root / "validation/metrics.json"
    recovery = _read(root / "models/flexible/synthetic_recovery.csv", pd.DataFrame())
    ceiling_validated = not recovery.empty and recovery.passed.astype(bool).all()
    validation = (
        json.loads(validation_metrics_path.read_text(encoding="utf-8"))
        if validation_metrics_path.exists()
        else None
    )
    best = _best(comparison, exclude=("linear_interactions", "gam", "mlp", "gru"))
    best_loto = (
        loto.groupby("model", as_index=False).r2.mean().sort_values("r2", ascending=False).iloc[0]
        if not loto.empty
        else None
    )
    best_residual = residuals.iloc[0] if not residuals.empty else None
    top_effect = marginals.iloc[0] if not marginals.empty else None
    top_interaction = interactions.iloc[0] if not interactions.empty else None
    latent = (
        comparison[
            (comparison.model == "latent_motivation")
            & (comparison.split == "interpolation_test")
        ]
        if not comparison.empty
        else pd.DataFrame()
    )
    dual = (
        comparison[(comparison.model == "dual_history") & (comparison.split == "interpolation_test")]
        if not comparison.empty
        else pd.DataFrame()
    )
    generic = (
        comparison[
            (comparison.model == "generic_sequential_choice")
            & (comparison.split == "interpolation_test")
        ]
        if not comparison.empty
        else pd.DataFrame()
    )
    semantic_roots = behavior.paired_condition_id.astype(str).str.replace(
        r":s\d+$", "", regex=True
    )
    semantic_count = int(semantic_roots.nunique())
    coverage = behavior.assign(_semantic_root=semantic_roots).groupby(
        "task_family"
    )._semantic_root.nunique()
    lines = [
        "# Automated Cognitive Discovery Report",
        "",
        f"Run contains **{semantic_count:,} semantic conditions**, "
        f"**{len(behavior):,} counterbalanced observations**, and "
        f"**{behavior.task_family.nunique()} task families**.",
        "",
        "## Registered questions",
        "",
        f"1. **Sample size.** {semantic_count:,} semantic conditions were sampled.",
        f"2. **Coverage.** Task counts range from {coverage.min()} "
        f"to {coverage.max()} semantic conditions.",
        (
            f"3. **Largest marginal dimension.** "
            f"{top_effect.factor + ' / ' + str(top_effect.level) if top_effect is not None else 'Not yet estimated'}."
        ),
        (
            f"4. **Strongest interaction.** "
            f"{top_interaction.interaction if top_interaction is not None else 'Not yet estimated'}."
        ),
        f"5. **Best held-out cognitive theory.** {best.model if best is not None else 'Not yet fitted'}.",
        (
            f"6. **Distance to flexible ceiling.** Explainable fraction = "
            f"{float(best.explainable_variance_fraction):.3f}."
            if best is not None and ceiling_validated
            else "6. **Distance to flexible ceiling.** Flexible-model recovery has not yet cleared the registered gate."
        ),
        f"7. **Best unseen-task theory.** {best_loto.model if best_loto is not None else 'Not yet evaluated'}.",
        (
            f"8. **Latent motivational state.** Best interpolation macro R² = "
            f"{latent.macro_r2.max():.3f}."
            if not latent.empty
            else "8. **Latent motivational state.** Not yet evaluated."
        ),
        (
            f"9. **Recent history contribution.** Dual-history macro R² = "
            f"{dual.macro_r2.max():.3f}."
            if not dual.empty
            else "9. **Recent history contribution.** Not yet evaluated."
        ),
        (
            "10. **Action versus outcome history.** Compare the separately registered "
            "choice-perseveration, outcome-history, and dual-history rows in the tournament artifact."
        ),
        (
            f"11. **Generic sequential choice.** Macro R² = {generic.macro_r2.max():.3f}."
            if not generic.empty
            else "11. **Generic sequential choice.** Not yet evaluated."
        ),
        (
            f"12. **Systematic residual structure.** "
            f"{best_residual.interaction if best_residual is not None else 'None evaluated'}."
        ),
        (
            f"13. **Cross-family residual generality.** {int(best_residual.families)} families."
            if best_residual is not None
            else "13. **Cross-family residual generality.** Not yet evaluated."
        ),
        (
            f"14. **Independent validation.** Macro R² = {validation['macro_r2']:.3f}."
            if validation
            else "14. **Independent validation.** Pending independent collection."
        ),
        (
            f"15. **Mechanistic follow-up candidate.** "
            f"{best.model if best is not None else 'No computation has yet earned follow-up'}."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "This report concerns behavior and computational prediction only. No activation, probe, "
            "steering, patching, or localization analysis is part of this run."
        ),
    ]
    if metadata:
        lines.extend(["", "## Provenance", "", "```json", json.dumps(metadata, indent=2, sort_keys=True), "```"])
    path = root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

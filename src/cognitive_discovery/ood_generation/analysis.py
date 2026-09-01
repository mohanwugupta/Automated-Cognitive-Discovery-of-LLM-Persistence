"""Censoring-aware analyses and compact figures for free-generation OOD tests."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cox_alpha_model(
    frame: pd.DataFrame,
    *,
    alpha_column: str = "alpha",
    duration_column: str = "survival_time",
    event_column: str = "eos_emitted",
    strata_column: str | None = "prompt_id",
    confidence: float = 0.95,
) -> dict:
    """Fit a one-covariate Cox model with Breslow ties and optional prompt strata."""

    required = {alpha_column, duration_column, event_column}
    if strata_column:
        required.add(strata_column)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"survival columns are absent: {sorted(missing)}")
    data = frame.dropna(subset=list(required)).copy()
    data[event_column] = data[event_column].astype(bool)
    if len(data) < 3 or not data[event_column].any():
        raise ValueError("Cox model requires at least one observed EOS event")
    groups = (
        data.groupby(strata_column, sort=False) if strata_column else [("all", data)]
    )

    def score_information(beta):
        score = 0.0
        information = 0.0
        log_likelihood = 0.0
        for _, part in groups:
            x = part[alpha_column].to_numpy(dtype=float)
            duration = part[duration_column].to_numpy(dtype=float)
            event = part[event_column].to_numpy(dtype=bool)
            for event_time in np.unique(duration[event]):
                event_mask = event & (duration == event_time)
                risk_mask = duration >= event_time
                d = int(event_mask.sum())
                risk_x = x[risk_mask]
                linear = beta * risk_x
                offset = float(np.max(linear))
                weights = np.exp(linear - offset)
                denominator = float(weights.sum())
                mean = float(np.sum(weights * risk_x) / denominator)
                variance = float(
                    np.sum(weights * np.square(risk_x - mean)) / denominator
                )
                score += float(x[event_mask].sum()) - d * mean
                information += d * variance
                log_likelihood += beta * float(x[event_mask].sum()) - d * (
                    offset + np.log(denominator)
                )
        return score, information, log_likelihood

    beta = 0.0
    converged = False
    for iteration in range(100):
        score, information, _ = score_information(beta)
        if information <= 1e-12:
            raise RuntimeError("Cox information matrix is singular")
        step = float(np.clip(score / information, -2.0, 2.0))
        beta += step
        if abs(step) < 1e-9:
            converged = True
            break
    score, information, log_likelihood = score_information(beta)
    standard_error = float(np.sqrt(1.0 / information))
    z = NormalDist().inv_cdf(0.5 + float(confidence) / 2)
    return {
        "coefficient": float(beta),
        "standard_error": standard_error,
        "ci_lower": float(beta - z * standard_error),
        "ci_upper": float(beta + z * standard_error),
        "hazard_ratio": float(np.exp(beta)),
        "events": int(data[event_column].sum()),
        "censored": int((~data[event_column]).sum()),
        "observations": int(len(data)),
        "log_likelihood": float(log_likelihood),
        "score_at_solution": float(score),
        "converged": converged,
        "ties": "breslow",
        "stratified_by": strata_column,
    }


def kaplan_meier(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha, part in frame.groupby("alpha", sort=True):
        durations = part.survival_time.to_numpy(dtype=int)
        events = part.eos_emitted.to_numpy(dtype=bool)
        survival = 1.0
        rows.append(
            {
                "alpha": float(alpha),
                "time": 0,
                "survival": 1.0,
                "at_risk": len(part),
                "events": 0,
            }
        )
        for time in sorted(np.unique(durations)):
            at_risk = int((durations >= time).sum())
            observed = int(((durations == time) & events).sum())
            if at_risk and observed:
                survival *= 1.0 - observed / at_risk
            rows.append(
                {
                    "alpha": float(alpha),
                    "time": int(time),
                    "survival": float(survival),
                    "at_risk": at_risk,
                    "events": observed,
                }
            )
    return pd.DataFrame(rows)


def _rmst(part: pd.DataFrame, tau: int) -> float:
    durations = part.survival_time.to_numpy(dtype=int)
    events = part.eos_emitted.to_numpy(dtype=bool)
    survival = 1.0
    area = 0.0
    for time in range(1, int(tau) + 1):
        area += survival
        at_risk = int((durations >= time).sum())
        observed = int(((durations == time) & events).sum())
        if at_risk:
            survival *= 1.0 - observed / at_risk
    return float(area)


def dose_response(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tau = int(frame.survival_time.max())
    rows = []
    for alpha, part in frame.groupby("alpha", sort=True):
        observed = part[part.eos_emitted.astype(bool)].survival_time
        rows.append(
            {
                "alpha": float(alpha),
                "runs": int(len(part)),
                "eos_fraction": float(part.eos_emitted.astype(bool).mean()),
                "median_observed_tokens": float(part.generated_tokens.median()),
                "median_eos_time": (
                    float(observed.median()) if len(observed) else np.nan
                ),
                "restricted_mean_survival": _rmst(part, tau),
                "severe_degeneration_fraction": float(
                    part.severe_degeneration.astype(bool).mean()
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("alpha")
    correlation = spearmanr(table.alpha, table.restricted_mean_survival).statistic
    changes = np.diff(table.restricted_mean_survival.to_numpy())
    summary = {
        "spearman_alpha_rmst": float(correlation),
        "nondecreasing_adjacent_doses": int((changes >= 0).sum()),
        "adjacent_comparisons": int(len(changes)),
        "endpoint_rmst_change": float(
            table.restricted_mean_survival.iloc[-1]
            - table.restricted_mean_survival.iloc[0]
        ),
        "monotonic": bool(
            correlation >= 0.8
            and table.restricted_mean_survival.iloc[-1]
            > table.restricted_mean_survival.iloc[0]
        ),
        "rmst_horizon": tau,
    }
    return table, summary


def prompt_effects(frame: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    rows = []
    for prompt_id, part in frame.groupby("prompt_id", sort=True):
        try:
            result = cox_alpha_model(
                part,
                strata_column=None,
                confidence=confidence,
            )
            rows.append({"prompt_id": prompt_id, **result})
        except (ValueError, RuntimeError):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "coefficient": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "events": int(part.eos_emitted.astype(bool).sum()),
                    "observations": int(len(part)),
                }
            )
    return pd.DataFrame(rows)


def immediate_random_specificity(frame: pd.DataFrame) -> dict:
    """Compare the true E EOS-logit slope with preregistered random subspaces."""

    slopes = []
    for direction_id, part in frame.groupby("direction_id"):
        x = part.alpha.to_numpy(dtype=float)
        y = part.eos_logit.to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if np.unique(x).size > 1 else np.nan
        slopes.append(
            {
                "direction_id": direction_id,
                "control_role": str(part.control_role.iloc[0]),
                "eos_logit_slope": slope,
            }
        )
    table = pd.DataFrame(slopes)
    true_rows = table[table.control_role == "frozen_E"]
    random_rows = table[table.control_role == "random_subspace"]
    if len(true_rows) != 1 or not len(random_rows):
        raise ValueError(
            "random specificity requires one E direction and random controls"
        )
    true_slope = float(true_rows.eos_logit_slope.iloc[0])
    random_p = float(
        (1 + int((random_rows.eos_logit_slope <= true_slope).sum()))
        / (len(random_rows) + 1)
    )
    return {
        "true_eos_logit_slope": true_slope,
        "random_directions": int(len(random_rows)),
        "random_p": random_p,
        "passed": bool(true_slope < 0 and random_p < 0.05),
        "slopes": table,
    }


def quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "repeated_4gram_fraction",
        "repeated_8gram_fraction",
        "repeated_sentence_fraction",
        "lexical_diversity",
        "punctuation_fraction",
        "severe_degeneration",
    ]
    return (
        frame.groupby(
            ["control_role", "prompt_family", "regime", "alpha"], as_index=False
        )[columns]
        .mean(numeric_only=True)
        .sort_values(["control_role", "prompt_family", "regime", "alpha"])
    )


def make_figures(
    root,
    primary: pd.DataFrame,
    token_events: pd.DataFrame,
    survival: pd.DataFrame,
    doses: pd.DataFrame,
    prompts: pd.DataFrame,
    random_slopes: pd.DataFrame,
    quality: pd.DataFrame,
    pulse: pd.DataFrame,
) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    figure_root = Path(root) / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for alpha, part in survival.groupby("alpha"):
        ax.step(part.time, part.survival, where="post", label=f"α={alpha:g}")
    ax.set(
        xlabel="Generation decision step",
        ylabel="P(not yet terminated)",
        title="OOD generation survival",
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    paths.append(figure_root / "figure1_survival.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    hazard = token_events.groupby(["alpha", "token_bin"], as_index=False).p_eos.mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for alpha, part in hazard.groupby("alpha"):
        ax.plot(part.token_bin, part.p_eos, label=f"α={alpha:g}")
    ax.set(
        xlabel="Generation-step bin",
        ylabel="Mean P(EOS)",
        title="EOS hazard by frozen E dose",
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    paths.append(figure_root / "figure2_eos_hazard.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(doses.alpha, doses.restricted_mean_survival, marker="o")
    ax.set(
        xlabel="Frozen E dose (α)",
        ylabel="Restricted mean duration",
        title="Generation dose response",
    )
    fig.tight_layout()
    paths.append(figure_root / "figure3_dose_response.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    shown = prompts.dropna(subset=["coefficient"])
    ax.errorbar(
        range(len(shown)),
        shown.coefficient,
        yerr=np.vstack(
            (shown.coefficient - shown.ci_lower, shown.ci_upper - shown.coefficient)
        ),
        fmt="o",
        capsize=3,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(shown)), shown.prompt_id, rotation=45, ha="right")
    ax.set(ylabel="Cox α coefficient", title="Prompt-wise E effects")
    fig.tight_layout()
    paths.append(figure_root / "figure4_prompt_effects.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    random = random_slopes[random_slopes.control_role == "random_subspace"]
    true = random_slopes[random_slopes.control_role == "frozen_E"].eos_logit_slope.iloc[
        0
    ]
    ax.hist(random.eos_logit_slope, bins=20, alpha=0.8)
    ax.axvline(true, color="red", linewidth=2, label="Frozen E")
    ax.set(
        xlabel="Immediate EOS-logit slope",
        ylabel="Random directions",
        title="Matched random-subspace null",
    )
    ax.legend()
    fig.tight_layout()
    paths.append(figure_root / "figure5_random_null.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(
        primary.generated_tokens,
        primary.repeated_4gram_fraction,
        c=primary.alpha,
        cmap="coolwarm",
        alpha=0.5,
    )
    ax.set(
        xlabel="Generated tokens",
        ylabel="Repeated 4-gram fraction",
        title="Quality versus length",
    )
    fig.tight_layout()
    paths.append(figure_root / "figure6_quality_length.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    comparison = [primary.assign(intervention="continuous")]
    if len(pulse):
        comparison.append(pulse.assign(intervention="pulse"))
    shown = (
        pd.concat(comparison)
        .groupby(["intervention", "alpha"], as_index=False)
        .generated_tokens.median()
    )
    for intervention, part in shown.groupby("intervention"):
        ax.plot(part.alpha, part.generated_tokens, marker="o", label=intervention)
    ax.set(
        xlabel="Dose (α)",
        ylabel="Median generated tokens",
        title="Continuous versus pulse intervention",
    )
    ax.legend()
    fig.tight_layout()
    paths.append(figure_root / "figure7_pulse_vs_continuous.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)
    return paths

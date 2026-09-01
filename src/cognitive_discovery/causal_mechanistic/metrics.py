"""Metrics whose target is a frozen computational counterfactual."""

from __future__ import annotations

import numpy as np


def _aligned(predicted_change, observed_change, baseline_change=None):
    predicted = np.asarray(predicted_change, dtype=float).reshape(-1)
    observed = np.asarray(observed_change, dtype=float).reshape(-1)
    baseline = (
        np.zeros_like(predicted)
        if baseline_change is None
        else np.asarray(baseline_change, dtype=float).reshape(-1)
    )
    if predicted.shape != observed.shape or predicted.shape != baseline.shape:
        raise ValueError("counterfactual effects and baseline changes must align")
    if not len(predicted):
        raise ValueError("counterfactual metrics require non-empty arrays")
    if not (
        np.isfinite(predicted).all()
        and np.isfinite(observed).all()
        and np.isfinite(baseline).all()
    ):
        raise ValueError("counterfactual metrics require finite arrays")
    return predicted, observed, baseline


def global_counterfactual_recovery(
    predicted_change, observed_change, *, baseline_change=None
) -> float:
    """Aggregate R2-style recovery without unstable per-example division."""

    predicted, observed, baseline = _aligned(
        predicted_change, observed_change, baseline_change
    )
    numerator = float(np.sum(np.square(observed - predicted)))
    denominator = float(np.sum(np.square(baseline - predicted)))
    if denominator <= np.finfo(float).eps:
        return 1.0 if numerator <= np.finfo(float).eps else float("nan")
    return float(1.0 - numerator / denominator)


def thresholded_counterfactual_diagnostics(
    predicted_change, observed_change, *, threshold: float = 0.10
) -> dict:
    """Descriptive per-example CFR after a preregistered effect-size filter."""

    predicted, observed, _ = _aligned(predicted_change, observed_change)
    threshold = float(threshold)
    if threshold < 0:
        raise ValueError("counterfactual threshold must be non-negative")
    retained = np.abs(predicted) >= threshold
    values = counterfactual_recovery(observed[retained], predicted[retained])
    return {
        "thresholded_cfr_threshold": threshold,
        "thresholded_cfr_retained": int(retained.sum()),
        "thresholded_cfr_fraction_retained": float(retained.mean()),
        "thresholded_cfr_median": (
            float(np.median(values)) if len(values) else float("nan")
        ),
        "thresholded_cfr_q25": (
            float(np.quantile(values, 0.25)) if len(values) else float("nan")
        ),
        "thresholded_cfr_q75": (
            float(np.quantile(values, 0.75)) if len(values) else float("nan")
        ),
    }


def counterfactual_recovery(observed_change, predicted_change, *, epsilon=1e-8):
    """Per-example normalized counterfactual recovery (CFR).

    With effects expressed relative to the base LLM state, a no-op scores zero,
    an exact counterfactual scores one, and a reversed effect scores below zero.
    """

    observed = np.asarray(observed_change, dtype=float)
    predicted = np.asarray(predicted_change, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted counterfactual effects must align")
    denominator = np.square(predicted) + float(epsilon)
    return 1.0 - np.square(observed - predicted) / denominator


def counterfactual_metrics(
    predicted_change,
    observed_change,
    *,
    epsilon=1e-8,
    baseline_change=None,
    threshold=0.10,
):
    predicted, observed, baseline = _aligned(
        predicted_change, observed_change, baseline_change
    )
    design = np.column_stack((np.ones(len(predicted)), predicted))
    intercept, slope = np.linalg.lstsq(design, observed, rcond=None)[0]
    correlation = (
        float(np.corrcoef(predicted, observed)[0, 1])
        if len(predicted) > 1 and np.std(predicted) > 1e-12 and np.std(observed) > 1e-12
        else float("nan")
    )
    nonzero = np.abs(predicted) > 1e-10
    recovered_fraction = (
        float(np.median(observed[nonzero] / predicted[nonzero]))
        if nonzero.any()
        else float("nan")
    )
    return {
        "global_cfr": global_counterfactual_recovery(
            predicted, observed, baseline_change=baseline
        ),
        "correlation": correlation,
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": float(np.sqrt(np.mean(np.square(observed - predicted)))),
        "sign_accuracy": float(np.mean(np.sign(observed) == np.sign(predicted))),
        "mean_cfr": float(
            np.mean(counterfactual_recovery(observed, predicted, epsilon=epsilon))
        ),
        "median_cfr": float(
            np.median(counterfactual_recovery(observed, predicted, epsilon=epsilon))
        ),
        "fraction_behavioral_effect_recovered": recovered_fraction,
        "examples": int(len(predicted)),
        **thresholded_counterfactual_diagnostics(
            predicted, observed, threshold=threshold
        ),
    }


def counterfactual_specificity(target_cfr: float, control_cfrs) -> float:
    controls = np.asarray(control_cfrs, dtype=float)
    finite = controls[np.isfinite(controls)]
    return float(target_cfr - finite.max()) if len(finite) else float("nan")

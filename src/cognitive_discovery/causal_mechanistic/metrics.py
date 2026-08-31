"""Metrics whose target is a frozen computational counterfactual."""

from __future__ import annotations

import numpy as np


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


def counterfactual_metrics(predicted_change, observed_change, *, epsilon=1e-8):
    predicted = np.asarray(predicted_change, dtype=float).reshape(-1)
    observed = np.asarray(observed_change, dtype=float).reshape(-1)
    if predicted.shape != observed.shape or not len(predicted):
        raise ValueError("counterfactual metrics require aligned non-empty arrays")
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
    }


def counterfactual_specificity(target_cfr: float, control_cfrs) -> float:
    controls = np.asarray(control_cfrs, dtype=float)
    finite = controls[np.isfinite(controls)]
    return float(target_cfr - finite.max()) if len(finite) else float("nan")

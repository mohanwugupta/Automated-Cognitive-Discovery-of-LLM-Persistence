"""Versioned canonical metrics used by compact regression replay."""

from __future__ import annotations

import numpy as np


def global_cfr_v1(target_effect, observed_effect, *, baseline_effect=None) -> float:
    """Aggregate counterfactual recovery with a zero-effect default baseline."""

    target = np.asarray(target_effect, dtype=float).reshape(-1)
    observed = np.asarray(observed_effect, dtype=float).reshape(-1)
    baseline = (
        np.zeros_like(target)
        if baseline_effect is None
        else np.asarray(baseline_effect, dtype=float).reshape(-1)
    )
    if target.shape != observed.shape or target.shape != baseline.shape:
        raise ValueError("target, observed, and baseline effects must align")
    if not len(target):
        raise ValueError("global_cfr_v1 requires non-empty arrays")
    if not (np.isfinite(target).all() and np.isfinite(observed).all() and np.isfinite(baseline).all()):
        raise ValueError("global_cfr_v1 requires finite arrays")
    numerator = float(np.sum(np.square(observed - target)))
    denominator = float(np.sum(np.square(baseline - target)))
    if denominator <= np.finfo(float).eps:
        return 1.0 if numerator <= np.finfo(float).eps else float("nan")
    return float(1.0 - numerator / denominator)

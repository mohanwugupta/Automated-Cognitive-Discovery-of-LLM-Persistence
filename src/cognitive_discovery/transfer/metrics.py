"""Frozen transfer estimand and cluster-bootstrap inference."""

from __future__ import annotations

import numpy as np

from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics
from cognitive_discovery.reproducibility.metrics import global_cfr_v1


def _cluster_bootstrap(target, observed, groups, *, samples, seed):
    target, observed, groups = np.asarray(target, float), np.asarray(observed, float), np.asarray(groups, str)
    unique = np.unique(groups)
    if not len(unique):
        raise ValueError("bootstrap requires at least one independent group")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(samples)):
        draw = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in draw])
        values.append(global_cfr_v1(target[indices], observed[indices]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize_transfer(
    frame, *, model, theory, source_task, target_task, layer, rank,
    response_mapping="pooled", bootstrap_samples=2000, seed=27001,
    random_cfrs=None, n_train=None, n_selection=0, controller_hash=None,
    split_hash=None,
):
    required = {"predicted_counterfactual_effect", "neural_counterfactual_effect", "contrast_id"}
    missing = required - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"transfer evaluation lacks required nonempty fields: {sorted(missing)}")
    target = frame.predicted_counterfactual_effect.to_numpy(float)
    observed = frame.neural_counterfactual_effect.to_numpy(float)
    metric = counterfactual_metrics(target, observed)
    low, high = _cluster_bootstrap(target, observed, frame.contrast_id, samples=bootstrap_samples, seed=seed)
    null = np.asarray([] if random_cfrs is None else random_cfrs, dtype=float)
    null = null[np.isfinite(null)]
    random_p = float((1 + np.sum(null >= metric["global_cfr"])) / (1 + len(null))) if len(null) else float("nan")
    return {
        "model": model, "theory": theory, "source_task": source_task,
        "target_task": target_task, "response_mapping": str(response_mapping),
        "layer": int(layer), "rank": int(rank),
        "n_train": int(frame.attrs.get("n_train", 0) if n_train is None else n_train),
        "n_selection": int(n_selection), "n_test": int(len(frame)),
        "global_cfr": metric["global_cfr"], "correlation": metric["correlation"],
        "slope": metric["slope"], "bootstrap_low": low, "bootstrap_high": high,
        "random_mean": float(np.mean(null)) if len(null) else float("nan"),
        "random_max": float(np.max(null)) if len(null) else float("nan"),
        "random_p": random_p, "mapping_gap": float("nan"),
        "controller_hash": controller_hash, "split_hash": split_hash,
        "endpoint_id": "cognitive_counterfactual_recovery", "metric_id": "global_cfr_v1",
    }

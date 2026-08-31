"""Known-ground-truth systems guarding counterfactual mechanistic inference."""

from __future__ import annotations

import numpy as np

from .das import fit_synthetic_alignment
from .interventions import interchange_subspace
from .metrics import counterfactual_metrics


def _effects(base, source, basis, readout):
    edited = interchange_subspace(base, source, basis)
    return (edited - base) @ readout


def run_synthetic_validation(seed: int = 19) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    n, dimension = 192, 12
    base = rng.normal(size=(n, dimension))
    source = rng.normal(size=(n, dimension))
    teacher, _ = np.linalg.qr(rng.normal(size=(dimension, 4)))

    # A: a one-dimensional causal variable is recovered by causal optimization.
    readout_a = teacher[:, 0]
    target_a = (source - base) @ readout_a
    learned_a = fit_synthetic_alignment(
        base, source, target_a, readout_a, rank=1, epochs=180, seed=seed
    )
    observed_a = _effects(base, source, learned_a, readout_a)
    metric_a = counterfactual_metrics(target_a, observed_a)

    # B: four independent linear causal coordinates require the complete 4-D
    # distributed teacher when evaluated jointly rather than as one scalar readout.
    target_b = (source - base) @ teacher
    full_b = (interchange_subspace(base, source, teacher) - base) @ teacher
    rank1_z = teacher[:, [0]]
    rank1_b = (interchange_subspace(base, source, rank1_z) - base) @ teacher
    metric_b4 = counterfactual_metrics(target_b, full_b)
    metric_b1 = counterfactual_metrics(target_b, rank1_b)

    # C: tasks share causal semantics through different rotations.
    task_bases = {
        task: np.linalg.qr(rng.normal(size=(dimension, 2)))[0]
        for task in ("task_a", "task_b", "task_c")
    }
    task_scores, shared_scores = [], []
    shared = task_bases["task_a"]
    for basis in task_bases.values():
        readout = basis[:, 0]
        target = (source - base) @ readout
        task_scores.append(
            counterfactual_metrics(target, _effects(base, source, basis, readout))[
                "mean_cfr"
            ]
        )
        shared_scores.append(
            counterfactual_metrics(target, _effects(base, source, shared, readout))[
                "mean_cfr"
            ]
        )

    # D: a correlated feature is highly decodable but absent from the readout.
    latent_base = rng.normal(size=n)
    latent_source = rng.normal(size=n)
    correlated = latent_base + rng.normal(scale=0.05, size=n)
    correlated_source = latent_source + rng.normal(scale=0.05, size=n)
    base_d = base.copy()
    source_d = source.copy()
    base_d[:, 0], base_d[:, 1] = latent_base, correlated
    source_d[:, 0], source_d[:, 1] = latent_source, correlated_source
    causal = base_d[:, 0]
    probe_r2 = float(np.corrcoef(causal, correlated)[0, 1] ** 2)
    noncausal_target = correlated_source - correlated
    noncausal_edit = interchange_subspace(base_d, source_d, np.eye(dimension)[:, [1]])
    noncausal_observed = noncausal_edit[:, 0] - base_d[:, 0]
    metric_d = counterfactual_metrics(noncausal_target, noncausal_observed)

    # E: direct decision control moves behavior, but violates task-specific theory.
    task = np.arange(n) % 3
    behavioral_slopes = np.array([0.6, -0.2, 0.05])
    delta = source[:, 2] - base[:, 2]
    target_e = behavioral_slopes[task] * delta
    downstream = 0.4 * delta
    metric_e = counterfactual_metrics(target_e, downstream)
    return [
        {
            "case": "A_1D_causal",
            "expected": "rank1_recovered",
            "passed": bool(
                metric_a["correlation"] > 0.9 and metric_a["mean_cfr"] > 0.7
            ),
            **metric_a,
        },
        {
            "case": "B_4D_distributed",
            "expected": "rank4_beats_rank1",
            "passed": bool(metric_b4["rmse"] < metric_b1["rmse"]),
            "rank1_rmse": metric_b1["rmse"],
            "rank4_rmse": metric_b4["rmse"],
        },
        {
            "case": "C_task_specific_rotations",
            "expected": "task_specific_beats_shared",
            "passed": bool(np.mean(task_scores) > np.mean(shared_scores)),
            "shared_mean_cfr": float(np.mean(shared_scores)),
            "task_specific_mean_cfr": float(np.mean(task_scores)),
        },
        {
            "case": "D_decodable_noncausal",
            "expected": "high_decoding_low_causal_recovery",
            "passed": bool(probe_r2 > 0.9 and metric_d["mean_cfr"] < 0.1),
            "probe_r2": probe_r2,
            "mean_cfr": metric_d["mean_cfr"],
        },
        {
            "case": "E_downstream_decision",
            "expected": "causal_effect_without_theory_correspondence",
            "passed": bool(metric_e["correlation"] < 0.8),
            **metric_e,
        },
    ]

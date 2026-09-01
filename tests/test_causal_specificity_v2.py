import inspect
import json

import numpy as np
import pandas as pd
import pytest

from cognitive_discovery.causal_mechanistic.interventions import (
    orthogonal_residual,
    remove_subspace,
)
from cognitive_discovery.causal_mechanistic.das import save_alignment
from cognitive_discovery.causal_mechanistic.metrics import (
    counterfactual_metrics,
    global_counterfactual_recovery,
)
from cognitive_discovery.causal_specificity.bootstrap import (
    bootstrap_metric_intervals,
    bootstrap_vsi,
    paired_bootstrap_difference,
)
from cognitive_discovery.causal_specificity.controls import (
    orthonormal_random_subspaces,
    pair_row_hash,
    shuffle_sources,
    shuffle_targets,
)
from cognitive_discovery.causal_specificity.geometry import subspace_geometry
from cognitive_discovery.causal_specificity.pipeline import (
    aggregate_specificity_run,
    _fit_necessity_coefficients,
    _identifier,
    _intervention_rows,
    finalize_specificity_run,
)
from cognitive_discovery.data.storage import write_records


def test_global_cfr_perfect_noop_wrong_way_and_near_zero_stability():
    predicted = np.array([0.5, -1.0, 2.0])
    assert global_counterfactual_recovery(predicted, predicted) == pytest.approx(1.0)
    assert global_counterfactual_recovery(predicted, np.zeros(3)) == pytest.approx(0.0)
    assert global_counterfactual_recovery(predicted, -predicted) < 0
    tiny = np.array([1e-12, -1e-13, 0.7, -0.4])
    value = global_counterfactual_recovery(tiny, np.zeros(4))
    assert np.isfinite(value)
    assert value == pytest.approx(0.0)


def test_thresholded_cfr_is_descriptive_and_reports_retention():
    metrics = counterfactual_metrics(
        [0.01, 0.2, -0.4], [100.0, 0.2, -0.4], threshold=0.10
    )
    assert metrics["global_cfr"] < 0
    assert metrics["thresholded_cfr_retained"] == 2
    assert metrics["thresholded_cfr_fraction_retained"] == pytest.approx(2 / 3)
    assert metrics["thresholded_cfr_median"] == pytest.approx(1.0)


def _effect_frame(observed=None):
    predicted = np.array([0.4, 0.4, -0.6, -0.6, 0.8, 0.8])
    return pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(6)],
            "contrast_id": ["a", "a", "b", "b", "c", "c"],
            "task_family": ["t1", "t1", "t1", "t1", "t2", "t2"],
            "response_mapping": ["x", "y", "x", "y", "x", "y"],
            "predicted_counterfactual_effect": predicted,
            "neural_counterfactual_effect": (
                predicted if observed is None else np.asarray(observed)
            ),
        }
    )


def test_bootstrap_clusters_semantic_pairs_and_paired_rows():
    candidate = _effect_frame()
    control = _effect_frame(np.zeros(6))
    point, intervals = bootstrap_metric_intervals(candidate, samples=80, seed=4)
    assert point["global_cfr"] == pytest.approx(1.0)
    assert next(row for row in intervals if row["metric"] == "global_cfr")[
        "ci_lower"
    ] == pytest.approx(1.0)
    difference = paired_bootstrap_difference(candidate, control, samples=80, seed=4)
    assert difference["row_identity_verified"]
    assert difference["ci_lower"] > 0
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap_difference(candidate, control.iloc[:-1], samples=10)


def test_shuffled_sources_preserve_strata_and_change_donor():
    rows = []
    for mapping in ("x", "y"):
        for index, score in enumerate((0.0, 0.1, 1.0, 1.1)):
            rows.append(
                {
                    "pair_id": f"{mapping}{index}",
                    "source_condition_id": f"s{mapping}{index}",
                    "task_family": "task",
                    "response_mapping": mapping,
                    "current_state_score": score,
                }
            )
    original = pd.DataFrame(rows)
    shuffled = shuffle_sources(original, seed=2)
    assert (shuffled.source_condition_id != shuffled.original_source_condition_id).all()
    donor = original.set_index("pair_id")
    for row in shuffled.itertuples():
        source_row = donor.loc[row.shuffled_source_pair_id]
        assert source_row.task_family == row.task_family
        assert source_row.response_mapping == row.response_mapping


def test_shuffled_targets_change_semantic_mapping_but_keep_label_duplicates_together():
    frame = _effect_frame()
    frame["task_family"] = "task"
    shuffled = shuffle_targets(frame, seed=9)
    assert (
        shuffled.shuffled_target_contrast_id.astype(str)
        != shuffled.contrast_id.astype(str)
    ).all()
    for _, part in shuffled.groupby("contrast_id"):
        assert part.predicted_counterfactual_effect.nunique() == 1
        assert part.shuffled_target_contrast_id.nunique() == 1


def test_random_subspaces_are_orthonormal_and_reproducible():
    first = orthonormal_random_subspaces(9, 3, 12, seed=3)
    second = orthonormal_random_subspaces(9, 3, 12, seed=3)
    assert np.array_equal(first, second)
    for basis in first:
        assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-10)


def test_cross_variable_artifact_ids_are_distinct_and_targets_cannot_enter_editor():
    raw = {
        "target_variable": "outcome_history",
        "theory": "outcome_history",
        "layer": 30,
        "rank": 2,
    }
    contextual = {
        "target_variable": "contextual_outcome_history",
        "theory": "latent_context",
        "layer": 30,
        "rank": 8,
    }
    assert _identifier(0, raw) != _identifier(1, contextual)
    parameters = inspect.signature(_intervention_rows).parameters
    assert "predicted_counterfactual_effect" not in parameters
    assert "theory" not in parameters


def test_vsi_bootstrap_recovers_variable_specificity():
    matched = _effect_frame()
    unrelated = _effect_frame(np.zeros(6))
    result = bootstrap_vsi(matched, [unrelated], samples=80, seed=5)
    assert result["vsi"] == pytest.approx(1.0)
    assert result["ci_lower"] > 0


def test_neutralization_changes_only_target_subspace():
    state = np.array([2.0, -1.0, 4.0, 0.5])
    basis = np.eye(4)[:, :2]
    neutralized = remove_subspace(state, basis, reference=np.array([0.25, -0.25]))
    assert np.allclose(neutralized @ basis, [0.25, -0.25])
    assert np.allclose(
        orthogonal_residual(neutralized, basis), orthogonal_residual(state, basis)
    )


def test_necessity_control_coefficients_reproduce_linear_baseline():
    rng = np.random.default_rng(7)
    n = 120
    frame = pd.DataFrame(
        {
            feature: rng.normal(size=n)
            for feature in (
                "success_evidence",
                "progress_evidence",
                "disengagement_value",
                "action_history",
                "outcome_history",
                "contextual_outcome_history",
            )
        }
    )
    frame["condition_id"] = [f"c{i}" for i in range(n)]
    frame["task_family"] = np.where(np.arange(n) % 2, "a", "b")
    frame["response_mapping"] = np.where(np.arange(n) % 2, "x", "y")
    frame["persistence_logit"] = 2.0 * frame.outcome_history - frame.progress_evidence
    coefficients, r2 = _fit_necessity_coefficients(frame, "persistence_logit")
    assert r2 > 0.999
    assert coefficients["outcome_history"] > 0
    assert coefficients["progress_evidence"] < 0


def test_subspace_geometry_reports_angles_overlap_and_hash_is_order_invariant():
    left = np.eye(5)[:, :2]
    right = np.eye(5)[:, 1:3]
    result = subspace_geometry(left, right)
    assert result["projection_overlap"] == pytest.approx(0.5)
    assert min(result["principal_angles_degrees"]) == pytest.approx(0.0)
    assert pair_row_hash(["b", "a"]) == pair_row_hash(["a", "b"])


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _synthetic_effect_rows(target, artifact_id, theory, observed_scale=1.0):
    rows = []
    predicted = (0.4, 0.4, -0.6, -0.6, 0.8, 0.8)
    for index, value in enumerate(predicted):
        rows.append(
            {
                "pair_id": f"{target}:p{index}",
                "contrast_id": f"{target}:c{index // 2}",
                "base_condition_id": f"{target}:b{index}",
                "source_condition_id": f"{target}:s{index}",
                "target_variable": target,
                "counterfactual_subtype": "synthetic",
                "task_family": "t1" if index < 4 else "t2",
                "response_mapping": "x" if index % 2 == 0 else "y",
                "pair_split": "mech_pair_test",
                "predicted_counterfactual_effect": value,
                "neural_counterfactual_effect": observed_scale * value,
                "intervention_norm": 1.0,
                "intervention_type": "frozen_DAS",
                "artifact_id": artifact_id,
                "theory": theory,
            }
        )
    return rows


def test_aggregate_and_post_specificity_necessity_smoke(tmp_path):
    root = tmp_path / "specificity"
    config = {
        "output_root": str(root),
        "seed": 4,
        "cfr": {"threshold": 0.10},
        "bootstrap": {"samples": 30, "confidence": 0.8},
    }
    specifications = (
        ("raw", "outcome_history", "outcome_history", np.eye(4)[:, :1]),
        (
            "context",
            "contextual_outcome_history",
            "latent_context",
            np.eye(4)[:, 1:2],
        ),
    )
    jobs = []
    for index, (artifact_id, target, theory, basis) in enumerate(specifications):
        artifact = save_alignment(
            root / "frozen_models/alignments" / f"{artifact_id}.safetensors",
            basis,
            metadata={"synthetic": True},
        )
        jobs.append(
            {
                "job_index": index,
                "artifact_id": artifact_id,
                "target_variable": target,
                "theory": theory,
                "layer": index,
                "rank": 1,
                "artifact": str(artifact),
                "sha256": "synthetic",
            }
        )
        shard = root / "shards" / f"candidate_{index:03d}"
        _write_json(
            shard / "audit.json",
            {"candidate_control_row_identity": True},
        )
        candidate_rows = _synthetic_effect_rows(target, artifact_id, theory)
        write_records(candidate_rows, shard / "candidate_effects.parquet")
        controls = []
        for control in (
            "persistence_state",
            "persistence_output",
            "generic_value",
            "task_id",
            "response_mapping",
        ):
            for row in _synthetic_effect_rows(target, artifact_id, theory, 0.0):
                controls.append({**row, "control": control})
        write_records(controls, shard / "named_controls.parquet")
        write_records(
            _synthetic_effect_rows(target, artifact_id, theory, 0.0),
            shard / "shuffled_source.parquet",
        )
        write_records(
            _synthetic_effect_rows(target, artifact_id, theory, 0.0),
            shard / "shuffled_target.parquet",
        )
        cross_rows = []
        for cross_target in (
            "outcome_history",
            "contextual_outcome_history",
            "action_history",
        ):
            scale = 1.0 if cross_target == target else 0.0
            for scoring_theory in (
                "outcome_history",
                "latent_context",
                "dual_history",
            ):
                for row in _synthetic_effect_rows(
                    cross_target, artifact_id, scoring_theory, scale
                ):
                    cross_rows.append(
                        {
                            **row,
                            "scoring_theory": scoring_theory,
                            "subspace_target": target,
                            "subspace_theory": theory,
                        }
                    )
            persistence_id = f"{artifact_id}__persistence_state"
            for scoring_theory in (
                "outcome_history",
                "latent_context",
                "dual_history",
            ):
                for row in _synthetic_effect_rows(
                    cross_target, persistence_id, scoring_theory, 0.0
                ):
                    cross_rows.append(
                        {
                            **row,
                            "scoring_theory": scoring_theory,
                            "subspace_target": "persistence_state",
                            "subspace_theory": "direct_persistence_control",
                            "intervention_type": "persistence_state",
                        }
                    )
        write_records(cross_rows, shard / "cross_variable.parquet")
        pd.DataFrame(
            {
                "artifact_id": [artifact_id] * 500,
                "random_index": np.arange(500),
                "global_cfr": np.linspace(-1.0, -0.1, 500),
                "correlation": np.zeros(500),
                "rmse": np.ones(500),
            }
        ).to_csv(shard / "random_subspaces.csv", index=False)
        save_alignment(
            shard / "persistence_state.safetensors",
            np.eye(4)[:, 2:3],
            metadata={"synthetic": True},
        )
    _write_json(root / "frozen_models/evaluation_jobs.json", jobs)
    _write_json(root / "gates.json", {})
    _write_json(root / "run_metadata.json", {})
    (root / "corrected_metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "layer": [0, 1],
            "target_variable": ["outcome_history", "contextual_outcome_history"],
            "theory": ["outcome_history", "latent_context"],
            "pair_split": ["mech_pair_validation"] * 2,
            "global_cfr": [0.8, 0.7],
        }
    ).to_csv(root / "corrected_metrics/global_cfr.csv", index=False)

    aggregate = aggregate_specificity_run(config, output=root)
    assert aggregate["level5a_passed"] == 2
    assert aggregate["necessity_jobs"] == 2
    metric_columns = set(
        pd.read_csv(root / "corrected_metrics/specificity_metrics.csv").columns
    )
    assert metric_columns.isdisjoint(
        {"mean_cfr", "median_cfr", "fraction_behavioral_effect_recovered"}
    )

    rng = np.random.default_rng(8)
    for job in jobs:
        target = job["target_variable"]
        endpoints = pd.DataFrame(
            {
                feature: rng.normal(size=100)
                for feature in (
                    "success_evidence",
                    "progress_evidence",
                    "disengagement_value",
                    "action_history",
                    "outcome_history",
                    "contextual_outcome_history",
                )
            }
        )
        other = 0.1 * endpoints.progress_evidence
        endpoints["condition_id"] = [f"{job['artifact_id']}:c{i}" for i in range(100)]
        endpoints["task_family"] = np.where(np.arange(100) % 2, "t1", "t2")
        endpoints["response_mapping"] = np.where(np.arange(100) % 2, "x", "y")
        endpoints["persistence_logit"] = 2.0 * endpoints[target] + other
        endpoints["neutralized_persistence_logit"] = other
        endpoints["artifact_id"] = job["artifact_id"]
        endpoints["target_variable"] = target
        endpoints["specificity_passed_before_necessity"] = True
        write_records(
            endpoints.to_dict("records"),
            root
            / "necessity/shards"
            / f"necessity_{int(job['job_index']):03d}.parquet",
        )
    result = finalize_specificity_run(config, output=root)
    assert result["level5b_passed"] == 2
    assert (root / "report.md").exists()
    assert (root / "figures/figure7_necessity.png").exists()

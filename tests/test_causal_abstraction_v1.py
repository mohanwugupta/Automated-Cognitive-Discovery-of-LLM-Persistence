import inspect
import json

import numpy as np
import pandas as pd
import pytest

from cognitive_discovery.causal_abstraction.analysis import (
    identify_synthetic_abstraction,
    natural_model_comparison,
    paired_cfr_difference,
)
from cognitive_discovery.causal_abstraction.design import (
    CONTRAST_FAMILIES,
    generate_candidate_conditions,
    shuffle_abstraction_targets,
    validate_contrast_manifest,
)
from cognitive_discovery.causal_abstraction.pipeline import (
    _interchange_effect,
    aggregate_abstraction_run,
)
from cognitive_discovery.causal_abstraction.variables import (
    ABSTRACTIONS,
    FrozenAbstractionBank,
)
from cognitive_discovery.causal_mechanistic.pipeline import _forward, _trial
from cognitive_discovery.data.storage import write_records


def _validated_contrast_rows():
    common = {
        "candidate_variables_frozen": True,
        "H_match_tolerance": 0.1,
        "E_match_tolerance": 0.1,
        "upstream_distance": 1.0,
        "delta_A": 0.0,
    }
    rows = [
        {
            **common,
            "contrast_family": CONTRAST_FAMILIES[0],
            "delta_O": 0.0,
            "delta_O_star": 1.0,
            "delta_H": 0.5,
            "delta_E": 0.5,
        },
        {
            **common,
            "contrast_family": CONTRAST_FAMILIES[1],
            "delta_O": 1.0,
            "delta_O_star": 0.1,
            "delta_H": 0.4,
            "delta_E": 0.4,
        },
        {
            **common,
            "contrast_family": CONTRAST_FAMILIES[2],
            "delta_O": 1.0,
            "delta_O_star": 0.5,
            "delta_H": 0.05,
            "delta_E": 0.05,
        },
        {
            **common,
            "contrast_family": CONTRAST_FAMILIES[3],
            "delta_O": 0.5,
            "delta_O_star": 0.5,
            "delta_H": 0.05,
            "delta_E": 1.0,
        },
        {
            **common,
            "contrast_family": CONTRAST_FAMILIES[4],
            "delta_O": 1.0,
            "delta_O_star": 0.5,
            "delta_H": 0.5,
            "delta_E": 0.05,
        },
    ]
    for route in ("history", "continuation_cost"):
        rows.append(
            {
                **common,
                "contrast_family": CONTRAST_FAMILIES[5],
                "delta_O": 1.0 if route == "history" else 0.0,
                "delta_O_star": 0.5 if route == "history" else 0.0,
                "delta_H": 0.5 if route == "history" else 0.0,
                "delta_E": 1.0,
                "effect_match_group": "match-1",
                "cause_route": route,
            }
        )
    return pd.DataFrame(rows)


def test_all_six_contrast_equalities_and_inequalities_are_enforced():
    frame = _validated_contrast_rows()
    validate_contrast_manifest(frame)
    invalid = frame.copy()
    invalid.loc[0, "delta_O"] = 0.5
    with pytest.raises(ValueError, match="same_raw_different_context"):
        validate_contrast_manifest(invalid)


def test_candidate_generator_uses_legal_existing_grammar_and_contexts():
    config = {
        "seed": 4,
        "design": {
            "candidate_conditions": 280,
            "minimum_candidate_conditions": 280,
        },
    }
    conditions, metadata = generate_candidate_conditions(config)
    assert len(conditions) == 280
    assert metadata.semantic_sha256.nunique() == 280
    assert metadata.contextual_condition.any()
    assert all(condition.history.length == 3 for condition in conditions)


def test_bare_abstraction_condition_uses_shared_mechanistic_forward_adapter():
    config = {
        "seed": 4,
        "design": {
            "candidate_conditions": 1,
            "minimum_candidate_conditions": 1,
        },
    }
    conditions, _ = generate_candidate_conditions(config)
    condition = conditions[0]
    trial = _trial(condition)

    class RecordingRunner:
        def forward(
            self,
            messages,
            labels,
            *,
            positive_label,
            editors,
            capture_layers,
        ):
            self.call = {
                "messages": messages,
                "labels": labels,
                "positive_label": positive_label,
                "editors": editors,
                "capture_layers": capture_layers,
            }
            return "forward-result"

    runner = RecordingRunner()
    result = _forward(runner, condition, capture_layers=(16, 28))

    assert result == "forward-result"
    assert runner.call["messages"] == list(trial.messages)
    assert runner.call["labels"] == condition.response_mapping.labels
    assert runner.call["positive_label"] == condition.response_mapping.continue_label
    assert runner.call["editors"] is None
    assert runner.call["capture_layers"] == (16, 28)


def test_shuffled_targets_preserve_marginals_and_pair_response_mappings():
    rows = []
    for semantic_index in range(20):
        for mapping in ("continue_x", "continue_y"):
            rows.append(
                {
                    "pair_id": f"p{semantic_index}:{mapping}",
                    "semantic_contrast_id": f"c{semantic_index}",
                    "task_family": "task",
                    "response_mapping": mapping,
                    "predicted_E_effect": float(semantic_index - 10),
                }
            )
    frame = pd.DataFrame(rows)
    shuffled = shuffle_abstraction_targets(
        frame, prediction_column="predicted_E_effect", seed=8
    )
    assert np.array_equal(
        np.sort(frame.predicted_E_effect), np.sort(shuffled.predicted_E_effect)
    )
    assert (shuffled.semantic_contrast_id != shuffled.shuffled_target_contrast_id).all()
    assert (
        shuffled.groupby("semantic_contrast_id").shuffled_target_contrast_id.nunique()
        == 1
    ).all()


@pytest.mark.parametrize("implemented", ABSTRACTIONS)
def test_synthetic_systems_recover_the_implemented_abstraction(implemented):
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        {
            "semantic_contrast_id": [f"c{i}" for i in range(120)],
            "task_family": np.where(np.arange(120) % 2, "a", "b"),
        }
    )
    for abstraction in ABSTRACTIONS:
        frame[
            {
                "O": "predicted_O_effect",
                "O_star": "predicted_O_star_effect",
                "H": "predicted_H_effect",
                "E": "predicted_E_effect",
            }[abstraction]
        ] = rng.normal(size=len(frame))
    frame["neural_counterfactual_effect"] = frame[
        {
            "O": "predicted_O_effect",
            "O_star": "predicted_O_star_effect",
            "H": "predicted_H_effect",
            "E": "predicted_E_effect",
        }[implemented]
    ]
    assert identify_synthetic_abstraction(frame) == implemented


def test_natural_geometry_models_are_fit_only_on_frozen_training_pairs():
    rng = np.random.default_rng(3)
    rows = []
    for index in range(100):
        delta_o = rng.normal()
        rows.append(
            {
                "pair_split": (
                    "abstraction_train" if index < 70 else "abstraction_test"
                ),
                "delta_O": delta_o,
                "delta_O_star": rng.normal(),
                "delta_H": rng.normal(),
                "delta_E": rng.normal(),
                "projection_distance": abs(delta_o),
            }
        )
    table = natural_model_comparison(pd.DataFrame(rows))
    assert table.iloc[0].abstraction == "O"
    assert table.iloc[0].r2_pair > 0.99


def test_paired_cfr_detects_true_over_shuffled_prediction():
    predicted = np.linspace(-1.0, 1.0, 40)
    frame = pd.DataFrame(
        {
            "semantic_contrast_id": [f"c{i}" for i in range(40)],
            "task_family": np.where(np.arange(40) % 2, "a", "b"),
            "true": predicted,
            "shuffle": predicted[::-1],
            "observed": predicted,
        }
    )
    result = paired_cfr_difference(
        frame,
        candidate_prediction="true",
        control_prediction="shuffle",
        candidate_observed="observed",
        samples=80,
        seed=2,
    )
    assert result["ci_lower"] > 0


def test_neural_editor_cannot_receive_or_update_computational_targets():
    parameters = inspect.signature(_interchange_effect).parameters
    assert not any(name.startswith("predicted_") for name in parameters)
    assert (
        "neural" not in inspect.signature(FrozenAbstractionBank.score_pairs).parameters
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_aggregate_smoke_identifies_known_integrated_evidence_system(tmp_path):
    root = tmp_path / "abstraction"
    config = {
        "output_root": str(root),
        "seed": 5,
        "bootstrap": {"samples": 30, "confidence": 0.8},
        "design": {"contrasts_per_family": 200},
        "controls": {"shuffle_magnitude_strata": 3},
        "gates": {
            "maximum_delta_correlation": 0.9,
            "minimum_effect_sd": 0.1,
            "random_p_max": 0.05,
            "minimum_task_fraction": 0.6,
        },
    }
    artifact_id = "frozen_primary"
    _write_json(
        root / "frozen_das/manifest.json",
        [
            {
                "artifact_id": artifact_id,
                "primary": True,
                "layer": 28,
                "rank": 2,
                "scope": "shared",
            }
        ],
    )
    jobs = [
        {"job_index": index, "contrast_family": family}
        for index, family in enumerate(CONTRAST_FAMILIES)
    ]
    _write_json(root / "design/evaluation_jobs.json", jobs)
    _write_json(
        root / "run_metadata.json",
        {
            "identifiability": {
                "semantic_conditions": 2400,
                "minimum_family_count": 200,
                "maximum_absolute_delta_correlation": 0.2,
                "minimum_discriminating_effect_sd": 0.5,
            }
        },
    )
    rng = np.random.default_rng(22)
    all_contrasts = []
    for job in jobs:
        natural, interchange, depth, controls = [], [], [], []
        for semantic_index in range(30):
            split = "abstraction_train" if semantic_index < 20 else "abstraction_test"
            latent = rng.normal()
            predictions = {
                "predicted_O_effect": rng.normal(),
                "predicted_O_star_effect": rng.normal(),
                "predicted_H_effect": rng.normal(),
                "predicted_E_effect": latent,
            }
            deltas = {
                "delta_O": rng.normal(),
                "delta_O_star": rng.normal(),
                "delta_H": rng.normal(),
                "delta_E": latent,
            }
            semantic_id = f"{job['job_index']}:c{semantic_index}"
            for mapping in range(2):
                pair_id = f"{semantic_id}:m{mapping}"
                identity = {
                    "pair_id": pair_id,
                    "semantic_contrast_id": semantic_id,
                    "task_family": "task",
                    "response_mapping": f"continue_{'x' if mapping == 0 else 'y'}",
                    "pair_split": split,
                    "contrast_family": job["contrast_family"],
                    **predictions,
                    **deltas,
                }
                all_contrasts.append(identity)
                natural.append(
                    {
                        **identity,
                        "artifact_id": artifact_id,
                        "projection_distance": abs(latent),
                        "signed_projection_delta": latent,
                    }
                )
                interchange.append(
                    {
                        **identity,
                        "artifact_id": artifact_id,
                        "neural_counterfactual_effect": latent,
                    }
                )
                for layer in (16, 20, 24, 28, 30):
                    depth.append(
                        {
                            **identity,
                            "artifact_id": f"whole_state_L{layer}",
                            "layer": layer,
                            "neural_counterfactual_effect": latent,
                        }
                    )
                for control in ("persistence_state", "persistence_output"):
                    controls.append(
                        {
                            **identity,
                            "artifact_id": control,
                            "intervention_type": control,
                            "neural_counterfactual_effect": 0.0,
                        }
                    )
        shard = root / "shards" / f"family_{job['job_index']:02d}"
        write_records(natural, shard / "natural_geometry.parquet")
        write_records(
            [
                {
                    "condition_id": f"coverage:{job['job_index']}",
                    "artifact_id": artifact_id,
                    "coordinates": "[0, 0]",
                }
            ],
            shard / "coverage_projections.parquet",
        )
        write_records(interchange, shard / "interchange_results.parquet")
        write_records(depth, shard / "depth_results.parquet")
        write_records(controls, shard / "control_results.parquet")
        random_rows = []
        for abstraction in ABSTRACTIONS:
            random_rows.append(
                {
                    "contrast_family": job["contrast_family"],
                    "random_index": -1,
                    "control_role": "candidate_matched_subset",
                    "abstraction": abstraction,
                    "global_cfr": 1.0 if abstraction == "E" else -1.0,
                }
            )
            for index in range(10):
                random_rows.append(
                    {
                        "contrast_family": job["contrast_family"],
                        "random_index": index,
                        "control_role": "random_subspace",
                        "abstraction": abstraction,
                        "global_cfr": 0.0,
                    }
                )
        pd.DataFrame(random_rows).to_csv(shard / "random_subspaces.csv", index=False)
        _write_json(
            shard / "audit.json",
            {
                "computational_targets_frozen_before_neural": True,
                "primary_retrained": False,
                "full_activations_saved": False,
            },
        )
    write_records(all_contrasts, root / "design/contrast_manifest.parquet")
    result = aggregate_abstraction_run(config, output=root)
    assert result["winner"] == "E"
    assert result["causal_abstraction_passed"]
    assert (root / "report.md").exists()
    assert len(list((root / "figures").glob("*.png"))) == 7

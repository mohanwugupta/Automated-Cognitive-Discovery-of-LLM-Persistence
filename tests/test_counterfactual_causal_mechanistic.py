import pickle

import numpy as np
import pandas as pd
import pytest

from cognitive_discovery.causal_mechanistic.counterfactuals import (
    FrozenTheoryBank,
    build_counterfactual_pairs,
)
from cognitive_discovery.causal_mechanistic.das import (
    audit_learning_splits,
    fit_synthetic_alignment,
)
from cognitive_discovery.causal_mechanistic.interventions import (
    interchange_subspace,
    intervention_norm,
    orthogonal_residual,
    replace_whole_state,
)
from cognitive_discovery.causal_mechanistic.metrics import (
    counterfactual_metrics,
    counterfactual_recovery,
)
from cognitive_discovery.causal_mechanistic.pipeline import (
    _specificity_control_assessment,
)
from cognitive_discovery.causal_mechanistic.synthetic import (
    run_synthetic_validation,
)
from cognitive_discovery.mechanistic.dataset.matched_conditions import (
    compile_mechanistic_conditions,
)


class DummyFrozenTheory:
    def predict(self, frame):
        values = []
        for outcomes in frame.history_outcomes:
            values.append(float(sum(outcomes)))
        return np.asarray(values)


def _frozen_bank(tmp_path):
    root = tmp_path / "frozen"
    package = root / "dummy"
    package.mkdir(parents=True)
    (root / "frozen_architectures.json").write_text('["dummy"]\n')
    with (package / "model.pkl").open("wb") as handle:
        pickle.dump(DummyFrozenTheory(), handle)
    (package / "model_spec.json").write_text('{"architecture":"dummy"}\n')
    (package / "parameters.csv").write_text("parameter,estimate\nx,1\n")
    (package / "training_condition_hashes.json").write_text("[]\n")
    return FrozenTheoryBank(root), root


def test_counterfactual_pairs_are_deterministic_intended_and_split_safe(tmp_path):
    bank, _ = _frozen_bank(tmp_path)
    config = {"dataset": {"smoke_contrasts": 35}, "seed": 17}
    records = compile_mechanistic_conditions(config, smoke=True)
    first_pairs, first_predictions = build_counterfactual_pairs(records, bank)
    second_pairs, second_predictions = build_counterfactual_pairs(records, bank)
    pd.testing.assert_frame_equal(first_pairs, second_pairs)
    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    assert first_pairs.source_base_semantically_matched.all()
    assert (first_pairs.variable_delta.abs() > 0).all()
    assert not first_pairs.behavioral_validation_row.any()
    assert set(first_pairs.pair_split) <= {
        "mech_pair_train",
        "mech_pair_validation",
        "mech_pair_test",
        "mech_task_holdout",
    }
    contextual = first_pairs[first_pairs.counterfactual_subtype == "context_isolated"]
    by_id = {record.condition.condition_id: record for record in records}
    for pair in contextual.itertuples():
        assert (
            by_id[pair.base_condition_id].targets["outcome_history"]
            == by_id[pair.source_condition_id].targets["outcome_history"]
        )


def test_frozen_theory_bank_detects_external_update(tmp_path):
    bank, root = _frozen_bank(tmp_path)
    frame = pd.DataFrame(
        {
            "history_outcomes": [[1, -1]],
            "task_family": ["bandit"],
            "response_mapping": ['{"continue":"X","disengage":"Y"}'],
        }
    )
    assert bank.predict("dummy", frame)[0] == 0
    (root / "dummy/parameters.csv").write_text("parameter,estimate\nx,2\n")
    with pytest.raises(RuntimeError, match="modified"):
        bank.assert_unchanged()


def test_frozen_theory_bank_detects_in_memory_model_update(tmp_path):
    bank, _ = _frozen_bank(tmp_path)
    bank.models["dummy"].tampered = True
    with pytest.raises(RuntimeError, match="in_memory:dummy"):
        bank.assert_unchanged()


def test_whole_state_and_empty_or_full_subspace_limits_are_exact():
    base = np.array([1.0, -2.0, 3.0])
    source = np.array([-4.0, 5.0, 6.0])
    assert np.array_equal(replace_whole_state(base, base), base)
    assert np.array_equal(replace_whole_state(base, source), source)
    empty = np.empty((3, 0))
    assert np.array_equal(interchange_subspace(base, source, empty), base)
    assert np.allclose(interchange_subspace(base, source, np.eye(3)), source)


def test_subspace_interchange_preserves_orthogonal_complement_and_logs_norm():
    base = np.array([1.0, -2.0, 3.0, 0.5])
    source = np.array([-4.0, 5.0, 6.0, -1.0])
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    edited = interchange_subspace(base, source, basis)
    assert np.allclose(edited[:2], source[:2])
    assert np.allclose(
        orthogonal_residual(edited, basis), orthogonal_residual(base, basis)
    )
    assert np.isclose(
        intervention_norm(base, edited), np.linalg.norm(source[:2] - base[:2])
    )


def test_das_split_audit_rejects_test_holdout_and_behavioral_validation():
    audit_learning_splits(["mech_pair_train", "mech_pair_validation"])
    for forbidden in ("mech_pair_test", "mech_task_holdout", "behavioral_validation"):
        with pytest.raises(ValueError, match="forbidden"):
            audit_learning_splits(["mech_pair_train", forbidden])


def test_distributed_teacher_is_recovered_and_shuffled_source_degrades():
    rng = np.random.default_rng(8)
    base = rng.normal(size=(160, 8))
    source = rng.normal(size=(160, 8))
    teacher = rng.normal(size=8)
    teacher /= np.linalg.norm(teacher)
    target = (source - base) @ teacher
    learned = fit_synthetic_alignment(
        base, source, target, teacher, rank=1, epochs=180, seed=8
    )
    correct = (interchange_subspace(base, source, learned) - base) @ teacher
    shuffled = (
        interchange_subspace(base, np.roll(source, 1, axis=0), learned) - base
    ) @ teacher
    correct_metrics = counterfactual_metrics(target, correct)
    shuffled_metrics = counterfactual_metrics(target, shuffled)
    assert correct_metrics["correlation"] > 0.9
    assert correct_metrics["mean_cfr"] > shuffled_metrics["mean_cfr"]


def test_counterfactual_recovery_perfect_noop_and_reversed():
    predicted = np.array([0.5, -1.0, 2.0])
    assert np.allclose(counterfactual_recovery(predicted, predicted), 1.0)
    assert np.allclose(counterfactual_recovery(np.zeros(3), predicted), 0.0, atol=1e-7)
    assert (counterfactual_recovery(-predicted, predicted) < 0).all()


def test_specificity_gate_requires_and_beats_every_named_control():
    names = [
        "persistence_state",
        "persistence_output",
        "generic_value",
        "task_id",
        "response_mapping",
        "unrelated_variable:action_history",
        "unrelated_variable:generic_value",
    ]
    controls = pd.DataFrame(
        {"control": names, "mean_cfr": np.linspace(-0.2, 0.2, len(names))}
    )
    assert _specificity_control_assessment(0.7, controls)["passed"]
    missing = _specificity_control_assessment(
        0.7, controls[controls.control != "task_id"]
    )
    assert not missing["passed"]
    assert missing["missing_controls"] == ["task_id"]
    controls.loc[controls.control == "generic_value", "mean_cfr"] = 0.8
    assert not _specificity_control_assessment(0.7, controls)["passed"]


def test_all_five_synthetic_guardrail_cases_pass():
    rows = run_synthetic_validation(seed=22)
    assert {row["case"][0] for row in rows} == set("ABCDE")
    assert all(row["passed"] for row in rows)

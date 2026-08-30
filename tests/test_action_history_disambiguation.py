import numpy as np
import pandas as pd

from cognitive_discovery.action_history.directions import (
    equal_norm_displacement,
    normalize,
    orthogonalize_to_subspace,
)
from cognitive_discovery.action_history.gradients import (
    finite_difference_directional_derivative,
    semantic_persistence_logit,
)
from cognitive_discovery.action_history.matching import (
    MatchConfig,
    build_matched_pairs,
)
from cognitive_discovery.action_history.residualization import ActionResidualizer
from cognitive_discovery.mechanistic.steering.intervene import add_direction


def test_action_orthogonalization_is_normalized_order_invariant_and_orthogonal():
    action = np.array([1.0, 2.0, 3.0, -1.0])
    decision = np.array([1.0, 0.0, 1.0, 0.0])
    gradient = np.array([0.0, 1.0, 1.0, 1.0])
    first, first_basis = orthogonalize_to_subspace(action, [decision, gradient])
    second, second_basis = orthogonalize_to_subspace(action, [gradient, decision])
    assert np.isclose(np.linalg.norm(first), 1.0, atol=1e-12)
    assert np.allclose(first, second, atol=1e-12)
    assert np.linalg.norm(first_basis.T @ first) < 1e-12
    assert np.linalg.norm(second_basis.T @ second) < 1e-12
    assert np.isclose(np.linalg.norm(normalize(action)), 1.0, atol=1e-12)


def _residual_frame(seed=4, count=300):
    rng = np.random.default_rng(seed)
    decision = rng.normal(size=count)
    evidence = rng.normal(size=count)
    independent_signal = rng.normal(size=count)
    action = 1.7 * decision - 0.8 * evidence + independent_signal
    return pd.DataFrame(
        {
            "condition_id": [f"c-{index}" for index in range(count)],
            "persistence_logit": decision,
            "outcome_history": evidence,
            "contextual_outcome_history": evidence * 0.5,
            "success_evidence": evidence,
            "progress_evidence": np.nan,
            "continuation_cost": 0.0,
            "disengagement_value": 0.0,
            "continuation_value": 0.0,
            "task_family": np.where(np.arange(count) % 2, "bandit", "waiting"),
            "response_mapping": np.where(
                np.arange(count) % 2, "continue_x", "continue_y"
            ),
            "action_history": action,
            "independent_signal": independent_signal,
        }
    )


def test_residualizer_is_train_only_orthogonal_and_recovers_independent_signal():
    frame = _residual_frame()
    train, heldout = frame.iloc[:220], frame.iloc[220:].copy()
    model = ActionResidualizer().fit(train)
    coefficient = model.coefficient_.copy()
    row_hash = model.fit_row_hash_
    residual = model.residualize(train)
    design = model._design(train, fitting=False)
    for column in design[:, 1:].T:
        if np.std(column) > 1e-10:
            assert abs(np.corrcoef(column, residual)[0, 1]) < 1e-10
    heldout["action_history"] += 10_000
    model.residualize(heldout)
    assert np.array_equal(model.coefficient_, coefficient)
    assert model.fit_row_hash_ == row_hash
    assert set(model.fit_condition_ids_) == set(train.condition_id)
    assert not set(heldout.condition_id) & set(model.fit_condition_ids_)
    assert np.corrcoef(residual, train.independent_signal)[0, 1] > 0.95


def test_semantic_gradient_matches_finite_difference_and_sign():
    weights = np.array([[2.0, -1.0], [-3.0, 4.0], [0.5, 0.1]])
    point = np.array([0.2, -0.4, 0.7])
    direction = np.array([0.3, 0.8, -0.5])

    def persistence(value):
        logits = value @ weights
        return semantic_persistence_logit(logits, {"X": 0, "Y": 1}, "Y")

    analytic_gradient = weights[:, 1] - weights[:, 0]
    finite = finite_difference_directional_derivative(persistence, point, direction)
    assert np.isclose(finite, analytic_gradient @ direction, atol=1e-8)
    logits = np.array([8.0, 3.0])
    assert semantic_persistence_logit(logits, {"X": 0, "Y": 1}, "X") == 5.0
    assert semantic_persistence_logit(logits, {"X": 0, "Y": 1}, "Y") == -5.0


def test_equal_norm_steering_zero_and_dose_sign():
    left = equal_norm_displacement([2.0, 0.0, 0.0], 1.0, 0.7)
    right = equal_norm_displacement([0.0, -5.0, 0.0], 1.0, 0.7)
    assert np.isclose(np.linalg.norm(left), np.linalg.norm(right))
    state = np.array([1.0, 2.0, 3.0])
    direction = np.array([2.0, -1.0, 0.5])
    assert np.array_equal(add_direction(state, direction, 0), state)
    plus = add_direction(state, direction, 0.6) - state
    minus = add_direction(state, direction, -0.6) - state
    assert np.allclose(plus, -minus)
    assert plus @ normalize(direction) > 0
    assert minus @ normalize(direction) < 0


def _matching_frame():
    rows = []
    for task in ("bandit", "waiting"):
        for mapping in ("continue_x", "continue_y"):
            prefix = f"{task}-{mapping}"
            for suffix, action, decision in (
                ("history-high", 2.0, 0.05),
                ("history-low", 0.0, 0.00),
                ("decision-high", 1.0, 2.0),
                ("decision-low", 1.0, 0.0),
            ):
                rows.append(
                    {
                        "condition_id": f"{prefix}-{suffix}",
                        "task_family": task,
                        "response_mapping": mapping,
                        "split": "test",
                        "action_history": action,
                        "persistence_logit": decision,
                        "success_evidence": 0.0,
                        "progress_evidence": 0.0,
                        "continuation_cost": 0.0,
                        "disengagement_value": 0.0,
                        "continuation_value": 0.0,
                        "outcome_history": 0.0,
                        "contextual_outcome_history": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_matching_enforces_calipers_task_boundaries_and_mapping_balance():
    frame = _matching_frame()
    config = MatchConfig(
        decision_caliper=0.1,
        history_min_delta=1.5,
        history_caliper=1e-8,
        decision_min_delta=1.0,
        nuisance_max_distance=0.1,
    )
    pairs = build_matched_pairs(frame, config)
    history = pairs[pairs.contrast_kind == "history_decision_matched"]
    decision = pairs[pairs.contrast_kind == "decision_history_matched"]
    assert (history.abs_decision_delta <= config.decision_caliper).all()
    assert (history.abs_action_delta >= config.history_min_delta).all()
    assert (decision.abs_action_delta <= config.history_caliper).all()
    assert (decision.abs_decision_delta >= config.decision_min_delta).all()
    source = frame.set_index("condition_id")
    for pair in pairs.itertuples():
        assert source.loc[pair.high_condition_id].task_family == pair.task_family
        assert source.loc[pair.low_condition_id].task_family == pair.task_family
        assert (
            source.loc[pair.high_condition_id].response_mapping == pair.response_mapping
        )
        assert (
            source.loc[pair.low_condition_id].response_mapping == pair.response_mapping
        )
    counts = pairs.groupby(["contrast_kind", "response_mapping"]).size().unstack()
    assert (counts.continue_x == counts.continue_y).all()

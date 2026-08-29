from cognitive_discovery.analysis.loto import leave_one_task_out
from cognitive_discovery.analysis.residual_discovery import discover_residuals
from cognitive_discovery.models.synthetic import generate_teacher_data


def test_loto_never_fits_target_task():
    train, test = generate_teacher_data(
        "dynamic_reevaluation", n_train=490, n_test=210, seed=44, noise=0.04
    )
    frame = __import__("pandas").concat([train, test], ignore_index=True)
    result = leave_one_task_out(frame, models=["dynamic_reevaluation"])
    assert set(result.heldout_task) == set(frame.task_family)
    assert (result.target_rows > 0).all()


def test_residual_discovery_finds_cross_family_interaction():
    train, test = generate_teacher_data(
        "dynamic_reevaluation", n_train=700, n_test=200, seed=19, noise=0.02
    )
    frame = __import__("pandas").concat([train, test], ignore_index=True)
    frame["persistence_logit"] += (
        1.7
        * frame["history_outcome_1"]
        * frame["factor_progress_evidence"].map(
            {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        )
    )
    result = discover_residuals(frame, base_model="dynamic_reevaluation")
    assert result.iloc[0].families >= 2
    assert result.iloc[0].heldout_r2_gain > 0


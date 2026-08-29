import numpy as np
import pandas as pd

from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS
from cognitive_discovery.models.fitting import compare_models, fit_model
from cognitive_discovery.models.synthetic import generate_teacher_data


def test_hypothesis_bank_is_complete():
    expected = {
        "intercept",
        "immediate_state",
        "dynamic_reevaluation",
        "choice_perseveration",
        "outcome_history",
        "dual_history",
        "latent_motivation",
        "latent_context",
        "option_termination",
        "meta_control",
        "task_set_reinstatement",
    }
    assert expected <= set(COGNITIVE_MODELS)


def test_dynamic_reevaluation_teacher_is_recovered():
    train, test = generate_teacher_data(
        "dynamic_reevaluation", n_train=500, n_test=200, seed=2, noise=0.03
    )
    result = compare_models(
        train,
        test,
        models=["intercept", "choice_perseveration", "dynamic_reevaluation"],
        sharing="fully_shared",
    )
    assert result.iloc[0].model == "dynamic_reevaluation"
    assert result.iloc[0].r2 > 0.95


def test_hierarchical_sharing_predicts_all_tasks():
    train, test = generate_teacher_data(
        "dual_history", n_train=420, n_test=140, seed=7, noise=0.05
    )
    fit = fit_model(train, "dual_history", sharing="hierarchical")
    prediction = fit.predict(test)
    assert prediction.shape == (len(test),)
    assert np.isfinite(prediction).all()


def test_missing_constructs_use_presence_indicators():
    frame = pd.DataFrame(
        {
            "task_family": ["a", "b", "a", "b"],
            "persistence_logit": [0.0, 1.0, 0.2, 0.8],
            "factor_progress_evidence": ["positive", None, "negative", None],
        }
    )
    fit = fit_model(frame, ["factor_progress_evidence"], sharing="fully_shared")
    assert any(name.endswith("__available") for name in fit.feature_names)


import pandas as pd

from cognitive_discovery.models.fitting import fit_model


def test_intercept_predicts_training_mean_without_cognitive_features():
    values = [1., 2., 3., 4., 5.] * 2
    frame = pd.DataFrame({"task_family": ["a"] * 10, "persistence_logit": values, "response_mapping": [{"continue": "X", "disengage": "Y"}] * 10})
    fit = fit_model(frame, "intercept", alphas=(1e-8,))
    assert list(fit.predict(frame)) == [3.] * 10

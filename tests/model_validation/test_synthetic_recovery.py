from cognitive_discovery.model_validation.pipeline import parameter_recovery


def test_synthetic_parameter_recovery_is_numerically_identifiable():
    result = parameter_recovery(n=420, seed=91, noise=.01)
    assert not result.empty
    assert result.predictive_r2.min() > .99
    assert result.bias.abs().quantile(.95) < .05

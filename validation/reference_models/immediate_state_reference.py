"""Reference implementation for ``immediate_state`` (no production imports)."""

FEATURES = (
    "factor_continuation_value", "factor_disengagement_value",
    "factor_continuation_cost", "factor_progress_evidence",
    "factor_success_evidence", "factor_uncertainty", "factor_prior_investment",
    "factor_controllability", "factor_environmental_stability", "factor_goal_continuity",
)


def features(row):
    return {name: float(row[name]) for name in FEATURES}


def predict(row, parameters, intercept=0.0):
    values = features(row)
    return float(intercept + sum(float(parameters.get(name, 0.0)) * value for name, value in values.items()))


def counterfactual(base, source, parameters):
    return predict(source, parameters) - predict(base, parameters)

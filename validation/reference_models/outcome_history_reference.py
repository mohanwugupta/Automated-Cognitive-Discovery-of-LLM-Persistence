"""Reference implementation for exponentially weighted outcome history."""

DECAY = 0.7
IMMEDIATE = (
    "factor_continuation_value", "factor_disengagement_value", "factor_continuation_cost",
    "factor_progress_evidence", "factor_success_evidence", "factor_uncertainty",
    "factor_prior_investment", "factor_controllability", "factor_environmental_stability",
    "factor_goal_continuity",
)


def outcome_trace(outcomes, decay=DECAY):
    ordered = [float(value) for value in outcomes]
    return float(sum((float(decay) ** lag) * value for lag, value in enumerate(reversed(ordered))))


def features(row):
    outcomes = list(row.get("history_outcomes", ()))
    values = {name: float(row[name]) for name in IMMEDIATE}
    values["history_outcome_1"] = float(outcomes[-1]) if outcomes else float("nan")
    values["history_outcome_kernel"] = outcome_trace(outcomes) if outcomes else float("nan")
    return values


def predict(row, parameters, intercept=0.0):
    values = features(row)
    return float(intercept + sum(float(parameters.get(name, 0.0)) * value for name, value in values.items()))


def counterfactual(base, source, parameters):
    return predict(source, parameters) - predict(base, parameters)

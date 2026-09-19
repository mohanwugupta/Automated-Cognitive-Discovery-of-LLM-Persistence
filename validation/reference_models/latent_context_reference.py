"""Reference cue-weighted context-relevant outcome-history model."""

DECAY = 0.7
IMMEDIATE = (
    "factor_continuation_value", "factor_disengagement_value", "factor_continuation_cost",
    "factor_progress_evidence", "factor_success_evidence", "factor_uncertainty",
    "factor_prior_investment", "factor_controllability", "factor_environmental_stability",
    "factor_goal_continuity",
)


def _trace(values, decay=DECAY):
    return float(sum((float(decay) ** lag) * float(value) for lag, value in enumerate(reversed(list(values)))))


def contextual_trace(row):
    recent = _trace(row.get("history_outcomes", ()))
    reliability = float(row["context_cue_probability"])
    if row.get("context_change_point") == "change_point":
        matched = 0.0
    elif row["context_context_return"] == "A":
        matched = _trace(row.get("context_a_history_outcomes", ()))
    elif row["context_context_return"] == "B":
        matched = _trace(row.get("context_b_history_outcomes", ()))
    else:
        matched = 0.0
    return reliability * matched + (1.0 - reliability) * recent


def features(row):
    values = {name: float(row[name]) for name in IMMEDIATE}
    trace = contextual_trace(row)
    values.update({
        "context_relevant_outcome_kernel": trace,
        "context_relevant_outcome_kernel*factor_goal_continuity": trace * float(row["factor_goal_continuity"]),
        "context_relevant_outcome_kernel*factor_environmental_stability": trace * float(row["factor_environmental_stability"]),
    })
    return values


def predict(row, parameters, intercept=0.0):
    values = features(row)
    return float(intercept + sum(float(parameters.get(name, 0.0)) * value for name, value in values.items()))


def counterfactual(base, source, parameters):
    return predict(source, parameters) - predict(base, parameters)

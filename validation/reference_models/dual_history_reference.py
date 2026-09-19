"""Reference implementation for separate action and outcome traces."""

DECAY = 0.7
IMMEDIATE = (
    "factor_continuation_value", "factor_disengagement_value", "factor_continuation_cost",
    "factor_progress_evidence", "factor_success_evidence", "factor_uncertainty",
    "factor_prior_investment", "factor_controllability", "factor_environmental_stability",
    "factor_goal_continuity",
)


def _action(value):
    return 1.0 if str(value).lower() == "continue" else -1.0


def _trace(values, decay=DECAY):
    return float(sum((float(decay) ** lag) * value for lag, value in enumerate(reversed(values))))


def features(row):
    actions = [_action(value) for value in row.get("history_actions", ())]
    outcomes = [float(value) for value in row.get("history_outcomes", ())]
    values = {name: float(row[name]) for name in IMMEDIATE}
    values.update({
        "history_action_1": actions[-1] if actions else float("nan"),
        "history_action_kernel": _trace(actions) if actions else float("nan"),
        "history_outcome_1": outcomes[-1] if outcomes else float("nan"),
        "history_outcome_kernel": _trace(outcomes) if outcomes else float("nan"),
    })
    return values


def predict(row, parameters, intercept=0.0):
    values = features(row)
    return float(intercept + sum(float(parameters.get(name, 0.0)) * value for name, value in values.items()))


def counterfactual(base, source, parameters):
    return predict(source, parameters) - predict(base, parameters)

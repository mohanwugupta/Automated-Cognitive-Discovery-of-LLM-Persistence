"""Small Bernoulli models with a frozen no-correction continuation policy.

Each prompt contributes one observation, including for the separate fractional
policy endpoint. Token probability mass is never converted to pseudo-trials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from cognitive_discovery.models.features import _series


MODELS = ("baseline", "integrated", "gate", "descriptive")


def features(rows):
    frame = pd.DataFrame(rows)
    return np.column_stack(
        [
            np.ones(len(frame)),
            frame.incentive.to_numpy(float),
            np.nan_to_num(_series(frame, "history_outcome_kernel")),
        ]
    )


def predict(rows, model, beta, correction_parameter=0.0, epsilon=0.001):
    frame = pd.DataFrame(rows)
    corrected = frame.correction.to_numpy(float)
    logits = features(frame) @ np.asarray(beta)
    base = expit(logits)
    if model == "integrated":
        return expit(logits - correction_parameter * corrected)
    if model == "gate":
        return (1 - corrected) * base + corrected * (
            (1 - correction_parameter) * base + correction_parameter * epsilon
        )
    if model == "descriptive":
        return np.where(corrected, correction_parameter, expit(beta[0]))
    if model != "baseline":
        raise ValueError(f"unknown model: {model}")
    return base


def loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-10, 1 - 1e-10)
    y = np.asarray(y, float)
    return -y * np.log(p) - (1 - y) * np.log1p(-p)


def fit(rows, target="choice_continue", *, epsilon=0.001):
    frame = pd.DataFrame(rows)
    frame = frame[np.isfinite(pd.to_numeric(frame[target], errors="coerce"))].copy()
    base = frame[frame.correction.eq(0)]
    corrected = frame[frame.correction.eq(1)]
    if len(base) < 3 or len(corrected) < 2:
        return {"status": "insufficient_valid_observations", "models": {}}
    x, y = features(base), base[target].to_numpy(float)
    result = minimize(
        lambda beta: float(loss(y, expit(x @ beta)).sum()),
        np.zeros(3),
        jac=lambda beta: x.T @ (expit(x @ beta) - y),
        bounds=[(-20, 20)] * 3,
        method="L-BFGS-B",
    )
    if not result.success:
        return {"status": "optimization_failed", "models": {}}
    beta = result.x.tolist()
    models = {"baseline": {"beta": beta, "correction_parameter": 0.0}}
    yc = corrected[target].to_numpy(float)
    for name, bounds in (("integrated", (0.0, 40.0)), ("gate", (0.0, 1.0))):
        objective = lambda value: float(
            loss(yc, predict(corrected, name, beta, value, epsilon)).sum()
        )
        optimized = minimize_scalar(objective, bounds=bounds, method="bounded")
        if not optimized.success:
            return {"status": "optimization_failed", "models": {}}
        value = min([*bounds, float(optimized.x)], key=objective)
        models[name] = {"beta": beta, "correction_parameter": value}
    mean = float(np.clip(y.mean(), 1e-8, 1 - 1e-8))
    models["descriptive"] = {
        "beta": [float(np.log(mean / (1 - mean))), 0.0, 0.0],
        "correction_parameter": float(yc.mean()),
    }
    return {
        "status": "complete",
        "models": models,
        "epsilon": epsilon,
        "n_baseline": len(base),
        "n_corrected": len(corrected),
        "baseline_fit_on": "no-correction fitting trials only; shared and frozen for M1/M2",
        "features": ["intercept", "incentive", "history_outcome_kernel (decay=0.7)"],
        "beta_bounds": [-20, 20],
        "kappa_bounds": [0, 40],
    }


def model_predictions(rows, fitted):
    return {
        name: predict(rows, name, **parameters, epsilon=fitted["epsilon"])
        for name, parameters in fitted["models"].items()
    }


def cross_validate(rows, target, *, epsilon):
    """Leave an entire history/incentive cell out, grouping all correction twins."""
    frame = pd.DataFrame(rows)
    predictions = []
    for (history, incentive), test in frame.groupby(["history", "incentive"]):
        train = frame[~(frame.history.eq(history) & frame.incentive.eq(incentive))]
        fitted = fit(train, target, epsilon=epsilon)
        if fitted["status"] != "complete":
            return pd.DataFrame()
        out = test[
            ["trial_id", "variant", "history", "incentive", "correction", target]
        ].copy()
        for name, p in model_predictions(test, fitted).items():
            out[f"p_{name}"] = p
            out[f"loss_{name}"] = loss(out[target], p)
        predictions.append(out)
    return pd.concat(predictions, ignore_index=True)


def comparison(cv, config, *, bootstrap=True):
    if cv.empty:
        return {"winner": None, "reason": "insufficient valid observations"}
    corrected = cv[cv.correction.eq(1)].dropna(subset=["loss_integrated", "loss_gate"])
    if corrected.empty:
        return {"winner": None, "reason": "no valid corrected observations"}
    scores = {name: float(corrected[f"loss_{name}"].mean()) for name in MODELS}
    margin = config["analysis"]["practical_log_loss_margin"]
    candidate = min(("integrated", "gate"), key=scores.get)
    other = "gate" if candidate == "integrated" else "integrated"
    advantages = {}
    for opponent in (other, "baseline", "descriptive"):
        differences = (
            corrected.assign(
                difference=corrected[f"loss_{opponent}"]
                - corrected[f"loss_{candidate}"]
            )
            .groupby("variant")
            .difference.mean()
            .to_numpy()
        )
        mean = float(differences.mean())
        lower, upper = mean, mean
        if bootstrap and len(differences) >= 2:
            rng = np.random.default_rng(config["seeds"]["bootstrap"])
            values = rng.choice(
                differences,
                size=(config["analysis"]["bootstrap_replicates"], len(differences)),
            ).mean(1)
            lower, upper = map(float, np.quantile(values, [0.025, 0.975]))
        advantages[opponent] = {
            "mean_log_loss_advantage": mean,
            "cluster_bootstrap_95_ci": [lower, upper],
        }
    distinguishable = advantages[other]["cluster_bootstrap_95_ci"][0] > margin
    return {
        "winner": candidate if distinguishable else None,
        "reason": (
            "discriminating held-out predictions"
            if distinguishable
            else "candidate predictions unresolved at the registered margin"
        ),
        "corrected_log_loss": scores,
        "candidate": candidate,
        "advantages": advantages,
        "beats_simpler_baselines": all(
            advantages[name]["cluster_bootstrap_95_ci"][0] > margin
            for name in ("baseline", "descriptive")
        ),
    }


def recovery(rows, config, fitted=None):
    """Recover both generators with sampled actions at the actual design budget.

    A reference test checks plumbing; a local test around each real fitted
    candidate gates interpretation. Near equivalence is allowed to fail.
    """
    frame = pd.DataFrame(rows).copy()
    epsilon = config["analysis"]["epsilon"]
    rng = np.random.default_rng(config["seeds"]["recovery"])
    parameters = {
        "integrated": {"beta": [-0.5, 0.7, 0.3], "correction_parameter": 2.0},
        "gate": {"beta": [-0.5, 0.7, 0.3], "correction_parameter": 0.8},
    }
    if fitted is not None:
        if fitted["status"] != "complete":
            return {
                "passed": False,
                "reason": "choice models unavailable",
                "generators": {},
            }
        parameters = {name: fitted["models"][name] for name in ("integrated", "gate")}
    results = {}
    for generator, params in parameters.items():
        p = predict(frame, generator, **params, epsilon=epsilon)
        recovered = 0
        for _ in range(config["analysis"]["recovery_replicates"]):
            frame["synthetic_y"] = rng.binomial(1, p)
            cv = cross_validate(frame, "synthetic_y", epsilon=epsilon)
            recovered += comparison(cv, config, bootstrap=False)["winner"] == generator
        rate = recovered / config["analysis"]["recovery_replicates"]
        results[generator] = {
            "recovery_rate": rate,
            "parameters": params,
            "passed": rate >= config["analysis"]["recovery_threshold"],
        }
    return {
        "passed": all(r["passed"] for r in results.values()),
        "generators": results,
        "replicates": config["analysis"]["recovery_replicates"],
        "sampling": "one Bernoulli action per prompt; four held-out cells; unresolved counts as failure",
    }

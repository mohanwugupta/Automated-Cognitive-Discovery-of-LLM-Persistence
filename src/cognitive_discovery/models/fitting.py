"""Continuous persistence-logit fitting under task-specific/shared/hierarchical assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import RidgeCV
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, r2_score

from .cognitive.registry import COGNITIVE_MODELS
from .features import FeatureEncoder


@dataclass
class ConstantRegressor:
    mean: float

    @property
    def coef_(self):
        return np.empty(0, dtype=float)

    def predict(self, matrix):
        return np.full(len(matrix), self.mean, dtype=float)


def _features(model) -> tuple[str, ...]:
    if isinstance(model, str):
        try:
            return COGNITIVE_MODELS[model].features
        except KeyError as error:
            raise ValueError(f"unknown cognitive model: {model}") from error
    return tuple(model)


def _expand_hierarchical(matrix, tasks, known_tasks):
    tasks = np.asarray(tasks, dtype=str)
    onehot = np.column_stack([(tasks == task).astype(float) for task in known_tasks])
    deviations = [matrix * onehot[:, [index]] for index in range(len(known_tasks))]
    return np.column_stack((matrix, onehot, *deviations))


@dataclass
class ModelFit:
    model: str
    sharing: str
    encoder: FeatureEncoder
    estimator: object
    feature_names: tuple[str, ...]
    known_tasks: tuple[str, ...]

    @property
    def coefficients(self):
        if isinstance(self.estimator, dict):
            return {task: fit.coef_.copy() for task, fit in self.estimator.items()}
        return self.estimator.coef_.copy()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix, _ = self.encoder.transform(frame)
        if self.sharing == "fully_shared":
            return np.asarray(self.estimator.predict(matrix), dtype=float)
        if self.sharing == "hierarchical":
            design = _expand_hierarchical(matrix, frame.task_family, self.known_tasks)
            return np.asarray(self.estimator.predict(design), dtype=float)
        prediction = np.full(len(frame), np.nan)
        tasks = frame.task_family.astype(str).to_numpy()
        for task in np.unique(tasks):
            if task not in self.estimator:
                raise ValueError(f"task-specific model cannot predict unseen task {task}")
            mask = tasks == task
            prediction[mask] = self.estimator[task].predict(matrix[mask])
        return prediction


def _ridge(x, y, alphas):
    if x.shape[1] == 0:
        return ConstantRegressor(float(np.mean(y)))
    cv = min(5, max(2, len(y) // 10))
    return RidgeCV(alphas=np.asarray(alphas, dtype=float), cv=cv).fit(x, y)


def fit_model(
    frame: pd.DataFrame,
    model,
    *,
    sharing: str = "fully_shared",
    target: str = "persistence_logit",
    alphas=(0.001, 0.01, 0.1, 1.0, 10.0),
) -> ModelFit:
    if target not in frame:
        raise ValueError(f"target column is absent: {target}")
    if "task_family" not in frame:
        raise ValueError("task_family is required for sharing assumptions")
    # Response mapping is retained as a nuisance-control variable for every
    # theory but is never interpreted as part of its cognitive mechanism.
    features = (*_features(model), "response_mapping")
    model_name = model if isinstance(model, str) else "custom"
    encoder = FeatureEncoder(features)
    matrix, names = encoder.fit_transform(frame)
    target_values = frame[target].to_numpy(dtype=float)
    tasks = frame.task_family.astype(str).to_numpy()
    known_tasks = tuple(sorted(np.unique(tasks)))
    if sharing == "fully_shared":
        estimator = _ridge(matrix, target_values, alphas)
    elif sharing == "hierarchical":
        design = _expand_hierarchical(matrix, tasks, known_tasks)
        estimator = _ridge(design, target_values, alphas)
    elif sharing == "task_specific":
        estimator = {}
        for task in known_tasks:
            mask = tasks == task
            estimator[task] = _ridge(matrix[mask], target_values[mask], alphas)
    else:
        raise ValueError(f"unknown sharing assumption: {sharing}")
    return ModelFit(model_name, sharing, encoder, estimator, names, known_tasks)


def regression_metrics(observed, predicted) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if np.std(observed) > 0 and np.std(predicted) > 0
        else float("nan")
    )
    return {
        "r2": float(r2_score(observed, predicted)),
        "mse": float(mean_squared_error(observed, predicted)),
        "correlation": correlation,
    }


def task_macro_metrics(frame: pd.DataFrame, prediction) -> dict[str, float]:
    scored = frame[["task_family", "persistence_logit"]].copy()
    scored["prediction"] = np.asarray(prediction)
    rows = [
        regression_metrics(part.persistence_logit, part.prediction)
        for _, part in scored.groupby("task_family")
    ]
    summary = {}
    for key in ("r2", "mse", "correlation"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        summary[f"macro_{key}"] = float(finite.mean()) if len(finite) else float("nan")
    return summary


def sampled_decision_metrics(frame: pd.DataFrame, prediction) -> dict[str, float]:
    if "sampled_action" not in frame:
        return {}
    probability = np.clip(expit(np.asarray(prediction, dtype=float)), 1e-7, 1 - 1e-7)
    observed = (frame.sampled_action.astype(str) == "continue").astype(int).to_numpy()
    rows = []
    for task, indices in frame.groupby("task_family").groups.items():
        positions = frame.index.get_indexer(indices)
        rows.append(
            (
                log_loss(observed[positions], probability[positions], labels=[0, 1]),
                brier_score_loss(observed[positions], probability[positions]),
            )
        )
    return {
        "macro_decision_log_loss": float(np.mean([row[0] for row in rows])),
        "macro_decision_brier": float(np.mean([row[1] for row in rows])),
    }


def compare_models(train, test, *, models=None, sharing="fully_shared") -> pd.DataFrame:
    rows = []
    for model in models or COGNITIVE_MODELS:
        fit = fit_model(train, model, sharing=sharing)
        prediction = fit.predict(test)
        metrics = regression_metrics(test.persistence_logit, prediction)
        macro = task_macro_metrics(test, prediction)
        decisions = sampled_decision_metrics(test.reset_index(drop=True), prediction)
        rows.append(
            {
                "model": model,
                "sharing": sharing,
                **metrics,
                **macro,
                **decisions,
                "parameters": int(
                    sum(len(value) for value in fit.coefficients.values())
                    if isinstance(fit.coefficients, dict)
                    else len(np.ravel(fit.coefficients))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["macro_r2", "r2"], ascending=False
    ).reset_index(drop=True)

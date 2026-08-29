"""Regularized interaction and spline/GAM-like static ceilings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from cognitive_discovery.hierarchy.task_descriptors import DescriptorEncoder
from cognitive_discovery.models.features import FeatureEncoder, all_observable_features
from cognitive_discovery.models.fitting import regression_metrics


@dataclass
class StaticFlexibleFit:
    kind: str
    sharing: str
    prediction: np.ndarray
    metrics: dict[str, float]
    model: object
    feature_names: tuple[str, ...]


def _ridge():
    return RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0), cv=5)


def _task_blocks(matrix, tasks, known_tasks):
    onehot = np.column_stack([(tasks == task).astype(float) for task in known_tasks])
    return np.column_stack(
        [matrix * onehot[:, [index]] for index in range(len(known_tasks))]
    )


def _descriptor_blocks(matrix, tasks, encoder):
    values = encoder.transform_rows(tasks)
    return np.column_stack(
        [matrix * values[:, [index]] for index in range(values.shape[1])]
    )


def fit_static_flexible(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    kind: str,
    sharing: str = "fully_shared",
    target: str = "persistence_logit",
):
    features = all_observable_features(train)
    encoder = FeatureEncoder(features)
    x_train, names = encoder.fit_transform(train)
    x_test, _ = encoder.transform(test)
    if kind == "linear":
        from sklearn.preprocessing import FunctionTransformer

        transformer = FunctionTransformer(validate=False)
    elif kind == "linear_interactions":
        transformer = PolynomialFeatures(2, include_bias=False)
    elif kind == "gam":
        transformer = SplineTransformer(n_knots=4, degree=2)
    else:
        raise ValueError(
            "static flexible model must be linear, linear_interactions, or gam"
        )
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)
    tasks_train = train.task_family.astype(str).to_numpy()
    tasks_test = test.task_family.astype(str).to_numpy()
    known_tasks = tuple(sorted(set(tasks_train)))
    y = train[target].to_numpy(dtype=float)
    descriptor_encoder = None
    if sharing == "fully_shared":
        model = _ridge().fit(x_train, y)
        prediction = model.predict(x_test)
    elif sharing == "task_specific":
        model = {
            task: _ridge().fit(x_train[tasks_train == task], y[tasks_train == task])
            for task in known_tasks
        }
        prediction = np.full(len(test), np.nan)
        for task in sorted(set(tasks_test)):
            if task not in model:
                raise ValueError(f"task-specific flexible model cannot predict {task}")
            mask = tasks_test == task
            prediction[mask] = model[task].predict(x_test[mask])
    elif sharing == "hierarchical":
        train_design = np.column_stack(
            (x_train, _task_blocks(x_train, tasks_train, known_tasks))
        )
        test_design = np.column_stack(
            (x_test, _task_blocks(x_test, tasks_test, known_tasks))
        )
        model = _ridge().fit(train_design, y)
        prediction = model.predict(test_design)
    elif sharing == "task_embedding":
        descriptor_encoder = DescriptorEncoder().fit(known_tasks)
        train_design = np.column_stack(
            (
                x_train,
                _descriptor_blocks(x_train, tasks_train, descriptor_encoder),
                _task_blocks(x_train, tasks_train, known_tasks),
            )
        )
        test_design = np.column_stack(
            (
                x_test,
                _descriptor_blocks(x_test, tasks_test, descriptor_encoder),
                _task_blocks(x_test, tasks_test, known_tasks),
            )
        )
        model = _ridge().fit(train_design, y)
        prediction = model.predict(test_design)
    else:
        raise ValueError(f"unknown flexible sharing structure: {sharing}")
    return StaticFlexibleFit(
        kind,
        sharing,
        prediction,
        regression_metrics(test[target], prediction),
        {
            "estimator": model,
            "transformer": transformer,
            "encoder": encoder,
            "known_tasks": known_tasks,
            "descriptor_encoder": descriptor_encoder,
        },
        names,
    )

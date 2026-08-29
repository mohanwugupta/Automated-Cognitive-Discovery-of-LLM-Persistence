"""Regularized interaction and spline/GAM-like static ceilings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from cognitive_discovery.models.features import FeatureEncoder, all_observable_features
from cognitive_discovery.models.fitting import regression_metrics


@dataclass
class StaticFlexibleFit:
    kind: str
    prediction: np.ndarray
    metrics: dict[str, float]
    model: object
    feature_names: tuple[str, ...]


def fit_static_flexible(train: pd.DataFrame, test: pd.DataFrame, *, kind: str):
    features = all_observable_features(train)
    encoder = FeatureEncoder(features)
    x_train, names = encoder.fit_transform(train)
    x_test, _ = encoder.transform(test)
    ridge = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0), cv=5)
    if kind == "linear_interactions":
        model = make_pipeline(PolynomialFeatures(2, include_bias=False), ridge)
    elif kind == "gam":
        model = make_pipeline(SplineTransformer(n_knots=4, degree=2), ridge)
    else:
        raise ValueError("static flexible model must be linear_interactions or gam")
    model.fit(x_train, train.persistence_logit)
    prediction = model.predict(x_test)
    return StaticFlexibleFit(
        kind,
        prediction,
        regression_metrics(test.persistence_logit, prediction),
        model,
        names,
    )

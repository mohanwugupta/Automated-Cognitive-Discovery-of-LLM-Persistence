"""Leakage-safe scalar/low-dimensional ridge probes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class RidgeProbe:
    alpha: float = 1.0
    rank: int = 1
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    components_: np.ndarray | None = None
    estimator_: Ridge | None = None

    def fit(self, matrix, target):
        x = np.asarray(matrix, dtype=float)
        y = np.asarray(target, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("probe training requires at least two aligned rows")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        standardized = (x - self.mean_) / self.scale_
        maximum_rank = min(int(self.rank), standardized.shape[1])
        if maximum_rank < standardized.shape[1]:
            _, _, vt = np.linalg.svd(standardized, full_matrices=False)
            self.components_ = vt[:maximum_rank]
            standardized = standardized @ self.components_.T
        else:
            self.components_ = np.eye(standardized.shape[1])
        self.estimator_ = Ridge(alpha=float(self.alpha)).fit(standardized, y)
        return self

    def transform(self, matrix) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise RuntimeError("probe has not been fitted")
        x = np.asarray(matrix, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != len(self.mean_):
            raise ValueError("probe input dimension changed")
        return ((x - self.mean_) / self.scale_) @ self.components_.T

    def predict(self, matrix) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("probe has not been fitted")
        return np.asarray(self.estimator_.predict(self.transform(matrix)), dtype=float)


def probe_metrics(observed, predicted) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0
        else float("nan")
    )
    return {
        "r2": (
            float(r2_score(observed, predicted)) if len(observed) > 1 else float("nan")
        ),
        "pearson_r": correlation,
        "mse": float(mean_squared_error(observed, predicted)),
    }


def contrast_sign_accuracy(frame, prediction, *, group="contrast_id") -> float:
    scored = frame[[group, "contrast_member"]].copy()
    scored["prediction"] = np.asarray(prediction, dtype=float)
    means = scored.groupby([group, "contrast_member"]).prediction.mean().unstack()
    if -1 not in means or 1 not in means or means.empty:
        return float("nan")
    return float((means[1] > means[-1]).mean())

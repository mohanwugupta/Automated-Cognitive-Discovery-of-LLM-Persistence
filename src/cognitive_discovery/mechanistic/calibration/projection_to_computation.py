"""Train-only calibration from neural projection to computational units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..representations.ridge_probe import probe_metrics


@dataclass
class DirectionCalibration:
    intercept: float
    slope: float
    target_mean: float
    target_scale: float

    @classmethod
    def fit(
        cls,
        projection,
        target,
        *,
        target_mean: float | None = None,
        target_scale: float | None = None,
    ):
        x = np.asarray(projection, dtype=float)
        y = np.asarray(target, dtype=float)
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("calibration requires at least two aligned observations")
        target_mean = float(y.mean()) if target_mean is None else float(target_mean)
        target_scale = float(y.std()) if target_scale is None else float(target_scale)
        if target_scale < 1e-8:
            raise ValueError("calibration target has no variance")
        z = (y - target_mean) / target_scale
        design = np.column_stack((np.ones(len(x)), x))
        intercept, slope = np.linalg.lstsq(design, z, rcond=None)[0]
        if abs(slope) < 1e-10:
            raise ValueError(
                "neural projection has zero computational calibration slope"
            )
        return cls(float(intercept), float(slope), target_mean, target_scale)

    def predict_standardized(self, projection) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(projection, dtype=float)

    def predict(self, projection) -> np.ndarray:
        return self.target_mean + self.target_scale * self.predict_standardized(
            projection
        )

    def alpha_for_delta(self, standardized_delta) -> np.ndarray:
        return np.asarray(standardized_delta, dtype=float) / self.slope

    def evaluate(self, projection, target) -> dict[str, float]:
        return probe_metrics(target, self.predict(projection))

    def to_dict(self) -> dict[str, float]:
        return {
            "intercept": self.intercept,
            "slope": self.slope,
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
        }

    @classmethod
    def from_dict(cls, row):
        return cls(
            float(row["intercept"]),
            float(row["slope"]),
            float(row["target_mean"]),
            float(row["target_scale"]),
        )

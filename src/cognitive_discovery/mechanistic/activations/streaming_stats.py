"""Numerically stable sufficient statistics; no example activation bank."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np


class StreamingMean:
    def __init__(self):
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, values) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or not len(matrix):
            raise ValueError("streamed values must be a non-empty vector or matrix")
        batch_count = len(matrix)
        batch_mean = matrix.mean(axis=0)
        batch_m2 = np.square(matrix - batch_mean).sum(axis=0)
        if self.mean is None:
            self.count = batch_count
            self.mean = batch_mean.copy()
            self.m2 = batch_m2.copy()
            return
        if matrix.shape[1] != len(self.mean):
            raise ValueError("streamed feature dimension changed")
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / total
        self.count = total

    @property
    def variance(self) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("streaming mean has no observations")
        return self.m2 / max(self.count - 1, 1)


@dataclass(frozen=True)
class RidgeSolution:
    coefficient: np.ndarray
    intercept: np.ndarray | float
    x_mean: np.ndarray
    x_scale: np.ndarray

    def predict(self, matrix) -> np.ndarray:
        x = np.asarray(matrix, dtype=float)
        return np.asarray(self.intercept + x @ self.coefficient, dtype=float)


class StreamingRidge:
    """Exact centered ridge sufficient statistics, optionally standardized."""

    def __init__(self):
        self.count = 0
        self.sum_x: np.ndarray | None = None
        self.sum_y: np.ndarray | None = None
        self.xtx: np.ndarray | None = None
        self.xty: np.ndarray | None = None
        self.sum_x2: np.ndarray | None = None

    def update(self, matrix, target) -> None:
        x = np.asarray(matrix, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError("ridge update requires aligned non-empty 2D arrays")
        if self.sum_x is None:
            self.sum_x = np.zeros(x.shape[1], dtype=np.float64)
            self.sum_y = np.zeros(y.shape[1], dtype=np.float64)
            self.xtx = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
            self.xty = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
            self.sum_x2 = np.zeros(x.shape[1], dtype=np.float64)
        if x.shape[1] != len(self.sum_x) or y.shape[1] != len(self.sum_y):
            raise ValueError("ridge dimensions changed between batches")
        self.count += len(x)
        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.sum_x2 += np.square(x).sum(axis=0)

    def solve(self, alpha: float = 1.0, *, standardize: bool = False) -> RidgeSolution:
        if self.count < 1 or self.sum_x is None:
            raise RuntimeError("ridge statistics have no observations")
        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count
        centered_xx = self.xtx - self.count * np.outer(mean_x, mean_x)
        centered_xy = self.xty - self.count * np.outer(mean_x, mean_y)
        scale = np.ones_like(mean_x)
        if standardize:
            variance = self.sum_x2 / self.count - np.square(mean_x)
            scale = np.sqrt(np.maximum(variance, 0.0))
            scale[scale < 1e-8] = 1.0
            centered_xx = centered_xx / np.outer(scale, scale)
            centered_xy = centered_xy / scale[:, None]
        coefficient_scaled = np.linalg.solve(
            centered_xx + float(alpha) * np.eye(len(mean_x)), centered_xy
        )
        coefficient = coefficient_scaled / scale[:, None]
        intercept = mean_y - mean_x @ coefficient
        coefficient = coefficient[:, 0] if coefficient.shape[1] == 1 else coefficient
        intercept = float(intercept[0]) if len(intercept) == 1 else intercept
        return RidgeSolution(coefficient, intercept, mean_x, scale)


class LayerTemporaryCache:
    """Scratch cache whose entire layer directory is removed on exit."""

    def __init__(self, scratch: str | Path, layer: int):
        self.scratch = Path(scratch)
        self.layer = int(layer)
        self.path: Path | None = None

    def __enter__(self) -> Path:
        parent = self.scratch / "temporary_activations"
        parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(
            tempfile.mkdtemp(prefix=f"layer-{self.layer:03d}-", dir=parent)
        )
        return self.path

    def __exit__(self, exc_type, exc, traceback):
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
        return False

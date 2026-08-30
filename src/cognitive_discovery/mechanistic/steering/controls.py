"""Seeded matched random and orthogonal direction controls."""

from __future__ import annotations

import numpy as np


def random_directions(dimension: int, count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(size=(int(count), int(dimension)))
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def orthogonalize(direction, reference) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    reference = reference / np.linalg.norm(reference)
    residual = vector - reference * np.dot(vector, reference)
    norm = np.linalg.norm(residual)
    if norm <= 1e-12:
        raise ValueError("direction is collinear with the reference")
    return residual / norm


def matched_orthogonal_direction(reference, *, seed: int) -> np.ndarray:
    random = random_directions(len(reference), 1, seed=seed)[0]
    return orthogonalize(random, reference)

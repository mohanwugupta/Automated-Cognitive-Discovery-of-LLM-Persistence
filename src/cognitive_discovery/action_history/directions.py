"""Direction geometry for separating action history from decision readout."""

from __future__ import annotations

import numpy as np


def normalize(vector, *, tolerance: float = 1e-12) -> np.ndarray:
    """Return an exactly unit-normalized float64 vector."""

    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= tolerance:
        raise ValueError("direction must be finite and nonzero")
    unit = value / norm
    # A second normalization removes the final few ulps for long vectors.
    return unit / np.linalg.norm(unit)


def orthonormal_basis(vectors, *, tolerance: float = 1e-10) -> np.ndarray:
    """Return a QR basis for a column span with rank-deficiency protection."""

    values = [np.asarray(vector, dtype=np.float64).reshape(-1) for vector in vectors]
    if not values:
        return np.empty((0, 0), dtype=np.float64)
    if len({len(value) for value in values}) != 1:
        raise ValueError("control directions must share one dimension")
    matrix = np.column_stack(values)
    if not np.isfinite(matrix).all():
        raise ValueError("control directions must be finite")
    _, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or singular[0] <= tolerance:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    rank = int(np.sum(singular > tolerance * singular[0]))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return basis[:, :rank]


def orthogonalize_to_subspace(direction, controls, *, tolerance: float = 1e-10):
    """Remove the span of ``controls`` and return a unit residual plus basis."""

    vector = normalize(direction)
    controls = list(controls)
    if not controls:
        return vector, np.empty((len(vector), 0), dtype=np.float64)
    basis = orthonormal_basis(controls, tolerance=tolerance)
    if basis.shape[0] != len(vector):
        raise ValueError("direction and control subspace dimensions differ")
    residual = vector - basis @ (basis.T @ vector)
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= tolerance:
        raise ValueError("action direction lies entirely in the control subspace")
    return normalize(residual), basis


def cosine_similarity(left, right) -> float:
    return float(normalize(left) @ normalize(right))


def pairwise_similarity(directions: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    names = sorted(directions)
    for left_index, left in enumerate(names):
        for right in names[left_index:]:
            rows.append(
                {
                    "direction_a": left,
                    "direction_b": right,
                    "cosine": cosine_similarity(directions[left], directions[right]),
                }
            )
    return rows


def equal_norm_displacement(
    direction, dose: float, activation_scale: float
) -> np.ndarray:
    """Construct the PRD's matched residual-stream displacement."""

    if not np.isfinite(activation_scale) or activation_scale <= 0:
        raise ValueError("activation scale must be positive")
    return float(dose) * float(activation_scale) * normalize(direction)

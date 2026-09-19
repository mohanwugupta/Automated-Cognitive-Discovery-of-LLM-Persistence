"""Controller-subspace geometry without activation-bank persistence."""

from __future__ import annotations

import numpy as np


def orthonormal_basis(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("controller basis must be a finite nonempty matrix")
    q, _ = np.linalg.qr(matrix)
    return q[:, : matrix.shape[1]]


def controller_geometry(left, right) -> dict:
    left, right = orthonormal_basis(left), orthonormal_basis(right)
    if left.shape[0] != right.shape[0]:
        raise ValueError("controller bases must share hidden dimension")
    singular = np.clip(np.linalg.svd(left.T @ right, compute_uv=False), 0.0, 1.0)
    angles = np.arccos(singular)
    overlap = float(np.square(singular).sum() / max(left.shape[1], right.shape[1]))
    projection = left @ left.T @ right
    residual = float(np.linalg.norm(right - projection, ord="fro") / np.linalg.norm(right, ord="fro"))
    # Orthogonal Procrustes residual is descriptive and does not use outcomes.
    u, _, vt = np.linalg.svd(left.T @ right, full_matrices=False)
    aligned = left @ (u @ vt)
    procrustes = float(np.linalg.norm(aligned - right[:, : aligned.shape[1]], ord="fro")) if aligned.shape[1] <= right.shape[1] else float("nan")
    return {
        "mean_principal_angle_radians": float(angles.mean()),
        "max_principal_angle_radians": float(angles.max()),
        "subspace_overlap": overlap,
        "mean_canonical_correlation": float(singular.mean()),
        "cross_projection_residual": residual,
        "procrustes_residual": procrustes,
    }

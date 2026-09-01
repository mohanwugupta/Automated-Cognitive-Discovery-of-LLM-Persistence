"""Secondary geometric comparisons among frozen neural subspaces."""

from __future__ import annotations

import numpy as np


def subspace_geometry(left, right) -> dict:
    left_q = np.linalg.qr(np.asarray(left, dtype=float), mode="reduced")[0]
    right_q = np.linalg.qr(np.asarray(right, dtype=float), mode="reduced")[0]
    if left_q.shape[0] != right_q.shape[0]:
        raise ValueError("subspaces must share hidden-state dimensionality")
    correlations = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
    correlations = np.clip(correlations, 0.0, 1.0)
    angles = np.degrees(np.arccos(correlations))
    overlap = float(
        np.square(np.linalg.norm(left_q.T @ right_q, ord="fro"))
        / min(left_q.shape[1], right_q.shape[1])
    )
    return {
        "canonical_correlations": correlations.tolist(),
        "principal_angles_degrees": angles.tolist(),
        "mean_canonical_correlation": float(correlations.mean()),
        "minimum_principal_angle_degrees": float(angles.min()),
        "projection_overlap": overlap,
    }

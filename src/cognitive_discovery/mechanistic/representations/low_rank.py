"""Registered ranks for the optional low-dimensional extension."""

from __future__ import annotations

from .ridge_probe import RidgeProbe


ALLOWED_RANKS = (1, 2, 4)


def fit_registered_rank(matrix, target, *, rank: int, alpha: float = 1.0) -> RidgeProbe:
    if int(rank) not in ALLOWED_RANKS:
        raise ValueError(f"rank must be one of {ALLOWED_RANKS}")
    return RidgeProbe(alpha=alpha, rank=int(rank)).fit(matrix, target)

"""Residual-stream direction interventions."""

from __future__ import annotations

import numpy as np


def _unit(direction):
    if hasattr(direction, "detach"):
        norm = direction.norm()
        if float(norm) <= 1e-12:
            raise ValueError("direction must be nonzero")
        return direction / norm
    vector = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("direction must be nonzero")
    return vector / norm


def add_direction(state, direction, alpha):
    """Add a unit direction; alpha is therefore the exact projection change."""

    if hasattr(state, "detach"):
        import torch

        vector = torch.as_tensor(direction, dtype=state.dtype, device=state.device)
        unit = _unit(vector)
        magnitude = torch.as_tensor(alpha, dtype=state.dtype, device=state.device)
        if magnitude.ndim == 0:
            return state + magnitude * unit
        return state + magnitude[:, None] * unit
    return np.asarray(state) + np.asarray(alpha)[..., None] * _unit(direction)


def direction_editor(direction, alpha):
    return lambda state: add_direction(state, direction, alpha)


def projection_change(before, after, direction) -> np.ndarray:
    unit = _unit(direction)
    return np.asarray(after - before, dtype=float) @ np.asarray(unit, dtype=float)

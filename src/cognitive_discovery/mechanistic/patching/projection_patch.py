"""Direction-specific activation patching with orthogonal state preserved."""

from __future__ import annotations

import numpy as np


def _unit_like(direction, target):
    if hasattr(target, "detach"):
        import torch

        vector = torch.as_tensor(direction, dtype=target.dtype, device=target.device)
        norm = vector.norm()
        if float(norm) <= 1e-12:
            raise ValueError("direction must be nonzero")
        return vector / norm
    vector = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("direction must be nonzero")
    return vector / norm


def projection_patch(target_state, source_state, direction):
    """Replace only ``d·h_target`` with ``d·h_source``."""

    unit = _unit_like(direction, target_state)
    if hasattr(target_state, "detach"):
        import torch

        source = torch.as_tensor(
            source_state, device=target_state.device, dtype=target_state.dtype
        )
        difference = (source * unit).sum(dim=-1) - (target_state * unit).sum(dim=-1)
        return target_state + difference[..., None] * unit
    target = np.asarray(target_state, dtype=float)
    source = np.asarray(source_state, dtype=float)
    difference = source @ unit - target @ unit
    return target + difference[..., None] * unit


def projection_replacement_error(patched, source, direction) -> float:
    unit = np.asarray(direction, dtype=float)
    unit /= np.linalg.norm(unit)
    return float(np.max(np.abs(np.asarray(patched) @ unit - np.asarray(source) @ unit)))


def orthogonal_residual(state, direction):
    vector = np.asarray(direction, dtype=float)
    vector /= np.linalg.norm(vector)
    values = np.asarray(state, dtype=float)
    return values - (values @ vector)[..., None] * vector

"""Exact whole-state and low-rank interchange interventions."""

from __future__ import annotations

import numpy as np


def _orthonormal(columns, *, like=None):
    if hasattr(columns, "detach") or hasattr(like, "detach"):
        import torch

        value = torch.as_tensor(
            columns,
            dtype=getattr(like, "dtype", None),
            device=getattr(like, "device", None),
        )
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2:
            raise ValueError("subspace basis must have shape [hidden, rank]")
        if value.shape[1] == 0:
            return value
        return torch.linalg.qr(value.float(), mode="reduced").Q.to(value.dtype)
    value = np.asarray(columns, dtype=float)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2:
        raise ValueError("subspace basis must have shape [hidden, rank]")
    if value.shape[1] == 0:
        return value
    return np.linalg.qr(value, mode="reduced")[0]


def replace_whole_state(base, source):
    if hasattr(base, "detach"):
        import torch

        source_value = torch.as_tensor(source, dtype=base.dtype, device=base.device)
        if source_value.shape == base.shape[1:]:
            source_value = source_value.expand_as(base)
        if source_value.shape != base.shape:
            raise ValueError("source and base state shapes differ")
        return source_value
    base_value, source_value = np.asarray(base), np.asarray(source)
    if source_value.shape == base_value.shape[1:]:
        source_value = np.broadcast_to(source_value, base_value.shape)
    if base_value.shape != source_value.shape:
        raise ValueError("source and base state shapes differ")
    return source_value.copy()


def interchange_subspace(base, source, basis):
    """Replace source coordinates only inside an orthonormal subspace."""

    q = _orthonormal(basis, like=base)
    if hasattr(base, "detach"):
        import torch

        source_value = torch.as_tensor(source, dtype=base.dtype, device=base.device)
        if source_value.shape == base.shape[1:]:
            source_value = source_value.expand_as(base)
        if source_value.shape != base.shape:
            raise ValueError("source and base state shapes differ")
        return base + ((source_value - base) @ q) @ q.transpose(-1, -2)
    base_value = np.asarray(base, dtype=float)
    source_value = np.asarray(source, dtype=float)
    if source_value.shape == base_value.shape[1:]:
        source_value = np.broadcast_to(source_value, base_value.shape)
    if source_value.shape != base_value.shape:
        raise ValueError("source and base state shapes differ")
    return base_value + ((source_value - base_value) @ q) @ q.T


def remove_subspace(state, basis, reference=None):
    """Replace subspace coordinates by zero or a frozen reference coordinate."""

    q = _orthonormal(basis, like=state)
    if hasattr(state, "detach"):
        import torch

        coordinates = state @ q
        target = torch.zeros_like(coordinates)
        if reference is not None:
            target = torch.as_tensor(reference, dtype=state.dtype, device=state.device)
        return state + (target - coordinates) @ q.transpose(-1, -2)
    value = np.asarray(state, dtype=float)
    coordinates = value @ q
    target = np.zeros_like(coordinates) if reference is None else np.asarray(reference)
    return value + (target - coordinates) @ q.T


def subspace_coordinates(state, basis):
    q = _orthonormal(basis, like=state)
    return state @ q


def orthogonal_residual(state, basis):
    q = _orthonormal(basis, like=state)
    return (
        state - (state @ q) @ q.transpose(-1, -2)
        if hasattr(state, "detach")
        else state - (state @ q) @ q.T
    )


def intervention_norm(before, after) -> float:
    difference = after - before
    if hasattr(difference, "detach"):
        return float(difference.detach().float().norm().cpu())
    return float(np.linalg.norm(np.asarray(difference, dtype=float)))

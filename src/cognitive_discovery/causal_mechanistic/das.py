"""Low-rank distributed alignment search primitives and compact I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .interventions import interchange_subspace


class DASAlignment:
    """A trainable distributed rank-k subspace with QR parameterization."""

    def __init__(self, hidden_size: int, rank: int, *, seed: int = 0, device=None):
        import torch

        hidden_size, rank = int(hidden_size), int(rank)
        if hidden_size < 1 or rank < 1 or rank > hidden_size:
            raise ValueError("DAS rank must be between one and hidden size")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.randn(hidden_size, rank, generator=generator) / np.sqrt(
            hidden_size
        )
        self.parameter = torch.nn.Parameter(initial.to(device=device))
        self.rank = rank

    def basis(self):
        import torch

        return torch.linalg.qr(self.parameter.float(), mode="reduced").Q.to(
            self.parameter.dtype
        )

    def parameters(self):
        return (self.parameter,)

    def edit(self, base, source):
        return interchange_subspace(base, source, self.basis())

    def numpy_basis(self) -> np.ndarray:
        return self.basis().detach().float().cpu().numpy().copy()

    def load_numpy_basis(self, basis) -> None:
        import torch

        value = torch.as_tensor(
            basis, dtype=self.parameter.dtype, device=self.parameter.device
        )
        if value.shape != self.parameter.shape:
            raise ValueError("saved DAS basis shape differs from alignment")
        with torch.no_grad():
            self.parameter.copy_(value)


def save_alignment(path: str | Path, basis, *, metadata: dict) -> Path:
    from safetensors.numpy import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(basis, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("DAS basis must be a matrix")
    save_file(
        {"basis": np.ascontiguousarray(value)},
        str(path),
        metadata={
            "protocol": "counterfactual_causal_mechanistic_v1",
            "run_metadata": json.dumps(metadata, sort_keys=True),
        },
    )
    return path


def load_alignment(path: str | Path) -> np.ndarray:
    from safetensors.numpy import load_file

    return np.asarray(load_file(str(path))["basis"], dtype=np.float32)


def audit_learning_splits(pair_splits, *, heldout_tasks=()):
    observed = set(map(str, pair_splits))
    forbidden = observed & {
        "mech_pair_test",
        "mech_task_holdout",
        "behavioral_validation",
    }
    if forbidden:
        raise ValueError(f"DAS learning received forbidden splits: {sorted(forbidden)}")
    if observed - {"mech_pair_train", "mech_pair_validation"}:
        raise ValueError(f"unknown mechanistic learning split: {sorted(observed)}")
    if heldout_tasks:
        raise ValueError("held-out task rows cannot contribute to DAS learning")


def fit_synthetic_alignment(
    base,
    source,
    target_change,
    readout,
    *,
    rank: int,
    epochs: int = 300,
    learning_rate: float = 0.05,
    seed: int = 0,
):
    """Recover a known linear causal subspace without an LLM (TDD/simulation)."""

    import torch

    base_t = torch.as_tensor(base, dtype=torch.float32)
    source_t = torch.as_tensor(source, dtype=torch.float32)
    target_t = torch.as_tensor(target_change, dtype=torch.float32)
    readout_t = torch.as_tensor(readout, dtype=torch.float32)
    alignment = DASAlignment(base_t.shape[1], rank, seed=seed)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=float(learning_rate))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        edited = alignment.edit(base_t, source_t)
        observed = (edited - base_t) @ readout_t
        loss = torch.mean(torch.square(observed - target_t))
        loss.backward()
        optimizer.step()
    return alignment.numpy_basis()

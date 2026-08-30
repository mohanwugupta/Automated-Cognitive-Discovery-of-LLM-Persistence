"""Paired mean-difference directions and compact safetensors I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..activations.streaming_stats import StreamingMean


class MeanDifferenceDirections:
    def __init__(self):
        self._statistics: dict[tuple[str, int], StreamingMean] = {}

    def update(self, target: str, layer: int, differences) -> None:
        self._statistics.setdefault((str(target), int(layer)), StreamingMean()).update(
            differences
        )

    def directions(self) -> tuple[dict[str, dict[int, np.ndarray]], list[dict]]:
        directions: dict[str, dict[int, np.ndarray]] = {}
        metadata = []
        for (target, layer), statistic in sorted(self._statistics.items()):
            mean = np.asarray(statistic.mean, dtype=np.float64)
            norm = float(np.linalg.norm(mean))
            if norm <= 1e-12:
                raise RuntimeError(
                    f"zero paired mean difference for {target}/layer {layer}"
                )
            directions.setdefault(target, {})[layer] = (mean / norm).astype(np.float32)
            metadata.append(
                {
                    "target": target,
                    "layer": layer,
                    "pairs": statistic.count,
                    "mean_difference_norm": norm,
                }
            )
        return directions, metadata


def _key(target: str, layer: int) -> str:
    return f"{target}.layer_{int(layer):03d}"


def save_directions(
    path: str | Path,
    directions: dict[str, dict[int, np.ndarray]],
    *,
    metadata: dict | None = None,
) -> Path:
    from safetensors.numpy import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        _key(target, layer): np.ascontiguousarray(vector, dtype=np.float32)
        for target, layers in directions.items()
        for layer, vector in layers.items()
    }
    if not tensors:
        raise ValueError("no directions were supplied")
    header = {"protocol": str((metadata or {}).get("protocol", "mechanistic_v1"))}
    if metadata is not None:
        header["run_metadata"] = json.dumps(metadata, sort_keys=True)
    save_file(tensors, str(path), metadata=header)
    return path


def load_directions(path: str | Path) -> dict[str, dict[int, np.ndarray]]:
    from safetensors.numpy import load_file

    tensors = load_file(str(path))
    output: dict[str, dict[int, np.ndarray]] = {}
    for key, vector in tensors.items():
        target, layer_text = key.rsplit(".layer_", 1)
        output.setdefault(target, {})[int(layer_text)] = np.asarray(
            vector, dtype=np.float32
        )
    return output


def project(states, direction) -> np.ndarray:
    matrix = np.asarray(states, dtype=float)
    vector = np.asarray(direction, dtype=float)
    return np.asarray(matrix @ vector, dtype=float)


def preliminary_candidate_layers(metadata, *, count: int = 4) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    for row in metadata:
        selected.setdefault(str(row["target"]), []).append(row)
    return {
        target: [
            int(row["layer"])
            for row in sorted(
                rows, key=lambda item: item["mean_difference_norm"], reverse=True
            )[:count]
        ]
        for target, rows in selected.items()
    }

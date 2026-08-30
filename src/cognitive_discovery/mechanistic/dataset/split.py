"""Contrast-grouped mechanistic splits."""

from __future__ import annotations

import hashlib


def mechanistic_split(
    contrast_id: str,
    task_family: str,
    *,
    discovery_tasks,
    heldout_tasks,
    seed: int,
) -> str:
    if task_family in set(heldout_tasks):
        return "transfer"
    if task_family not in set(discovery_tasks):
        raise ValueError(
            f"task is in neither discovery nor held-out set: {task_family}"
        )
    digest = hashlib.sha256(f"{seed}:{contrast_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"

"""Frozen, low-dimensional ontology of task *structure*, not task state.

The definitions in this module are deliberately data independent.  They must
not be inferred from Round-1 outcomes because they are used for zero-shot task
prediction.  Binary values use the positive interpretation in
``DESCRIPTOR_DEFINITIONS``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DESCRIPTOR_DEFINITIONS = {
    "outcome_dependence": "continuing can causally influence later task outcomes",
    "evidence_accumulation": "observations accumulate into task-relevant evidence",
    "explicit_progress": "the interface exposes a meaningful progress signal",
    "reward_stationarity": "recent outcomes predict future opportunities",
    "absorbing_disengagement": "disengagement terminates the current pursuit",
    "effort_accumulation": "continuation consumes cumulative effort or resources",
}
DESCRIPTOR_NAMES = tuple(DESCRIPTOR_DEFINITIONS)

# These values are the preregistered Round-2 ontology.  Changes require a new
# protocol version; they are never optimized against behavioral outcomes.
TASK_DESCRIPTORS: dict[str, dict[str, int]] = {
    "bandit": {
        "outcome_dependence": 1,
        "evidence_accumulation": 1,
        "explicit_progress": 0,
        "reward_stationarity": 1,
        "absorbing_disengagement": 0,
        "effort_accumulation": 0,
    },
    "foraging": {
        "outcome_dependence": 1,
        "evidence_accumulation": 1,
        "explicit_progress": 0,
        "reward_stationarity": 0,
        "absorbing_disengagement": 1,
        "effort_accumulation": 0,
    },
    "solvability": {
        "outcome_dependence": 1,
        "evidence_accumulation": 1,
        "explicit_progress": 1,
        "reward_stationarity": 1,
        "absorbing_disengagement": 1,
        "effort_accumulation": 0,
    },
    "information_sampling": {
        "outcome_dependence": 0,
        "evidence_accumulation": 1,
        "explicit_progress": 0,
        "reward_stationarity": 1,
        "absorbing_disengagement": 1,
        "effort_accumulation": 0,
    },
    "waiting": {
        "outcome_dependence": 0,
        "evidence_accumulation": 1,
        "explicit_progress": 1,
        "reward_stationarity": 0,
        "absorbing_disengagement": 1,
        "effort_accumulation": 0,
    },
    "effort": {
        "outcome_dependence": 1,
        "evidence_accumulation": 0,
        "explicit_progress": 1,
        "reward_stationarity": 1,
        "absorbing_disengagement": 1,
        "effort_accumulation": 1,
    },
    "debugging": {
        "outcome_dependence": 1,
        "evidence_accumulation": 1,
        "explicit_progress": 1,
        "reward_stationarity": 0,
        "absorbing_disengagement": 1,
        "effort_accumulation": 1,
    },
}


def validate_descriptors(descriptors=TASK_DESCRIPTORS) -> None:
    for task, values in descriptors.items():
        if tuple(values) != DESCRIPTOR_NAMES:
            raise ValueError(f"descriptor order/schema mismatch for {task}")
        invalid = {name: value for name, value in values.items() if value not in (0, 1)}
        if invalid:
            raise ValueError(f"task descriptors must be binary: {task}: {invalid}")


def descriptor_frame(tasks=None, descriptors=TASK_DESCRIPTORS) -> pd.DataFrame:
    """Return the frozen ontology in a stable order."""

    validate_descriptors(descriptors)
    selected = sorted(descriptors) if tasks is None else [str(task) for task in tasks]
    unknown = set(selected) - set(descriptors)
    if unknown:
        raise ValueError(f"missing frozen task descriptors: {sorted(unknown)}")
    return pd.DataFrame(
        [{"task_family": task, **descriptors[task]} for task in selected]
    )


@dataclass
class DescriptorEncoder:
    """Source-task-only centering used by ontology-conditioned fits."""

    names: tuple[str, ...] = DESCRIPTOR_NAMES
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, tasks) -> "DescriptorEncoder":
        raw = descriptor_frame(tasks)[list(self.names)].to_numpy(dtype=float)
        self.means = raw.mean(axis=0)
        self.scales = raw.std(axis=0)
        self.scales[self.scales < 1e-8] = 1.0
        return self

    def transform(self, tasks) -> np.ndarray:
        if self.means is None or self.scales is None:
            raise RuntimeError("descriptor encoder has not been fitted")
        raw = descriptor_frame(tasks)[list(self.names)].to_numpy(dtype=float)
        return (raw - self.means) / self.scales

    def transform_rows(self, tasks) -> np.ndarray:
        """Transform a task label per observation, preserving row order."""

        labels = np.asarray(tasks, dtype=str)
        unique = tuple(sorted(set(labels)))
        encoded = self.transform(unique)
        lookup = {task: encoded[index] for index, task in enumerate(unique)}
        return np.vstack([lookup[task] for task in labels])

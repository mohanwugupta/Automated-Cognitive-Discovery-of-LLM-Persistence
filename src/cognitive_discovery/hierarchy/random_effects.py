"""Regularized M1--M4 hierarchy for a shared cognitive architecture.

The implementation is an empirical-Bayes/ridge approximation: task deviations
and ontology coefficients receive stronger shrinkage through scaled design
blocks.  Crucially, M4's descriptor encoder and coefficients are fitted only on
source tasks, so held-out outcomes cannot leak into ontology zero-shot fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS
from cognitive_discovery.models.features import FeatureEncoder

from .task_descriptors import DESCRIPTOR_NAMES, DescriptorEncoder


VARIANT_NAMES = {
    "M1": "fully_shared",
    "M2": "task_specific",
    "M3": "random_effects",
    "M4": "ontology_conditioned",
}


def _architecture_features(architecture) -> tuple[str, ...]:
    if isinstance(architecture, str):
        if architecture not in COGNITIVE_MODELS:
            raise ValueError(f"unknown cognitive architecture: {architecture}")
        return COGNITIVE_MODELS[architecture].features
    return tuple(architecture)


def _fit_ridge(x, y, alphas, *, fit_intercept=False):
    alphas = tuple(float(value) for value in alphas)
    if len(alphas) == 1:
        return Ridge(alpha=alphas[0], fit_intercept=fit_intercept).fit(x, y)
    cv = min(5, max(2, len(y) // 20))
    return RidgeCV(alphas=np.asarray(alphas), cv=cv, fit_intercept=fit_intercept).fit(
        x, y
    )


def _with_intercept(matrix: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(matrix), dtype=float), matrix))


def _interaction_blocks(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.column_stack([x * values[:, [index]] for index in range(values.shape[1])])


@dataclass
class HierarchicalFit:
    architecture: str
    variant: str
    encoder: FeatureEncoder
    estimator: object
    feature_names: tuple[str, ...]
    known_tasks: tuple[str, ...]
    descriptor_encoder: DescriptorEncoder | None = None
    descriptor_names: tuple[str, ...] = DESCRIPTOR_NAMES
    descriptor_scale: float = 0.35
    random_effect_scale: float = 0.5
    target: str = "persistence_logit"

    @property
    def sharing(self) -> str:
        return VARIANT_NAMES[self.variant]

    @property
    def outcome_tasks(self) -> tuple[str, ...]:
        """Tasks whose outcomes contributed to this fit (leakage audit)."""

        return self.known_tasks

    def feature_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        matrix, _ = self.encoder.transform(frame)
        return _with_intercept(matrix)

    def _task_onehot(self, tasks) -> np.ndarray:
        labels = np.asarray(tasks, dtype=str)
        return np.column_stack(
            [(labels == task).astype(float) for task in self.known_tasks]
        )

    def _design(self, frame: pd.DataFrame, *, include_random=True) -> np.ndarray:
        x = self.feature_matrix(frame)
        tasks = frame.task_family.astype(str).to_numpy()
        if self.variant == "M1":
            return x
        if self.variant == "M3":
            onehot = self._task_onehot(tasks)
            random = (
                _interaction_blocks(x, onehot) * self.random_effect_scale
                if include_random
                else np.zeros((len(x), len(self.known_tasks) * x.shape[1]))
            )
            return np.column_stack((x, random))
        if self.variant == "M4":
            if self.descriptor_encoder is None:
                raise RuntimeError("M4 fit is missing its descriptor encoder")
            z = self.descriptor_encoder.transform_rows(tasks)
            ontology = _interaction_blocks(x, z) * self.descriptor_scale
            onehot = self._task_onehot(tasks)
            random = (
                _interaction_blocks(x, onehot) * self.random_effect_scale
                if include_random
                else np.zeros((len(x), len(self.known_tasks) * x.shape[1]))
            )
            return np.column_stack((x, ontology, random))
        raise ValueError("task-specific fits do not have one shared design matrix")

    def predict(self, frame: pd.DataFrame, *, include_random=True) -> np.ndarray:
        if self.variant != "M2":
            return np.asarray(
                self.estimator.predict(
                    self._design(frame, include_random=include_random)
                ),
                dtype=float,
            )
        x = self.feature_matrix(frame)
        labels = frame.task_family.astype(str).to_numpy()
        prediction = np.full(len(frame), np.nan, dtype=float)
        for task in sorted(set(labels)):
            if task not in self.estimator:
                raise ValueError(f"M2 cannot predict unseen task {task}")
            mask = labels == task
            prediction[mask] = self.estimator[task].predict(x[mask])
        return prediction

    def coefficient_components(self):
        """Return ``mu``, ``B``, and task deviations on coefficient scale."""

        p = len(self.feature_names)
        if self.variant == "M2":
            task = {name: fit.coef_.copy() for name, fit in self.estimator.items()}
            return None, None, task
        coef = np.asarray(self.estimator.coef_, dtype=float)
        mu = coef[:p]
        if self.variant == "M1":
            return mu, None, {task: np.zeros(p) for task in self.known_tasks}
        if self.variant == "M3":
            raw = coef[p:].reshape(len(self.known_tasks), p)
            return (
                mu,
                None,
                {
                    task: raw[index] * self.random_effect_scale
                    for index, task in enumerate(self.known_tasks)
                },
            )
        b_end = p + len(self.descriptor_names) * p
        raw_b = coef[p:b_end].reshape(len(self.descriptor_names), p).T
        b = raw_b * self.descriptor_scale
        raw_u = coef[b_end:].reshape(len(self.known_tasks), p)
        return (
            mu,
            b,
            {
                task: raw_u[index] * self.random_effect_scale
                for index, task in enumerate(self.known_tasks)
            },
        )

    def task_parameters(self, tasks=None, *, include_random=True) -> pd.DataFrame:
        selected = (
            self.known_tasks if tasks is None else tuple(str(task) for task in tasks)
        )
        mu, b, deviations = self.coefficient_components()
        rows = []
        if self.variant == "M2":
            for task in selected:
                if task not in deviations:
                    continue
                for index, feature in enumerate(self.feature_names):
                    rows.append(
                        {
                            "architecture": self.architecture,
                            "variant": self.variant,
                            "task_family": task,
                            "parameter": feature,
                            "mu": np.nan,
                            "ontology_effect": np.nan,
                            "random_effect": np.nan,
                            "estimate": deviations[task][index],
                        }
                    )
            return pd.DataFrame(rows)
        for task in selected:
            ontology = np.zeros_like(mu)
            if b is not None:
                z = self.descriptor_encoder.transform([task])[0]
                ontology = b @ z
            random = (
                deviations.get(task, np.zeros_like(mu))
                if include_random
                else np.zeros_like(mu)
            )
            estimate = mu + ontology + random
            for index, feature in enumerate(self.feature_names):
                rows.append(
                    {
                        "architecture": self.architecture,
                        "variant": self.variant,
                        "task_family": task,
                        "parameter": feature,
                        "mu": mu[index],
                        "ontology_effect": ontology[index],
                        "random_effect": random[index],
                        "estimate": estimate[index],
                    }
                )
        return pd.DataFrame(rows)


def fit_hierarchical_model(
    frame: pd.DataFrame,
    architecture,
    *,
    variant: str = "M4",
    target: str = "persistence_logit",
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    descriptor_scale: float = 0.35,
    random_effect_scale: float = 0.5,
) -> HierarchicalFit:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"variant must be one of {sorted(VARIANT_NAMES)}")
    if target not in frame:
        raise ValueError(f"target column is absent: {target}")
    if "task_family" not in frame:
        raise ValueError("task_family is required")
    features = (*_architecture_features(architecture), "response_mapping")
    encoder = FeatureEncoder(features)
    raw, names = encoder.fit_transform(frame.reset_index(drop=True))
    x = _with_intercept(raw)
    feature_names = ("intercept", *names)
    y = frame[target].to_numpy(dtype=float)
    tasks = frame.task_family.astype(str).to_numpy()
    known_tasks = tuple(sorted(set(tasks)))
    architecture_name = architecture if isinstance(architecture, str) else "custom"
    descriptors = None
    if variant == "M1":
        estimator = _fit_ridge(x, y, alphas)
    elif variant == "M2":
        estimator = {
            task: _fit_ridge(x[tasks == task], y[tasks == task], alphas)
            for task in known_tasks
        }
    elif variant == "M3":
        onehot = np.column_stack(
            [(tasks == task).astype(float) for task in known_tasks]
        )
        design = np.column_stack(
            (x, _interaction_blocks(x, onehot) * float(random_effect_scale))
        )
        estimator = _fit_ridge(design, y, alphas)
    else:
        descriptors = DescriptorEncoder().fit(known_tasks)
        z = descriptors.transform_rows(tasks)
        onehot = np.column_stack(
            [(tasks == task).astype(float) for task in known_tasks]
        )
        design = np.column_stack(
            (
                x,
                _interaction_blocks(x, z) * float(descriptor_scale),
                _interaction_blocks(x, onehot) * float(random_effect_scale),
            )
        )
        estimator = _fit_ridge(design, y, alphas)
    return HierarchicalFit(
        architecture=str(architecture_name),
        variant=variant,
        encoder=encoder,
        estimator=estimator,
        feature_names=feature_names,
        known_tasks=known_tasks,
        descriptor_encoder=descriptors,
        descriptor_scale=float(descriptor_scale),
        random_effect_scale=float(random_effect_scale),
        target=target,
    )

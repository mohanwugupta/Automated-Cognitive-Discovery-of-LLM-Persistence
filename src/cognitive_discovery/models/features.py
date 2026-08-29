"""Leakage-safe, explicit-missingness feature encoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
import pandas as pd

from cognitive_discovery.data.validation import assert_no_future_leakage
from cognitive_discovery.ontology.factors import ORDINAL_VALUE


def _number(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        if value in ORDINAL_VALUE:
            return float(ORDINAL_VALUE[value])
        try:
            return float(value)
        except ValueError:
            return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _history_array(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return list(value) if isinstance(value, (tuple, list, np.ndarray)) else []


def _series(frame: pd.DataFrame, name: str) -> np.ndarray:
    if "*" in name:
        left, right = name.split("*", 1)
        return _series(frame, left) * _series(frame, right)
    if name == "continuation_advantage":
        if name in frame:
            return np.asarray([_number(value) for value in frame[name]], dtype=float)
        return (
            _series(frame, "factor_continuation_value")
            - _series(frame, "factor_disengagement_value")
            - _series(frame, "factor_continuation_cost")
        )
    if name.startswith("history_action"):
        direct = frame[name] if name in frame else None
        if direct is not None:
            return np.asarray([_number(value) for value in direct], dtype=float)
        histories = frame.get("history_actions", pd.Series([()] * len(frame)))
        values = []
        for raw in histories:
            history = _history_array(raw)
            encoded = [1.0 if str(action).lower() == "continue" else -1.0 for action in history]
            if name == "history_action_1":
                values.append(encoded[-1] if encoded else float("nan"))
            else:
                values.append(
                    sum((0.7**lag) * value for lag, value in enumerate(reversed(encoded)))
                    if encoded
                    else float("nan")
                )
        return np.asarray(values, dtype=float)
    if name.startswith("history_outcome"):
        direct = frame[name] if name in frame else None
        if direct is not None:
            return np.asarray([_number(value) for value in direct], dtype=float)
        histories = frame.get("history_outcomes", pd.Series([()] * len(frame)))
        values = []
        for raw in histories:
            history = [_number(value) for value in _history_array(raw)]
            if name == "history_outcome_1":
                values.append(history[-1] if history else float("nan"))
            else:
                values.append(
                    sum((0.7**lag) * value for lag, value in enumerate(reversed(history)))
                    if history
                    else float("nan")
                )
        return np.asarray(values, dtype=float)
    if name == "latent_state":
        current = (
            _series(frame, "factor_continuation_value")
            - _series(frame, "factor_continuation_cost")
            + _series(frame, "factor_progress_evidence")
        )
        history = np.nan_to_num(_series(frame, "history_outcome_kernel"))
        return current + 0.7 * history
    if name == "response_mapping":
        if name not in frame:
            return np.full(len(frame), np.nan)
        values = []
        for raw in frame[name]:
            try:
                mapping = json.loads(raw) if isinstance(raw, str) else dict(raw)
                values.append(1.0 if str(mapping["continue"]) < str(mapping["disengage"]) else -1.0)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                values.append(float("nan"))
        return np.asarray(values, dtype=float)
    if name not in frame:
        return np.full(len(frame), np.nan)
    return np.asarray([_number(value) for value in frame[name]], dtype=float)


@dataclass
class FeatureEncoder:
    features: tuple[str, ...]
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def _raw(self, frame: pd.DataFrame):
        assert_no_future_leakage(self.features)
        columns, names = [], []
        for name in self.features:
            values = _series(frame, name)
            available = np.isfinite(values)
            columns.extend((np.where(available, values, 0.0), available.astype(float)))
            names.extend((name, f"{name}__available"))
        matrix = (
            np.column_stack(columns).astype(float)
            if columns
            else np.empty((len(frame), 0), dtype=float)
        )
        return matrix, tuple(names)

    def fit_transform(self, frame: pd.DataFrame):
        matrix, names = self._raw(frame)
        self.means = matrix.mean(axis=0) if matrix.shape[1] else np.empty(0)
        self.scales = matrix.std(axis=0) if matrix.shape[1] else np.empty(0)
        self.scales[self.scales < 1e-8] = 1.0
        return (matrix - self.means) / self.scales, names

    def transform(self, frame: pd.DataFrame):
        if self.means is None or self.scales is None:
            raise RuntimeError("feature encoder has not been fitted")
        matrix, names = self._raw(frame)
        if matrix.shape[1] != len(self.means):
            raise ValueError("feature definitions changed after fitting")
        return (matrix - self.means) / self.scales, names


def all_observable_features(frame: pd.DataFrame) -> tuple[str, ...]:
    factors = sorted(
        column
        for column in frame
        if column.startswith("factor_") and not column.startswith("factor_available_")
    )
    return tuple(
        factors
        + [
            "continuation_advantage",
            "history_action_1",
            "history_action_kernel",
            "history_outcome_1",
            "history_outcome_kernel",
            "response_mapping",
        ]
    )

"""Train-only residualization of action history against the current decision."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
import pandas as pd


DEFAULT_NUMERIC_PREDICTORS = (
    "persistence_logit",
    "outcome_history",
    "contextual_outcome_history",
    "success_evidence",
    "progress_evidence",
    "continuation_cost",
    "disengagement_value",
    "continuation_value",
)
DEFAULT_CATEGORICAL_PREDICTORS = ("task_family", "response_mapping")


@dataclass
class ActionResidualizer:
    """OLS nuisance model whose schema and fitting rows are permanently frozen."""

    numeric_predictors: tuple[str, ...] = DEFAULT_NUMERIC_PREDICTORS
    categorical_predictors: tuple[str, ...] = DEFAULT_CATEGORICAL_PREDICTORS
    target: str = "action_history"
    numeric_means_: dict[str, float] = field(default_factory=dict)
    categorical_levels_: dict[str, tuple[str, ...]] = field(default_factory=dict)
    feature_names_: tuple[str, ...] = ()
    coefficient_: np.ndarray | None = None
    fit_condition_ids_: tuple[str, ...] = ()
    fit_row_hash_: str | None = None

    def _validate_columns(self, frame: pd.DataFrame, *, include_target: bool) -> None:
        required = set(self.numeric_predictors) | set(self.categorical_predictors)
        if include_target:
            required.add(self.target)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"residualization columns are missing: {missing}")

    def _design(self, frame: pd.DataFrame, *, fitting: bool) -> np.ndarray:
        self._validate_columns(frame, include_target=False)
        columns = [np.ones(len(frame), dtype=np.float64)]
        names = ["intercept"]
        for name in self.numeric_predictors:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            missing = ~np.isfinite(values)
            if fitting:
                observed = values[~missing]
                self.numeric_means_[name] = (
                    float(observed.mean()) if len(observed) else 0.0
                )
            if name not in self.numeric_means_:
                raise RuntimeError("residualizer has not been fitted")
            filled = np.where(missing, self.numeric_means_[name], values)
            columns.extend((filled, missing.astype(float)))
            names.extend((name, f"{name}__missing"))
        for name in self.categorical_predictors:
            values = frame[name].fillna("<missing>").astype(str)
            if fitting:
                self.categorical_levels_[name] = tuple(sorted(values.unique()))
            if name not in self.categorical_levels_:
                raise RuntimeError("residualizer has not been fitted")
            # Keep all levels. np.linalg.lstsq handles the redundant intercept;
            # doing so makes the serialized transformation transparent.
            for level in self.categorical_levels_[name]:
                columns.append((values == level).to_numpy(dtype=float))
                names.append(f"{name}={level}")
        if fitting:
            self.feature_names_ = tuple(names)
        elif tuple(names) != self.feature_names_:
            raise RuntimeError("residualization design schema changed")
        return np.column_stack(columns)

    def fit(self, frame: pd.DataFrame):
        self._validate_columns(frame, include_target=True)
        if len(frame) < 2:
            raise ValueError("residualization requires at least two training rows")
        design = self._design(frame, fitting=True)
        target = pd.to_numeric(frame[self.target], errors="raise").to_numpy(dtype=float)
        self.coefficient_ = np.linalg.lstsq(design, target, rcond=None)[0]
        identifiers = (
            frame["condition_id"].astype(str).tolist()
            if "condition_id" in frame
            else [str(index) for index in frame.index]
        )
        self.fit_condition_ids_ = tuple(identifiers)
        payload = "\n".join(self.fit_condition_ids_).encode("utf-8")
        self.fit_row_hash_ = hashlib.sha256(payload).hexdigest()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.coefficient_ is None:
            raise RuntimeError("residualizer has not been fitted")
        return self._design(frame, fitting=False) @ self.coefficient_

    def residualize(self, frame: pd.DataFrame) -> np.ndarray:
        self._validate_columns(frame, include_target=True)
        observed = pd.to_numeric(frame[self.target], errors="raise").to_numpy(
            dtype=float
        )
        return observed - self.predict(frame)

    def to_dict(self) -> dict:
        if self.coefficient_ is None:
            raise RuntimeError("residualizer has not been fitted")
        return {
            "target": self.target,
            "numeric_predictors": list(self.numeric_predictors),
            "categorical_predictors": list(self.categorical_predictors),
            "numeric_means": self.numeric_means_,
            "categorical_levels": {
                key: list(value) for key, value in self.categorical_levels_.items()
            },
            "feature_names": list(self.feature_names_),
            "coefficient": self.coefficient_.tolist(),
            "fit_condition_ids": list(self.fit_condition_ids_),
            "fit_row_hash": self.fit_row_hash_,
            "fit_split": "train",
        }

    @classmethod
    def from_dict(cls, payload: dict):
        model = cls(
            numeric_predictors=tuple(payload["numeric_predictors"]),
            categorical_predictors=tuple(payload["categorical_predictors"]),
            target=str(payload["target"]),
        )
        model.numeric_means_ = {
            str(key): float(value) for key, value in payload["numeric_means"].items()
        }
        model.categorical_levels_ = {
            str(key): tuple(map(str, value))
            for key, value in payload["categorical_levels"].items()
        }
        model.feature_names_ = tuple(payload["feature_names"])
        model.coefficient_ = np.asarray(payload["coefficient"], dtype=float)
        model.fit_condition_ids_ = tuple(map(str, payload["fit_condition_ids"]))
        model.fit_row_hash_ = str(payload["fit_row_hash"])
        return model

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

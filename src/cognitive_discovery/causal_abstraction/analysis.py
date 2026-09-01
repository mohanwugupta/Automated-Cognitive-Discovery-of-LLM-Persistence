"""Inference utilities for abstraction-level causal dissociations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics

from .variables import ABSTRACTIONS, PREDICTION_COLUMNS


STABLE_METRICS = (
    "global_cfr",
    "correlation",
    "slope",
    "intercept",
    "rmse",
    "sign_accuracy",
)


def _clusters(frame: pd.DataFrame):
    column = (
        "semantic_contrast_id" if "semantic_contrast_id" in frame else "contrast_id"
    )
    return column, frame[column].astype(str)


def _resampled_indices(frame: pd.DataFrame, *, samples: int, seed: int):
    cluster_column, cluster_values = _clusters(frame)
    reset = frame.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    grouped = {}
    for task, task_part in reset.groupby("task_family"):
        ids = sorted(task_part[cluster_column].astype(str).unique())
        grouped[str(task)] = {
            value: task_part.index[
                task_part[cluster_column].astype(str) == value
            ].to_numpy()
            for value in ids
        }
    for _ in range(int(samples)):
        indices = []
        for mapping in grouped.values():
            ids = np.asarray(list(mapping), dtype=object)
            chosen = rng.choice(ids, size=len(ids), replace=True)
            indices.extend(index for value in chosen for index in mapping[value])
        yield np.asarray(indices, dtype=int)


def abstraction_metric_table(
    frame: pd.DataFrame,
    *,
    observed_column: str = "neural_counterfactual_effect",
    threshold: float = 0.10,
) -> pd.DataFrame:
    rows = []
    for abstraction in ABSTRACTIONS:
        metrics = counterfactual_metrics(
            frame[PREDICTION_COLUMNS[abstraction]],
            frame[observed_column],
            threshold=threshold,
        )
        rows.append(
            {
                "abstraction": abstraction,
                **{name: metrics[name] for name in STABLE_METRICS},
                "examples": metrics["examples"],
                "thresholded_cfr_retained": metrics["thresholded_cfr_retained"],
                "thresholded_cfr_fraction_retained": metrics[
                    "thresholded_cfr_fraction_retained"
                ],
                "thresholded_cfr_median": metrics["thresholded_cfr_median"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_abstraction_metrics(
    frame: pd.DataFrame,
    *,
    observed_column: str = "neural_counterfactual_effect",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    threshold: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = abstraction_metric_table(
        frame, observed_column=observed_column, threshold=threshold
    )
    draws = {
        (abstraction, metric): []
        for abstraction in ABSTRACTIONS
        for metric in STABLE_METRICS
    }
    reset = frame.reset_index(drop=True)
    for indices in _resampled_indices(reset, samples=samples, seed=seed):
        table = abstraction_metric_table(
            reset.iloc[indices],
            observed_column=observed_column,
            threshold=threshold,
        ).set_index("abstraction")
        for key in draws:
            value = table.loc[key[0], key[1]]
            if np.isfinite(value):
                draws[key].append(float(value))
    alpha = (1.0 - float(confidence)) / 2.0
    intervals = []
    for (abstraction, metric), values in draws.items():
        values = np.asarray(values, dtype=float)
        intervals.append(
            {
                "abstraction": abstraction,
                "metric": metric,
                "estimate": float(
                    point.loc[point.abstraction == abstraction, metric].iloc[0]
                ),
                "ci_lower": (
                    float(np.quantile(values, alpha)) if len(values) else np.nan
                ),
                "ci_upper": (
                    float(np.quantile(values, 1.0 - alpha)) if len(values) else np.nan
                ),
                "bootstrap_valid": len(values),
                "bootstrap_samples": int(samples),
            }
        )
    return point, pd.DataFrame(intervals)


def paired_abstraction_differences(
    frame: pd.DataFrame,
    winner: str,
    *,
    observed_column: str = "neural_counterfactual_effect",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Winner-minus-alternative CFR on identical cluster-bootstrap samples."""

    if winner not in ABSTRACTIONS:
        raise ValueError(f"unknown abstraction: {winner}")

    def scores(part):
        return {
            abstraction: counterfactual_metrics(
                part[PREDICTION_COLUMNS[abstraction]],
                part[observed_column],
                threshold=threshold,
            )["global_cfr"]
            for abstraction in ABSTRACTIONS
        }

    point = scores(frame)
    draws = {value: [] for value in ABSTRACTIONS if value != winner}
    reset = frame.reset_index(drop=True)
    for indices in _resampled_indices(reset, samples=samples, seed=seed):
        values = scores(reset.iloc[indices])
        for alternative in draws:
            difference = values[winner] - values[alternative]
            if np.isfinite(difference):
                draws[alternative].append(float(difference))
    alpha = (1.0 - float(confidence)) / 2.0
    rows = []
    for alternative, values in draws.items():
        values = np.asarray(values, dtype=float)
        rows.append(
            {
                "winner": winner,
                "alternative": alternative,
                "estimate_difference": point[winner] - point[alternative],
                "ci_lower": (
                    float(np.quantile(values, alpha)) if len(values) else np.nan
                ),
                "ci_upper": (
                    float(np.quantile(values, 1.0 - alpha)) if len(values) else np.nan
                ),
                "bootstrap_valid": len(values),
            }
        )
    return pd.DataFrame(rows)


def paired_cfr_difference(
    frame: pd.DataFrame,
    *,
    candidate_prediction: str,
    control_prediction: str | None = None,
    candidate_observed: str = "neural_counterfactual_effect",
    control_observed: str | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    threshold: float = 0.10,
) -> dict:
    """Paired CFR difference when either predictions or neural effects differ."""

    control_prediction = control_prediction or candidate_prediction
    control_observed = control_observed or candidate_observed

    def difference(part):
        candidate = counterfactual_metrics(
            part[candidate_prediction],
            part[candidate_observed],
            threshold=threshold,
        )["global_cfr"]
        control = counterfactual_metrics(
            part[control_prediction],
            part[control_observed],
            threshold=threshold,
        )["global_cfr"]
        return float(candidate - control)

    point = difference(frame)
    draws = []
    reset = frame.reset_index(drop=True)
    for indices in _resampled_indices(reset, samples=samples, seed=seed):
        value = difference(reset.iloc[indices])
        if np.isfinite(value):
            draws.append(value)
    draws = np.asarray(draws, dtype=float)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "estimate_difference": point,
        "ci_lower": float(np.quantile(draws, alpha)) if len(draws) else np.nan,
        "ci_upper": (float(np.quantile(draws, 1.0 - alpha)) if len(draws) else np.nan),
        "bootstrap_valid": len(draws),
        "row_identity_verified": True,
    }


def _linear_fit(train_x, train_y, test_x, test_y):
    train_x = np.asarray(train_x, dtype=float).reshape(len(train_y), -1)
    test_x = np.asarray(test_x, dtype=float).reshape(len(test_y), -1)
    train_y, test_y = np.asarray(train_y, dtype=float), np.asarray(test_y, dtype=float)
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_z, test_z = (train_x - mean) / scale, (test_x - mean) / scale
    design = np.column_stack((np.ones(len(train_z)), train_z))
    coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
    prediction = np.column_stack((np.ones(len(test_z)), test_z)) @ coefficients
    denominator = float(np.sum(np.square(test_y - test_y.mean())))
    r2 = (
        float(1.0 - np.sum(np.square(test_y - prediction)) / denominator)
        if denominator > 1e-12
        else np.nan
    )
    return coefficients, prediction, r2


def natural_model_comparison(
    frame: pd.DataFrame,
    *,
    response: str = "projection_distance",
) -> pd.DataFrame:
    train = frame[frame.pair_split == "abstraction_train"]
    test = frame[frame.pair_split == "abstraction_test"]
    if train.empty or test.empty:
        raise ValueError("natural model comparison requires frozen train/test pairs")
    rows = []
    for abstraction in ABSTRACTIONS:
        column = f"delta_{abstraction}"
        coefficients, _, r2 = _linear_fit(
            train[[column]].abs(),
            train[response],
            test[[column]].abs(),
            test[response],
        )
        rows.append(
            {
                "abstraction": abstraction,
                "r2_pair": r2,
                "intercept": coefficients[0],
                "standardized_coefficient": coefficients[1],
                "train_pairs": len(train),
                "test_pairs": len(test),
            }
        )
    return pd.DataFrame(rows).sort_values("r2_pair", ascending=False)


def factorial_regression(
    frame: pd.DataFrame,
    *,
    response: str = "projection_distance",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    predictors = [f"delta_{value}" for value in ABSTRACTIONS]
    train = frame[frame.pair_split == "abstraction_train"].copy()
    test = frame[frame.pair_split == "abstraction_test"].copy()
    coefficients, _, full_r2 = _linear_fit(
        train[predictors], train[response], test[predictors], test[response]
    )
    partial = {}
    for index, predictor in enumerate(predictors):
        reduced = [value for value in predictors if value != predictor]
        _, _, reduced_r2 = _linear_fit(
            train[reduced], train[response], test[reduced], test[response]
        )
        partial[predictor] = full_r2 - reduced_r2
    rng = np.random.default_rng(int(seed))
    draws = {predictor: [] for predictor in predictors}
    for _ in range(int(samples)):
        indices = rng.choice(len(train), size=len(train), replace=True)
        sampled = train.iloc[indices]
        values, _, _ = _linear_fit(
            sampled[predictors],
            sampled[response],
            test[predictors],
            test[response],
        )
        for index, predictor in enumerate(predictors):
            draws[predictor].append(float(values[index + 1]))
    alpha = (1.0 - confidence) / 2.0
    return pd.DataFrame(
        [
            {
                "abstraction": predictor.removeprefix("delta_"),
                "standardized_coefficient": float(coefficients[index + 1]),
                "partial_r2": partial[predictor],
                "full_heldout_r2": full_r2,
                "ci_lower": float(np.quantile(draws[predictor], alpha)),
                "ci_upper": float(np.quantile(draws[predictor], 1.0 - alpha)),
            }
            for index, predictor in enumerate(predictors)
        ]
    )


def convergence_index(frame: pd.DataFrame, *, seed: int = 0) -> dict:
    same_e = frame[frame.contrast_family == "same_total_evidence"]
    comparison = frame[frame.contrast_family != "same_total_evidence"]
    if same_e.empty or comparison.empty:
        return {"convergence_index": np.nan, "same_e_distance": np.nan}
    rng = np.random.default_rng(int(seed))
    matched = []
    for row in same_e.itertuples():
        candidates = comparison[comparison.task_family == row.task_family]
        if candidates.empty:
            candidates = comparison
        matched.append(
            float(
                candidates.iloc[
                    int(rng.integers(0, len(candidates)))
                ].projection_distance
            )
        )
    same_distance = float(same_e.projection_distance.mean())
    random_distance = float(np.mean(matched))
    return {
        "convergence_index": (
            float(1.0 - same_distance / random_distance)
            if random_distance > 1e-12
            else np.nan
        ),
        "same_e_distance": same_distance,
        "random_matched_distance": random_distance,
        "same_e_pairs": len(same_e),
    }


def identify_synthetic_abstraction(frame: pd.DataFrame) -> str:
    """Return the abstraction with maximal stable CFR in a known synthetic system."""

    table = abstraction_metric_table(frame)
    return str(table.sort_values("global_cfr", ascending=False).iloc[0].abstraction)

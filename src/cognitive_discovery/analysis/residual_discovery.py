"""Nested, stability-selected residual discovery with held-out confirmation."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from cognitive_discovery.models.features import FeatureEncoder
from cognitive_discovery.models.fitting import fit_model
from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model


CANDIDATE_INTERACTIONS = (
    "history_outcome_1*factor_progress_evidence",
    "history_outcome_kernel*factor_goal_continuity",
    "factor_continuation_cost*factor_progress_evidence",
    "factor_prior_investment*factor_goal_continuity",
    "history_action_1*factor_progress_evidence",
    "history_action_kernel*factor_environmental_stability",
)


def _group_column(frame):
    if "paired_condition_id" in frame:
        return "paired_condition_id"
    if "episode_id" in frame:
        return "episode_id"
    return None


def _nested_split(frame, seed, fraction=0.65):
    rng = np.random.default_rng(int(seed))
    column = _group_column(frame)
    groups = (
        frame[column].astype(str).to_numpy()
        if column is not None
        else np.arange(len(frame)).astype(str)
    )
    unique = np.asarray(sorted(set(groups)))
    rng.shuffle(unique)
    boundary = min(len(unique) - 1, max(1, int(float(fraction) * len(unique))))
    selection_groups = set(unique[:boundary])
    mask = np.asarray([group in selection_groups for group in groups])
    selection = frame.loc[mask].reset_index(drop=True)
    validation = frame.loc[~mask].reset_index(drop=True)
    if selection.empty or validation.empty:
        raise ValueError(
            "nested residual discovery needs at least two independent groups"
        )
    return selection, validation, column


def _group_digest(frame, column):
    values = (
        frame[column].astype(str).unique()
        if column is not None
        else frame.index.astype(str).to_numpy()
    )
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()[:16]


def discover_residuals(
    frame: pd.DataFrame,
    *,
    base_model: str,
    interactions=CANDIDATE_INTERACTIONS,
    seed: int = 0,
    bootstraps: int = 200,
    stability_threshold: float = 0.60,
    support_threshold: float = 0.95,
    base_variant: str = "fully_shared",
) -> pd.DataFrame:
    """Select on one subset and evaluate once on a disjoint subset."""

    selection, validation, group_column = _nested_split(frame, seed)
    base = (
        fit_hierarchical_model(selection, base_model, variant=base_variant)
        if base_variant in {"M1", "M2", "M3", "M4"}
        else fit_model(selection, base_model, sharing=base_variant)
    )
    selection_residual = selection.persistence_logit.to_numpy() - base.predict(
        selection
    )
    validation_residual = validation.persistence_logit.to_numpy() - base.predict(
        validation
    )
    null_validation_mse = float(
        np.mean((validation_residual - float(np.mean(selection_residual))) ** 2)
    )
    rng = np.random.default_rng(int(seed) + 13)
    selection_groups = (
        selection[group_column].astype(str).to_numpy()
        if group_column is not None
        else selection.index.astype(str).to_numpy()
    )
    unique_selection = np.asarray(sorted(set(selection_groups)))
    validation_groups = (
        validation[group_column].astype(str).to_numpy()
        if group_column is not None
        else validation.index.astype(str).to_numpy()
    )
    unique_validation = np.asarray(sorted(set(validation_groups)))

    rows = []
    for interaction in interactions:
        encoder = FeatureEncoder((interaction,))
        x_selection, names = encoder.fit_transform(selection)
        x_validation, _ = encoder.transform(validation)
        coefficients, selection_gains = [], []
        for _ in range(int(bootstraps)):
            sampled = rng.choice(
                unique_selection, size=len(unique_selection), replace=True
            )
            positions = np.concatenate(
                [np.flatnonzero(selection_groups == group) for group in sampled]
            )
            model = Ridge(alpha=0.01).fit(
                x_selection[positions], selection_residual[positions]
            )
            prediction = model.predict(x_selection[positions])
            null = float(
                np.mean(
                    (
                        selection_residual[positions]
                        - float(np.mean(selection_residual[positions]))
                    )
                    ** 2
                )
            )
            mse = float(np.mean((selection_residual[positions] - prediction) ** 2))
            coefficients.append(float(model.coef_[0]))
            selection_gains.append(1.0 - mse / null if null > 1e-12 else 0.0)
        coefficient_values = np.asarray(coefficients)
        median_sign = np.sign(np.median(coefficient_values))
        stable = np.mean(
            (np.asarray(selection_gains) > 0)
            & (np.sign(coefficient_values) == median_sign)
        )
        selected = bool(stable >= float(stability_threshold))
        heldout_gain = float("nan")
        validation_mse = float("nan")
        gain_low = gain_high = support = float("nan")
        coefficient = float(np.median(coefficient_values))
        effect_tasks = 0
        if selected:
            model = Ridge(alpha=0.01).fit(x_selection, selection_residual)
            prediction = model.predict(x_validation)
            validation_mse = float(np.mean((validation_residual - prediction) ** 2))
            heldout_gain = (
                1.0 - validation_mse / null_validation_mse
                if null_validation_mse > 1e-12
                else float("nan")
            )
            coefficient = float(model.coef_[0])
            gains = []
            for _ in range(int(bootstraps)):
                sampled = rng.choice(
                    unique_validation, size=len(unique_validation), replace=True
                )
                positions = np.concatenate(
                    [np.flatnonzero(validation_groups == group) for group in sampled]
                )
                residual = validation_residual[positions]
                predicted = prediction[positions]
                null = float(
                    np.mean((residual - float(np.mean(selection_residual))) ** 2)
                )
                mse = float(np.mean((residual - predicted) ** 2))
                gains.append(1.0 - mse / null if null > 1e-12 else float("nan"))
            gains = np.asarray(gains, dtype=float)
            gains = gains[np.isfinite(gains)]
            if len(gains):
                gain_low, gain_high = np.quantile(gains, (0.025, 0.975))
                support = float(np.mean(gains > 0))
            for _, indices in validation.groupby("task_family").groups.items():
                positions = validation.index.get_indexer(indices)
                if len(positions) >= 2 and np.std(prediction[positions]) > 1e-8:
                    effect_tasks += 1
        discovered = bool(
            selected
            and np.isfinite(heldout_gain)
            and heldout_gain > 0
            and (
                (np.isfinite(gain_low) and gain_low > 0)
                or (np.isfinite(support) and support >= float(support_threshold))
            )
        )
        rows.append(
            {
                "interaction": interaction,
                "selected_for_validation": selected,
                "stability_probability": float(stable),
                "heldout_r2_gain": heldout_gain,
                "heldout_gain_ci_low": gain_low,
                "heldout_gain_ci_high": gain_high,
                "bootstrap_positive_probability": support,
                "residual_mse": validation_mse,
                "coefficient": coefficient,
                "families": effect_tasks,
                "discovered": discovered,
                "conclusion": (
                    "discovered held-out residual structure"
                    if discovered
                    else "No tested residual interaction improved held-out prediction."
                ),
                "selection_rows": len(selection),
                "validation_rows": len(validation),
                "selection_group_digest": _group_digest(selection, group_column),
                "validation_group_digest": _group_digest(validation, group_column),
                "feature_names": ";".join(names),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["discovered", "heldout_r2_gain", "stability_probability"],
            ascending=[False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

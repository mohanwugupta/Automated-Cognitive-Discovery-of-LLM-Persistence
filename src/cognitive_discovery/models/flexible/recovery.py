"""Synthetic teacher gate required before treating flexible predictors as ceilings."""

from __future__ import annotations

import pandas as pd
import numpy as np

from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model
from cognitive_discovery.models.features import FeatureEncoder
from cognitive_discovery.models.synthetic import generate_teacher_data
from cognitive_discovery.models.fitting import fit_model

from .neural import fit_flexible_model
from .static import fit_static_flexible


def validate_flexible_ceilings(config: dict, *, smoke: bool = False) -> pd.DataFrame:
    train, test = generate_teacher_data(
        "dynamic_reevaluation",
        n_train=240 if smoke else 500,
        n_test=80 if smoke else 180,
        seed=int(config["base_seed"]) + 991,
        noise=0.01,
    )
    # Distill a fitted cognitive teacher rather than exposing the generator's
    # formula directly to the flexible models.
    teacher = fit_model(train, "dynamic_reevaluation", sharing="fully_shared")
    train = train.copy()
    test = test.copy()
    train["persistence_logit"] = teacher.predict(train)
    test["persistence_logit"] = teacher.predict(test)
    threshold = float(config["models"].get("flexible_validation_minimum_r2", 0.90))
    rows = []
    for kind in config["models"].get("flexible", []):
        if kind in {"linear_interactions", "gam"}:
            fit = fit_static_flexible(train, test, kind=kind)
            epochs = None
        else:
            settings = config["models"][kind]
            fit = fit_flexible_model(
                train,
                test,
                kind=kind,
                seed=int(config["base_seed"]) + 991,
                hidden_size=8 if smoke else min(32, int(settings["hidden_size"])),
                learning_rate=float(settings["learning_rate"]),
                max_epochs=5 if smoke else min(120, int(settings["max_epochs"])),
                patience=2 if smoke else min(15, int(settings["patience"])),
            )
            epochs = fit.selected_epochs
        rows.append(
            {
                "model": kind,
                "teacher": "fitted_dynamic_reevaluation",
                "r2": fit.metrics["r2"],
                "mse": fit.metrics["mse"],
                "correlation": fit.metrics["correlation"],
                "selected_epochs": epochs,
                "threshold": threshold,
                "passed": bool(fit.metrics["r2"] >= threshold),
                "smoke": bool(smoke),
            }
        )
    return pd.DataFrame(rows)


def validate_matched_flexible_ceilings(
    records: pd.DataFrame,
    config: dict,
    *,
    smoke: bool = False,
) -> pd.DataFrame:
    """Recover hierarchical teachers on the real Round-1 design matrix.

    This is the validity gate used by Round 2.  Targets are teacher predictions,
    so success is measured against an approximately noiseless R2 >= .98 target.
    """

    if "split" in records and {"discovery", "interpolation_test"} <= set(records.split):
        train = records[records.split == "discovery"].reset_index(drop=True)
        test = records[records.split == "interpolation_test"].reset_index(drop=True)
    else:
        boundary = max(1, int(0.75 * len(records)))
        train = records.iloc[:boundary].reset_index(drop=True)
        test = records.iloc[boundary:].reset_index(drop=True)
    teachers = {}
    for architecture in ("dual_history", "latent_context"):
        teacher = fit_hierarchical_model(train, architecture, variant="M4")
        teachers[f"fitted_{architecture}"] = (
            teacher.predict(train),
            teacher.predict(test),
        )
    encoder = FeatureEncoder(
        (
            "factor_continuation_value",
            "factor_continuation_cost",
            "factor_progress_evidence",
            "history_outcome_kernel",
        )
    )
    x_train, _ = encoder.fit_transform(train)
    x_test, _ = encoder.transform(test)
    teachers["nonlinear_interaction"] = (
        1.2 * x_train[:, 0] * x_train[:, 4] - 0.8 * x_train[:, 2] * x_train[:, 6],
        1.2 * x_test[:, 0] * x_test[:, 4] - 0.8 * x_test[:, 2] * x_test[:, 6],
    )
    sharing_modes = tuple(
        config.get("models", {}).get(
            "flexible_sharing",
            ("fully_shared", "task_specific", "hierarchical", "task_embedding"),
        )
    )
    threshold = float(config.get("models", {}).get("round2_teacher_minimum_r2", 0.98))
    rows = []
    for teacher_name, (train_target, test_target) in teachers.items():
        teacher_train, teacher_test = train.copy(), test.copy()
        teacher_train["teacher_target"] = np.asarray(train_target)
        teacher_test["teacher_target"] = np.asarray(test_target)
        for sharing in sharing_modes:
            fit = fit_static_flexible(
                teacher_train,
                teacher_test,
                kind="linear_interactions",
                sharing=sharing,
                target="teacher_target",
            )
            rows.append(
                {
                    "model": "linear_interactions",
                    "sharing": sharing,
                    "teacher": teacher_name,
                    **fit.metrics,
                    "delta_r2_from_teacher": 1.0 - fit.metrics["r2"],
                    "threshold": threshold,
                    "passed": bool(fit.metrics["r2"] >= threshold),
                    "real_round1_design": True,
                }
            )
        if not smoke:
            for kind in ("mlp", "gru"):
                settings = config["models"][kind]
                for sharing in sharing_modes:
                    neural_sharing = {
                        "task_specific": "task_specific_heads",
                        "hierarchical": "shared_trunk_task_head",
                    }.get(sharing, sharing)
                    fit = fit_flexible_model(
                        teacher_train,
                        teacher_test,
                        kind=kind,
                        target="teacher_target",
                        sharing=neural_sharing,
                        seed=int(config["base_seed"]) + 177,
                        hidden_size=int(settings["hidden_size"]),
                        learning_rate=float(settings["learning_rate"]),
                        max_epochs=int(settings["max_epochs"]),
                        patience=int(settings["patience"]),
                    )
                    rows.append(
                        {
                            "model": kind,
                            "sharing": sharing,
                            "teacher": teacher_name,
                            **fit.metrics,
                            "delta_r2_from_teacher": 1.0 - fit.metrics["r2"],
                            "threshold": threshold,
                            "passed": bool(fit.metrics["r2"] >= threshold),
                            "real_round1_design": True,
                        }
                    )
    return pd.DataFrame(rows)

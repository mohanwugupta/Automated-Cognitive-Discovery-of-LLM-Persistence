"""Synthetic teacher gate required before treating flexible predictors as ceilings."""

from __future__ import annotations

import pandas as pd

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

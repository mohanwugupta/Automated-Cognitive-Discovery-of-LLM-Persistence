"""Round-2 comparison of architecture families under M1--M4 sharing."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cognitive_discovery.hierarchy.few_shot import evaluate_few_shot_adaptation
from cognitive_discovery.hierarchy.random_effects import (
    VARIANT_NAMES,
    fit_hierarchical_model,
)
from cognitive_discovery.hierarchy.variance_decomposition import (
    parameter_sign_consistency,
    variance_decomposition,
)
from cognitive_discovery.models.fitting import regression_metrics, task_macro_metrics
from cognitive_discovery.analysis.parameter_structure import (
    bootstrap_parameter_intervals,
)


DEFAULT_ARCHITECTURES = (
    "immediate_state",
    "choice_perseveration",
    "outcome_history",
    "dual_history",
    "dynamic_reevaluation",
    "latent_context",
    "latent_motivation",
)


def compare_hierarchical_architectures(
    train: pd.DataFrame,
    tests: dict[str, pd.DataFrame],
    *,
    architectures=DEFAULT_ARCHITECTURES,
    variants=("M1", "M2", "M3", "M4"),
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
):
    rows, fits = [], {}
    for architecture in architectures:
        for variant in variants:
            fit = fit_hierarchical_model(
                train, architecture, variant=variant, alphas=alphas
            )
            fits[(architecture, variant)] = fit
            for split, test in tests.items():
                prediction = fit.predict(test)
                rows.append(
                    {
                        "architecture": architecture,
                        "variant": variant,
                        "sharing": VARIANT_NAMES[variant],
                        "split": split,
                        **regression_metrics(test.persistence_logit, prediction),
                        **task_macro_metrics(test, prediction),
                        "source_tasks": len(fit.outcome_tasks),
                    }
                )
    comparison = pd.DataFrame(rows).sort_values(
        ["split", "macro_r2", "r2"], ascending=[True, False, False]
    )
    return comparison.reset_index(drop=True), fits


def hierarchical_loto(
    frame: pd.DataFrame,
    *,
    architectures=DEFAULT_ARCHITECTURES,
    variants=("M1", "M3", "M4"),
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    sensitivity_label="with_information_sampling",
) -> pd.DataFrame:
    rows = []
    for heldout in sorted(frame.task_family.astype(str).unique()):
        source = frame[frame.task_family.astype(str) != heldout].reset_index(drop=True)
        target = frame[frame.task_family.astype(str) == heldout].reset_index(drop=True)
        for architecture in architectures:
            for variant in variants:
                fit = fit_hierarchical_model(
                    source, architecture, variant=variant, alphas=alphas
                )
                if heldout in fit.outcome_tasks:
                    raise RuntimeError("held-out task leaked into hierarchy fit")
                prediction = fit.predict(target, include_random=False)
                rows.append(
                    {
                        "architecture": architecture,
                        "variant": variant,
                        "sharing": VARIANT_NAMES[variant],
                        "heldout_task": heldout,
                        "sensitivity": sensitivity_label,
                        "source_tasks": source.task_family.nunique(),
                        "source_rows": len(source),
                        "target_rows": len(target),
                        "heldout_outcomes_used_for_population": False,
                        **regression_metrics(target.persistence_logit, prediction),
                    }
                )
    return pd.DataFrame(rows)


def paired_architecture_bootstrap(
    test: pd.DataFrame,
    prediction_a,
    prediction_b,
    *,
    architecture_a: str,
    architecture_b: str,
    bootstraps: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired group bootstrap of R2 differences, overall and per task."""

    scored = test[["task_family", "persistence_logit"]].reset_index(drop=True).copy()
    scored["prediction_a"] = np.asarray(prediction_a, dtype=float)
    scored["prediction_b"] = np.asarray(prediction_b, dtype=float)
    if "paired_condition_id" in test:
        scored["group"] = test.paired_condition_id.astype(str).to_numpy()
    elif "episode_id" in test:
        scored["group"] = test.episode_id.astype(str).to_numpy()
    else:
        scored["group"] = np.arange(len(test)).astype(str)
    rng = np.random.default_rng(int(seed))
    rows = []
    for task in ("__macro__", *sorted(scored.task_family.astype(str).unique())):
        part = scored if task == "__macro__" else scored[scored.task_family == task]
        groups = np.asarray(sorted(part.group.unique()))
        values = []
        for _ in range(int(bootstraps)):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            positions = np.concatenate(
                [np.flatnonzero(part.group.to_numpy() == group) for group in sampled]
            )
            observed = part.persistence_logit.to_numpy()[positions]
            a = part.prediction_a.to_numpy()[positions]
            b = part.prediction_b.to_numpy()[positions]
            values.append(
                regression_metrics(observed, a)["r2"]
                - regression_metrics(observed, b)["r2"]
            )
        values = np.asarray(values, dtype=float)
        rows.append(
            {
                "architecture_a": architecture_a,
                "architecture_b": architecture_b,
                "task_family": task,
                "delta_r2_mean": float(np.mean(values)),
                "delta_r2_ci_low": float(np.quantile(values, 0.025)),
                "delta_r2_ci_high": float(np.quantile(values, 0.975)),
                "probability_a_better": float(np.mean(values > 0)),
                "observationally_equivalent": bool(
                    np.quantile(values, 0.025) <= 0 <= np.quantile(values, 0.975)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_hierarchical_analysis(
    records: pd.DataFrame,
    config: dict,
    *,
    output_directory: str | Path,
    smoke: bool = False,
):
    settings = config.get("hierarchy", {})
    architectures = tuple(settings.get("architectures", DEFAULT_ARCHITECTURES))
    variants = tuple(settings.get("variants", ("M1", "M2", "M3", "M4")))
    alphas = tuple(settings.get("ridge_alphas", (0.01, 0.1, 1.0, 10.0, 100.0)))
    train = records[records.split == "discovery"].reset_index(drop=True)
    tests = {
        split: records[records.split == split].reset_index(drop=True)
        for split in ("interpolation_test", "structural_test")
    }
    comparison, fits = compare_hierarchical_architectures(
        train, tests, architectures=architectures, variants=variants, alphas=alphas
    )
    interpolation = comparison[comparison.split == "interpolation_test"]
    best = interpolation.sort_values(["macro_r2", "r2"], ascending=False).iloc[0]
    best_fit = fits[(best.architecture, best.variant)]

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    with (output / "best_hierarchical_model.pkl").open("wb") as handle:
        pickle.dump(best_fit, handle)

    parameter_fit = fits.get((str(best.architecture), "M4"), best_fit)
    parameters = parameter_fit.task_parameters()
    parameters.to_csv(output / "task_parameters.csv", index=False)
    if parameter_fit.variant == "M4":
        mu, b, _ = parameter_fit.coefficient_components()
        coefficient_rows = []
        for parameter_index, parameter in enumerate(parameter_fit.feature_names):
            for descriptor_index, descriptor in enumerate(
                parameter_fit.descriptor_names
            ):
                coefficient_rows.append(
                    {
                        "architecture": parameter_fit.architecture,
                        "parameter": parameter,
                        "descriptor": descriptor,
                        "coefficient": b[parameter_index, descriptor_index],
                    }
                )
        pd.DataFrame(coefficient_rows).to_csv(
            output / "ontology_coefficients.csv", index=False
        )
        variance_decomposition(parameter_fit).to_csv(
            output / "variance_decomposition.csv", index=False
        )
    parameter_sign_consistency(parameters).to_csv(
        output / "parameter_sign_consistency.csv", index=False
    )
    intervals = bootstrap_parameter_intervals(
        train,
        architecture=str(parameter_fit.architecture),
        bootstraps=5 if smoke else int(settings.get("parameter_bootstraps", 100)),
        seed=int(config["base_seed"]) + 313,
    )
    intervals.to_csv(output / "task_parameter_intervals.csv", index=False)

    loto = hierarchical_loto(
        train, architectures=architectures, variants=("M1", "M3", "M4"), alphas=alphas
    )
    without_information = train[
        train.task_family != "information_sampling"
    ].reset_index(drop=True)
    if without_information.task_family.nunique() > 1:
        sensitivity = hierarchical_loto(
            without_information,
            architectures=architectures,
            variants=("M1", "M3", "M4"),
            alphas=alphas,
            sensitivity_label="without_information_sampling",
        )
        loto = pd.concat([loto, sensitivity], ignore_index=True)
    loto.to_csv(output / "loto.csv", index=False)

    few_shot_architectures = tuple(
        settings.get("few_shot_architectures", ("dual_history", "latent_context"))
    )
    sample_sizes = (
        (0, 1, 4)
        if smoke
        else tuple(settings.get("few_shot_conditions", (0, 1, 4, 8, 16, 32, 64)))
    )
    few_shot = pd.concat(
        [
            evaluate_few_shot_adaptation(
                train,
                architecture=architecture,
                sample_sizes=sample_sizes,
                seed=int(config["base_seed"]),
                alphas=alphas,
            )
            for architecture in few_shot_architectures
        ],
        ignore_index=True,
    )
    few_shot.to_csv(output / "few_shot.csv", index=False)

    bootstrap_rows = []
    if {"dual_history", "latent_context"} <= set(architectures):
        for split, test in tests.items():
            dual = fits[("dual_history", "M4")].predict(test)
            context = fits[("latent_context", "M4")].predict(test)
            result = paired_architecture_bootstrap(
                test,
                dual,
                context,
                architecture_a="dual_history",
                architecture_b="latent_context",
                bootstraps=20 if smoke else int(settings.get("bootstraps", 500)),
                seed=int(config["base_seed"]),
            )
            result["split"] = split
            bootstrap_rows.extend(result.to_dict("records"))
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output / "architecture_bootstrap.csv", index=False)
    return {
        "comparison": comparison,
        "fits": fits,
        "best_fit": best_fit,
        "parameters": parameters,
        "loto": loto,
        "few_shot": few_shot,
        "bootstrap": bootstrap,
    }

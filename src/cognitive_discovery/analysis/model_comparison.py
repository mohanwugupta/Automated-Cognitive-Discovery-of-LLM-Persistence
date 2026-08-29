"""End-to-end cognitive tournament, flexible ceilings, and explainable fraction."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS
from cognitive_discovery.models.fitting import (
    compare_models,
    fit_model,
    sampled_decision_metrics,
    task_macro_metrics,
)
from cognitive_discovery.models.flexible.neural import fit_flexible_model
from cognitive_discovery.models.flexible.static import fit_static_flexible


def run_model_tournament(
    records: pd.DataFrame,
    config: dict,
    *,
    output_directory: str | Path | None = None,
    smoke: bool = False,
):
    train = records[records.split == "discovery"].reset_index(drop=True)
    tests = {
        split: records[records.split == split].reset_index(drop=True)
        for split in ("interpolation_test", "structural_test")
    }
    if train.empty or any(test.empty for test in tests.values()):
        raise ValueError("model tournament requires all three fixed partitions")
    requested = config["models"].get("cognitive", list(COGNITIVE_MODELS))
    sharing_modes = config["models"].get("sharing", ["fully_shared"])
    rows, fits = [], {}
    for sharing in sharing_modes:
        for split, test in tests.items():
            result = compare_models(train, test, models=requested, sharing=sharing)
            result["split"] = split
            rows.extend(result.to_dict("records"))
        for model in requested:
            fits[(model, sharing)] = fit_model(train, model, sharing=sharing)

    flexible_rows = []
    flexible_sharing = config["models"].get(
        "flexible_sharing",
        ["fully_shared", "task_specific", "hierarchical", "task_embedding"],
    )
    for kind in config["models"].get("flexible", []):
        if smoke and kind == "gru":
            continue
        for flexible_structure in flexible_sharing:
            neural_structure = {
                "task_specific": "task_specific_heads",
                "hierarchical": "shared_trunk_task_head",
            }.get(flexible_structure, flexible_structure)
            for split, test in tests.items():
                if kind in {"linear_interactions", "gam"}:
                    fit = fit_static_flexible(
                        train, test, kind=kind, sharing=flexible_structure
                    )
                else:
                    settings = config["models"][kind]
                    fit = fit_flexible_model(
                        train,
                        test,
                        kind=kind,
                        seed=int(config["base_seed"]),
                        hidden_size=8 if smoke else int(settings["hidden_size"]),
                        learning_rate=float(settings["learning_rate"]),
                        max_epochs=5 if smoke else int(settings["max_epochs"]),
                        patience=2 if smoke else int(settings["patience"]),
                        sharing=neural_structure,
                    )
                macro = task_macro_metrics(test, fit.prediction)
                flexible_rows.append(
                    {
                        "model": kind,
                        "sharing": flexible_structure,
                        "split": split,
                        **fit.metrics,
                        **macro,
                        **sampled_decision_metrics(test, fit.prediction),
                        "parameters": None,
                    }
                )
    comparison = pd.DataFrame([*rows, *flexible_rows])
    for split, part in comparison.groupby("split"):
        null = part[part.model == "intercept"].macro_r2.max()
        flexible = part[
            part.model.isin(config["models"].get("flexible", []))
        ].macro_r2.max()
        denominator = flexible - null
        mask = comparison.split == split
        comparison.loc[mask, "explainable_variance_fraction"] = (
            (comparison.loc[mask, "macro_r2"] - null) / denominator
            if denominator > 1e-12
            else float("nan")
        )
    cognitive = comparison[comparison.model.isin(requested)]
    best_name = (
        cognitive[cognitive.split == "interpolation_test"]
        .sort_values(["macro_r2", "r2"], ascending=False)
        .iloc[0]
    )
    best_fit = fits[(best_name.model, best_name.sharing)]
    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_directory / "model_comparison.csv", index=False)
        with (output_directory / "best_cognitive_model.pkl").open("wb") as handle:
            pickle.dump(best_fit, handle)
        (output_directory / "best_model.txt").write_text(
            f"{best_name.model}\t{best_name.sharing}\n", encoding="utf-8"
        )
    return (
        comparison.sort_values(["split", "macro_r2"], ascending=[True, False]),
        best_fit,
    )

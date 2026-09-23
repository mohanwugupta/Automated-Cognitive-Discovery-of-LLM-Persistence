"""Cross-task computational map M[a,t,m] and behavior-only survivor sets."""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model
from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS
from cognitive_discovery.models.flexible.neural import fit_flexible_model
from cognitive_discovery.models.flexible.static import fit_static_flexible
from cognitive_discovery.replication.survivors import select_behavioral_survivors


SHARING_VARIANTS = {
    "fully_shared": "M1", "task_specific": "M2",
    "hierarchical": "M3", "ontology_conditioned": "M4",
}


def _metrics(frame: pd.DataFrame, prediction, *, model: str, sharing: str, split: str):
    enriched = frame.assign(__prediction=np.asarray(prediction, dtype=float))
    rows = []
    for task, part in [("all", enriched), *enriched.groupby("task_family")]:
        rows.append({
            "model": model, "sharing": sharing, "split": split, "task": task,
            "mse": float(mean_squared_error(part.persistence_logit, part["__prediction"])),
            "r2": float(r2_score(part.persistence_logit, part["__prediction"])),
            "n": int(len(part)),
        })
    return rows


def fit_computational_map(
    observations: pd.DataFrame,
    config: dict,
    *,
    output: str | Path,
) -> dict:
    required = {"analysis_split", "task_family", "persistence_logit", "response_mapping"}
    missing = required - set(observations)
    if missing:
        raise ValueError(f"computational observations missing: {sorted(missing)}")
    train = observations[observations.analysis_split == "train"].reset_index(drop=True)
    selection = observations[observations.analysis_split == "selection"].reset_index(drop=True)
    test = observations[observations.analysis_split == "test"].reset_index(drop=True)
    if train.empty or selection.empty or test.empty:
        raise ValueError("computational map requires frozen train/selection/test splits")
    root = Path(output)
    frozen_root = root / "frozen_models"
    frozen_root.mkdir(parents=True, exist_ok=True)
    bank = [name for name in config["computational"]["model_bank"] if name in COGNITIVE_MODELS]
    unknown = set(config["computational"]["model_bank"]) - set(bank)
    if unknown:
        raise ValueError(f"computational implementations are absent: {sorted(unknown)}")
    rows, fits = [], {}
    for model in bank:
        for sharing in config["computational"]["sharing_structures"]:
            variant = SHARING_VARIANTS[sharing]
            fit = fit_hierarchical_model(train, model, variant=variant)
            fits[(model, sharing)] = fit
            for split, part in (("selection", selection), ("test", test)):
                rows.extend(_metrics(
                    part, fit.predict(part), model=model, sharing=sharing, split=split
                ))
    flexible_settings = config["computational"].get("flexible_settings", {})
    for model in config["computational"].get("flexible_baselines", []):
        for split, part in (("selection", selection), ("test", test)):
            if model == "linear_interactions":
                fit = fit_static_flexible(
                    train, part, kind=model, sharing="fully_shared"
                )
            elif model in {"mlp", "gru"}:
                fit = fit_flexible_model(
                    train, part, kind=model, sharing="fully_shared",
                    seed=int(config["seeds"]["computational_fit"]),
                    hidden_size=int(flexible_settings.get("hidden_size", 64)),
                    learning_rate=float(flexible_settings.get("learning_rate", .001)),
                    max_epochs=int(flexible_settings.get("max_epochs", 200)),
                    patience=int(flexible_settings.get("patience", 20)),
                )
            else:
                raise ValueError(f"unknown flexible computational baseline: {model}")
            rows.extend(_metrics(
                part, fit.prediction, model=model, sharing="fully_shared", split=split
            ))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / "parameter_sharing_comparison.csv", index=False)
    per_task = metrics[metrics.task != "all"].copy()
    per_task.to_csv(root / "per_task_metrics.csv", index=False)
    aggregate = metrics[metrics.task == "all"].copy()
    aggregate.to_csv(root / "model_comparison.csv", index=False)
    selection_rows = aggregate[
        (aggregate.split == "selection") & aggregate.model.isin(bank)
    ].sort_values(
        ["model", "mse", "sharing"]
    ).drop_duplicates("model")
    score = dict(zip(selection_rows.model, selection_rows.mse))
    survivor = select_behavioral_survivors(
        score, metric=config["computational"]["survivor_metric"],
        direction=config["computational"]["survivor_direction"],
        equivalence_rule="absolute_score_margin",
        equivalence_margin=float(config["computational"]["practical_equivalence_margin"]),
    )
    survivor["frozen_before_neural_execution"] = True
    survivor["parameterization_by_model"] = {
        row.model: row.sharing for row in selection_rows.itertuples(index=False)
        if row.model in survivor["behavioral_survivor_set"]
    }
    (root / "survivor_set.json").write_text(
        json.dumps(survivor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for model in survivor["behavioral_survivor_set"]:
        sharing = survivor["parameterization_by_model"][model]
        destination = frozen_root / model
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "model.pkl").open("wb") as handle:
            pickle.dump(fits[(model, sharing)], handle, protocol=5)
        training_groups = sorted(train.semantic_group.astype(str).unique())
        (destination / "training_groups.json").write_text(
            json.dumps(training_groups, indent=2) + "\n", encoding="utf-8"
        )
        (destination / "model_spec.json").write_text(json.dumps({
            "architecture": model, "sharing": sharing,
            "variant": SHARING_VARIANTS[sharing],
            "selection_metric": config["computational"]["survivor_metric"],
            "neural_results_used": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return survivor


def evaluate_frozen_validation(
    observations: pd.DataFrame, *, computational_root: str | Path
) -> pd.DataFrame:
    """Evaluate the frozen behavior-only survivors without refitting."""

    required = {"semantic_group", "task_family", "persistence_logit"}
    missing = required - set(observations)
    if missing:
        raise ValueError(f"final validation observations lack: {sorted(missing)}")
    root = Path(computational_root)
    survivor = json.loads((root / "survivor_set.json").read_text(encoding="utf-8"))
    rows = []
    for model in survivor["behavioral_survivor_set"]:
        model_root = root / "frozen_models" / model
        training = set(json.loads((model_root / "training_groups.json").read_text(encoding="utf-8")))
        overlap = training & set(observations.semantic_group.astype(str))
        if overlap:
            raise ValueError("untouched final validation overlaps computational training groups")
        with (model_root / "model.pkl").open("rb") as handle:
            fit = pickle.load(handle)
        prediction = fit.predict(observations)
        rows.extend(_metrics(
            observations, prediction, model=model,
            sharing=survivor["parameterization_by_model"][model],
            split="untouched_final_validation",
        ))
    result = pd.DataFrame(rows)
    result["refit"] = False
    result["selection_used"] = False
    result.to_csv(root / "final_validation.csv", index=False)
    return result

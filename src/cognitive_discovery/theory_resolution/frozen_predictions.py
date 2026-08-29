"""Immutable model packages and prediction-only access."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cognitive_discovery.design.manifests import semantic_hash_from_row
from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS
from cognitive_discovery.sampling.theory_disagreement import (
    assert_outcome_free_candidates,
)


PREDICTION_FUNCTION = '''"""Generated prediction-only entry point for a frozen cognitive theory."""
import pickle
from pathlib import Path


def load_model(directory=None):
    root = Path(directory or Path(__file__).parent)
    with (root / "model.pkl").open("rb") as handle:
        return pickle.load(handle)


def predict(frame, directory=None):
    return load_model(directory).predict(frame)
'''


def _training_hashes(records: pd.DataFrame) -> list[str]:
    frame = records
    if "terminated" in frame and frame.terminated.astype(bool).any():
        frame = frame[frame.terminated.astype(bool)]
    return sorted({semantic_hash_from_row(row) for _, row in frame.iterrows()})


def _fit_digest(fit) -> str:
    return hashlib.sha256(pickle.dumps(fit, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def freeze_surviving_models(
    records: pd.DataFrame,
    comparison: pd.DataFrame,
    fits: dict[tuple[str, str], object],
    directory: str | Path,
    *,
    viability_delta: float = 0.03,
    candidate_architectures=("dual_history", "latent_context", "outcome_history"),
    primary_variant: str = "M4",
) -> pd.DataFrame:
    """Freeze theoretically distinct architectures within ΔR² of the best."""

    heldout = comparison[comparison.split == "interpolation_test"].copy()
    if heldout.empty:
        raise ValueError("model freeze requires interpolation-test metrics")
    global_best = float(heldout.macro_r2.max())
    architecture_best = (
        heldout.groupby("architecture", as_index=False).macro_r2.max()
    )
    allowed = set(candidate_architectures)
    architecture_best = architecture_best[
        architecture_best.architecture.astype(str).isin(allowed)
    ].copy()
    architecture_best["delta_from_best"] = global_best - architecture_best.macro_r2
    architecture_best["predictively_viable"] = (
        architecture_best.delta_from_best <= float(viability_delta)
    )
    architecture_best["distinct_interpretation"] = True
    architecture_best["survives"] = (
        architecture_best.predictively_viable & architecture_best.distinct_interpretation
    )
    survivors = architecture_best[architecture_best.survives].architecture.astype(str).tolist()
    if not survivors:
        raise RuntimeError("no theoretically distinct architecture met the freeze criterion")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    hashes = _training_hashes(records)
    for architecture in survivors:
        architecture_root = root / architecture
        architecture_root.mkdir(parents=True, exist_ok=True)
        hierarchy_rows = heldout[
            (heldout.architecture == architecture)
            & heldout.variant.isin(["M3", "M4"])
        ].sort_values(["macro_r2", "r2"], ascending=False)
        best_hierarchical = (
            str(hierarchy_rows.iloc[0].variant) if not hierarchy_rows.empty else primary_variant
        )
        variants = tuple(dict.fromkeys(("M2", best_hierarchical, primary_variant)))
        if (architecture, primary_variant) not in fits:
            raise RuntimeError(f"missing primary frozen fit: {architecture}/{primary_variant}")
        primary = fits[(architecture, primary_variant)]
        with (architecture_root / "model.pkl").open("wb") as handle:
            pickle.dump(primary, handle)
        variant_root = architecture_root / "variants"
        variant_root.mkdir(exist_ok=True)
        for variant in variants:
            if (architecture, variant) not in fits:
                continue
            with (variant_root / f"{variant}.pkl").open("wb") as handle:
                pickle.dump(fits[(architecture, variant)], handle)
        parameters = primary.task_parameters()
        parameters.to_csv(architecture_root / "parameters.csv", index=False)
        parameters.to_csv(architecture_root / "task_parameter_table.csv", index=False)
        (architecture_root / "training_condition_hashes.json").write_text(
            json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
        )
        specification = COGNITIVE_MODELS[architecture]
        (architecture_root / "model_spec.json").write_text(
            json.dumps(
                {
                    "architecture": architecture,
                    "hypothesis": specification.hypothesis,
                    "description": specification.description,
                    "features": specification.features,
                    "primary_variant": primary_variant,
                    "preserved_variants": variants,
                    "viability_delta_macro_r2": float(
                        architecture_best.loc[
                            architecture_best.architecture == architecture,
                            "delta_from_best",
                        ].iloc[0]
                    ),
                    "fit_sha256": _fit_digest(primary),
                    "training_condition_count": len(hashes),
                    "round3_outcomes_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (architecture_root / "prediction_function.py").write_text(
            PREDICTION_FUNCTION, encoding="utf-8"
        )
    architecture_best.to_csv(root / "freeze_decisions.csv", index=False)
    (root / "frozen_architectures.json").write_text(
        json.dumps(survivors, indent=2) + "\n", encoding="utf-8"
    )
    return architecture_best


def load_frozen_models(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    manifest = root / "frozen_architectures.json"
    if not manifest.exists():
        raise FileNotFoundError(f"frozen architecture manifest is absent: {manifest}")
    architectures = json.loads(manifest.read_text(encoding="utf-8"))
    models = {}
    for architecture in architectures:
        with (root / architecture / "model.pkl").open("rb") as handle:
            models[str(architecture)] = pickle.load(handle)
    return models


def predict_frozen_models(
    candidates: pd.DataFrame, directory: str | Path
) -> dict[str, np.ndarray]:
    assert_outcome_free_candidates(candidates)
    return {
        architecture: np.asarray(model.predict(candidates), dtype=float)
        for architecture, model in load_frozen_models(directory).items()
    }

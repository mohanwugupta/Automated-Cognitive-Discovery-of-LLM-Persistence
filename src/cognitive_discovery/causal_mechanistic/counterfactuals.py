"""Immutable theory bank and deterministic computational counterfactuals."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
from types import MappingProxyType

import numpy as np
import pandas as pd

from cognitive_discovery.mechanistic.targets.behavioral_targets import (
    compute_condition_targets,
)


SPLIT_NAMES = {
    "train": "mech_pair_train",
    "validation": "mech_pair_validation",
    "test": "mech_pair_test",
    "transfer": "mech_task_holdout",
}
PRIMARY_VARIABLES = ("outcome_history", "contextual_outcome_history")
CONTROL_VARIABLES = ("action_history", "generic_value")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_sha256(value) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


class FrozenTheoryBank:
    """Prediction-only access that detects any mutation of frozen model packages."""

    def __init__(self, frozen_root: str | Path):
        self.root = Path(frozen_root)
        manifest = self.root / "frozen_architectures.json"
        if not manifest.exists():
            raise FileNotFoundError(f"frozen theory manifest is absent: {manifest}")
        self.architectures = tuple(
            map(str, json.loads(manifest.read_text(encoding="utf-8")))
        )
        self._files = tuple(
            path
            for architecture in self.architectures
            for path in (
                self.root / architecture / "model.pkl",
                self.root / architecture / "model_spec.json",
                self.root / architecture / "parameters.csv",
                self.root / architecture / "training_condition_hashes.json",
            )
        )
        missing = [str(path) for path in self._files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"frozen theory files are absent: {missing}")
        self.hashes = {
            str(path.relative_to(self.root)): _sha256(path) for path in self._files
        }
        self._models = {}
        for architecture in self.architectures:
            with (self.root / architecture / "model.pkl").open("rb") as handle:
                self._models[architecture] = pickle.load(handle)
        self._model_hashes = {
            architecture: _object_sha256(model)
            for architecture, model in self._models.items()
        }

    @property
    def models(self):
        """Read-only architecture mapping; mutations are detected by the audit."""

        return MappingProxyType(self._models)

    def predict(self, architecture: str, frame: pd.DataFrame) -> np.ndarray:
        if architecture not in self._models:
            raise ValueError(f"unknown frozen architecture: {architecture}")
        prediction = np.asarray(self._models[architecture].predict(frame), dtype=float)
        self.assert_unchanged()
        return prediction

    def assert_unchanged(self) -> None:
        current = {
            str(path.relative_to(self.root)): _sha256(path) for path in self._files
        }
        changed = sorted(
            name for name in self.hashes if current[name] != self.hashes[name]
        )
        changed.extend(
            f"in_memory:{architecture}"
            for architecture, model in self._models.items()
            if _object_sha256(model) != self._model_hashes[architecture]
        )
        if changed:
            raise RuntimeError(f"frozen behavioral theory was modified: {changed}")

    def manifest(self) -> dict:
        return {
            "architectures": list(self.architectures),
            "file_sha256": self.hashes,
            "in_memory_model_sha256": self._model_hashes,
            "prediction_only": True,
        }


def condition_feature_row(condition) -> dict:
    row = {
        "condition_id": condition.condition_id,
        "task_family": condition.task_family,
        "response_mapping": json.dumps(
            condition.response_mapping.to_dict(), sort_keys=True
        ),
        "history_actions": list(condition.history.actions),
        "history_outcomes": list(condition.history.outcomes),
        "history_length": int(condition.history.length),
        "history_valence": condition.history.valence,
    }
    row.update(
        {f"factor_{name}": value for name, value in condition.semantic_factors.items()}
    )
    if condition.contextual_history:
        row.update(
            {
                f"context_{name}": value
                for name, value in condition.contextual_history.items()
            }
        )
    return row


def _counterfactual_frame(base: dict, source: dict, variable: str, subtype: str):
    counterfactual = dict(base)
    if variable == "outcome_history":
        counterfactual["history_outcomes"] = list(source["history_outcomes"])
        counterfactual["history_length"] = len(counterfactual["history_outcomes"])
    elif variable == "contextual_outcome_history":
        if subtype == "context_isolated":
            context_names = set(
                name for name in base if name.startswith("context_")
            ) | set(name for name in source if name.startswith("context_"))
            for name in context_names:
                if name in source:
                    counterfactual[name] = source[name]
                else:
                    counterfactual.pop(name, None)
        else:
            counterfactual["history_outcomes"] = list(source["history_outcomes"])
            counterfactual["history_length"] = len(counterfactual["history_outcomes"])
    elif variable == "action_history":
        counterfactual["history_actions"] = list(source["history_actions"])
        counterfactual["history_length"] = len(counterfactual["history_actions"])
    elif variable == "generic_value":
        counterfactual["factor_success_evidence"] = source.get(
            "factor_success_evidence"
        )
    else:
        raise ValueError(f"unsupported counterfactual variable: {variable}")
    return pd.DataFrame([counterfactual])


def _pair_specs(record) -> list[tuple[str, str, str]]:
    family = record.contrast_family
    split = record.condition.split
    if family == "outcome_history":
        specs = [("outcome_history", "history_isolated", "primary")]
        # Context-specific renderers do not exist for the transfer tasks. Their
        # ordinary history contrasts provide the preregistered held-out/weak-effect
        # test for O*, but are never used to learn its neural subspace.
        if split == "transfer":
            specs.append(
                ("contextual_outcome_history", "heldout_broad_history", "primary")
            )
        return specs
    if family == "contextual_history":
        return [("contextual_outcome_history", "context_isolated", "primary")]
    if family == "action_history":
        return [("action_history", "action_isolated", "specificity_control")]
    if family == "current_value_control":
        return [("generic_value", "current_value_isolated", "specificity_control")]
    return []


def _validate_pair(base, source, variable: str, subtype: str, *, tolerance=1e-10):
    base_targets = compute_condition_targets(base.condition)
    source_targets = compute_condition_targets(source.condition)
    if base.condition.task_family != source.condition.task_family:
        raise ValueError("counterfactual pair crosses task families")
    if (
        base.condition.response_mapping.mapping_id
        != source.condition.response_mapping.mapping_id
    ):
        raise ValueError("counterfactual pair crosses response mappings")
    intended = {
        "outcome_history": "outcome_history",
        "contextual_outcome_history": "contextual_outcome_history",
        "action_history": "action_history",
        "generic_value": "success_evidence",
    }[variable]
    if (
        abs(float(source_targets[intended]) - float(base_targets[intended]))
        <= tolerance
    ):
        raise ValueError(f"counterfactual pair does not change {intended}")
    controls = (
        "success_evidence",
        "progress_evidence",
        "continuation_cost",
        "disengagement_value",
        "continuation_value",
    )
    if variable != "generic_value":
        for name in controls:
            left, right = float(base_targets[name]), float(source_targets[name])
            if (
                np.isfinite(left)
                and np.isfinite(right)
                and abs(left - right) > tolerance
            ):
                raise ValueError(f"counterfactual nuisance changed: {name}")
    if (
        subtype == "context_isolated"
        and abs(
            float(source_targets["outcome_history"])
            - float(base_targets["outcome_history"])
        )
        > tolerance
    ):
        raise ValueError("context-isolated pair changed raw outcome history")
    return base_targets, source_targets


def build_counterfactual_pairs(records, bank: FrozenTheoryBank):
    """Create pair and long-form frozen-prediction tables deterministically."""

    grouped = defaultdict(list)
    for record in records:
        key = (
            record.contrast_id,
            record.condition.response_mapping.mapping_id,
        )
        grouped[key].append(record)
    pair_rows, examples = [], []
    for (contrast_id, mapping), group in sorted(grouped.items()):
        if len(group) != 2:
            continue
        source = next((record for record in group if record.contrast_member == 1), None)
        base = next((record for record in group if record.contrast_member == -1), None)
        if source is None or base is None:
            continue
        for variable, subtype, role in _pair_specs(source):
            base_targets, source_targets = _validate_pair(
                base, source, variable, subtype
            )
            base_row = condition_feature_row(base.condition)
            source_row = condition_feature_row(source.condition)
            counterfactual = _counterfactual_frame(
                base_row, source_row, variable, subtype
            )
            base_frame = pd.DataFrame([base_row])
            source_frame = pd.DataFrame([source_row])
            intended = {
                "outcome_history": "outcome_history",
                "contextual_outcome_history": "contextual_outcome_history",
                "action_history": "action_history",
                "generic_value": "success_evidence",
            }[variable]
            pair_id = f"cf:{variable}:{contrast_id}:{mapping}"
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "contrast_id": contrast_id,
                    "target_variable": variable,
                    "counterfactual_subtype": subtype,
                    "scientific_role": role,
                    "base_condition_id": base.condition.condition_id,
                    "source_condition_id": source.condition.condition_id,
                    "task_family": base.condition.task_family,
                    "response_mapping": mapping,
                    "pair_split": SPLIT_NAMES[base.condition.split],
                    "base_variable": float(base_targets[intended]),
                    "source_variable": float(source_targets[intended]),
                    "variable_delta": float(
                        source_targets[intended] - base_targets[intended]
                    ),
                    "source_base_semantically_matched": True,
                    "behavioral_validation_row": False,
                }
            )
            examples.append(
                {
                    "pair_id": pair_id,
                    "target_variable": variable,
                    "counterfactual_subtype": subtype,
                    "task_family": base.condition.task_family,
                    "pair_split": SPLIT_NAMES[base.condition.split],
                    "base": base_frame.iloc[0].to_dict(),
                    "source": source_frame.iloc[0].to_dict(),
                    "counterfactual": counterfactual.iloc[0].to_dict(),
                }
            )
    bank.assert_unchanged()
    pairs = pd.DataFrame(pair_rows).sort_values("pair_id").reset_index(drop=True)
    prediction_rows = []
    base_frame = pd.DataFrame([example["base"] for example in examples])
    source_frame = pd.DataFrame([example["source"] for example in examples])
    counterfactual_frame = pd.DataFrame(
        [example["counterfactual"] for example in examples]
    )
    for architecture in bank.architectures:
        base_predictions = bank.predict(architecture, base_frame)
        source_predictions = bank.predict(architecture, source_frame)
        cf_predictions = bank.predict(architecture, counterfactual_frame)
        repeated = bank.predict(architecture, counterfactual_frame)
        if not np.array_equal(cf_predictions, repeated):
            raise RuntimeError(
                "behavioral counterfactual recomputation is not deterministic"
            )
        for example, base_prediction, source_prediction, cf_prediction in zip(
            examples, base_predictions, source_predictions, cf_predictions
        ):
            prediction_rows.append(
                {
                    **{
                        key: example[key]
                        for key in (
                            "pair_id",
                            "target_variable",
                            "counterfactual_subtype",
                            "task_family",
                            "pair_split",
                        )
                    },
                    "theory": architecture,
                    "behavioral_base_logit": float(base_prediction),
                    "behavioral_source_logit": float(source_prediction),
                    "behavioral_counterfactual_logit": float(cf_prediction),
                    "predicted_counterfactual_effect": float(
                        cf_prediction - base_prediction
                    ),
                }
            )
    predictions = (
        pd.DataFrame(prediction_rows)
        .sort_values(["pair_id", "theory"])
        .reset_index(drop=True)
    )
    if pairs.pair_id.duplicated().any():
        raise ValueError("counterfactual pair IDs are not unique")
    return pairs, predictions


def select_neural_pairs(pairs: pd.DataFrame, predictions: pd.DataFrame, config: dict):
    """Preselect broad, balanced pairs from frozen predictions only."""

    settings = config.get("counterfactuals", {})
    per_task_split = int(settings.get("pairs_per_task_split", 6))
    primary_theories = config.get("theory_targets", {})
    scores = []
    for row in pairs.itertuples():
        theories = tuple(primary_theories.get(row.target_variable, ()))
        part = predictions[
            (predictions.pair_id == row.pair_id) & predictions.theory.isin(theories)
        ]
        effects = part.predicted_counterfactual_effect.to_numpy(dtype=float)
        scores.append(
            {
                "pair_id": row.pair_id,
                "predicted_effect_magnitude": (
                    float(np.max(np.abs(effects))) if len(effects) else 0.0
                ),
                "theory_disagreement": (
                    float(np.ptp(effects)) if len(effects) > 1 else 0.0
                ),
            }
        )
    scored = pairs.merge(pd.DataFrame(scores), on="pair_id", how="left")
    scored["selection_score"] = (
        scored.predicted_effect_magnitude + scored.theory_disagreement
    )
    scored["selected_for_neural"] = False
    eligible = scored[
        scored.scientific_role.eq("primary")
        | scored.pair_split.isin(("mech_pair_test", "mech_task_holdout"))
    ]
    for _, part in eligible.groupby(
        ["target_variable", "pair_split", "task_family", "response_mapping"]
    ):
        # Half per response mapping is achieved by grouping on mapping. The
        # deterministic sort preserves counterfactual magnitude and disagreement.
        take = max(1, per_task_split // 2)
        chosen = part.sort_values(
            ["selection_score", "pair_id"], ascending=[False, True]
        ).head(take)
        scored.loc[chosen.index, "selected_for_neural"] = True
    # Controls use a smaller fixed balanced subset for specificity tests only.
    control_take = int(settings.get("control_pairs_per_task_split", 2))
    controls = scored[scored.scientific_role.eq("specificity_control")]
    for _, part in controls.groupby(
        ["target_variable", "pair_split", "task_family", "response_mapping"]
    ):
        chosen = part.sort_values("pair_id").head(max(1, control_take // 2))
        scored.loc[chosen.index, "selected_for_neural"] = True
    return scored.sort_values("pair_id").reset_index(drop=True)

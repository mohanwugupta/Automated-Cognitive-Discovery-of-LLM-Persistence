"""Frozen computational variables and counterfactual predictions for O/O*/H/E."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cognitive_discovery.causal_mechanistic.counterfactuals import (
    FrozenTheoryBank,
    condition_feature_row,
)
from cognitive_discovery.mechanistic.targets.behavioral_targets import (
    compute_condition_targets,
)


ABSTRACTIONS = ("O", "O_star", "H", "E")
PREDICTION_COLUMNS = {
    "O": "predicted_O_effect",
    "O_star": "predicted_O_star_effect",
    "H": "predicted_H_effect",
    "E": "predicted_E_effect",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _neutral_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Set both history regressors to zero without changing availability flags."""

    output = frame.copy()
    for name in (
        "history_action_1",
        "history_action_kernel",
        "history_outcome_1",
        "history_outcome_kernel",
    ):
        output[name] = 0.0
    return output


def _replace_raw_history(base: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    output = base.copy()
    for name in ("history_actions", "history_outcomes", "history_length"):
        output[name] = source[name].to_numpy()
    return output


def _replace_contextual_history(
    base: pd.DataFrame, source: pd.DataFrame
) -> pd.DataFrame:
    """Create O* counterfactual rows while retaining the base current state."""

    output = _replace_raw_history(base, source)
    contextual = sorted(
        set(name for name in base if name.startswith("context_"))
        | set(name for name in source if name.startswith("context_"))
    )
    for name in contextual:
        if name in source:
            output[name] = source[name].to_numpy()
        elif name in output:
            output = output.drop(columns=name)
    return output


class FrozenAbstractionBank:
    """Prediction-only O/O*/H/E definitions backed by frozen behavioral models."""

    def __init__(
        self,
        frozen_root: str | Path,
        *,
        reference_architecture: str = "dual_history",
    ):
        self.root = Path(frozen_root)
        self.bank = FrozenTheoryBank(self.root)
        self.reference_architecture = str(reference_architecture)
        required = {"dual_history", "latent_context", "outcome_history"}
        if not required.issubset(self.bank.architectures):
            raise ValueError(
                "abstraction discovery requires frozen dual, contextual, and raw-history models"
            )
        if self.reference_architecture not in self.bank.architectures:
            raise ValueError("reference architecture is not in the frozen theory bank")

    def manifest(self) -> dict:
        definitions = {
            "O": "Frozen exponentially weighted raw outcome history.",
            "O_star": "Frozen cue-weighted context-relevant outcome history.",
            "A": "Frozen exponentially weighted action history.",
            "H": (
                "Reference-model prediction minus its prediction with action and "
                "outcome history regressors fixed to zero."
            ),
            "E": "Full pre-decision prediction of the frozen reference model.",
            "D": "Frozen reference persistence logit; numerically E in this parameterization.",
        }
        return {
            "reference_architecture": self.reference_architecture,
            "abstractions": list(ABSTRACTIONS),
            "definitions": definitions,
            "theory_bank": self.bank.manifest(),
            "definitions_frozen_before_neural_intervention": True,
        }

    def assert_unchanged(self) -> None:
        self.bank.assert_unchanged()

    def score_conditions(self, conditions, *, chunk_size: int = 5000) -> pd.DataFrame:
        """Calculate and freeze the PRD candidate-variable table."""

        rows = []
        conditions = list(conditions)
        for condition in conditions:
            targets = compute_condition_targets(condition)
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "task_family": condition.task_family,
                    "response_mapping": condition.response_mapping.mapping_id,
                    "O": float(targets["outcome_history"]),
                    "O_star": float(targets["contextual_outcome_history"]),
                    "A": float(targets["action_history"]),
                    "C": float(targets["continuation_cost"]),
                    "V_stop": float(targets["disengagement_value"]),
                    "P": float(targets["progress_evidence"]),
                    "S": float(targets["success_evidence"]),
                }
            )
        feature_rows = [condition_feature_row(condition) for condition in conditions]
        values = {architecture: [] for architecture in self.bank.architectures}
        history_contribution = []
        for start in range(0, len(feature_rows), int(chunk_size)):
            frame = pd.DataFrame(feature_rows[start : start + int(chunk_size)])
            for architecture in self.bank.architectures:
                values[architecture].extend(self.bank.predict(architecture, frame))
            full = self.bank.predict(self.reference_architecture, frame)
            neutral = self.bank.predict(
                self.reference_architecture, _neutral_history(frame)
            )
            history_contribution.extend(full - neutral)
        output = pd.DataFrame(rows)
        output["H"] = np.asarray(history_contribution, dtype=float)
        output["E"] = np.asarray(values[self.reference_architecture], dtype=float)
        output["D"] = output.E
        for architecture, predictions in values.items():
            output[f"D_{architecture}"] = np.asarray(predictions, dtype=float)
        self.assert_unchanged()
        return output

    def score_pairs(self, pair_rows: pd.DataFrame, records_by_id: dict) -> pd.DataFrame:
        """Generate every abstraction prediction before any neural intervention."""

        pairs = pair_rows.reset_index(drop=True).copy()
        base_conditions = [records_by_id[value] for value in pairs.base_condition_id]
        source_conditions = [
            records_by_id[value] for value in pairs.source_condition_id
        ]
        base = pd.DataFrame([condition_feature_row(value) for value in base_conditions])
        source = pd.DataFrame(
            [condition_feature_row(value) for value in source_conditions]
        )
        raw_cf = _replace_raw_history(base, source)
        contextual_cf = _replace_contextual_history(base, source)

        raw_base = self.bank.predict("outcome_history", base)
        raw_effect = self.bank.predict("outcome_history", raw_cf) - raw_base
        contextual_base = self.bank.predict("latent_context", base)
        contextual_effect = (
            self.bank.predict("latent_context", contextual_cf) - contextual_base
        )
        reference_base = self.bank.predict(self.reference_architecture, base)
        reference_source = self.bank.predict(self.reference_architecture, source)
        base_h = reference_base - self.bank.predict(
            self.reference_architecture, _neutral_history(base)
        )
        source_h = reference_source - self.bank.predict(
            self.reference_architecture, _neutral_history(source)
        )
        pairs[PREDICTION_COLUMNS["O"]] = raw_effect
        pairs[PREDICTION_COLUMNS["O_star"]] = contextual_effect
        pairs[PREDICTION_COLUMNS["H"]] = source_h - base_h
        pairs[PREDICTION_COLUMNS["E"]] = reference_source - reference_base
        pairs["behavioral_base_logit"] = reference_base
        pairs["behavioral_source_logit"] = reference_source
        pairs["computational_targets_frozen_before_neural"] = True
        self.assert_unchanged()
        return pairs

    def file_hashes(self) -> dict[str, str]:
        return {
            str(path.relative_to(self.root)): _sha256(path)
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
        }


def write_frozen_manifest(path: str | Path, manifest: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path

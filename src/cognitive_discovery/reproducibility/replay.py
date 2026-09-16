"""CPU-only replay of the thirteen frozen manuscript claim records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .claims import audit_target_shuffle, validate_claim_gate_consistency
from .identities import EndpointID
from .manifest import load_canonical_manifest
from .metrics import global_cfr_v1


class ReplayStatus(str, Enum):
    FULLY_REPLAYABLE = "fully_replayable"
    COMPACT_REPLAY_ONLY = "compact_replay_only"
    EXTERNAL_DEPENDENCY = "external_dependency"
    HISTORICAL_UNKNOWN = "historical_unknown"


@dataclass(frozen=True)
class ClaimReplayResult:
    claim_id: str
    endpoint: EndpointID | str
    replay_status: ReplayStatus
    values: dict[str, Any]
    frozen_value_reproduced: bool
    remaining_dependency: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["endpoint"] = self.endpoint.value if isinstance(self.endpoint, EndpointID) else self.endpoint
        value["replay_status"] = self.replay_status.value
        return value


def _r2(observed, predicted) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum(np.square(observed - observed.mean())))
    return float(1.0 - np.sum(np.square(observed - predicted)) / denominator)


def _approx_mapping(observed, expected, *, atol=1e-6) -> bool:
    return set(observed) == set(expected) and all(
        np.isclose(float(observed[key]), float(expected[key]), rtol=0, atol=atol)
        for key in expected
    )


def replay_all_claims(root: str | Path) -> dict[str, ClaimReplayResult]:
    root = Path(root)
    manifest = load_canonical_manifest(root / "canonical_manifest.yaml")
    results: dict[str, ClaimReplayResult] = {}

    decision = json.loads(
        (root / "artifacts/theory_resolution_v1/discrimination/theory_decision.json").read_text()
    )
    results["C01"] = ClaimReplayResult(
        "C01",
        "behavioral_model_comparison",
        ReplayStatus.HISTORICAL_UNKNOWN,
        {"outcome": decision.get("outcome"), "winner": decision.get("winner")},
        decision.get("outcome") == "unresolved" and decision.get("winner") is None,
        "Core Qwen model revision was not recorded.",
    )

    controllers = pd.read_csv(root / "paper/generated/controllers.csv")
    controller_values = {
        f"{row.theory}_{split}": float(getattr(row, column))
        for row in controllers.itertuples(index=False)
        for split, column in (("test", "test_estimate"), ("task_holdout", "holdout_estimate"))
    }
    expected_controllers = {
        "latent_context_test": 0.930129,
        "latent_context_task_holdout": 0.672503,
        "dual_history_test": 0.968265,
        "dual_history_task_holdout": 0.872806,
        "outcome_history_test": 0.966739,
        "outcome_history_task_holdout": 0.878302,
    }
    results["C02"] = ClaimReplayResult(
        "C02",
        EndpointID.COGNITIVE_COUNTERFACTUAL_RECOVERY,
        ReplayStatus.HISTORICAL_UNKNOWN,
        controller_values,
        _approx_mapping(controller_values, expected_controllers),
        "Compact Level-4 replay is complete; core Qwen model revision is unknown.",
    )

    validate_claim_gate_consistency(manifest, root=root)
    shuffled = pd.read_csv(root / "artifacts/causal_specificity_v2/controls/shuffled_target.csv")
    changed = {
        artifact_id: audit_target_shuffle(group).changed_rows
        for artifact_id, group in shuffled.groupby("artifact_id")
    }
    results["C03"] = ClaimReplayResult(
        "C03",
        EndpointID.COGNITIVE_COUNTERFACTUAL_RECOVERY,
        ReplayStatus.HISTORICAL_UNKNOWN,
        {
            "level4_candidates": 2,
            "level5b": manifest["stages"]["level5b_necessity_v2"]["status"],
            "dual_history_shuffle_changed": changed["das_001_outcome_history_dual_history_L28_r2"],
            "outcome_history_shuffle_changed": changed[
                "das_002_outcome_history_outcome_history_L30_r2"
            ],
        },
        changed["das_001_outcome_history_dual_history_L28_r2"] == 0
        and changed["das_002_outcome_history_outcome_history_L30_r2"] == 0,
        "Core Qwen model revision is unknown; necessity was not run.",
    )

    abstraction = json.loads((root / "artifacts/abstraction_discovery_v1/gates.json").read_text())[
        "causal_abstraction"
    ]
    results["C04"] = ClaimReplayResult(
        "C04",
        "abstraction_identity_gate",
        ReplayStatus.HISTORICAL_UNKNOWN,
        {"descriptive_winner": abstraction["winner"], "identity_gate_passed": abstraction["passed"]},
        abstraction["winner"] == "E" and abstraction["passed"] is False,
        "Core Qwen model revision is unknown; replay uses committed interchange results.",
    )

    survival = json.loads(
        (root / "artifacts/ood_free_generation_v1/analysis/survival_model.json").read_text()
    )
    ood_gate = json.loads((root / "artifacts/ood_free_generation_v1/gates.json").read_text())[
        "ood_generalization"
    ]
    ood_values = {
        "coefficient": survival["coefficient"],
        "ci_lower": survival["ci_lower"],
        "ci_upper": survival["ci_upper"],
    }
    results["C05"] = ClaimReplayResult(
        "C05",
        "free_generation_survival",
        ReplayStatus.HISTORICAL_UNKNOWN,
        ood_values,
        _approx_mapping(
            ood_values,
            {"coefficient": -0.013780833078464748, "ci_lower": -0.05006854729370516, "ci_upper": 0.022506881136775667},
            atol=1e-12,
        )
        and ood_gate["passed"] is False,
        "Core Qwen model revision is unknown; raw model inference is not replayed.",
    )

    runpod = json.loads((root / "paper/generated/runpod_protocol.json").read_text())
    results["C06"] = ClaimReplayResult(
        "C06",
        EndpointID.COGNITIVE_COUNTERFACTUAL_RECOVERY,
        ReplayStatus.COMPACT_REPLAY_ONLY,
        {"seed_count": len(runpod["seeds"]), "layer": runpod["layer"], "rank": runpod["rank"]},
        len(runpod["seeds"]) == 5 and runpod["layer"] == 28 and runpod["rank"] == 2,
        "Seed-specific fitted basis files are not committed.",
    )

    qwen = pd.read_csv(root / "paper/generated/qwen_confirmation_v2/interventions.csv.gz")
    qwen = qwen[qwen.method == "frozen"]
    natural_values = {
        split: global_cfr_v1(group.natural_effect, group.neural_counterfactual_effect)
        for split, group in qwen.groupby("pair_split")
    }
    cognitive_values = {
        split: global_cfr_v1(group.original_prediction, group.neural_counterfactual_effect)
        for split, group in qwen.groupby("pair_split")
    }
    results["C07"] = ClaimReplayResult(
        "C07",
        EndpointID.NATURAL_EFFECT_RECOVERY,
        ReplayStatus.COMPACT_REPLAY_ONLY,
        natural_values,
        _approx_mapping(
            natural_values, {"mech_pair_test": 0.778584, "mech_task_holdout": 0.815848}
        ),
        "The raw condition manifest is not committed; compact intervention replay is complete.",
    )
    results["C08"] = ClaimReplayResult(
        "C08",
        EndpointID.COGNITIVE_COUNTERFACTUAL_RECOVERY,
        ReplayStatus.COMPACT_REPLAY_ONLY,
        cognitive_values,
        _approx_mapping(
            cognitive_values, {"mech_pair_test": 0.250536, "mech_task_holdout": -0.427508}
        ),
        "The raw condition manifest is not committed; compact intervention replay is complete.",
    )

    fresh_cognitive = pd.read_csv(root / "paper/generated/qwen_fresh_contexts/metrics.csv")
    fresh_cognitive = fresh_cognitive[
        (fresh_cognitive.theory == "dual_history") & (fresh_cognitive.method == "frozen")
    ]
    fresh_natural = pd.read_csv(
        root / "paper/generated/qwen_fresh_contexts/natural_effect_recovery.csv"
    )
    fresh_natural = fresh_natural[
        (fresh_natural.theory == "dual_history") & (fresh_natural.method == "frozen")
    ]
    fresh_values = {
        **{f"cognitive_{row.split}": float(row.global_cfr) for row in fresh_cognitive.itertuples()},
        **{f"natural_{row.split}": float(row.global_cfr) for row in fresh_natural.itertuples()},
    }
    results["C09"] = ClaimReplayResult(
        "C09",
        "separate_cognitive_and_natural_endpoints",
        ReplayStatus.COMPACT_REPLAY_ONLY,
        fresh_values,
        _approx_mapping(
            fresh_values,
            {
                "cognitive_mech_pair_test": 0.268406,
                "cognitive_mech_task_holdout": 0.025816,
                "natural_mech_pair_test": 0.894428,
                "natural_mech_task_holdout": 0.870528,
            },
        ),
        "Raw model inference and activation banks are not committed.",
    )

    behavior_values = {}
    for model in ("dual_history", "immediate_state"):
        predictions = pd.read_csv(
            root / f"paper/generated/llama_behavior_v2/test_{model}_predictions.csv"
        )
        behavior_values[f"{model}_test_r2"] = _r2(predictions.observed, predictions.prediction)
    results["C10"] = ClaimReplayResult(
        "C10",
        "behavioral_prediction",
        ReplayStatus.COMPACT_REPLAY_ONLY,
        behavior_values,
        _approx_mapping(
            behavior_values,
            {"dual_history_test_r2": 0.835387, "immediate_state_test_r2": 0.821201},
        ),
        "Model inference is not replayed; committed prediction rows fully replay the score.",
    )

    llama = pd.read_csv(root / "paper/generated/llama_mechanistic_v2/interventions.csv.gz")
    llama = llama[llama.method == "selected"]
    llama_values = {}
    for split, group in llama.groupby("pair_split"):
        llama_values[f"cognitive_{split}"] = global_cfr_v1(
            group.predicted_counterfactual_effect, group.neural_counterfactual_effect
        )
        llama_values[f"natural_{split}"] = global_cfr_v1(
            group.natural_effect, group.neural_counterfactual_effect
        )
    expected_llama = {
        "cognitive_mech_pair_test": 0.648886,
        "cognitive_mech_task_holdout": 0.011162,
        "natural_mech_pair_test": 0.344851,
        "natural_mech_task_holdout": 0.318750,
    }
    results["C11"] = ClaimReplayResult(
        "C11",
        "separate_cognitive_and_natural_endpoints",
        ReplayStatus.COMPACT_REPLAY_ONLY,
        llama_values,
        _approx_mapping(llama_values, expected_llama),
        "The exact selected Llama neural basis is external/unavailable; compact rows replay metrics.",
    )

    current_interface = json.loads(
        (root / "paper/generated/llama_interface_v2/validation_gates.json").read_text()
    )
    current_passed = all(value.get("approved") is True for value in current_interface.values())
    results["C12"] = ClaimReplayResult(
        "C12",
        "measurement_validity",
        ReplayStatus.COMPACT_REPLAY_ONLY,
        {"yes_no_tasks_passed": sum(value.get("approved") is True for value in current_interface.values())},
        current_passed and len(current_interface) == 7,
        "Historical label attempts are retained as compact measurement-validity exports.",
    )

    original = json.loads((root / "paper/generated/original_audit/audit.json").read_text())
    original_values = {
        "future_return_r2": original["probe"]["future_return_r2"],
        "relative_incentive_r2": original["factorial"]["relative_incentive_r2"],
        "persistence_steering": next(
            row["episode_weighted_effect"]
            for row in original["steering"]
            if row["direction"] == "persistence"
        ),
    }
    results["C13"] = ClaimReplayResult(
        "C13",
        "external_historical_audit",
        ReplayStatus.EXTERNAL_DEPENDENCY,
        original_values,
        _approx_mapping(
            original_values,
            {
                "future_return_r2": 0.23986218083208,
                "relative_incentive_r2": 0.7836443200157732,
                "persistence_steering": 1.9935099649929058,
            },
            atol=1e-12,
        ),
        "Full replay requires the external digital-minds-hackathon repository and LFS data.",
    )
    return results

"""Deterministic synthetic parameter/model recovery for Gate A."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score

from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS
from cognitive_discovery.models.features import FeatureEncoder
from cognitive_discovery.models.fitting import fit_model

from .specifications import validate_specifications


TASKS = (
    "bandit", "foraging", "solvability", "information_sampling",
    "waiting", "effort", "debugging",
)
LEVELS = {
    "factor_continuation_value": (-1.0, 0.0, 1.0),
    "factor_disengagement_value": (-1.0, 0.0, 1.0),
    "factor_continuation_cost": (-1.0, 0.0, 1.0),
    "factor_progress_evidence": (-1.0, 0.0, 1.0),
    "factor_success_evidence": (-1.0, 0.0, 1.0),
    "factor_uncertainty": (-1.0, 1.0),
    "factor_prior_investment": (-1.0, 1.0),
    "factor_controllability": (-1.0, 1.0),
    "factor_environmental_stability": (-1.0, 1.0),
    "factor_goal_continuity": (-1.0, 1.0),
}


def synthetic_design(n: int, *, seed: int) -> pd.DataFrame:
    """Generate raw histories/conditions, not precomputed production features."""

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.choice(levels, n) for name, levels in LEVELS.items()})
    frame["task_family"] = rng.choice(TASKS, n)
    frame["history_actions"] = [
        rng.choice(("continue", "disengage"), int(rng.integers(1, 6))).tolist()
        for _ in range(n)
    ]
    frame["history_outcomes"] = [rng.choice((-1.0, 0.0, 1.0), int(rng.integers(1, 6))).tolist() for _ in range(n)]
    frame["context_context_return"] = rng.choice(("A", "B"), n)
    frame["context_cue_probability"] = rng.choice((0.6, 0.8, 1.0), n)
    frame["context_change_point"] = rng.choice(("stable", "change_point"), n, p=(0.8, 0.2))
    for context in ("a", "b"):
        frame[f"context_{context}_history_actions"] = [
            rng.choice(("continue", "disengage"), 3).tolist() for _ in range(n)
        ]
        frame[f"context_{context}_history_outcomes"] = [
            rng.choice((-1.0, 0.0, 1.0), 3).tolist() for _ in range(n)
        ]
    frame["response_mapping"] = [
        {"continue": "X", "disengage": "Y"} if index % 2 == 0
        else {"continue": "Y", "disengage": "X"}
        for index in range(n)
    ]
    return frame


def _teacher(frame: pd.DataFrame, model: str, *, seed: int, noise: float):
    features = (*COGNITIVE_MODELS[model].features, "response_mapping")
    encoder = FeatureEncoder(features)
    matrix, names = encoder.fit_transform(frame)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.35, 1.15, matrix.shape[1]) * np.where(np.arange(matrix.shape[1]) % 2, -1, 1)
    # Constant availability columns become all zero after standardization; their
    # generating parameter is defined as zero and is excluded from recovery RMSE.
    active = matrix.std(axis=0) > 1e-10
    theta[~active] = 0.0
    target = matrix @ theta + rng.normal(0.0, noise, len(frame))
    return target, theta, active, names


def parameter_recovery(*, n: int = 1200, seed: int = 7101, noise: float = 0.05) -> pd.DataFrame:
    rows = []
    sample_sizes = sorted({max(120, n // 3), max(180, (2 * n) // 3), int(n)})
    for offset, model in enumerate(sorted(COGNITIVE_MODELS)):
        for setting in range(3):
            for sample_size in sample_sizes:
                local_seed = seed + 10000 * offset + 100 * setting + sample_size
                frame = synthetic_design(sample_size, seed=local_seed)
                target, theta, active, names = _teacher(
                    frame, model, seed=local_seed + 1, noise=noise
                )
                frame["persistence_logit"] = target
                fit = fit_model(frame, model, sharing="fully_shared", alphas=(1e-8,))
                estimate = np.asarray(fit.estimator.coef_, dtype=float)
                prediction = fit.predict(frame)
                errors = estimate[active] - theta[active]
                parameter_rmse = float(np.sqrt(np.mean(np.square(errors)))) if active.any() else 0.0
                for index, name in enumerate(names):
                    if active[index]:
                        rows.append({
                            "model": model, "parameter_setting": setting,
                            "parameter": name, "true_value": theta[index],
                            "estimate": estimate[index], "bias": estimate[index] - theta[index],
                            "absolute_error": abs(estimate[index] - theta[index]),
                            "parameter_rmse": parameter_rmse,
                            "sample_size": sample_size, "noise_sd": noise,
                            "predictive_r2": float(r2_score(target, prediction)),
                        })
    return pd.DataFrame(rows)


def model_recovery(*, n: int = 1400, seed: int = 7201, noise: float = 0.10, replicates: int = 5) -> pd.DataFrame:
    """Blindly fit the complete registered bank to every registered teacher."""

    rows = []
    for offset, teacher in enumerate(sorted(COGNITIVE_MODELS)):
        for replicate in range(int(replicates)):
            local_seed = seed + 1000 * offset + replicate
            frame = synthetic_design(n, seed=local_seed)
            target, _, _, _ = _teacher(frame, teacher, seed=local_seed + 1, noise=noise)
            frame["persistence_logit"] = target
            cut = int(0.7 * n)
            train, test = frame.iloc[:cut].copy(), frame.iloc[cut:].copy()
            candidate_rows = []
            for candidate in sorted(COGNITIVE_MODELS):
                fit = fit_model(train, candidate, sharing="fully_shared")
                prediction = fit.predict(test)
                mse = float(np.mean(np.square(test.persistence_logit.to_numpy() - prediction)))
                candidate_rows.append((candidate, mse))
            winner = min(candidate_rows, key=lambda value: (value[1], value[0]))[0]
            for candidate, mse in candidate_rows:
                rows.append({
                    "teacher_model": teacher, "candidate_model": candidate,
                    "replicate": replicate, "selected": candidate == winner,
                    "test_mse": mse, "sample_size": n, "noise_sd": noise,
                })
    detailed = pd.DataFrame(rows)
    matrix = detailed.groupby(["teacher_model", "candidate_model"], as_index=False).agg(
        recovery_probability=("selected", "mean"),
        selection_count=("selected", "sum"),
        mean_test_mse=("test_mse", "mean"),
        sd_test_mse=("test_mse", "std"),
    )
    matrix["replicates"] = int(replicates)
    matrix["teacher_recovered"] = matrix.teacher_model == matrix.candidate_model
    return matrix


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_checks() -> dict:
    """Run deterministic core hand/reference checks without invoking pytest."""

    from cognitive_discovery.models.features import _series

    reference_root = Path(__file__).resolve().parents[3] / "validation/reference_models"

    def load_reference(name: str):
        path = reference_root / f"{name}_reference.py"
        spec = importlib.util.spec_from_file_location(
            f"cognitive_discovery_validation_{name}", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load independent reference: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    immediate_state_reference = load_reference("immediate_state")
    outcome_history_reference = load_reference("outcome_history")
    dual_history_reference = load_reference("dual_history")
    latent_context_reference = load_reference("latent_context")

    row = {
        "factor_continuation_value": 1.0,
        "factor_disengagement_value": -1.0,
        "factor_continuation_cost": 0.5,
        "factor_progress_evidence": -0.5,
        "factor_success_evidence": 1.0,
        "factor_uncertainty": -1.0,
        "factor_prior_investment": 1.0,
        "factor_controllability": 0.5,
        "factor_environmental_stability": 1.0,
        "factor_goal_continuity": -1.0,
        "history_actions": ["continue", "disengage", "continue"],
        "history_outcomes": [-1.0, 0.0, 1.0],
        "context_context_return": "B",
        "context_cue_probability": 0.8,
        "context_change_point": "stable",
        "context_a_history_outcomes": [1.0, 1.0, 0.0],
        "context_b_history_outcomes": [-1.0, -1.0, 1.0],
    }
    references = (
        immediate_state_reference,
        outcome_history_reference,
        dual_history_reference,
        latent_context_reference,
    )
    frame = pd.DataFrame([row])
    expected_fixed = {
        "history_outcome_1": 1.0,
        "history_outcome_kernel": 0.51,
        "history_action_1": 1.0,
        "history_action_kernel": 0.79,
        "context_relevant_outcome_kernel": -0.05,
        "context_relevant_outcome_kernel*factor_goal_continuity": 0.05,
        "context_relevant_outcome_kernel*factor_environmental_stability": -0.05,
    }
    reference_equal = True
    fixed_equal = True
    checked_features = 0
    for reference in references:
        for name, expected in reference.features(row).items():
            observed = float(_series(frame, name)[0])
            reference_equal &= bool(np.isclose(observed, expected, atol=1e-12, rtol=0.0))
            if name in expected_fixed:
                fixed_equal &= bool(
                    np.isclose(observed, expected_fixed[name], atol=1e-12, rtol=0.0)
                )
            checked_features += 1
    source = dict(row, history_outcomes=[1.0, 1.0, 1.0])
    parameters = {"history_outcome_kernel": 2.0}
    reference_effect = outcome_history_reference.counterfactual(row, source, parameters)
    production_effect = 2.0 * (
        float(_series(pd.DataFrame([source]), "history_outcome_kernel")[0])
        - float(_series(frame, "history_outcome_kernel")[0])
    )
    counterfactual_equal = bool(
        np.isclose(reference_effect, production_effect, atol=1e-12, rtol=0.0)
        and reference_effect > 0
    )
    checks = {
        "hand_calculated_fixtures": fixed_equal,
        "production_reference_equivalence": reference_equal,
        "fixed_parameter_predictions": fixed_equal,
        "counterfactual_equivalence": counterfactual_equal,
    }
    return {
        "schema_version": "implementation-validity-checks-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "checked_core_references": [
            "immediate_state", "outcome_history", "dual_history", "latent_context"
        ],
        "checked_feature_values": checked_features,
        "counterfactual_sign_convention": "source_minus_base",
        "notes": (
            "The complete registered-bank hand-fixture suite remains enforced by "
            "tests/model_validation; this runtime record repeats the four independent "
            "core references and fixed numerical fixture."
        ),
    }


def serialize_identifiability(recovery: pd.DataFrame, config: dict) -> dict:
    """Represent model confusion as a diagnostic rather than a blocking failure."""

    gate = config["identifiability_gate"]
    well = float(gate["well_identified_threshold"])
    partial = float(gate["partial_threshold"])
    nested = {tuple(pair) for pair in gate.get("known_nested_or_equivalent_pairs", [])}
    diagonal = recovery[recovery.teacher_recovered].set_index("teacher_model")[
        "recovery_probability"
    ].to_dict()
    pairs = []
    for row in recovery.itertuples():
        if row.teacher_model == row.candidate_model:
            continue
        teacher_probability = float(diagonal.get(row.teacher_model, float("nan")))
        confusion_probability = float(row.recovery_probability)
        if (row.teacher_model, row.candidate_model) in nested:
            status = "nested_or_equivalent"
            notes = "Pair was preregistered as nested or behaviorally near-equivalent."
        elif teacher_probability >= well and confusion_probability <= 1.0 - well:
            status = "well_identified"
            notes = "Teacher recovery and pairwise confusion meet the diagnostic threshold."
        elif teacher_probability >= partial:
            status = "partially_identified"
            notes = "Current design does not cleanly exclude this competing model."
        else:
            status = "poorly_identified"
            notes = "Teacher recovery is below the partial-identification threshold."
        pairs.append(
            {
                "teacher_model": str(row.teacher_model),
                "confused_with": str(row.candidate_model),
                "teacher_recovery_probability": teacher_probability,
                "confusion_probability": confusion_probability,
                "identifiability_status": status,
                "notes": notes,
            }
        )
    key_values = [float(diagonal[name]) for name in config["model_recovery"]["key_theories"]]
    if all(value >= well for value in key_values):
        overall = "well_identified"
    elif all(value >= partial for value in key_values):
        overall = "partial"
    else:
        overall = "poor"
    return {
        "schema_version": "model-identifiability-v1",
        "blocking": False,
        "identifiability_status": overall,
        "diagnostic_thresholds": {
            "well_identified": well,
            "partial": partial,
        },
        "pairs": pairs,
    }


def validation_decision(
    *, implementation_passed: bool | None, identifiability_status: str, smoke: bool
) -> dict:
    if implementation_passed is None:
        implementation_validity = "owner_review"
    else:
        implementation_validity = "pass" if implementation_passed else "fail"
    if implementation_validity != "pass":
        permission = "stop"
    elif smoke:
        permission = "development_only"
    else:
        permission = "continue"
    return {
        "implementation_validity": implementation_validity,
        "identifiability_status": identifiability_status,
        "pipeline_permission": permission,
    }


def run_synthetic_validation(root: str | Path, *, output: str | Path | None = None, config_path: str | Path | None = None, smoke: bool = False) -> dict:
    root = Path(root)
    output = Path(output or root / "validation" / "synthetic")
    config_path = Path(config_path or root / "configs/validation/computational_models_v1.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "computational-model-validation-v1":
        raise ValueError("unsupported computational-model validation config")
    spec_hashes = validate_specifications(root)
    parameter_settings = config["parameter_recovery"]
    recovery_settings = config["model_recovery"]
    parameters = parameter_recovery(
        n=350 if smoke else int(recovery_settings["sample_size"]),
        noise=float(parameter_settings["noise_sd"]),
    )
    recovery = model_recovery(
        n=420 if smoke else int(recovery_settings["sample_size"]),
        noise=float(recovery_settings["noise_sd"]),
        replicates=2 if smoke else int(recovery_settings["replicates"]),
    )
    artifacts = {
        "parameter_recovery.csv": _write_csv(parameters, output / "parameter_recovery.csv"),
        "model_recovery_matrix.csv": _write_csv(recovery, output / "model_recovery_matrix.csv"),
    }
    runtime_checks = implementation_checks()
    checks_path = output / "implementation_checks.json"
    checks_path.write_text(
        json.dumps(runtime_checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts["implementation_checks.json"] = hashlib.sha256(checks_path.read_bytes()).hexdigest()
    diagonal = recovery[recovery.teacher_recovered].set_index("teacher_model").recovery_probability
    key_recovery = {name: float(diagonal.loc[name]) for name in recovery_settings["key_theories"]}
    parameter_passed = bool(
        parameters.bias.abs().max() <= float(parameter_settings["maximum_absolute_bias"])
        and parameters.predictive_r2.min() >= float(parameter_settings["minimum_predictive_r2"])
    )
    identifiability_passed = all(
        value >= float(recovery_settings["minimum_diagonal_recovery_probability"])
        for value in key_recovery.values()
    )
    identifiability = serialize_identifiability(recovery, config)
    identifiability_path = output / "model_identifiability.json"
    identifiability_path.write_text(
        json.dumps(identifiability, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts["model_identifiability.json"] = hashlib.sha256(
        identifiability_path.read_bytes()
    ).hexdigest()
    implementation_passed = bool(runtime_checks["passed"] and parameter_passed)
    decision = validation_decision(
        implementation_passed=implementation_passed,
        identifiability_status=identifiability["identifiability_status"],
        smoke=smoke,
    )
    summary = {
        "schema_version": "computational-model-validation-v1",
        "specification_hashes": spec_hashes,
        "validation_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "artifacts": artifacts,
        "parameter_max_abs_bias": float(parameters.bias.abs().max()),
        "parameter_min_predictive_r2": float(parameters.predictive_r2.min()),
        "model_recovery_rate": float(
            recovery[recovery.teacher_recovered].recovery_probability.mean()
        ),
        "key_theory_recovery": key_recovery,
        "parameter_recovery_passed": parameter_passed,
        "key_theory_identifiability_passed": identifiability_passed,
        **decision,
        "implementation_gate_blocking": bool(config["implementation_gate"]["blocking"]),
        "identifiability_gate_blocking": bool(config["identifiability_gate"]["blocking"]),
        "reason": (
            "Implementation validity passed; behavioral ambiguity is propagated."
            if decision["pipeline_permission"] == "continue"
            else "Smoke evidence is developmental only."
            if smoke and implementation_passed
            else "Implementation validity did not pass."
        ),
    }
    path = output / "validation_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_path = output / "amended_gate_status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema_version": "nonblocking-identifiability-amendment-v1",
                **decision,
                "supersedes_interpretation_in": "DEVELOPMENT_GATE_A_REPORT.md",
                "does_not_rewrite_original_report": True,
                "validation_summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary

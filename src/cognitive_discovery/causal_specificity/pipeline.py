"""Frozen-DAS reanalysis with stable CFR and matched causal-specificity tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from cognitive_discovery.causal_mechanistic.das import load_alignment, save_alignment
from cognitive_discovery.causal_mechanistic.interventions import (
    interchange_subspace,
    intervention_norm,
    remove_subspace,
)
from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics
from cognitive_discovery.causal_mechanistic.pipeline import (
    _control_subspaces,
    _forward,
    _make_runner,
    _trial,
)
from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.mechanistic.dataset.matched_conditions import (
    load_mechanistic_manifest,
)
from cognitive_discovery.mechanistic.representations.mean_difference import (
    load_directions,
)
from cognitive_discovery.mechanistic.targets.behavioral_targets import (
    compute_condition_targets,
)

from .bootstrap import (
    bootstrap_metric_intervals,
    bootstrap_vsi,
    paired_bootstrap_difference,
)
from .controls import (
    orthonormal_random_subspaces,
    pair_row_hash,
    shuffle_sources,
    shuffle_targets,
)
from .geometry import subspace_geometry


PRIMARY_VARIABLES = ("outcome_history", "contextual_outcome_history")
CROSS_VARIABLES = (*PRIMARY_VARIABLES, "action_history")
NECESSITY_FEATURES = (
    "success_evidence",
    "progress_evidence",
    "disengagement_value",
    "action_history",
    "outcome_history",
    "contextual_outcome_history",
)
UNSTABLE_LEGACY_FIELDS = {
    "mean_cfr",
    "median_cfr",
    "fraction_behavioral_effect_recovered",
}


def _stable_only(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key not in UNSTABLE_LEGACY_FIELDS
    }


def _root(config: dict, output=None) -> Path:
    return Path(output or config.get("output_root", "artifacts/causal_specificity_v2"))


def _source(config: dict, source=None) -> Path:
    return Path(source or config.get("source_root", "artifacts/causal_mech_v1"))


def _json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _record_path(root: Path, relative: str) -> Path:
    path = root / relative
    if path.exists():
        return path
    alternate = path.with_suffix(".csv.gz")
    return alternate if alternate.exists() else path


def _identifier(index: int, row: dict) -> str:
    return (
        f"das_{int(index):03d}_{row['target_variable']}_{row['theory']}"
        f"_L{int(row['layer']):02d}_r{int(row['rank'])}"
    )


def _merge_pair_metadata(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    columns = ["pair_id", "contrast_id", "base_condition_id", "source_condition_id"]
    missing = [column for column in columns if column not in frame]
    metadata = pairs[columns].drop_duplicates("pair_id")
    if missing:
        frame = frame.merge(metadata, on="pair_id", how="left", validate="many_to_one")
    if frame.contrast_id.isna().any():
        raise ValueError("counterfactual rows could not be mapped to semantic pairs")
    return frame


def _corrected_source_metrics(root: Path, source: Path, config: dict):
    (root / "corrected_metrics").mkdir(parents=True, exist_ok=True)
    pairs = read_records(_record_path(source, "counterfactuals/pair_manifest.parquet"))
    threshold = float(config.get("cfr", {}).get("threshold", 0.10))
    localization = read_records(
        _record_path(source, "localization/whole_state_patching.parquet")
    )
    localization = _merge_pair_metadata(localization, pairs)
    localization_rows = []
    for keys, part in localization.groupby(
        ["layer", "target_variable", "theory", "pair_split"]
    ):
        localization_rows.append(
            {
                "layer": int(keys[0]),
                "target_variable": keys[1],
                "theory": keys[2],
                "pair_split": keys[3],
                **_stable_only(
                    counterfactual_metrics(
                        part.predicted_counterfactual_effect,
                        part.neural_counterfactual_effect,
                        threshold=threshold,
                    )
                ),
            }
        )
    corrected = pd.DataFrame(localization_rows)
    corrected.to_csv(root / "corrected_metrics/global_cfr.csv", index=False)

    recovery = read_records(
        _record_path(source, "validation/counterfactual_recovery.parquet")
    )
    recovery = _merge_pair_metadata(recovery, pairs)
    bootstrap_config = config.get("bootstrap", {})
    samples = int(bootstrap_config.get("samples", 2000))
    confidence = float(bootstrap_config.get("confidence", 0.95))
    seed = int(config.get("seed", 73001))
    level4_rows, interval_rows = [], []
    group_columns = [
        "candidate_index",
        "target_variable",
        "theory",
        "scope",
        "layer",
        "rank",
        "pair_split",
    ]
    for group_index, (keys, part) in enumerate(recovery.groupby(group_columns)):
        point, intervals = bootstrap_metric_intervals(
            part,
            samples=samples,
            confidence=confidence,
            seed=seed + group_index,
            threshold=threshold,
        )
        identity = dict(zip(group_columns, keys))
        level4_rows.append({**identity, **_stable_only(point)})
        interval_rows.extend({**identity, **row} for row in intervals)
    level4 = pd.DataFrame(level4_rows)
    level4.to_csv(root / "corrected_metrics/calibration.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(
        root / "corrected_metrics/bootstrap_intervals.csv", index=False
    )
    return corrected, level4


def prepare_specificity_reanalysis(
    config: dict,
    *,
    source: str | Path | None = None,
    output: str | Path | None = None,
):
    """Freeze the current Level-4 state and recompute existing results with CFR_G."""

    root, source_root = _root(config, output), _source(config, source)
    root.mkdir(parents=True, exist_ok=True)
    source_gates = json.loads((source_root / "gates.json").read_text())
    if not source_gates.get("causal_representation", {}).get("passed"):
        raise RuntimeError("specificity v2 requires a completed Level-4 source run")
    if source_gates.get("causal_specificity", {}).get("passed"):
        raise RuntimeError(
            "source run already claims Level 5; v2 expects unresolved specificity"
        )
    selected = json.loads(
        (source_root / "representations/selected_alignments.json").read_text()
    )
    validated = pd.read_csv(source_root / "validation/validated_alignments.csv")
    passing_indices = set(
        validated.loc[validated.passes_level4.astype(bool), "candidate_index"].astype(
            int
        )
    )
    frozen_root = root / "frozen_models/alignments"
    frozen_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(selected):
        if index not in passing_indices:
            continue
        artifact_id = _identifier(index, row)
        source_artifact = Path(row["artifact"])
        if not source_artifact.is_absolute() and not source_artifact.exists():
            source_artifact = (
                source_root / "representations/alignments" / source_artifact.name
            )
        if not source_artifact.exists():
            raise FileNotFoundError(
                f"selected DAS alignment is absent: {source_artifact}"
            )
        frozen_artifact = frozen_root / f"{artifact_id}.safetensors"
        shutil.copy2(source_artifact, frozen_artifact)
        manifest.append(
            {
                "candidate_index": int(index),
                "artifact_id": artifact_id,
                "target_variable": row["target_variable"],
                "theory": row["theory"],
                "scope": row["scope"],
                "task_family": row.get("task_family"),
                "layer": int(row["layer"]),
                "rank": int(row["rank"]),
                "artifact": str(frozen_artifact),
                "sha256": _sha256(frozen_artifact),
                "source_artifact": str(source_artifact),
                "primary_reanalysis_retrained": False,
            }
        )
    if not manifest:
        raise RuntimeError("no Level-4 DAS candidate was available to freeze")
    if len({row["artifact_id"] for row in manifest}) != len(manifest):
        raise RuntimeError("frozen DAS artifact identifiers are not unique")
    _json(root / "frozen_models/das_manifest.json", manifest)
    shutil.copy2(
        source_root / "behavioral_handoff/model_hash.json",
        root / "frozen_models/behavioral_hash.json",
    )
    hashes = {
        relative: _sha256(source_root / relative)
        for relative in (
            "counterfactuals/pair_manifest.parquet",
            "counterfactuals/predicted_effects.parquet",
            "counterfactuals/mechanistic_conditions.jsonl",
            "localization/whole_state_patching.parquet",
            "validation/counterfactual_recovery.parquet",
            "gates.json",
        )
    }
    _json(root / "frozen_models/source_artifact_hashes.json", hashes)
    corrected, level4 = _corrected_source_metrics(root, source_root, config)
    jobs = [{**row, "job_index": index} for index, row in enumerate(manifest)]
    _json(root / "frozen_models/evaluation_jobs.json", jobs)
    metadata = {
        "protocol_version": config.get("protocol_version", "causal_specificity_v2"),
        "git_commit": _git_commit(),
        "source_root": str(source_root),
        "source_highest_evidence_level": source_gates.get("highest_evidence_level"),
        "source_neural_hypothesis": source_gates.get(
            "neural_implementation_hypothesis"
        ),
        "source_specificity_passed": False,
        "source_necessity_passed": bool(
            source_gates.get("necessity", {}).get("passed", False)
        ),
        "frozen_das_count": len(manifest),
        "primary_reanalysis_retrained": False,
        "bootstrap_unit": "semantic contrast/source-base pair",
        "response_mapping_rows_resampled_together": True,
        "full_activations_saved": False,
        "random_subspaces": int(
            config.get("controls", {}).get("random_subspaces", 500)
        ),
    }
    _json(root / "run_metadata.json", metadata)
    gates = {
        "frozen_source_state": {
            "level4_passed": True,
            "specificity_unresolved": True,
            "necessity_unresolved": True,
            "circuit_not_eligible": True,
        },
        "corrected_level3": {"completed": True, "rows": len(corrected)},
        "corrected_level4": {"status": "awaiting_v2_aggregate"},
        "level5a_specificity": {"status": "awaiting_matched_controls"},
        "level5b_necessity": {"status": "awaiting_specificity"},
        "circuit": {"status": "not_eligible"},
    }
    _json(root / "gates.json", gates)
    return {
        "frozen_candidates": len(manifest),
        "corrected_localization_rows": len(corrected),
        "corrected_level4_rows": len(level4),
        "evaluation_jobs": len(jobs),
    }


def _pair_tables(source_root: Path):
    pairs = read_records(
        _record_path(source_root, "counterfactuals/pair_manifest.parquet")
    )
    predictions = read_records(
        _record_path(source_root, "counterfactuals/predicted_effects.parquet")
    )
    return pairs, predictions


def _selected_pairs(pairs, variables, splits):
    return (
        pairs[
            pairs.selected_for_neural.astype(bool)
            & pairs.target_variable.isin(tuple(variables))
            & pairs.pair_split.isin(tuple(splits))
        ]
        .sort_values("pair_id")
        .reset_index(drop=True)
    )


def _condition_score(record) -> float:
    targets = compute_condition_targets(record.condition)
    return float(
        targets["success_evidence"]
        + targets["progress_evidence"]
        + targets["continuation_value"]
        - targets["continuation_cost"]
        - targets["disengagement_value"]
    )


def _cache_endpoints(runner, condition_ids, records_by_id, layer: int):
    states, logits = {}, {}
    for condition_id in sorted(set(map(str, condition_ids))):
        result = _forward(
            runner, records_by_id[condition_id], capture_layers=(int(layer),)
        )
        states[condition_id] = result.states[int(layer)]
        logits[condition_id] = float(result.persistence_logit)
    return states, logits


def _intervention_rows(
    runner,
    basis,
    pairs,
    records_by_id,
    states,
    baseline_logits,
    *,
    layer: int,
    intervention_type: str,
    matched_norms=None,
):
    rows = []
    for pair in pairs.itertuples():
        source_state = states[pair.source_condition_id]
        required_norm = (
            float(matched_norms[pair.pair_id]) if matched_norms is not None else None
        )
        captured = {}

        def editor(
            state, source=source_state, subspace=basis, target_norm=required_norm
        ):
            edited = interchange_subspace(state, source, subspace)
            if target_norm is not None:
                difference = edited - state
                current = difference.float().norm()
                if float(current.detach().cpu()) > 1e-12:
                    edited = state + difference * (target_norm / current)
            captured["norm"] = intervention_norm(state, edited)
            return edited

        observed_logit = _forward(
            runner,
            records_by_id[pair.base_condition_id],
            layer=layer,
            editor=editor,
        ).persistence_logit
        baseline = float(baseline_logits[pair.base_condition_id])
        rows.append(
            {
                "pair_id": pair.pair_id,
                "contrast_id": pair.contrast_id,
                "base_condition_id": pair.base_condition_id,
                "source_condition_id": pair.source_condition_id,
                "target_variable": pair.target_variable,
                "counterfactual_subtype": pair.counterfactual_subtype,
                "task_family": pair.task_family,
                "response_mapping": pair.response_mapping,
                "pair_split": pair.pair_split,
                "base_persistence_logit": baseline,
                "intervened_persistence_logit": float(observed_logit),
                "neural_counterfactual_effect": float(observed_logit - baseline),
                "intervention_norm": float(captured.get("norm", np.nan)),
                "intervention_type": intervention_type,
                "layer": int(layer),
                "rank": int(np.asarray(basis).shape[1]),
            }
        )
    return pd.DataFrame(rows)


def _score_effects(effects, predictions, *, theories=None):
    selected = predictions[predictions.pair_id.isin(effects.pair_id)]
    if theories is not None:
        selected = selected[selected.theory.isin(tuple(theories))]
    scored = effects.merge(
        selected[["pair_id", "theory", "predicted_counterfactual_effect"]],
        on="pair_id",
        how="inner",
        validate="one_to_many",
    ).rename(columns={"theory": "scoring_theory"})
    return scored


def _generic_basis(source_root: Path, layer: int, hidden_size: int):
    metadata = json.loads((source_root / "run_metadata.json").read_text())
    mechanistic_root = Path(metadata["mechanistic_source_root"])
    path = mechanistic_root / "directions/generic_value_control.safetensors"
    if not path.exists():
        return np.empty((hidden_size, 0), dtype=float)
    directions = load_directions(path)
    vector = directions.get("generic_value", {}).get(int(layer))
    return (
        np.asarray(vector, dtype=float)[:, None]
        if vector is not None
        else np.empty((hidden_size, 0), dtype=float)
    )


def _condition_feature_row(record, condition_id, logit, neutralized):
    targets = compute_condition_targets(record.condition)
    return {
        "condition_id": condition_id,
        "task_family": record.condition.task_family,
        "response_mapping": record.condition.response_mapping.mapping_id,
        "persistence_logit": float(logit),
        "neutralized_persistence_logit": float(neutralized),
        **{feature: float(targets[feature]) for feature in NECESSITY_FEATURES},
    }


def evaluate_specificity_candidate(
    config: dict,
    *,
    job_index: int,
    source: str | Path | None = None,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Run all frozen-basis interventions for one candidate on one active GPU."""

    root, source_root = _root(config, output), _source(config, source)
    jobs = json.loads((root / "frozen_models/evaluation_jobs.json").read_text())
    job_index = int(job_index)
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"specificity job {job_index} is outside 0..{len(jobs) - 1}")
    candidate = jobs[job_index]
    if _sha256(Path(candidate["artifact"])) != candidate["sha256"]:
        raise RuntimeError("frozen DAS artifact changed after preparation")
    layer, rank = int(candidate["layer"]), int(candidate["rank"])
    basis = load_alignment(candidate["artifact"])
    pairs, predictions = _pair_tables(source_root)
    cross_pairs = _selected_pairs(
        pairs,
        CROSS_VARIABLES,
        ("mech_pair_test", "mech_task_holdout"),
    )
    target_test = cross_pairs[
        (cross_pairs.target_variable == candidate["target_variable"])
        & (cross_pairs.pair_split == "mech_pair_test")
    ].copy()
    if limit is not None:
        target_test = target_test.head(int(limit)).copy()
        allowed = set(target_test.pair_id)
        cross_pairs = cross_pairs[
            (cross_pairs.target_variable != candidate["target_variable"])
            | (cross_pairs.pair_id.isin(allowed))
        ].copy()
    if target_test.empty:
        raise RuntimeError(
            "frozen candidate has no untouched target-variable test pairs"
        )
    records = load_mechanistic_manifest(
        source_root / "counterfactuals/mechanistic_conditions.jsonl"
    )
    records_by_id = {record.condition.condition_id: record for record in records}
    cross_pairs["current_state_score"] = [
        _condition_score(records_by_id[value])
        for value in cross_pairs.base_condition_id
    ]
    target_test = cross_pairs[cross_pairs.pair_id.isin(target_test.pair_id)].copy()
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    condition_ids = list(cross_pairs.base_condition_id) + list(
        cross_pairs.source_condition_id
    )
    states, logits = _cache_endpoints(runner, condition_ids, records_by_id, layer)
    candidate_effects = _intervention_rows(
        runner,
        basis,
        cross_pairs,
        records_by_id,
        states,
        logits,
        layer=layer,
        intervention_type="frozen_DAS",
    )
    candidate_effects["artifact_id"] = candidate["artifact_id"]
    candidate_effects["subspace_target"] = candidate["target_variable"]
    candidate_effects["subspace_theory"] = candidate["theory"]
    cross_scored = _score_effects(candidate_effects, predictions)
    matched_effects = candidate_effects[
        candidate_effects.pair_id.isin(target_test.pair_id)
    ].copy()
    matched_scored = _score_effects(
        matched_effects, predictions, theories=(candidate["theory"],)
    ).rename(columns={"scoring_theory": "theory"})
    matched_norms = matched_effects.set_index("pair_id").intervention_norm.to_dict()

    train_pairs = _selected_pairs(
        pairs,
        (candidate["target_variable"],),
        ("mech_pair_train",),
    )
    train_ids = list(train_pairs.base_condition_id) + list(
        train_pairs.source_condition_id
    )
    train_states, train_logits = _cache_endpoints(
        runner, train_ids, records_by_id, layer
    )
    base_states = {
        condition_id: train_states[condition_id]
        for condition_id in train_pairs.base_condition_id.unique()
    }
    baseline_by_pair = {
        row.pair_id: train_logits[row.base_condition_id]
        for row in train_pairs.itertuples()
    }
    control_bases = _control_subspaces(
        base_states,
        train_pairs,
        baseline_by_pair,
        rank,
        seed=int(config.get("seed", 73001)) + job_index,
    )
    hidden_size = int(basis.shape[0])
    control_bases["generic_value"] = _generic_basis(source_root, layer, hidden_size)
    first_record = records_by_id[target_test.iloc[0].base_condition_id]
    first_trial = _trial(first_record)
    control_bases["persistence_output"] = runner.choice_output_direction(
        list(first_trial.messages),
        first_record.condition.response_mapping.labels,
        positive_label=first_record.condition.response_mapping.continue_label,
    )[:, None]
    control_bases.pop("random_reference", None)

    named_rows = []
    persistence_cross_rows = []
    for control, control_basis in control_bases.items():
        effects = _intervention_rows(
            runner,
            control_basis,
            target_test,
            records_by_id,
            states,
            logits,
            layer=layer,
            intervention_type=control,
        )
        scored = _score_effects(
            effects, predictions, theories=(candidate["theory"],)
        ).rename(columns={"scoring_theory": "theory"})
        scored["artifact_id"] = candidate["artifact_id"]
        scored["control"] = control
        named_rows.append(scored)
        if control == "persistence_state":
            persistence_effects = _intervention_rows(
                runner,
                control_basis,
                cross_pairs,
                records_by_id,
                states,
                logits,
                layer=layer,
                intervention_type="persistence_state",
            )
            persistence_effects["artifact_id"] = (
                f"{candidate['artifact_id']}__persistence_state"
            )
            persistence_effects["subspace_target"] = "persistence_state"
            persistence_effects["subspace_theory"] = "direct_persistence_control"
            persistence_cross_rows.append(
                _score_effects(persistence_effects, predictions)
            )
            save_alignment(
                root
                / "shards"
                / f"candidate_{job_index:03d}"
                / "persistence_state.safetensors",
                control_basis,
                metadata={
                    "candidate_artifact_id": candidate["artifact_id"],
                    "training_split": "mech_pair_train",
                    "target": "persistence_logit",
                    "retrained_DAS": False,
                },
            )
    named_controls = pd.concat(named_rows, ignore_index=True)

    shuffled_source_pairs = shuffle_sources(
        target_test,
        seed=int(config.get("seed", 73001)) + 1000 + job_index,
    )
    shuffled_source_effects = _intervention_rows(
        runner,
        basis,
        shuffled_source_pairs,
        records_by_id,
        states,
        logits,
        layer=layer,
        intervention_type="shuffled_source",
        matched_norms=matched_norms,
    )
    shuffled_source = _score_effects(
        shuffled_source_effects, predictions, theories=(candidate["theory"],)
    ).rename(columns={"scoring_theory": "theory"})
    shuffled_source["artifact_id"] = candidate["artifact_id"]
    shuffled_source = shuffled_source.merge(
        shuffled_source_pairs[
            [
                "pair_id",
                "original_source_condition_id",
                "shuffled_source_pair_id",
                "current_state_bin",
                "current_state_score",
            ]
        ],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    shuffled_target = shuffle_targets(
        matched_scored,
        seed=int(config.get("seed", 73001)) + 2000 + job_index,
    )
    shuffled_target["artifact_id"] = candidate["artifact_id"]
    shuffled_target["intervention_type"] = "shuffled_target"

    random_count = int(config.get("controls", {}).get("random_subspaces", 500))
    if random_count < 500:
        raise ValueError("specificity v2 requires at least 500 random subspaces")
    threshold = float(config.get("cfr", {}).get("threshold", 0.10))
    random_rows = []
    random_bases = orthonormal_random_subspaces(
        hidden_size,
        rank,
        random_count,
        seed=int(config.get("seed", 73001)) + 10007 * job_index,
    )
    for random_index, random_basis in enumerate(random_bases):
        effects = _intervention_rows(
            runner,
            random_basis,
            target_test,
            records_by_id,
            states,
            logits,
            layer=layer,
            intervention_type="random_subspace",
            matched_norms=matched_norms,
        )
        scored = _score_effects(effects, predictions, theories=(candidate["theory"],))
        metrics = counterfactual_metrics(
            scored.predicted_counterfactual_effect,
            scored.neural_counterfactual_effect,
            threshold=threshold,
        )
        random_rows.append(
            {
                "artifact_id": candidate["artifact_id"],
                "random_index": random_index,
                "layer": layer,
                "rank": rank,
                "row_set_sha256": pair_row_hash(scored.pair_id),
                "maximum_intervention_norm_error": float(
                    max(
                        abs(row.intervention_norm - matched_norms[row.pair_id])
                        for row in effects.itertuples()
                    )
                ),
                **{
                    key: metrics[key]
                    for key in (
                        "global_cfr",
                        "correlation",
                        "slope",
                        "intercept",
                        "rmse",
                        "sign_accuracy",
                        "examples",
                    )
                },
            }
        )

    shard_root = root / "shards" / f"candidate_{job_index:03d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate": write_records(
            matched_scored.to_dict("records"), shard_root / "candidate_effects.parquet"
        ),
        "named": write_records(
            named_controls.to_dict("records"), shard_root / "named_controls.parquet"
        ),
        "shuffled_source": write_records(
            shuffled_source.to_dict("records"),
            shard_root / "shuffled_source.parquet",
        ),
        "shuffled_target": write_records(
            shuffled_target.to_dict("records"),
            shard_root / "shuffled_target.parquet",
        ),
        "cross": write_records(
            pd.concat(
                [cross_scored, *persistence_cross_rows], ignore_index=True
            ).to_dict("records"),
            shard_root / "cross_variable.parquet",
        ),
    }
    pd.DataFrame(random_rows).to_csv(shard_root / "random_subspaces.csv", index=False)
    audit = {
        "artifact_id": candidate["artifact_id"],
        "frozen_alignment_sha256": candidate["sha256"],
        "primary_reanalysis_retrained": False,
        "target_test_rows": len(target_test),
        "target_test_row_sha256": pair_row_hash(target_test.pair_id),
        "random_subspaces": random_count,
        "candidate_control_row_identity": all(
            pair_row_hash(part.pair_id) == pair_row_hash(target_test.pair_id)
            for _, part in named_controls.groupby("control")
        ),
        "full_activations_saved": False,
    }
    _json(shard_root / "audit.json", audit)
    return {"job_index": job_index, "artifact_id": candidate["artifact_id"], **paths}


def evaluate_specificity_necessity(
    config: dict,
    *,
    job_index: int,
    source: str | Path | None = None,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Run task-conditioned neutralization only after Level 5A reassessment."""

    root, source_root = _root(config, output), _source(config, source)
    jobs = json.loads((root / "frozen_models/necessity_jobs.json").read_text())
    job_index = int(job_index)
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"necessity job {job_index} is outside 0..{len(jobs) - 1}")
    candidate = jobs[job_index]
    if _sha256(Path(candidate["artifact"])) != candidate["sha256"]:
        raise RuntimeError("frozen DAS artifact changed before necessity testing")
    basis = load_alignment(candidate["artifact"])
    layer, rank = int(candidate["layer"]), int(candidate["rank"])
    pairs, _ = _pair_tables(source_root)
    evaluation_pairs = _selected_pairs(
        pairs,
        CROSS_VARIABLES,
        ("mech_pair_test", "mech_task_holdout"),
    )
    train_pairs = _selected_pairs(
        pairs,
        (candidate["target_variable"],),
        ("mech_pair_train",),
    )
    if limit is not None:
        evaluation_pairs = evaluation_pairs.head(int(limit)).copy()
    records = load_mechanistic_manifest(
        source_root / "counterfactuals/mechanistic_conditions.jsonl"
    )
    records_by_id = {record.condition.condition_id: record for record in records}
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    evaluation_ids = list(evaluation_pairs.base_condition_id) + list(
        evaluation_pairs.source_condition_id
    )
    states, logits = _cache_endpoints(runner, evaluation_ids, records_by_id, layer)
    train_ids = list(train_pairs.base_condition_id) + list(
        train_pairs.source_condition_id
    )
    train_states, _ = _cache_endpoints(runner, train_ids, records_by_id, layer)
    q = np.linalg.qr(basis, mode="reduced")[0]
    unique_train_ids = sorted(set(train_ids))
    train_conditions = pd.DataFrame({"condition_id": unique_train_ids})
    train_conditions["task_family"] = [
        records_by_id[value].condition.task_family
        for value in train_conditions.condition_id
    ]
    task_neutral = {
        task: np.mean(
            np.vstack([train_states[value] @ q for value in part.condition_id]), axis=0
        )
        for task, part in train_conditions.groupby("task_family")
    }
    global_neutral = np.mean(
        np.vstack([train_states[value] @ q for value in unique_train_ids]), axis=0
    )
    rows = []
    for condition_id in sorted(set(evaluation_ids)):
        record = records_by_id[condition_id]
        neutral = task_neutral.get(record.condition.task_family, global_neutral)
        editor = lambda state, b=basis, reference=neutral: remove_subspace(
            state, b, reference
        )
        neutralized = _forward(
            runner, record, layer=layer, editor=editor
        ).persistence_logit
        rows.append(
            {
                "artifact_id": candidate["artifact_id"],
                "target_variable": candidate["target_variable"],
                "layer": layer,
                "rank": rank,
                "neutral_reference": "task_conditioned_training_mean",
                "specificity_passed_before_necessity": True,
                **_condition_feature_row(
                    record, condition_id, logits[condition_id], neutralized
                ),
            }
        )
    shard_root = root / "necessity/shards"
    path = write_records(rows, shard_root / f"necessity_{job_index:03d}.parquet")
    return {
        "job_index": job_index,
        "artifact_id": candidate["artifact_id"],
        "rows": len(rows),
        "path": path,
    }


def _load_shards(root: Path, jobs: list[dict], name: str) -> pd.DataFrame:
    frames = []
    for job in jobs:
        path = _record_path(
            root, f"shards/candidate_{int(job['job_index']):03d}/{name}.parquet"
        )
        if not path.exists():
            raise RuntimeError(f"specificity shard is incomplete: {path}")
        frames.append(read_records(path))
    return pd.concat(frames, ignore_index=True)


def _aggregate_metrics(frame, groups, *, threshold):
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            {
                **dict(zip(groups, keys)),
                **_stable_only(
                    counterfactual_metrics(
                        part.predicted_counterfactual_effect,
                        part.neural_counterfactual_effect,
                        threshold=threshold,
                    )
                ),
                "row_set_sha256": pair_row_hash(part.pair_id),
            }
        )
    return pd.DataFrame(rows)


def _fit_necessity_coefficients(frame: pd.DataFrame, response: str):
    data = frame.drop_duplicates("condition_id").copy()
    continuous = data[list(NECESSITY_FEATURES)].to_numpy(dtype=float)
    mean = continuous.mean(axis=0)
    scale = continuous.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    standardized = (continuous - mean) / scale
    nuisance = pd.get_dummies(
        data[["task_family", "response_mapping"]].astype(str), drop_first=True
    ).to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(data)), standardized, nuisance))
    outcome = data[response].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    fitted = design @ coefficients
    denominator = float(np.sum(np.square(outcome - outcome.mean())))
    r2 = (
        float(1.0 - np.sum(np.square(outcome - fitted)) / denominator)
        if denominator > 1e-12
        else np.nan
    )
    return (
        dict(zip(NECESSITY_FEATURES, coefficients[1 : 1 + len(NECESSITY_FEATURES)])),
        r2,
    )


def _necessity_coefficients(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for artifact_id, part in frame.groupby("artifact_id"):
        before, before_r2 = _fit_necessity_coefficients(part, "persistence_logit")
        after, after_r2 = _fit_necessity_coefficients(
            part, "neutralized_persistence_logit"
        )
        target = str(part.target_variable.iloc[0])
        for feature in NECESSITY_FEATURES:
            denominator = abs(before[feature])
            score = (
                1.0 - abs(after[feature]) / denominator
                if denominator > 1e-10
                else np.nan
            )
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "target_variable": target,
                    "feature": feature,
                    "baseline_coefficient": before[feature],
                    "neutralized_coefficient": after[feature],
                    "necessity_score": score,
                    "baseline_fit_r2": before_r2,
                    "neutralized_fit_r2": after_r2,
                    "control_estimator_reproduced_baseline": bool(
                        np.isfinite(before_r2) and before_r2 >= 0.5
                    ),
                }
            )
    return pd.DataFrame(rows)


def _geometry_table(root: Path, jobs: list[dict]) -> pd.DataFrame:
    subspaces = []
    for job in jobs:
        subspaces.append(
            {
                "artifact_id": job["artifact_id"],
                "target_variable": job["target_variable"],
                "theory": job["theory"],
                "layer": int(job["layer"]),
                "basis": load_alignment(job["artifact"]),
            }
        )
        persistence_path = (
            root
            / "shards"
            / f"candidate_{int(job['job_index']):03d}"
            / "persistence_state.safetensors"
        )
        subspaces.append(
            {
                "artifact_id": f"{job['artifact_id']}__persistence_state",
                "target_variable": "persistence_state",
                "theory": "direct_persistence_control",
                "layer": int(job["layer"]),
                "basis": load_alignment(persistence_path),
            }
        )
    rows = []
    for left_index, left in enumerate(subspaces):
        for right in subspaces[left_index + 1 :]:
            geometry = subspace_geometry(left["basis"], right["basis"])
            rows.append(
                {
                    "left_artifact_id": left["artifact_id"],
                    "right_artifact_id": right["artifact_id"],
                    "left_target": left["target_variable"],
                    "right_target": right["target_variable"],
                    "left_layer": left["layer"],
                    "right_layer": right["layer"],
                    "cross_layer_geometry_is_secondary": left["layer"]
                    != right["layer"],
                    **{
                        key: json.dumps(value) if isinstance(value, list) else value
                        for key, value in geometry.items()
                    },
                }
            )
    return pd.DataFrame(rows)


def _context_conflict(
    cross: pd.DataFrame,
    *,
    confidence: float,
    samples: int,
    seed: int,
    quantile: float = 0.75,
):
    context = cross[
        (cross.target_variable == "contextual_outcome_history")
        & (cross.pair_split == "mech_pair_test")
        & (cross.intervention_type == "frozen_DAS")
    ]
    rows, summaries = [], []
    rng = np.random.default_rng(int(seed))
    for artifact_id, part in context.groupby("artifact_id"):
        contextual = part[part.scoring_theory == "latent_context"][
            [
                "pair_id",
                "contrast_id",
                "task_family",
                "response_mapping",
                "neural_counterfactual_effect",
                "predicted_counterfactual_effect",
            ]
        ].rename(columns={"predicted_counterfactual_effect": "contextual_prediction"})
        for raw_theory in ("dual_history", "outcome_history"):
            raw = part[part.scoring_theory == raw_theory][
                ["pair_id", "predicted_counterfactual_effect"]
            ].rename(columns={"predicted_counterfactual_effect": "raw_prediction"})
            merged = contextual.merge(raw, on="pair_id", validate="one_to_one")
            if merged.empty:
                continue
            merged["prediction_disagreement"] = (
                merged.contextual_prediction - merged.raw_prediction
            ).abs()
            cutoff = float(merged.prediction_disagreement.quantile(quantile))
            merged = merged[merged.prediction_disagreement >= cutoff].copy()
            merged["raw_theory"] = raw_theory
            merged["artifact_id"] = artifact_id
            merged["mse_raw"] = np.square(
                merged.neural_counterfactual_effect - merged.raw_prediction
            )
            merged["mse_contextual"] = np.square(
                merged.neural_counterfactual_effect - merged.contextual_prediction
            )
            merged["delta_mse_raw_minus_contextual"] = (
                merged.mse_raw - merged.mse_contextual
            )
            rows.extend(merged.to_dict("records"))
            clusters = merged[["contrast_id", "task_family"]].drop_duplicates()
            draws = []
            for _ in range(int(samples)):
                pieces = []
                for task, task_clusters in clusters.groupby("task_family"):
                    ids = task_clusters.contrast_id.astype(str).to_numpy()
                    chosen = rng.choice(ids, size=len(ids), replace=True)
                    task_frame = merged[merged.task_family == task]
                    pieces.extend(
                        task_frame[task_frame.contrast_id.astype(str) == value]
                        for value in chosen
                    )
                sampled = pd.concat(pieces, ignore_index=True)
                draws.append(float(sampled.delta_mse_raw_minus_contextual.mean()))
            alpha = (1.0 - confidence) / 2.0
            summaries.append(
                {
                    "artifact_id": artifact_id,
                    "raw_theory": raw_theory,
                    "conflict_quantile": float(quantile),
                    "examples": len(merged),
                    "mean_delta_mse_raw_minus_contextual": float(
                        merged.delta_mse_raw_minus_contextual.mean()
                    ),
                    "ci_lower": float(np.quantile(draws, alpha)),
                    "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
                    "positive_favors_contextual_history": True,
                    "same_frozen_neural_results_scored_against_both_theories": True,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _bootstrap_candidate_and_controls(
    candidate,
    controls,
    *,
    samples,
    confidence,
    seed,
    threshold,
):
    point, intervals = bootstrap_metric_intervals(
        candidate,
        samples=samples,
        confidence=confidence,
        seed=seed,
        threshold=threshold,
    )
    comparisons = []
    for index, (control_name, part) in enumerate(controls.items()):
        difference = paired_bootstrap_difference(
            candidate,
            part,
            samples=samples,
            confidence=confidence,
            seed=seed + 100 + index,
            threshold=threshold,
        )
        comparisons.append({"control": control_name, **difference})
    return point, intervals, comparisons


def aggregate_specificity_run(
    config: dict,
    *,
    source: str | Path | None = None,
    output: str | Path | None = None,
):
    """Resolve Level 5A and preregister only passing candidates for necessity."""

    root = _root(config, output)
    jobs = json.loads((root / "frozen_models/evaluation_jobs.json").read_text())
    for job in jobs:
        audit_path = (
            root / "shards" / f"candidate_{int(job['job_index']):03d}" / "audit.json"
        )
        if not audit_path.exists():
            raise RuntimeError(f"specificity evaluation is incomplete: {audit_path}")
        audit = json.loads(audit_path.read_text())
        if not audit["candidate_control_row_identity"]:
            raise RuntimeError("candidate/control row identity audit failed")

    candidate = _load_shards(root, jobs, "candidate_effects")
    named = _load_shards(root, jobs, "named_controls")
    shuffled_source = _load_shards(root, jobs, "shuffled_source")
    shuffled_target = _load_shards(root, jobs, "shuffled_target")
    cross = _load_shards(root, jobs, "cross_variable")
    random_frames = [
        pd.read_csv(
            root
            / "shards"
            / f"candidate_{int(job['job_index']):03d}"
            / "random_subspaces.csv"
        )
        for job in jobs
    ]
    random_null = pd.concat(random_frames, ignore_index=True)

    controls_root = root / "controls"
    controls_root.mkdir(parents=True, exist_ok=True)
    write_records(
        random_null.to_dict("records"), controls_root / "random_subspaces.parquet"
    )
    shuffled_source.to_csv(controls_root / "shuffled_source.csv", index=False)
    shuffled_target.to_csv(controls_root / "shuffled_target.csv", index=False)
    named[named.control.isin(("persistence_state", "persistence_output"))].to_csv(
        controls_root / "persistence_control.csv", index=False
    )
    named[named.control == "generic_value"].to_csv(
        controls_root / "generic_value_control.csv", index=False
    )
    named[named.control == "task_id"].to_csv(
        controls_root / "task_id_control.csv", index=False
    )
    named[named.control == "response_mapping"].to_csv(
        controls_root / "response_mapping_control.csv", index=False
    )
    write_records(
        candidate.to_dict("records"), controls_root / "candidate_effects.parquet"
    )

    threshold = float(config.get("cfr", {}).get("threshold", 0.10))
    bootstrap_config = config.get("bootstrap", {})
    samples = int(bootstrap_config.get("samples", 2000))
    confidence = float(bootstrap_config.get("confidence", 0.95))
    seed = int(config.get("seed", 73001))
    metric_rows, interval_rows, comparison_rows = [], [], []
    gate_rows = []
    for candidate_index, job in enumerate(jobs):
        artifact_id = job["artifact_id"]
        candidate_rows = candidate[candidate.artifact_id == artifact_id]
        control_frames = {
            control: part
            for control, part in named[named.artifact_id == artifact_id].groupby(
                "control"
            )
        }
        control_frames["shuffled_source"] = shuffled_source[
            shuffled_source.artifact_id == artifact_id
        ]
        control_frames["shuffled_target"] = shuffled_target[
            shuffled_target.artifact_id == artifact_id
        ]
        unrelated_das = cross[
            (cross.artifact_id != artifact_id)
            & (cross.intervention_type == "frozen_DAS")
            & (cross.target_variable == job["target_variable"])
            & (cross.scoring_theory == job["theory"])
            & (cross.pair_split == "mech_pair_test")
        ]
        for other_id, part in unrelated_das.groupby("artifact_id"):
            control_frames[f"unrelated_DAS:{other_id}"] = part
        for control_name, part in control_frames.items():
            if pair_row_hash(part.pair_id) != pair_row_hash(candidate_rows.pair_id):
                raise RuntimeError(
                    f"{artifact_id}/{control_name} used a different test row set"
                )
        point, intervals, comparisons = _bootstrap_candidate_and_controls(
            candidate_rows,
            control_frames,
            samples=samples,
            confidence=confidence,
            seed=seed + candidate_index * 1000,
            threshold=threshold,
        )
        metric_rows.append(
            {
                "artifact_id": artifact_id,
                "target_variable": job["target_variable"],
                "theory": job["theory"],
                "comparison": "candidate",
                **_stable_only(point),
            }
        )
        for control_name, part in control_frames.items():
            control_metrics, control_intervals = bootstrap_metric_intervals(
                part,
                samples=samples,
                confidence=confidence,
                seed=seed + candidate_index * 1000 + 300 + len(metric_rows),
                threshold=threshold,
            )
            metric_rows.append(
                {
                    "artifact_id": artifact_id,
                    "target_variable": job["target_variable"],
                    "theory": job["theory"],
                    "comparison": control_name,
                    **_stable_only(control_metrics),
                }
            )
            interval_rows.extend(
                {"artifact_id": artifact_id, "comparison": control_name, **row}
                for row in control_intervals
            )
        interval_rows.extend(
            {"artifact_id": artifact_id, "comparison": "candidate", **row}
            for row in intervals
        )
        comparison_rows.extend(
            {"artifact_id": artifact_id, **row} for row in comparisons
        )
        random_part = random_null[random_null.artifact_id == artifact_id]
        random_p = float(
            (1 + int((random_part.global_cfr >= point["global_cfr"]).sum()))
            / (len(random_part) + 1)
        )
        cfr_interval = next(row for row in intervals if row["metric"] == "global_cfr")
        r_interval = next(row for row in intervals if row["metric"] == "correlation")
        difference_by_control = {row["control"]: row for row in comparisons}
        named_required = (
            "persistence_state",
            "persistence_output",
            "generic_value",
            "task_id",
            "response_mapping",
        )
        unrelated_required = [
            name for name in difference_by_control if name.startswith("unrelated_DAS:")
        ]
        gate_rows.append(
            {
                "artifact_id": artifact_id,
                "target_variable": job["target_variable"],
                "theory": job["theory"],
                "global_cfr": point["global_cfr"],
                "global_cfr_ci_lower": cfr_interval["ci_lower"],
                "correlation": point["correlation"],
                "correlation_ci_lower": r_interval["ci_lower"],
                "random_p": random_p,
                "random_correlation_p": float(
                    (
                        1
                        + int(
                            (
                                random_part.correlation.fillna(-np.inf)
                                >= point["correlation"]
                            ).sum()
                        )
                    )
                    / (len(random_part) + 1)
                ),
                "random_rmse_p": float(
                    (1 + int((random_part.rmse <= point["rmse"]).sum()))
                    / (len(random_part) + 1)
                ),
                "passes_recovery": bool(
                    cfr_interval["ci_lower"] > 0 and r_interval["ci_lower"] > 0
                ),
                "passes_random": bool(random_p < 0.05),
                "passes_shuffled_source": bool(
                    difference_by_control["shuffled_source"]["ci_lower"] > 0
                ),
                "passes_shuffled_target": bool(
                    difference_by_control["shuffled_target"]["ci_lower"] > 0
                ),
                "passes_named_decision_controls": bool(
                    all(
                        difference_by_control[name]["ci_lower"] > 0
                        for name in named_required
                    )
                ),
                "passes_unrelated_das_controls": bool(
                    unrelated_required
                    and all(
                        difference_by_control[name]["ci_lower"] > 0
                        for name in unrelated_required
                    )
                ),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(root / "corrected_metrics/specificity_metrics.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(
        root / "corrected_metrics/v2_bootstrap_intervals.csv", index=False
    )
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(
        controls_root / "paired_control_differences.csv", index=False
    )

    cross_root = root / "cross_variable"
    cross_root.mkdir(parents=True, exist_ok=True)
    cross_metrics = _aggregate_metrics(
        cross,
        [
            "artifact_id",
            "subspace_target",
            "subspace_theory",
            "target_variable",
            "scoring_theory",
            "pair_split",
        ],
        threshold=threshold,
    )
    cross_metrics.to_csv(cross_root / "specificity_matrix.csv", index=False)
    information_sampling = cross[
        (cross.task_family == "information_sampling")
        & (cross.pair_split == "mech_task_holdout")
        & (cross.intervention_type == "frozen_DAS")
    ].copy()
    information_rows = []
    for keys, part in information_sampling.groupby(
        ["artifact_id", "target_variable", "scoring_theory"]
    ):
        information_rows.append(
            {
                "artifact_id": keys[0],
                "target_variable": keys[1],
                "scoring_theory": keys[2],
                "mean_absolute_predicted_effect": float(
                    part.predicted_counterfactual_effect.abs().mean()
                ),
                "mean_absolute_neural_effect": float(
                    part.neural_counterfactual_effect.abs().mean()
                ),
                "mean_intervention_norm": float(part.intervention_norm.mean()),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                part.neural_counterfactual_effect
                                - part.predicted_counterfactual_effect
                            )
                        )
                    )
                ),
                "examples": len(part),
                "negative_control_interpretation": True,
            }
        )
    pd.DataFrame(information_rows).to_csv(
        cross_root / "information_sampling_negative_control.csv", index=False
    )
    geometry = _geometry_table(root, jobs)
    geometry.to_csv(cross_root / "subspace_geometry.csv", index=False)

    vsi_rows = []
    for index, job in enumerate(jobs):
        artifact_rows = cross[
            (cross.artifact_id == job["artifact_id"])
            & (cross.pair_split == "mech_pair_test")
        ]
        matched = artifact_rows[
            (artifact_rows.target_variable == job["target_variable"])
            & (artifact_rows.scoring_theory == job["theory"])
        ]
        unrelated = []
        for target in CROSS_VARIABLES:
            if target == job["target_variable"]:
                continue
            target_rows = artifact_rows[artifact_rows.target_variable == target]
            for _, theory_rows in target_rows.groupby("scoring_theory"):
                unrelated.append(theory_rows)
        vsi = bootstrap_vsi(
            matched,
            unrelated,
            samples=samples,
            confidence=confidence,
            seed=seed + 5000 + index,
            threshold=threshold,
        )
        vsi_rows.append(
            {
                "artifact_id": job["artifact_id"],
                "target_variable": job["target_variable"],
                "theory": job["theory"],
                **vsi,
            }
        )
    vsi_frame = pd.DataFrame(vsi_rows)
    vsi_frame.to_csv(cross_root / "variable_specificity.csv", index=False)
    context_rows, context_summary = _context_conflict(
        cross,
        confidence=confidence,
        samples=samples,
        seed=seed + 7000,
        quantile=float(
            config.get("cross_variable", {}).get("context_conflict_quantile", 0.75)
        ),
    )
    context_rows.to_csv(cross_root / "context_conflict.csv", index=False)
    context_summary.to_csv(cross_root / "context_conflict_summary.csv", index=False)

    gate_frame = pd.DataFrame(gate_rows).merge(
        vsi_frame[["artifact_id", "vsi", "ci_lower", "ci_upper"]].rename(
            columns={"ci_lower": "vsi_ci_lower", "ci_upper": "vsi_ci_upper"}
        ),
        on="artifact_id",
        validate="one_to_one",
    )
    gate_frame["passes_cross_variable_specificity"] = gate_frame.vsi_ci_lower > 0
    gate_frame["passes_level5a"] = gate_frame[
        [
            "passes_recovery",
            "passes_random",
            "passes_shuffled_source",
            "passes_shuffled_target",
            "passes_named_decision_controls",
            "passes_unrelated_das_controls",
            "passes_cross_variable_specificity",
        ]
    ].all(axis=1)
    gate_frame.to_csv(root / "candidate_gates_pre_necessity.csv", index=False)
    passing_ids = set(
        gate_frame.loc[gate_frame.passes_level5a.astype(bool), "artifact_id"]
    )
    necessity_jobs = []
    for job in jobs:
        if job["artifact_id"] in passing_ids:
            necessity_jobs.append({**job, "job_index": len(necessity_jobs)})
    _json(root / "frozen_models/necessity_jobs.json", necessity_jobs)
    gates = json.loads((root / "gates.json").read_text())
    gates["corrected_level4"] = {
        "passed": bool(gate_frame.passes_recovery.any()),
        "candidate_count": int(gate_frame.passes_recovery.sum()),
    }
    gates["level5a_specificity"] = {
        "passed": bool(gate_frame.passes_level5a.any()),
        "candidate_count": int(gate_frame.passes_level5a.sum()),
        "all_six_primary_criteria_required": True,
    }
    gates["level5b_necessity"] = {
        "status": (
            "awaiting_post_specificity_neutralization"
            if necessity_jobs
            else "not_tested_after_specificity_failure"
        )
    }
    gates["circuit"] = {"status": "not_eligible_pending_level5_resolution"}
    _json(root / "gates.json", gates)
    return {
        "candidates": len(gate_frame),
        "level4_reproduced": int(gate_frame.passes_recovery.sum()),
        "level5a_passed": int(gate_frame.passes_level5a.sum()),
        "necessity_jobs": len(necessity_jobs),
    }


def finalize_specificity_run(
    config: dict,
    *,
    source: str | Path | None = None,
    output: str | Path | None = None,
):
    """Finalize Level 5B only from conditionally submitted necessity shards."""

    root = _root(config, output)
    jobs = json.loads((root / "frozen_models/evaluation_jobs.json").read_text())
    necessity_jobs = json.loads(
        (root / "frozen_models/necessity_jobs.json").read_text()
    )
    gate_frame = pd.read_csv(root / "candidate_gates_pre_necessity.csv")
    metrics = pd.read_csv(root / "corrected_metrics/specificity_metrics.csv")
    cross_metrics = pd.read_csv(root / "cross_variable/specificity_matrix.csv")
    context_summary = pd.read_csv(root / "cross_variable/context_conflict_summary.csv")
    random_null = read_records(_record_path(root, "controls/random_subspaces.parquet"))
    candidate = read_records(_record_path(root, "controls/candidate_effects.parquet"))
    coefficient_columns = [
        "artifact_id",
        "target_variable",
        "feature",
        "baseline_coefficient",
        "neutralized_coefficient",
        "necessity_score",
        "baseline_fit_r2",
        "neutralized_fit_r2",
        "control_estimator_reproduced_baseline",
    ]
    necessity_root = root / "necessity"
    necessity_root.mkdir(parents=True, exist_ok=True)
    necessity_gate = []
    if necessity_jobs:
        frames = []
        for job in necessity_jobs:
            path = _record_path(
                root,
                f"necessity/shards/necessity_{int(job['job_index']):03d}.parquet",
            )
            if not path.exists():
                raise RuntimeError(
                    f"post-specificity necessity shard is absent: {path}"
                )
            frames.append(read_records(path))
        necessity = pd.concat(frames, ignore_index=True)
        if not necessity.specificity_passed_before_necessity.astype(bool).all():
            raise RuntimeError("necessity was run before its specificity gate")
        write_records(
            necessity.to_dict("records"),
            necessity_root / "neutralization_results.parquet",
        )
        coefficient_changes = _necessity_coefficients(necessity)
        for artifact_id, part in coefficient_changes.groupby("artifact_id"):
            target = str(part.target_variable.iloc[0])
            target_score = part.loc[part.feature == target, "necessity_score"]
            baseline_reproduced = bool(
                part.control_estimator_reproduced_baseline.astype(bool).all()
            )
            other_scores = part.loc[part.feature != target, "necessity_score"].dropna()
            necessity_gate.append(
                {
                    "artifact_id": artifact_id,
                    "target_necessity_score": (
                        float(target_score.iloc[0]) if len(target_score) else np.nan
                    ),
                    "median_other_necessity_score": (
                        float(other_scores.median()) if len(other_scores) else np.nan
                    ),
                    "passes_level5b": bool(
                        len(target_score)
                        and baseline_reproduced
                        and target_score.iloc[0] > 0
                        and (
                            not len(other_scores)
                            or target_score.iloc[0] > other_scores.median()
                        )
                    ),
                }
            )
    else:
        coefficient_changes = pd.DataFrame(columns=coefficient_columns)
    coefficient_changes.to_csv(necessity_root / "coefficient_changes.csv", index=False)
    if necessity_gate:
        gate_frame = gate_frame.merge(
            pd.DataFrame(necessity_gate), on="artifact_id", how="left"
        )
    else:
        gate_frame["target_necessity_score"] = np.nan
        gate_frame["median_other_necessity_score"] = np.nan
        gate_frame["passes_level5b"] = False
    gate_frame["passes_level5b"] = gate_frame.passes_level5b.fillna(False).astype(bool)
    gate_frame.to_csv(root / "candidate_gates.csv", index=False)
    _finalize_report(
        config,
        root,
        jobs,
        gate_frame,
        metrics,
        cross_metrics,
        context_summary,
        coefficient_changes,
        random_null,
        candidate,
    )
    return {
        "candidates": len(gate_frame),
        "level4_reproduced": int(gate_frame.passes_recovery.sum()),
        "level5a_passed": int(gate_frame.passes_level5a.sum()),
        "level5b_passed": int(gate_frame.passes_level5b.sum()),
        "report": root / "report.md",
    }


def _figures(
    root,
    gates,
    metrics,
    cross_metrics,
    context_summary,
    coefficients,
    random_null,
    candidate_effects,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []
    localization = pd.read_csv(root / "corrected_metrics/global_cfr.csv")
    shown = localization[
        (localization.pair_split == "mech_pair_validation")
        & localization.target_variable.isin(PRIMARY_VARIABLES)
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (target, theory), part in shown.groupby(["target_variable", "theory"]):
        ax.plot(part.layer, part.global_cfr, label=f"{target}/{theory}")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(xlabel="Layer", ylabel="Global CFR", title="Corrected coarse localization")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    paths.append(figure_root / "figure1_corrected_localization.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    best = gates.sort_values("global_cfr", ascending=False).iloc[0]
    effects = candidate_effects[candidate_effects.artifact_id == best.artifact_id]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        effects.predicted_counterfactual_effect,
        effects.neural_counterfactual_effect,
        alpha=0.7,
    )
    extent = max(
        float(effects.predicted_counterfactual_effect.abs().max()),
        float(effects.neural_counterfactual_effect.abs().max()),
        1e-3,
    )
    ax.plot([-extent, extent], [-extent, extent], "--", color="grey", label="identity")
    fit = metrics[
        (metrics.artifact_id == best.artifact_id) & (metrics.comparison == "candidate")
    ].iloc[0]
    x = np.linspace(-extent, extent, 100)
    ax.plot(x, fit.intercept + fit.slope * x, color="tab:red", label="fitted")
    ax.text(
        0.03,
        0.97,
        f"r={fit.correlation:.2f}\nslope={fit.slope:.2f}\nCFR_G={fit.global_cfr:.2f}\nRMSE={fit.rmse:.2f}",
        transform=ax.transAxes,
        va="top",
    )
    ax.set(
        xlabel="Frozen counterfactual effect",
        ylabel="Frozen DAS effect",
        title="Untouched-test counterfactual equivalence",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    paths.append(figure_root / "figure2_heldout_equivalence.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(gates), 1, figsize=(8, 3 * len(gates)), squeeze=False)
    for ax, gate in zip(axes[:, 0], gates.itertuples()):
        part = random_null[random_null.artifact_id == gate.artifact_id]
        ax.hist(part.global_cfr, bins=30, alpha=0.75)
        ax.axvline(gate.global_cfr, color="tab:red", label="candidate")
        ax.set(title=gate.artifact_id, xlabel="Global CFR", ylabel="random subspaces")
        ax.legend(frameon=False)
    fig.tight_layout()
    paths.append(figure_root / "figure3_random_null.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    control_plot = metrics.copy()
    control_intervals = pd.read_csv(
        root / "corrected_metrics/v2_bootstrap_intervals.csv"
    )
    cfr_intervals = control_intervals[control_intervals.metric == "global_cfr"][
        ["artifact_id", "comparison", "ci_lower", "ci_upper"]
    ]
    control_plot = control_plot.merge(
        cfr_intervals,
        on=["artifact_id", "comparison"],
        how="left",
        validate="one_to_one",
    )
    control_plot["label"] = (
        control_plot.artifact_id.str.slice(0, 12) + ":" + control_plot.comparison
    )
    errors = np.vstack(
        (
            np.maximum(0, control_plot.global_cfr - control_plot.ci_lower),
            np.maximum(0, control_plot.ci_upper - control_plot.global_cfr),
        )
    )
    ax.bar(range(len(control_plot)), control_plot.global_cfr, yerr=errors, capsize=2)
    ax.set_xticks(range(len(control_plot)), control_plot.label, rotation=70, ha="right")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(ylabel="Global CFR", title="Matched causal-specificity controls")
    fig.tight_layout()
    paths.append(figure_root / "figure4_specificity_controls.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    matrix_source = cross_metrics[
        (cross_metrics.pair_split == "mech_pair_test")
        & cross_metrics.scoring_theory.isin(
            ("outcome_history", "latent_context", "dual_history")
        )
    ].copy()
    matrix_source["target"] = (
        matrix_source.target_variable + ":" + matrix_source.scoring_theory
    )
    matrix = matrix_source.pivot_table(
        index="artifact_id", columns="target", values="global_cfr", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(matrix))))
    image = ax.imshow(
        matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1
    )
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=55, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Functional cross-variable specificity matrix")
    fig.colorbar(image, ax=ax, label="Global CFR")
    fig.tight_layout()
    paths.append(figure_root / "figure5_cross_variable_matrix.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(context_summary):
        labels = (
            context_summary.artifact_id.str.slice(0, 11)
            + "/"
            + context_summary.raw_theory
        )
        values = context_summary.mean_delta_mse_raw_minus_contextual
        lower = values - context_summary.ci_lower
        upper = context_summary.ci_upper - values
        ax.errorbar(range(len(values)), values, yerr=np.vstack((lower, upper)), fmt="o")
        ax.set_xticks(range(len(values)), labels, rotation=55, ha="right")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(
        ylabel="ΔMSE raw − contextual",
        title="Context-conflict theory discrimination",
    )
    fig.tight_layout()
    paths.append(figure_root / "figure6_context_conflict.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    if len(coefficients):
        positions = np.arange(len(coefficients))
        ax.scatter(positions, coefficients.baseline_coefficient, label="baseline", s=18)
        ax.scatter(
            positions,
            coefficients.neutralized_coefficient,
            label="neutralized",
            s=18,
        )
        ax.set_xticks(
            positions,
            coefficients.artifact_id.str.slice(0, 8) + ":" + coefficients.feature,
            rotation=75,
            ha="right",
        )
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "Necessity not run because no candidate passed Level 5A",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(ylabel="Standardized coefficient", title="Necessity coefficient changes")
    fig.tight_layout()
    paths.append(figure_root / "figure7_necessity.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)
    return paths


def _mechanistic_outcome(gates, context_summary):
    passing = gates[gates.passes_level5a.astype(bool)]
    if passing.empty:
        if gates.passes_random.any():
            return (
                "DAS_controller_without_variable_identity",
                "DAS identified an effective persistence controller but not a specific implementation of the proposed cognitive variable.",
            )
        return (
            "level4_only",
            "Stable evaluation retains at most a causal representation; variable-specific implementation remains unsupported.",
        )
    context = passing[passing.target_variable == "contextual_outcome_history"]
    raw = passing[passing.target_variable == "outcome_history"]
    if len(context) and not len(raw):
        necessary = bool(context.passes_level5b.fillna(False).any())
        return (
            "specific_contextual_history",
            (
                "A shared neural subspace causally implements context-sensitive outcome-history integration."
                if necessary
                else "A DAS subspace is causally specific to context-sensitive outcome history, but implementation wording is withheld because necessity failed or remains unsupported."
            ),
        )
    if len(raw) and not len(context):
        necessary = bool(raw.passes_level5b.fillna(False).any())
        return (
            "specific_raw_history",
            (
                "A shared neural subspace causally implements recent raw outcome history."
                if necessary
                else "A DAS subspace is causally specific to recent raw outcome history, but implementation wording is withheld because necessity failed or remains unsupported."
            ),
        )
    if len(raw) and len(context):
        return (
            "multiple_specific_or_shared_downstream_state",
            "Both history families pass specificity; functional interchange and context-conflict results determine whether they remain distinct or converge downstream.",
        )
    return "unresolved", "Causal specificity remains unresolved."


def _finalize_report(
    config,
    root,
    jobs,
    candidate_gates,
    metrics,
    cross_metrics,
    context_summary,
    coefficients,
    random_null,
    candidate_effects,
):
    level4 = bool(candidate_gates.passes_recovery.any())
    level5a = bool(candidate_gates.passes_level5a.any())
    level5b = bool(
        level5a
        and candidate_gates.loc[
            candidate_gates.passes_level5a.astype(bool), "passes_level5b"
        ]
        .fillna(False)
        .any()
    )
    outcome, conclusion = _mechanistic_outcome(candidate_gates, context_summary)
    highest = "5B" if level5b else ("5A" if level5a else ("4" if level4 else "3"))
    best = candidate_gates.sort_values("global_cfr", ascending=False).iloc[0]
    best_random = random_null[random_null.artifact_id == best.artifact_id]
    persistence = metrics[
        (metrics.artifact_id == best.artifact_id)
        & metrics.comparison.isin(("persistence_state", "persistence_output"))
    ]
    conflict_answer = (
        "contextual history"
        if len(context_summary)
        and context_summary.mean_delta_mse_raw_minus_contextual.mean() > 0
        else "raw history or unresolved"
    )
    holdout = cross_metrics[cross_metrics.pair_split == "mech_task_holdout"]
    holdout_match = holdout[
        holdout.apply(
            lambda row: any(
                row.artifact_id == job["artifact_id"]
                and row.target_variable == job["target_variable"]
                and row.scoring_theory == job["theory"]
                for job in jobs
            ),
            axis=1,
        )
    ]
    necessity_target = coefficients[
        coefficients.feature == coefficients.target_variable
    ]
    figures = _figures(
        root,
        candidate_gates,
        metrics,
        cross_metrics,
        context_summary,
        coefficients,
        random_null,
        candidate_effects,
    )
    gates = json.loads((root / "gates.json").read_text())
    gates.update(
        {
            "corrected_level4": {
                "passed": level4,
                "candidate_count": int(candidate_gates.passes_recovery.sum()),
                "criteria": "CFR_G and correlation bootstrap lower bounds exceed zero",
            },
            "level5a_specificity": {
                "passed": level5a,
                "candidate_count": int(candidate_gates.passes_level5a.sum()),
                "all_six_primary_criteria_required": True,
            },
            "level5b_necessity": {
                "passed": level5b,
                "evaluated_only_after_specificity": True,
            },
            "circuit": {
                "status": "eligible" if level5a else "not_eligible",
                "submitted": False,
            },
            "highest_evidence_level": highest,
            "mechanistic_outcome": outcome,
        }
    )
    _json(root / "gates.json", gates)
    report = [
        "# DAS Causal Specificity with Stable CFR",
        "",
        f"**Highest justified evidence level:** {highest}",
        "",
        f"**Conclusion:** {conclusion}",
        "",
        "## Automated questions",
        "",
        f"1. Corrected global CFR reproduces Level 4: **{'yes' if level4 else 'no'}**.",
        f"2. Best untouched-test candidate: `{best.artifact_id}` (layer {int(next(job['layer'] for job in jobs if job['artifact_id'] == best.artifact_id))}, rank {int(next(job['rank'] for job in jobs if job['artifact_id'] == best.artifact_id))}, CFR_G={best.global_cfr:.3f}).",
        f"3. Best candidate beats 500 matched random subspaces: **{'yes' if best.passes_random else 'no'}** (p={best.random_p:.4f}; null 95th={best_random.global_cfr.quantile(.95):.3f}).",
        f"4. It beats shuffled sources with paired-bootstrap support: **{'yes' if best.passes_shuffled_source else 'no'}**.",
        f"5. It beats shuffled targets with paired-bootstrap support: **{'yes' if best.passes_shuffled_target else 'no'}**.",
        f"6. It outperforms direct persistence controls: **{'yes' if best.passes_named_decision_controls else 'no'}** (maximum persistence-control CFR_G={persistence.global_cfr.max() if len(persistence) else np.nan:.3f}).",
        f"7. A raw-history subspace specifically recovers do(O): **{'yes' if ((candidate_gates.target_variable == 'outcome_history') & candidate_gates.passes_level5a).any() else 'no'}**.",
        f"8. A contextual-history subspace specifically recovers do(O*): **{'yes' if ((candidate_gates.target_variable == 'contextual_outcome_history') & candidate_gates.passes_level5a).any() else 'no'}**.",
        f"9. Functional interchangeability: **{outcome}**; geometry is reported only as secondary evidence.",
        f"10. Context-conflict interventions favor: **{conflict_answer}**.",
        f"11. Held-out-task matched recovery: mean CFR_G={holdout_match.global_cfr.mean() if len(holdout_match) else np.nan:.3f}; no target-task refitting occurred.",
        f"12. Target-selective necessity: **{'yes' if level5b else 'no'}** (mean target NS={necessity_target.necessity_score.mean() if len(necessity_target) else np.nan:.3f}).",
        f"13. Highest evidence level: **{highest}**.",
        f"14. Justified claim: **{conclusion}**",
        "15. Unsupported claims: circuit implementation and any variable identity whose candidate did not pass all Level-5A criteria.",
        "",
        "## Guardrails",
        "",
        "- The behavioral models, DAS layers/ranks/rotations, pair definitions, and splits were frozen before this reanalysis.",
        "- The primary analysis performed no DAS retraining.",
        "- Global CFR is primary; thresholded per-example CFR is descriptive only.",
        "- Bootstrap sampling uses semantic source/base pairs and keeps duplicated response mappings together.",
        "- Every specificity control uses the candidate's identical untouched-test rows.",
        "- Necessity is reported as Level 5B separately from causal specificity.",
        "- Circuit analysis remains blocked unless Level 5A passes.",
        "- No full activation bank was retained.",
    ]
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "report_generated": True,
            "highest_evidence_level": highest,
            "mechanistic_outcome": outcome,
            "figure_count": len(figures),
            "full_activations_saved": False,
        }
    )
    _json(metadata_path, metadata)

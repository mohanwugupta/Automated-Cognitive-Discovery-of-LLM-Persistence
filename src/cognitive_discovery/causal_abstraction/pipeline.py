"""Causal-abstraction workflow using frozen behavioral targets and DAS subspaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from cognitive_discovery.causal_mechanistic.das import load_alignment
from cognitive_discovery.causal_mechanistic.interventions import (
    interchange_subspace,
    intervention_norm,
    replace_whole_state,
)
from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics
from cognitive_discovery.causal_mechanistic.pipeline import (
    _forward,
    _make_runner,
    _trial,
)
from cognitive_discovery.causal_specificity.controls import (
    orthonormal_random_subspaces,
    pair_row_hash,
)
from cognitive_discovery.data.storage import read_records, write_records
from cognitive_discovery.design.manifests import load_condition_manifest

from .analysis import (
    STABLE_METRICS,
    abstraction_metric_table,
    bootstrap_abstraction_metrics,
    convergence_index,
    factorial_regression,
    natural_model_comparison,
    paired_abstraction_differences,
    paired_cfr_difference,
)
from .design import (
    CONTRAST_FAMILIES,
    condition_rows,
    expand_contrast_mappings,
    expand_selected_conditions,
    generate_candidate_conditions,
    select_identifiability_contrasts,
    shuffle_abstraction_targets,
    write_condition_jsonl,
)
from .variables import (
    ABSTRACTIONS,
    PREDICTION_COLUMNS,
    FrozenAbstractionBank,
    write_frozen_manifest,
)


def _root(config: dict, output=None) -> Path:
    return Path(
        output or config.get("output_root", "artifacts/abstraction_discovery_v1")
    )


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


def _source_path(path, source_root: Path, fallback: str) -> Path:
    value = Path(path)
    if value.exists():
        return value
    candidate = source_root / fallback / value.name
    return candidate if candidate.exists() else value


def _freeze_das_inputs(root: Path, specificity_root: Path, config: dict):
    gates = json.loads((specificity_root / "gates.json").read_text())
    if gates.get("highest_evidence_level") not in (4, "4"):
        raise RuntimeError("abstraction discovery requires the completed Level-4 run")
    if gates.get("level5a_specificity", {}).get("passed"):
        raise RuntimeError("this protocol expects unresolved one-variable specificity")
    source_jobs = json.loads(
        (specificity_root / "frozen_models/evaluation_jobs.json").read_text()
    )
    configured_primary = config.get("frozen_das", {}).get(
        "primary_artifact_id",
        "das_001_outcome_history_dual_history_L28_r2",
    )
    expected_hash = config.get("frozen_das", {}).get("expected_primary_sha256")
    frozen = []
    for source_job in source_jobs:
        source_artifact = _source_path(
            source_job["artifact"], specificity_root, "frozen_models/alignments"
        )
        if not source_artifact.exists():
            raise FileNotFoundError(f"frozen DAS artifact is absent: {source_artifact}")
        artifact = root / "frozen_das/alignments" / source_artifact.name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_artifact, artifact)
        digest = _sha256(artifact)
        if digest != source_job["sha256"]:
            raise RuntimeError("copied DAS artifact does not reproduce its frozen hash")
        source_control = (
            specificity_root
            / "shards"
            / f"candidate_{int(source_job['job_index']):03d}"
            / "persistence_state.safetensors"
        )
        control = root / "frozen_das/persistence_controls" / source_control.name
        if not source_control.exists():
            raise FileNotFoundError(
                f"persistence-control basis is absent: {source_control}"
            )
        control.parent.mkdir(parents=True, exist_ok=True)
        control = control.with_name(f"{source_job['artifact_id']}.safetensors")
        shutil.copy2(source_control, control)
        frozen.append(
            {
                **source_job,
                "artifact": str(artifact),
                "sha256": digest,
                "persistence_control": str(control),
                "persistence_control_sha256": _sha256(control),
                "primary": source_job["artifact_id"] == configured_primary,
                "retrained_for_abstraction_primary": False,
            }
        )
    primary = [row for row in frozen if row["primary"]]
    if len(primary) != 1:
        raise RuntimeError(
            "exactly one preregistered primary DAS candidate is required"
        )
    if expected_hash and primary[0]["sha256"] != expected_hash:
        raise RuntimeError(
            "primary DAS hash differs from the preregistered rank-2 solution"
        )
    _json(root / "frozen_das/manifest.json", frozen)
    return frozen, primary[0], gates


def _identifiability_metrics(
    contrasts: pd.DataFrame, selected_variables: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    columns = [f"delta_{value}" for value in ABSTRACTIONS]
    correlation = contrasts[columns].corr()
    rows = []
    for left in columns:
        for right in columns:
            rows.append(
                {
                    "record_type": "delta_correlation",
                    "left": left.removeprefix("delta_"),
                    "right": right.removeprefix("delta_"),
                    "value": float(correlation.loc[left, right]),
                }
            )
    for family, part in contrasts.groupby("contrast_family"):
        rows.append(
            {
                "record_type": "family_count",
                "left": family,
                "right": "contrasts",
                "value": int(len(part)),
            }
        )
        family_sds = {
            abstraction: float(part[f"delta_{abstraction}"].std())
            for abstraction in ABSTRACTIONS
        }
        for abstraction, value in family_sds.items():
            rows.append(
                {
                    "record_type": "effect_sd",
                    "left": family,
                    "right": abstraction,
                    "value": value,
                }
            )
    off_diagonal = correlation.to_numpy()[~np.eye(len(columns), dtype=bool)]
    summary = {
        "semantic_conditions": int(selected_variables.condition_id.nunique()),
        "contrast_rows": len(contrasts),
        "contrast_families": int(contrasts.contrast_family.nunique()),
        "maximum_absolute_delta_correlation": float(np.nanmax(np.abs(off_diagonal))),
        "minimum_family_count": int(contrasts.groupby("contrast_family").size().min()),
        # An identifiability family deliberately holds one abstraction fixed.
        # Require a broad discriminating effect in at least one preregistered
        # abstraction per family rather than penalizing the intended equality.
        "minimum_discriminating_effect_sd": float(
            min(
                max(
                    float(part[f"delta_{abstraction}"].std())
                    for abstraction in ABSTRACTIONS
                )
                for _, part in contrasts.groupby("contrast_family")
            )
        ),
    }
    return pd.DataFrame(rows), summary


def prepare_abstraction_run(
    config: dict,
    *,
    specificity_output: str | Path | None = None,
    theory_output: str | Path | None = None,
    output: str | Path | None = None,
):
    """Freeze all computational targets and select the dissociation dataset on CPU."""

    root = _root(config, output)
    specificity_root = Path(
        specificity_output
        or config.get("specificity_output", "artifacts/causal_specificity_v2")
    )
    theory_root = Path(
        theory_output or config.get("theory_output", "artifacts/theory_resolution_v1")
    )
    root.mkdir(parents=True, exist_ok=True)
    frozen, primary, source_gates = _freeze_das_inputs(root, specificity_root, config)
    bank = FrozenAbstractionBank(
        theory_root / "frozen_models",
        reference_architecture=config.get("behavioral", {}).get(
            "reference_architecture", "dual_history"
        ),
    )
    behavioral_manifest = bank.manifest()
    behavioral_manifest["file_sha256"] = bank.file_hashes()
    write_frozen_manifest(root / "behavioral/frozen_models.json", behavioral_manifest)

    conditions, design_metadata = generate_candidate_conditions(config)
    variables = bank.score_conditions(
        conditions,
        chunk_size=int(config.get("design", {}).get("prediction_chunk_size", 5000)),
    ).merge(design_metadata, on="condition_id", validate="one_to_one")
    condition_table = pd.DataFrame(condition_rows(conditions))
    candidate_pool = condition_table.merge(
        variables, on=["condition_id", "task_family"], validate="one_to_one"
    )
    candidate_path = write_records(
        candidate_pool.to_dict("records"), root / "design/candidate_pool.parquet"
    )
    contrasts = select_identifiability_contrasts(variables, config)
    selected_ids = set(contrasts.base_semantic_id) | set(contrasts.source_semantic_id)
    target_conditions = int(
        config.get("design", {}).get("target_semantic_conditions", 2400)
    )
    maximum_conditions = int(
        config.get("design", {}).get("maximum_semantic_conditions", 4000)
    )
    if len(selected_ids) > maximum_conditions:
        raise RuntimeError("matched contrasts exceed the maximum semantic dataset size")
    available = variables[~variables.condition_id.isin(selected_ids)].copy()
    coverage_needed = max(0, target_conditions - len(selected_ids))
    if coverage_needed > len(available):
        raise RuntimeError(
            "candidate pool cannot supply the preregistered coverage sample"
        )
    coverage = available.sample(
        n=coverage_needed, random_state=int(config.get("seed", 73001))
    )
    coverage = coverage[["condition_id", "task_family"]].copy()
    coverage["coverage_job_index"] = np.arange(len(coverage)) % len(CONTRAST_FAMILIES)
    coverage["selection_role"] = "coverage_random_validation"
    coverage.to_csv(root / "design/coverage_manifest.csv", index=False)
    selected_ids.update(coverage.condition_id)
    conditions_by_id = {condition.condition_id: condition for condition in conditions}
    selected_conditions = expand_selected_conditions(
        conditions_by_id, selected_ids, config
    )
    write_condition_jsonl(
        root / "design/mechanistic_conditions.jsonl", selected_conditions
    )
    selected_variables = variables[variables.condition_id.isin(selected_ids)].copy()
    write_records(
        selected_variables.to_dict("records"),
        root / "behavioral/abstraction_variables.parquet",
    )
    mapped = expand_contrast_mappings(contrasts)
    records_by_id = {
        condition.condition_id: condition for condition in selected_conditions
    }
    mapped = bank.score_pairs(mapped, records_by_id)
    contrast_path = write_records(
        mapped.to_dict("records"), root / "design/contrast_manifest.parquet"
    )
    metrics, identifiability = _identifiability_metrics(contrasts, selected_variables)
    metrics.to_csv(root / "design/identifiability_metrics.csv", index=False)
    _json(
        root / "design/prediction_matrix.json",
        {
            "same_raw_different_context": {"O": "no_change", "O_star": "change"},
            "same_context_different_raw": {"O": "change", "O_star": "no_change"},
            "same_history_contribution": {"H": "no_change"},
            "same_history_different_evidence": {"H": "no_change", "E": "change"},
            "same_total_evidence": {"E": "no_change"},
            "same_effect_different_cause": {
                "upstream_route": "different",
                "E_effect": "matched",
            },
            "frozen_before_neural": True,
        },
    )
    jobs = []
    for index, family in enumerate(CONTRAST_FAMILIES):
        family_rows = mapped[mapped.contrast_family == family]
        jobs.append(
            {
                "job_index": index,
                "contrast_family": family,
                "pair_rows": len(family_rows),
                "semantic_contrasts": int(family_rows.semantic_contrast_id.nunique()),
                "coverage_semantic_conditions": int(
                    (coverage.coverage_job_index == index).sum()
                ),
            }
        )
    _json(root / "design/evaluation_jobs.json", jobs)
    metadata = {
        "protocol_version": config.get("protocol_version", "causal_abstraction_v1"),
        "git_commit": _git_commit(),
        "source_specificity_root": str(specificity_root),
        "source_mechanistic_outcome": source_gates.get("mechanistic_outcome"),
        "behavioral_theory_root": str(theory_root),
        "primary_artifact_id": primary["artifact_id"],
        "primary_layer": int(primary["layer"]),
        "primary_rank": int(primary["rank"]),
        "primary_sha256": primary["sha256"],
        "primary_retrained": False,
        "candidate_conditions": len(candidate_pool),
        "selected_semantic_conditions": len(selected_ids),
        "coverage_semantic_conditions": len(coverage),
        "identifiability": identifiability,
        "full_activations_saved": False,
        "circuit_eligible": False,
    }
    _json(root / "run_metadata.json", metadata)
    _json(
        root / "gates.json",
        {
            "source_level4": True,
            "source_variable_identity": False,
            "design": {"completed": True, **identifiability},
            "causal_abstraction": {"status": "awaiting_frozen_das_interventions"},
            "depth_transformation": {"status": "awaiting_preselected_layers"},
            "circuit": {"status": "not_eligible"},
        },
    )
    return {
        "candidate_pool": candidate_path,
        "contrast_manifest": contrast_path,
        "candidate_conditions": len(candidate_pool),
        "selected_semantic_conditions": len(selected_ids),
        "contrast_rows": len(mapped),
        "evaluation_jobs": len(jobs),
    }


def _interchange_effect(
    runner,
    record,
    base_state,
    source_state,
    baseline_logit,
    *,
    layer: int,
    basis=None,
    target_norm=None,
):
    captured = {}

    def editor(state):
        edited = (
            replace_whole_state(state, source_state)
            if basis is None
            else interchange_subspace(state, source_state, basis)
        )
        if target_norm is not None:
            difference = edited - state
            current = difference.float().norm()
            if float(current.detach().cpu()) > 1e-12:
                edited = state + difference * (float(target_norm) / current)
        captured["norm"] = intervention_norm(state, edited)
        return edited

    changed = _forward(runner, record, layer=int(layer), editor=editor)
    return float(changed.persistence_logit - baseline_logit), float(captured["norm"])


def _row_predictions(row) -> dict:
    return {
        column: float(getattr(row, column)) for column in PREDICTION_COLUMNS.values()
    }


def evaluate_abstraction_job(
    config: dict,
    *,
    job_index: int,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
):
    """Run one contrast-family shard with continuously active Qwen forwards."""

    root = _root(config, output)
    jobs = json.loads((root / "design/evaluation_jobs.json").read_text())
    job_index = int(job_index)
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"abstraction job must lie in 0..{len(jobs) - 1}")
    job = jobs[job_index]
    pairs = read_records(_record_path(root, "design/contrast_manifest.parquet"))
    pairs = pairs[pairs.contrast_family == job["contrast_family"]].copy()
    if limit is not None:
        ids = pairs.semantic_contrast_id.drop_duplicates().head(int(limit))
        pairs = pairs[pairs.semantic_contrast_id.isin(ids)]
    claim_pairs = pairs[pairs.pair_split == "abstraction_test"].copy()
    records = load_condition_manifest(root / "design/mechanistic_conditions.jsonl")
    records_by_id = {record.condition_id: record for record in records}
    coverage = pd.read_csv(root / "design/coverage_manifest.csv")
    coverage = coverage[coverage.coverage_job_index == job_index]
    condition_ids = set(pairs.base_condition_id) | set(pairs.source_condition_id)
    for semantic_id in coverage.condition_id:
        condition_ids.update((f"{semantic_id}-m0", f"{semantic_id}-m1"))
    runner = _make_runner(
        config, model_path=model_path, revision=revision, online=online
    )
    layers = tuple(
        map(int, config.get("depth", {}).get("layers", (16, 20, 24, 28, 30)))
    )
    if not set(layers).issubset(set(range(runner.layer_count))):
        raise ValueError("preselected abstraction layer is outside the loaded model")
    states, logits = {}, {}
    for condition_id in sorted(condition_ids):
        result = _forward(runner, records_by_id[condition_id], capture_layers=layers)
        states[condition_id] = result.states
        logits[condition_id] = float(result.persistence_logit)
    candidates = json.loads((root / "frozen_das/manifest.json").read_text())
    bases = {}
    for candidate in candidates:
        if _sha256(Path(candidate["artifact"])) != candidate["sha256"]:
            raise RuntimeError("frozen DAS hash changed after preparation")
        bases[candidate["artifact_id"]] = load_alignment(candidate["artifact"])

    natural_rows, coverage_rows = [], []
    for pair in pairs.itertuples():
        for candidate in candidates:
            layer = int(candidate["layer"])
            basis = bases[candidate["artifact_id"]]
            base_z = states[pair.base_condition_id][layer] @ basis
            source_z = states[pair.source_condition_id][layer] @ basis
            delta = np.asarray(source_z - base_z, dtype=float)
            natural_rows.append(
                {
                    **pair._asdict(),
                    "artifact_id": candidate["artifact_id"],
                    "layer": layer,
                    "rank": int(candidate["rank"]),
                    "primary": bool(candidate["primary"]),
                    "projection_distance": float(np.linalg.norm(delta)),
                    "signed_projection_delta": float(delta.mean()),
                    "projection_delta": json.dumps(delta.tolist()),
                }
            )
        for layer in layers:
            difference = (
                states[pair.source_condition_id][layer]
                - states[pair.base_condition_id][layer]
            )
            natural_rows.append(
                {
                    **pair._asdict(),
                    "artifact_id": f"whole_state_L{layer}",
                    "layer": layer,
                    "rank": len(difference),
                    "primary": False,
                    "projection_distance": float(
                        np.linalg.norm(difference) / np.sqrt(len(difference))
                    ),
                    "signed_projection_delta": float(np.mean(difference)),
                    "projection_delta": None,
                }
            )
    for semantic_id in coverage.condition_id:
        for mapping_index in (0, 1):
            condition_id = f"{semantic_id}-m{mapping_index}"
            for candidate in candidates:
                layer = int(candidate["layer"])
                coordinates = (
                    states[condition_id][layer] @ bases[candidate["artifact_id"]]
                )
                coverage_rows.append(
                    {
                        "condition_id": condition_id,
                        "semantic_condition_id": semantic_id,
                        "task_family": records_by_id[condition_id].task_family,
                        "artifact_id": candidate["artifact_id"],
                        "layer": layer,
                        "coordinates": json.dumps(np.asarray(coordinates).tolist()),
                        "persistence_logit": logits[condition_id],
                    }
                )

    interchange_rows, depth_rows, control_rows = [], [], []
    candidate_norms = {}
    for pair in claim_pairs.itertuples():
        base_record = records_by_id[pair.base_condition_id]
        baseline = logits[pair.base_condition_id]
        for candidate in candidates:
            layer = int(candidate["layer"])
            effect, norm = _interchange_effect(
                runner,
                base_record,
                states[pair.base_condition_id][layer],
                states[pair.source_condition_id][layer],
                baseline,
                layer=layer,
                basis=bases[candidate["artifact_id"]],
            )
            interchange_rows.append(
                {
                    **pair._asdict(),
                    **_row_predictions(pair),
                    "artifact_id": candidate["artifact_id"],
                    "layer": layer,
                    "rank": int(candidate["rank"]),
                    "primary": bool(candidate["primary"]),
                    "intervention_type": "frozen_DAS",
                    "neural_counterfactual_effect": effect,
                    "intervention_norm": norm,
                }
            )
            if candidate["primary"]:
                candidate_norms[pair.pair_id] = norm
        for layer in layers:
            effect, norm = _interchange_effect(
                runner,
                base_record,
                states[pair.base_condition_id][layer],
                states[pair.source_condition_id][layer],
                baseline,
                layer=layer,
                basis=None,
            )
            depth_rows.append(
                {
                    **pair._asdict(),
                    **_row_predictions(pair),
                    "artifact_id": f"whole_state_L{layer}",
                    "layer": layer,
                    "intervention_type": "whole_state",
                    "neural_counterfactual_effect": effect,
                    "intervention_norm": norm,
                }
            )
        primary = next(value for value in candidates if value["primary"])
        layer = int(primary["layer"])
        persistence_basis = load_alignment(primary["persistence_control"])
        effect, norm = _interchange_effect(
            runner,
            base_record,
            states[pair.base_condition_id][layer],
            states[pair.source_condition_id][layer],
            baseline,
            layer=layer,
            basis=persistence_basis,
        )
        control_rows.append(
            {
                **pair._asdict(),
                **_row_predictions(pair),
                "artifact_id": "persistence_state",
                "layer": layer,
                "intervention_type": "persistence_state",
                "neural_counterfactual_effect": effect,
                "intervention_norm": norm,
            }
        )
        trial = _trial(base_record)
        output_basis = runner.choice_output_direction(
            list(trial.messages),
            base_record.response_mapping.labels,
            positive_label=base_record.response_mapping.continue_label,
        )[:, None]
        effect, norm = _interchange_effect(
            runner,
            base_record,
            states[pair.base_condition_id][layer],
            states[pair.source_condition_id][layer],
            baseline,
            layer=layer,
            basis=output_basis,
        )
        control_rows.append(
            {
                **pair._asdict(),
                **_row_predictions(pair),
                "artifact_id": "persistence_output",
                "layer": layer,
                "intervention_type": "persistence_output",
                "neural_counterfactual_effect": effect,
                "intervention_norm": norm,
            }
        )

    random_rows = []
    random_count = int(config.get("controls", {}).get("random_subspaces", 100))
    random_semantic = int(config.get("controls", {}).get("random_pairs_per_family", 5))
    random_ids = claim_pairs.semantic_contrast_id.drop_duplicates().head(
        random_semantic
    )
    random_pairs = claim_pairs[claim_pairs.semantic_contrast_id.isin(random_ids)]
    primary = next(value for value in candidates if value["primary"])
    primary_basis = bases[primary["artifact_id"]]
    primary_random_rows = pd.DataFrame(interchange_rows)
    primary_random_rows = primary_random_rows[
        (primary_random_rows.artifact_id == primary["artifact_id"])
        & primary_random_rows.pair_id.isin(random_pairs.pair_id)
    ]
    for abstraction in ABSTRACTIONS:
        metrics = counterfactual_metrics(
            primary_random_rows[PREDICTION_COLUMNS[abstraction]],
            primary_random_rows.neural_counterfactual_effect,
        )
        random_rows.append(
            {
                "job_index": job_index,
                "contrast_family": job["contrast_family"],
                "random_index": -1,
                "control_role": "candidate_matched_subset",
                "abstraction": abstraction,
                "layer": int(primary["layer"]),
                "rank": int(primary["rank"]),
                "row_set_sha256": pair_row_hash(primary_random_rows.pair_id),
                **{name: metrics[name] for name in STABLE_METRICS},
            }
        )
    random_bases = orthonormal_random_subspaces(
        primary_basis.shape[0],
        primary_basis.shape[1],
        random_count,
        seed=int(config.get("seed", 73001)) + 1000 * job_index,
    )
    for random_index, random_basis in enumerate(random_bases):
        effects = []
        for pair in random_pairs.itertuples():
            layer = int(primary["layer"])
            effect, norm = _interchange_effect(
                runner,
                records_by_id[pair.base_condition_id],
                states[pair.base_condition_id][layer],
                states[pair.source_condition_id][layer],
                logits[pair.base_condition_id],
                layer=layer,
                basis=random_basis,
                target_norm=candidate_norms[pair.pair_id],
            )
            effects.append(
                {
                    **pair._asdict(),
                    **_row_predictions(pair),
                    "neural_counterfactual_effect": effect,
                    "intervention_norm": norm,
                }
            )
        effect_frame = pd.DataFrame(effects)
        for abstraction in ABSTRACTIONS:
            metrics = counterfactual_metrics(
                effect_frame[PREDICTION_COLUMNS[abstraction]],
                effect_frame.neural_counterfactual_effect,
            )
            random_rows.append(
                {
                    "job_index": job_index,
                    "contrast_family": job["contrast_family"],
                    "random_index": random_index,
                    "control_role": "random_subspace",
                    "abstraction": abstraction,
                    "layer": int(primary["layer"]),
                    "rank": int(primary["rank"]),
                    "row_set_sha256": pair_row_hash(effect_frame.pair_id),
                    **{name: metrics[name] for name in STABLE_METRICS},
                }
            )

    shard = root / "shards" / f"family_{job_index:02d}"
    paths = {
        "natural": write_records(natural_rows, shard / "natural_geometry.parquet"),
        "coverage": write_records(
            coverage_rows, shard / "coverage_projections.parquet"
        ),
        "interchange": write_records(
            interchange_rows, shard / "interchange_results.parquet"
        ),
        "depth": write_records(depth_rows, shard / "depth_results.parquet"),
        "controls": write_records(control_rows, shard / "control_results.parquet"),
    }
    pd.DataFrame(random_rows).to_csv(shard / "random_subspaces.csv", index=False)
    _json(
        shard / "audit.json",
        {
            "job_index": job_index,
            "contrast_family": job["contrast_family"],
            "computational_targets_frozen_before_neural": bool(
                pairs.computational_targets_frozen_before_neural.astype(bool).all()
            ),
            "primary_alignment_sha256": primary["sha256"],
            "primary_retrained": False,
            "claim_pair_row_sha256": pair_row_hash(claim_pairs.pair_id),
            "full_activations_saved": False,
            "gpu_forward_phases_only": True,
        },
    )
    return {"job_index": job_index, "contrast_family": job["contrast_family"], **paths}


def _load_shards(root: Path, jobs: list[dict], name: str) -> pd.DataFrame:
    frames = []
    for job in jobs:
        path = _record_path(
            root, f"shards/family_{int(job['job_index']):02d}/{name}.parquet"
        )
        if not path.exists():
            raise RuntimeError(f"abstraction shard is incomplete: {path}")
        frames.append(read_records(path))
    return pd.concat(frames, ignore_index=True)


def _method_matrix(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, part in frame.groupby(group_column):
        metrics = abstraction_metric_table(part)
        for row in metrics.to_dict("records"):
            rows.append({group_column: group, **row})
    return pd.DataFrame(rows)


def _task_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, part in frame.groupby("task_family"):
        table = abstraction_metric_table(part)
        rows.extend({"task_family": task, **row} for row in table.to_dict("records"))
    return pd.DataFrame(rows)


def _control_comparisons(
    candidate: pd.DataFrame,
    controls: pd.DataFrame,
    winner: str,
    *,
    samples: int,
    confidence: float,
    seed: int,
):
    prediction = PREDICTION_COLUMNS[winner]
    rows = []
    candidate_values = candidate[
        [
            "pair_id",
            "semantic_contrast_id",
            "task_family",
            prediction,
            "neural_counterfactual_effect",
        ]
    ].rename(columns={"neural_counterfactual_effect": "candidate_observed"})
    for index, (control, part) in enumerate(controls.groupby("intervention_type")):
        control_values = part[["pair_id", "neural_counterfactual_effect"]].rename(
            columns={"neural_counterfactual_effect": "control_observed"}
        )
        merged = candidate_values.merge(
            control_values, on="pair_id", validate="one_to_one"
        )
        if len(merged) != len(candidate_values):
            raise RuntimeError("persistence control used a different claim row set")
        result = paired_cfr_difference(
            merged,
            candidate_prediction=prediction,
            candidate_observed="candidate_observed",
            control_observed="control_observed",
            samples=samples,
            confidence=confidence,
            seed=seed + index,
        )
        rows.append({"control": control, **result})
    return pd.DataFrame(rows)


def _depth_table(depth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, part in depth.groupby("layer"):
        table = abstraction_metric_table(part)
        rows.extend({"layer": int(layer), **row} for row in table.to_dict("records"))
    return pd.DataFrame(rows)


def _figures(
    root: Path,
    contrasts: pd.DataFrame,
    natural: pd.DataFrame,
    primary_id: str,
    primary_metrics: pd.DataFrame,
    primary_intervals: pd.DataFrame,
    cross_matrix: pd.DataFrame,
    depth_table: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []
    delta_columns = [f"delta_{value}" for value in ABSTRACTIONS]
    correlation = contrasts[delta_columns].corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(4), ABSTRACTIONS)
    ax.set_yticks(range(4), ABSTRACTIONS)
    for row in range(4):
        for column in range(4):
            ax.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
            )
    ax.set_title("Experimental identifiability of abstraction deltas")
    fig.colorbar(image, ax=ax, label="correlation")
    fig.tight_layout()
    paths.append(figure_root / "figure1_experimental_identifiability.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    primary_natural = natural[natural.artifact_id == primary_id]
    family_distance = primary_natural.groupby(
        "contrast_family"
    ).projection_distance.mean()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(family_distance)), family_distance)
    ax.set_xticks(
        range(len(family_distance)), family_distance.index, rotation=45, ha="right"
    )
    ax.set(
        ylabel="Mean frozen-DAS projection distance", title="Causal dissociation grid"
    )
    fig.tight_layout()
    paths.append(figure_root / "figure2_causal_dissociation_grid.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    cfr_ci = primary_intervals[primary_intervals.metric == "global_cfr"].set_index(
        "abstraction"
    )
    shown = primary_metrics.set_index("abstraction").loc[list(ABSTRACTIONS)]
    errors = np.vstack(
        (
            shown.global_cfr - cfr_ci.loc[shown.index].ci_lower,
            cfr_ci.loc[shown.index].ci_upper - shown.global_cfr,
        )
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(range(4), shown.global_cfr, yerr=np.maximum(errors, 0), capsize=4)
    ax.set_xticks(range(4), ABSTRACTIONS)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(
        ylabel="Global CFR",
        title="Same frozen DAS intervention, competing abstractions",
    )
    fig.tight_layout()
    paths.append(figure_root / "figure3_frozen_das_abstractions.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    matrix = cross_matrix.pivot_table(
        index="artifact_id", columns="abstraction", values="global_cfr", aggfunc="first"
    ).reindex(columns=ABSTRACTIONS)
    fig, ax = plt.subplots(figsize=(7.8, max(4.5, 0.55 * len(matrix))))
    image = ax.imshow(
        matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto"
    )
    ax.set_xticks(range(4), ABSTRACTIONS)
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set_title("Cross-abstraction stable-CFR matrix")
    fig.colorbar(image, ax=ax, label="Global CFR")
    fig.tight_layout()
    paths.append(figure_root / "figure4_cross_abstraction_matrix.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    same_e = primary_natural[primary_natural.contrast_family == "same_total_evidence"]
    other = primary_natural[primary_natural.contrast_family != "same_total_evidence"]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.boxplot(
        [same_e.projection_distance, other.projection_distance],
        tick_labels=["same E / different cause", "other matched pairs"],
    )
    ax.set(
        ylabel="Frozen-DAS projection distance", title="Same-effect causal convergence"
    )
    fig.tight_layout()
    paths.append(figure_root / "figure5_same_e_convergence.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    depth_matrix = depth_table.pivot(
        index="layer", columns="abstraction", values="global_cfr"
    ).reindex(columns=ABSTRACTIONS)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(
        depth_matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto"
    )
    ax.set_xticks(range(4), ABSTRACTIONS)
    ax.set_yticks(range(len(depth_matrix)), depth_matrix.index)
    ax.set(
        xlabel="Abstraction",
        ylabel="Layer",
        title="Abstraction through preselected depths",
    )
    fig.colorbar(image, ax=ax, label="Global CFR")
    fig.tight_layout()
    paths.append(figure_root / "figure6_abstraction_through_depth.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(shuffle_summary.comparison, shuffle_summary.global_cfr)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(ylabel="Global CFR", title="High-identifiability shuffled-target test")
    fig.tight_layout()
    paths.append(figure_root / "figure7_shuffled_target.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)
    return paths


def aggregate_abstraction_run(
    config: dict,
    *,
    output: str | Path | None = None,
):
    """Resolve the represented abstraction, render figures, and enforce the stop rule."""

    root = _root(config, output)
    jobs = json.loads((root / "design/evaluation_jobs.json").read_text())
    for job in jobs:
        audit_path = (
            root / "shards" / f"family_{int(job['job_index']):02d}" / "audit.json"
        )
        if not audit_path.exists():
            raise RuntimeError(f"abstraction evaluation is incomplete: {audit_path}")
        audit = json.loads(audit_path.read_text())
        if not audit["computational_targets_frozen_before_neural"]:
            raise RuntimeError("neural outcomes leaked into computational targets")
        if audit["primary_retrained"] or audit["full_activations_saved"]:
            raise RuntimeError("primary freeze/no-activation-bank guard failed")
    natural = _load_shards(root, jobs, "natural_geometry")
    coverage = _load_shards(root, jobs, "coverage_projections")
    interchange = _load_shards(root, jobs, "interchange_results")
    depth = _load_shards(root, jobs, "depth_results")
    controls = _load_shards(root, jobs, "control_results")
    random_null = pd.concat(
        [
            pd.read_csv(
                root
                / "shards"
                / f"family_{int(job['job_index']):02d}"
                / "random_subspaces.csv"
            )
            for job in jobs
        ],
        ignore_index=True,
    )
    contrasts = read_records(_record_path(root, "design/contrast_manifest.parquet"))
    frozen = json.loads((root / "frozen_das/manifest.json").read_text())
    primary = next(value for value in frozen if value["primary"])
    primary_rows = interchange[
        (interchange.artifact_id == primary["artifact_id"])
        & (interchange.pair_split == "abstraction_test")
    ].copy()
    settings = config.get("bootstrap", {})
    samples = int(settings.get("samples", 2000))
    confidence = float(settings.get("confidence", 0.95))
    seed = int(config.get("seed", 73001))
    threshold = float(config.get("cfr", {}).get("threshold", 0.10))
    primary_metrics, primary_intervals = bootstrap_abstraction_metrics(
        primary_rows,
        samples=samples,
        confidence=confidence,
        seed=seed,
        threshold=threshold,
    )
    winner = str(
        primary_metrics.sort_values("global_cfr", ascending=False).iloc[0].abstraction
    )
    pairwise = paired_abstraction_differences(
        primary_rows,
        winner,
        samples=samples,
        confidence=confidence,
        seed=seed + 100,
        threshold=threshold,
    )
    prediction = PREDICTION_COLUMNS[winner]
    shuffled = shuffle_abstraction_targets(
        primary_rows,
        prediction_column=prediction,
        seed=seed + 200,
        magnitude_strata=int(
            config.get("controls", {}).get("shuffle_magnitude_strata", 5)
        ),
    )
    shuffled["true_prediction"] = shuffled.original_predicted_effect
    shuffle_difference = paired_cfr_difference(
        shuffled,
        candidate_prediction="true_prediction",
        control_prediction=prediction,
        samples=samples,
        confidence=confidence,
        seed=seed + 300,
        threshold=threshold,
    )
    true_metrics = counterfactual_metrics(
        shuffled.true_prediction,
        shuffled.neural_counterfactual_effect,
        threshold=threshold,
    )
    shuffled_metrics = counterfactual_metrics(
        shuffled[prediction], shuffled.neural_counterfactual_effect, threshold=threshold
    )
    shuffle_summary = pd.DataFrame(
        [
            {
                "comparison": "true",
                **{name: true_metrics[name] for name in STABLE_METRICS},
            },
            {
                "comparison": "shuffled",
                **{name: shuffled_metrics[name] for name in STABLE_METRICS},
            },
        ]
    )
    control_differences = _control_comparisons(
        primary_rows,
        controls[controls.pair_split == "abstraction_test"],
        winner,
        samples=samples,
        confidence=confidence,
        seed=seed + 400,
    )
    cross_matrix = _method_matrix(
        interchange[interchange.pair_split == "abstraction_test"], "artifact_id"
    )
    control_matrix = _method_matrix(
        controls[controls.pair_split == "abstraction_test"], "artifact_id"
    )
    cross_matrix = pd.concat([cross_matrix, control_matrix], ignore_index=True)
    if not bool(config.get("secondary_das", {}).get("enabled", False)):
        cross_matrix = pd.concat(
            [
                cross_matrix,
                pd.DataFrame(
                    [
                        {
                            "artifact_id": artifact,
                            "abstraction": abstraction,
                            "global_cfr": np.nan,
                            "status": "optional_secondary_not_run",
                        }
                        for artifact in ("S_H", "S_E")
                        for abstraction in ABSTRACTIONS
                    ]
                ),
            ],
            ignore_index=True,
        )
    depth_table = _depth_table(depth[depth.pair_split == "abstraction_test"])
    task_metrics = _task_metrics(primary_rows)
    primary_natural = natural[natural.artifact_id == primary["artifact_id"]].copy()
    natural_comparison = natural_model_comparison(primary_natural)
    factorial = factorial_regression(
        primary_natural,
        samples=samples,
        confidence=confidence,
        seed=seed + 500,
    )
    convergence = convergence_index(primary_natural, seed=seed + 600)

    (root / "frozen_das").mkdir(parents=True, exist_ok=True)
    natural.to_csv(root / "frozen_das/natural_geometry.csv", index=False)
    write_records(
        coverage.to_dict("records"), root / "frozen_das/coverage_projections.parquet"
    )
    write_records(
        interchange.to_dict("records"), root / "frozen_das/interchange_results.parquet"
    )
    primary_metrics.to_csv(root / "frozen_das/abstraction_metrics.csv", index=False)
    primary_intervals.to_csv(root / "frozen_das/bootstrap_intervals.csv", index=False)
    pairwise.to_csv(root / "frozen_das/paired_abstraction_differences.csv", index=False)
    natural_comparison.to_csv(
        root / "frozen_das/natural_model_comparison.csv", index=False
    )
    factorial.to_csv(root / "frozen_das/factorial_regression.csv", index=False)
    _json(root / "frozen_das/convergence.json", convergence)
    tests_root = root / "abstraction_tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    names = {
        CONTRAST_FAMILIES[0]: "same_raw_context.csv",
        CONTRAST_FAMILIES[1]: "same_context_raw.csv",
        CONTRAST_FAMILIES[2]: "same_history_contribution.csv",
        CONTRAST_FAMILIES[3]: "same_history_different_evidence.csv",
        CONTRAST_FAMILIES[4]: "same_total_evidence.csv",
        CONTRAST_FAMILIES[5]: "same_effect_different_cause.csv",
    }
    merged_tests = primary_rows.merge(
        primary_natural[["pair_id", "projection_distance", "signed_projection_delta"]],
        on="pair_id",
        validate="one_to_one",
    )
    for family, filename in names.items():
        merged_tests[merged_tests.contrast_family == family].to_csv(
            tests_root / filename, index=False
        )
    (root / "depth").mkdir(parents=True, exist_ok=True)
    depth_table.to_csv(root / "depth/abstraction_by_layer.csv", index=False)
    task_metrics.to_csv(root / "depth/task_general_abstraction.csv", index=False)
    cross_matrix.to_csv(
        root / "abstraction_tests/cross_abstraction_matrix.csv", index=False
    )
    (root / "controls").mkdir(parents=True, exist_ok=True)
    shuffled.to_csv(root / "controls/shuffled_target.csv", index=False)
    shuffle_summary.to_csv(root / "controls/shuffled_target_summary.csv", index=False)
    pd.DataFrame([shuffle_difference]).to_csv(
        root / "controls/shuffled_target_difference.csv", index=False
    )
    controls[controls.intervention_type == "persistence_state"].to_csv(
        root / "controls/persistence_control.csv", index=False
    )
    controls[controls.intervention_type == "persistence_output"].to_csv(
        root / "controls/persistence_output_control.csv", index=False
    )
    control_differences.to_csv(root / "controls/control_differences.csv", index=False)
    random_null.to_csv(root / "controls/random_subspaces.csv", index=False)

    intervals = primary_intervals.set_index(["abstraction", "metric"])
    winner_cfr = intervals.loc[(winner, "global_cfr")]
    winner_r = intervals.loc[(winner, "correlation")]
    candidates_random = random_null[
        (random_null.control_role == "candidate_matched_subset")
        & (random_null.abstraction == winner)
    ][["contrast_family", "global_cfr"]].rename(columns={"global_cfr": "candidate_cfr"})
    random_winner = random_null[
        (random_null.control_role == "random_subspace")
        & (random_null.abstraction == winner)
    ].merge(candidates_random, on="contrast_family", validate="many_to_one")
    random_p = float(
        (1 + int((random_winner.global_cfr >= random_winner.candidate_cfr).sum()))
        / (len(random_winner) + 1)
    )
    metadata = json.loads((root / "run_metadata.json").read_text())
    identifiability = metadata["identifiability"]
    gates_config = config.get("gates", {})
    design_passed = bool(
        2000 <= identifiability["semantic_conditions"] <= 4000
        and identifiability["minimum_family_count"]
        >= int(config.get("design", {}).get("contrasts_per_family", 200))
        and identifiability["maximum_absolute_delta_correlation"]
        <= float(gates_config.get("maximum_delta_correlation", 0.90))
        and identifiability["minimum_discriminating_effect_sd"]
        >= float(gates_config.get("minimum_effect_sd", 0.10))
    )
    recovery_passed = bool(winner_cfr.ci_lower > 0 and winner_r.ci_lower > 0)
    alternatives_passed = bool(len(pairwise) and (pairwise.ci_lower > 0).all())
    shuffled_passed = bool(shuffle_difference["ci_lower"] > 0)
    controls_passed = bool(
        len(control_differences) and (control_differences.ci_lower > 0).all()
    )
    random_passed = bool(random_p < float(gates_config.get("random_p_max", 0.05)))
    task_winner = task_metrics[task_metrics.abstraction == winner]
    task_general = bool(
        len(task_winner)
        and (task_winner.global_cfr > 0).mean()
        >= float(gates_config.get("minimum_task_fraction", 0.60))
    )
    abstraction_passed = bool(
        design_passed
        and recovery_passed
        and alternatives_passed
        and shuffled_passed
        and controls_passed
        and random_passed
        and task_general
    )
    order = {value: index for index, value in enumerate(ABSTRACTIONS)}
    depth_winners = (
        depth_table.sort_values("global_cfr", ascending=False)
        .groupby("layer", as_index=False)
        .first()
        .sort_values("layer")
    )
    depth_sequence = [order[value] for value in depth_winners.abstraction]
    systematic_depth = bool(
        len(depth_sequence) >= 3 and np.all(np.diff(depth_sequence) >= 0)
    )
    outcome = (
        {
            "O": "raw_history_correspondence",
            "O_star": "context_transformed_correspondence",
            "H": "integrated_history_correspondence",
            "E": "general_persistence_evidence_correspondence",
        }[winner]
        if abstraction_passed
        else "no_candidate_abstraction_passed"
    )
    gates = {
        "source_level4": True,
        "design": {"passed": design_passed, **identifiability},
        "causal_abstraction": {
            "passed": abstraction_passed,
            "winner": winner,
            "outcome": outcome,
            "recovery_passed": recovery_passed,
            "neighbor_dissociations_passed": alternatives_passed,
            "shuffled_target_passed": shuffled_passed,
            "persistence_controls_passed": controls_passed,
            "random_subspaces_passed": random_passed,
            "random_p": random_p,
            "task_general": task_general,
        },
        "depth_transformation": {
            "systematic": systematic_depth,
            "sequence": depth_winners[["layer", "abstraction"]].to_dict("records"),
        },
        "circuit": {
            "status": "eligible" if abstraction_passed else "not_eligible",
            "submitted": False,
        },
        "highest_evidence": "C" if abstraction_passed else "B",
    }
    _json(root / "gates.json", gates)
    figures = _figures(
        root,
        contrasts,
        natural,
        primary["artifact_id"],
        primary_metrics,
        primary_intervals,
        cross_matrix,
        depth_table,
        shuffle_summary,
    )
    _report(
        root,
        gates,
        primary,
        primary_metrics,
        natural_comparison,
        convergence,
        depth_winners,
        control_differences,
        shuffle_difference,
    )
    metadata.update(
        {
            "winner": winner,
            "causal_abstraction_passed": abstraction_passed,
            "mechanistic_outcome": outcome,
            "figure_count": len(figures),
            "full_activations_saved": False,
            "circuit_eligible": abstraction_passed,
        }
    )
    _json(root / "run_metadata.json", metadata)
    return {
        "winner": winner,
        "causal_abstraction_passed": abstraction_passed,
        "outcome": outcome,
        "random_p": random_p,
        "figures": len(figures),
        "report": root / "report.md",
    }


def _report(
    root,
    gates,
    primary,
    metrics,
    natural_comparison,
    convergence,
    depth_winners,
    control_differences,
    shuffle_difference,
):
    result = gates["causal_abstraction"]
    winner = result["winner"]
    causal_best = metrics.set_index("abstraction").loc[winner]
    natural_best = natural_comparison.iloc[0]
    claim = {
        "raw_history_correspondence": "The causal neural state corresponds most closely to raw outcome history.",
        "context_transformed_correspondence": "The model context-transforms history before the identified causal stage.",
        "integrated_history_correspondence": "The causal state represents the integrated history contribution rather than separate history terms.",
        "general_persistence_evidence_correspondence": "History and other task evidence converge on a shared persistence-evidence state.",
        "no_candidate_abstraction_passed": "No candidate abstraction uniquely characterizes the frozen persistence controller.",
    }[result["outcome"]]
    same_context = root / "abstraction_tests/same_raw_context.csv"
    same_raw = root / "abstraction_tests/same_context_raw.csv"
    same_h = root / "abstraction_tests/same_history_contribution.csv"
    same_e = root / "abstraction_tests/same_total_evidence.csv"
    family_values = {
        "same_raw_context": pd.read_csv(same_context).projection_distance.mean(),
        "same_context_raw": pd.read_csv(same_raw).projection_distance.mean(),
        "same_h": pd.read_csv(same_h).projection_distance.mean(),
        "same_e": pd.read_csv(same_e).projection_distance.mean(),
    }
    depth_text = ", ".join(
        f"L{int(row.layer)}={row.abstraction}" for row in depth_winners.itertuples()
    )
    report = [
        "# Causal Abstraction Level of Persistence Computation",
        "",
        f"**Result:** {claim}",
        "",
        "## Automated questions",
        "",
        f"1. The dataset decorrelated O/O*/H/E sufficiently: **{'yes' if gates['design']['passed'] else 'no'}** (maximum |r|={gates['design']['maximum_absolute_delta_correlation']:.3f}).",
        f"2. Same-O/different-O* frozen-DAS distance: **{family_values['same_raw_context']:.3f}**.",
        f"3. Different-O/same-O* frozen-DAS distance: **{family_values['same_context_raw']:.3f}**.",
        f"4. Different histories with equal H converge: **{'yes' if family_values['same_h'] < np.nanmedian(list(family_values.values())) else 'no'}** (distance={family_values['same_h']:.3f}).",
        f"5. Different causal routes with equal E converge: **{'yes' if convergence['convergence_index'] > 0 else 'no'}** (CE={convergence['convergence_index']:.3f}).",
        f"6. Best natural-state geometry model: **{natural_best.abstraction}** (held-out R²={natural_best.r2_pair:.3f}).",
        f"7. Best frozen-DAS intervention abstraction: **{winner}** (CFR_G={causal_best.global_cfr:.3f}, r={causal_best.correlation:.3f}).",
        f"8. It beats shuffled targets: **{'yes' if result['shuffled_target_passed'] else 'no'}** (paired lower CI={shuffle_difference['ci_lower']:.3f}).",
        f"9. It beats persistence/output controls: **{'yes' if result['persistence_controls_passed'] else 'no'}** (minimum lower CI={control_differences.ci_lower.min():.3f}).",
        f"10. Systematic abstraction-through-depth pattern: **{'yes' if gates['depth_transformation']['systematic'] else 'no'}** ({depth_text}).",
        f"11. The winning abstraction generalizes across tasks: **{'yes' if result['task_general'] else 'no'}**.",
        f"12. Neural coordinates are shared across tasks: **{'supported' if result['task_general'] and primary['scope'] == 'shared' else 'not established'}**; no target-task refitting occurred.",
        f"13. Best-supported computational decomposition: **{result['outcome']}**.",
        f"14. Behavioral-theory refinement: **{claim}**",
        f"15. Justified mechanistic claim: **{claim if result['passed'] else 'Level-4 causal controller only; abstraction identity remains unresolved.'}**",
        "",
        "## Guardrails",
        "",
        "- Behavioral O/O*/H/E targets and the prediction matrix were frozen before neural evaluation.",
        "- The primary layer-28/rank-2 DAS rotation was hash-verified and never retrained.",
        "- Every abstraction scored the same neural interventions on identical untouched test rows.",
        "- Depth analysis used only preregistered layers 16/20/24/28/30.",
        "- Integrated H/E DAS training remains optional and was not used to choose the primary result.",
        "- No full activation bank was retained.",
        "- Circuit localization remains blocked unless the complete causal-abstraction gate passes.",
    ]
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

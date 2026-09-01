"""Pre-OOD freezing of the signed persistence-evidence intervention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cognitive_discovery.causal_mechanistic.das import load_alignment
from cognitive_discovery.causal_specificity.controls import orthonormal_random_subspaces
from cognitive_discovery.data.storage import read_records


FORBIDDEN_LENGTH_KEYS = {"max_length", "max_new_tokens", "min_length", "min_new_tokens"}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path: str | Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def resolve_record_path(path: Path) -> Path:
    if path.exists():
        return path
    alternate = path.with_suffix(".csv.gz")
    return alternate if alternate.exists() else path


def resolve_source_path(value: str | Path, source_root: Path, fallback: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = source_root / fallback / path.name
    return candidate if candidate.exists() else path


def validate_sampling_config(config: dict) -> dict:
    """Reject application-level length caps from the scientific configuration."""

    forbidden = sorted(FORBIDDEN_LENGTH_KEYS & set(config))
    if forbidden:
        raise ValueError(f"OOD sampling contains forbidden output cap(s): {forbidden}")
    output = {
        "do_sample": bool(config.get("do_sample", True)),
        "temperature": float(config.get("temperature", 1.0)),
        "top_p": float(config.get("top_p", 1.0)),
        "top_k": config.get("top_k"),
        "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
        "thinking_enabled": bool(config.get("thinking_enabled", False)),
        "count_all_autoregressive_tokens": True,
        "scientific_stop_rule": "canonical_eos_or_architectural_context_capacity",
        "application_token_cap": None,
        "context_safety_margin": int(config.get("context_safety_margin", 1)),
    }
    if not output["do_sample"]:
        raise ValueError("primary OOD decoding must remain stochastic")
    if output["temperature"] <= 0:
        raise ValueError("sampling temperature must be positive")
    if not 0 < output["top_p"] <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    if output["top_k"] not in (None, 0):
        raise ValueError(
            "top_k must be disabled for the frozen primary decoding policy"
        )
    if output["repetition_penalty"] != 1.0:
        raise ValueError("repetition penalty would alter continuation behavior")
    if output["thinking_enabled"]:
        raise ValueError("the frozen Qwen protocol uses thinking disabled")
    if output["context_safety_margin"] < 0:
        raise ValueError("context safety margin cannot be negative")
    return output


def fit_e_orientation(
    basis: np.ndarray,
    coordinates: np.ndarray,
    evidence: np.ndarray,
) -> dict:
    """Fit E only inside a frozen orthonormal DAS coordinate system."""

    basis = np.asarray(basis, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    evidence = np.asarray(evidence, dtype=np.float64).reshape(-1)
    if basis.ndim != 2 or coordinates.ndim != 2:
        raise ValueError("basis and DAS coordinates must both be matrices")
    if coordinates.shape != (len(evidence), basis.shape[1]):
        raise ValueError("structured E rows do not match the frozen DAS rank")
    if len(evidence) < max(20, 5 * basis.shape[1]):
        raise ValueError("too few pre-OOD structured rows to orient the DAS subspace")
    design = np.column_stack((np.ones(len(coordinates)), coordinates))
    coefficients = np.linalg.lstsq(design, evidence, rcond=None)[0]
    coordinate_weights = coefficients[1:]
    weight_norm = float(np.linalg.norm(coordinate_weights))
    if not np.isfinite(weight_norm) or weight_norm <= 1e-12:
        raise RuntimeError(
            "E has no orientable direction inside the frozen DAS subspace"
        )
    coordinate_direction = coordinate_weights / weight_norm
    signed_coordinates = coordinates @ coordinate_direction
    correlation = float(np.corrcoef(signed_coordinates, evidence)[0, 1])
    if correlation < 0:
        coordinate_direction *= -1
        signed_coordinates *= -1
        correlation *= -1
    hidden_direction = basis @ coordinate_direction
    hidden_direction /= np.linalg.norm(hidden_direction)
    sigma = float(np.std(signed_coordinates, ddof=1))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        raise RuntimeError("structured activation scale is zero or non-finite")
    fitted = design @ coefficients
    residual = float(np.sum(np.square(evidence - fitted)))
    total = float(np.sum(np.square(evidence - evidence.mean())))
    median = float(np.median(signed_coordinates))
    lower = evidence[signed_coordinates <= median]
    upper = evidence[signed_coordinates > median]
    sign_check = {
        "low_projection_mean_E": float(lower.mean()),
        "high_projection_mean_E": float(upper.mean()),
        "passed": bool(upper.mean() > lower.mean()),
    }
    if not sign_check["passed"]:
        raise RuntimeError(
            "signed E direction failed its pre-OOD high/low projection test"
        )
    return {
        "hidden_direction": hidden_direction.astype(np.float32),
        "coordinate_direction": coordinate_direction.astype(np.float32),
        "coordinate_weights": coordinate_weights.astype(np.float32),
        "intercept": float(coefficients[0]),
        "sigma_E": sigma,
        "correlation": correlation,
        "r2": float(1.0 - residual / total) if total > 0 else 0.0,
        "rows": int(len(evidence)),
        "sign_check": sign_check,
    }


def save_orientation(path: str | Path, orientation: dict, *, metadata: dict) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hidden_direction": torch.as_tensor(orientation["hidden_direction"]),
        "coordinate_direction": torch.as_tensor(orientation["coordinate_direction"]),
        "coordinate_weights": torch.as_tensor(orientation["coordinate_weights"]),
        "sigma_E": torch.tensor(float(orientation["sigma_E"]), dtype=torch.float32),
        "intercept": torch.tensor(float(orientation["intercept"]), dtype=torch.float32),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }
    torch.save(payload, path)
    return path


def load_orientation(path: str | Path) -> dict:
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return {
        "hidden_direction": payload["hidden_direction"].float().numpy(),
        "coordinate_direction": payload["coordinate_direction"].float().numpy(),
        "coordinate_weights": payload["coordinate_weights"].float().numpy(),
        "sigma_E": float(payload["sigma_E"]),
        "intercept": float(payload["intercept"]),
        "metadata": json.loads(payload["metadata_json"]),
    }


def prompt_rows(config: dict) -> list[dict]:
    rows = []
    for family, prompts in config.get("prompts", {}).items():
        if family not in {"open_ended", "fixed_topic", "continuation"}:
            continue
        for index, prompt in enumerate(prompts):
            prompt = str(prompt).strip()
            if not prompt:
                raise ValueError("OOD prompts cannot be empty")
            lowered = prompt.lower()
            forbidden = (
                "be concise",
                "detailed answer",
                "keep writing",
                "at least",
                "few sentences",
            )
            if any(value in lowered for value in forbidden):
                raise ValueError(f"prompt contains a length instruction: {prompt}")
            rows.append(
                {
                    "prompt_id": f"{family}_{index + 1:02d}",
                    "prompt_family": family,
                    "text": prompt,
                    "primary": family == "open_ended",
                }
            )
    primary = [row for row in rows if row["primary"]]
    if len(primary) < 2:
        raise ValueError(
            "the primary result must use multiple frozen open-ended prompts"
        )
    return rows


def freeze_protocol(config: dict, source_root: Path, root: Path) -> dict:
    """Freeze the complete intervention and analysis design before OOD inference."""

    gates = json.loads((source_root / "gates.json").read_text(encoding="utf-8"))
    abstraction = gates.get("causal_abstraction", {})
    if (
        config.get("entry", {}).get("require_e_winner", True)
        and abstraction.get("winner") != "E"
    ):
        raise RuntimeError(
            "OOD persistence validation requires the pre-OOD winner to be E"
        )
    if config.get("entry", {}).get(
        "require_abstraction_gate", False
    ) and not abstraction.get("passed"):
        raise RuntimeError(
            "the preregistered source causal-abstraction gate did not pass"
        )
    source_manifest_path = source_root / "frozen_das/manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    primary = [row for row in source_manifest if row.get("primary")]
    if len(primary) != 1:
        raise RuntimeError("exactly one frozen primary DAS artifact is required")
    primary = primary[0]
    expected_hash = config.get("frozen", {}).get("expected_das_sha256")
    source_das = resolve_source_path(
        primary["artifact"], source_root, "frozen_das/alignments"
    )
    observed_hash = sha256_file(source_das)
    if observed_hash != primary["sha256"] or (
        expected_hash and observed_hash != expected_hash
    ):
        raise RuntimeError(
            "primary DAS hash differs from the frozen source/preregistration"
        )
    frozen_root = root / "frozen"
    frozen_root.mkdir(parents=True, exist_ok=True)
    copied_das = frozen_root / source_das.name
    shutil.copy2(source_das, copied_das)
    basis = load_alignment(copied_das)
    if basis.shape[1] != int(primary["rank"]):
        raise RuntimeError("frozen DAS rank metadata does not match its basis")

    projection_path = resolve_record_path(
        source_root / "frozen_das/coverage_projections.parquet"
    )
    variable_path = resolve_record_path(
        source_root / "behavioral/abstraction_variables.parquet"
    )
    projections = read_records(projection_path)
    variables = read_records(variable_path)[["condition_id", "E"]].rename(
        columns={"condition_id": "semantic_condition_id"}
    )
    selected = projections[projections.artifact_id == primary["artifact_id"]].copy()
    selected["coordinates_array"] = selected.coordinates.map(
        lambda value: np.asarray(json.loads(value), dtype=float)
    )
    selected = selected.merge(
        variables, on="semantic_condition_id", validate="many_to_one"
    )
    coordinates = np.stack(selected.coordinates_array)
    orientation = fit_e_orientation(basis, coordinates, selected.E.to_numpy())
    orientation_metadata = {
        "source": "structured_persistence_coverage_only",
        "source_projection_sha256": sha256_file(projection_path),
        "source_behavioral_variables_sha256": sha256_file(variable_path),
        "source_das_sha256": observed_hash,
        "artifact_id": primary["artifact_id"],
        "layer": int(primary["layer"]),
        "rank": int(primary["rank"]),
        "activation_position": "final_prompt_token",
        "scale_definition": "sample_sd_of_signed_frozen_DAS_coordinate",
        "orientation_rows": orientation["rows"],
        "orientation_r2": orientation["r2"],
        "orientation_correlation": orientation["correlation"],
        "sign_check": orientation["sign_check"],
        "ood_data_used": False,
    }
    orientation_path = save_orientation(
        frozen_root / "e_orientation.pt", orientation, metadata=orientation_metadata
    )
    sampling = validate_sampling_config(config.get("sampling", {}))
    sampling_path = json_write(frozen_root / "sampling_config.json", sampling)
    prompts = prompt_rows(config)
    prompt_path = json_write(
        frozen_root / "prompt_manifest.json",
        {"prompts": prompts, "selected_before_ood": True},
    )
    doses = [
        float(value)
        for value in config.get("intervention", {}).get("doses", (-2, -1, 0, 1, 2))
    ]
    if doses != sorted(set(doses)) or 0.0 not in doses:
        raise ValueError(
            "frozen intervention doses must be unique, sorted, and include zero"
        )
    analysis_plan = {
        "execution_profile": config.get(
            "execution_profile", "compute_efficient_initial"
        ),
        "model": config.get("model"),
        "model_revision": config.get("model_revision"),
        "seed": int(config.get("seed", 94001)),
        "entry": config.get("entry", {}),
        "intervention": {**config.get("intervention", {}), "doses": doses},
        "design": config.get("design", {}),
        "pulse": config.get("pulse", {}),
        "secondary": config.get("secondary", {}),
        "controls": config.get("controls", {}),
        "runtime": config.get("runtime", {}),
        "analysis": config.get("analysis", {}),
        "gates": config.get("gates", {}),
        "exclusions": {
            "eos": "observed_event",
            "context_limit": "right_censored",
            "infrastructure_timeout": "right_censored",
            "infrastructure_oom": "right_censored",
            "post_ood_prompt_or_dose_selection": "forbidden",
        },
        "primary_population": "open_ended_continuous_frozen_E",
        "primary_outcome": "canonical_EOS_hazard",
        "selected_before_ood": True,
    }
    analysis_path = json_write(frozen_root / "analysis_plan.json", analysis_plan)

    random_count = int(config.get("controls", {}).get("random_subspaces", 100))
    random_bases = orthonormal_random_subspaces(
        basis.shape[0],
        basis.shape[1],
        random_count,
        seed=int(config.get("seed", 94001)) + 17,
    ).astype(np.float32)
    rng = np.random.default_rng(int(config.get("seed", 94001)) + 18)
    coordinate_axes = rng.normal(size=(random_count, basis.shape[1]))
    coordinate_axes /= np.linalg.norm(coordinate_axes, axis=1, keepdims=True)
    random_directions = np.einsum("ndk,nk->nd", random_bases, coordinate_axes).astype(
        np.float32
    )
    random_path = frozen_root / "random_subspaces.npz"
    np.savez_compressed(
        random_path,
        bases=random_bases,
        directions=random_directions,
        sigma_E=np.asarray(orientation["sigma_E"], dtype=np.float32),
    )
    das_manifest = {
        **primary,
        "source_artifact": str(source_das),
        "artifact": str(copied_das),
        "sha256": sha256_file(copied_das),
        "orientation_artifact": str(orientation_path),
        "orientation_sha256": sha256_file(orientation_path),
        "orientation": orientation_metadata,
        "retrained_for_ood": False,
    }
    manifest_path = json_write(frozen_root / "das_manifest.json", das_manifest)
    hashes = {
        path.name: sha256_file(path)
        for path in (
            copied_das,
            orientation_path,
            sampling_path,
            prompt_path,
            analysis_path,
            random_path,
            manifest_path,
        )
    }
    hashes_path = json_write(frozen_root / "hashes.json", hashes)
    return {
        "primary": das_manifest,
        "prompts": prompts,
        "doses": doses,
        "sampling": sampling,
        "orientation": orientation_metadata,
        "hashes": hashes,
        "hashes_path": hashes_path,
    }


def verify_frozen_protocol(root: str | Path) -> dict:
    root = Path(root)
    frozen_root = root / "frozen"
    hashes = json.loads((frozen_root / "hashes.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        path = frozen_root / name
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen OOD artifact hash mismatch: {name}")
    manifest = json.loads(
        (frozen_root / "das_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("retrained_for_ood") or manifest["orientation"].get(
        "ood_data_used"
    ):
        raise RuntimeError("OOD data leaked into the frozen neural object")
    return manifest

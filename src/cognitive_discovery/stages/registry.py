"""Validation and discovery for Stage-3 canonical experiment configurations."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from ..reproducibility.identities import EndpointID, MetricID
from ..reproducibility.manifest import CanonicalManifestError, sha256_file


SCHEMA_VERSION = "canonical-stage-v1"
EXPECTED_CONFIG_STAGES = (
    "discovery_v1",
    "discovery_v2",
    "theory_resolution_v1",
    "mechanistic_manifest_v1",
    "mechanistic_probe_steering_v1",
    "action_history_disambiguation_v1",
    "causal_mech_v1",
    "causal_specificity_v2",
    "abstraction_discovery_v1",
    "ood_free_generation_v1",
    "qwen_fixed_setting_stability_v1",
    "qwen_fresh_contexts_v1",
    "qwen_confirmation_v2",
    "llama_interface_v2",
    "llama_behavior_v2",
    "llama_mechanistic_v2",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StageConfigError(ValueError):
    pass


def load_stage_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageConfigError(f"stage config must be a mapping: {path}")
    return value


def _relative(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StageConfigError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise StageConfigError(f"{field} must stay inside the repository")
    return path


def _sha_or_status(value: Mapping[str, Any], field: str) -> None:
    status = value.get("status")
    if status == "not_applicable":
        return
    digest = value.get("sha256")
    if status not in {"frozen", "historical_unknown", "external_required"}:
        raise StageConfigError(f"{field} has invalid status {status!r}")
    if digest is not None and (not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
        raise StageConfigError(f"{field}.sha256 must be a lowercase SHA-256 or null")


def validate_stage_config(
    config: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    root: str | Path | None = None,
    verify_inputs: bool = False,
) -> None:
    required = {
        "schema_version",
        "stage_id",
        "pipeline_phase",
        "status",
        "upstream",
        "model",
        "endpoint_id",
        "metric_id",
        "split_design",
        "behavioral_object",
        "neural_object",
        "seeds",
        "scientific_defaults",
        "execution",
        "output",
    }
    missing = required - set(config)
    if missing:
        raise StageConfigError(f"stage config missing keys: {sorted(missing)}")
    if config["schema_version"] != SCHEMA_VERSION:
        raise StageConfigError("unsupported stage config schema_version")
    stage_id = config["stage_id"]
    if stage_id not in manifest["stages"]:
        raise StageConfigError(f"unknown canonical stage {stage_id!r}")
    stage = manifest["stages"][stage_id]
    if config["status"] != stage["status"]:
        raise StageConfigError(f"{stage_id} status differs from canonical manifest")

    upstream = config["upstream"]
    if not isinstance(upstream, list):
        raise StageConfigError("upstream must be a list")
    upstream_ids = []
    for entry in upstream:
        if not isinstance(entry, dict) or entry.get("stage_id") not in manifest["stages"]:
            raise StageConfigError(f"{stage_id} has an invalid upstream entry")
        if not isinstance(entry.get("artifact_ids"), list):
            raise StageConfigError(f"{stage_id} upstream artifact_ids must be a list")
        upstream_ids.append(entry["stage_id"])
    if upstream_ids != stage.get("depends_on", []):
        raise StageConfigError(f"{stage_id} upstream stages differ from canonical manifest")

    model = config["model"]
    model_ref = stage.get("model_ref")
    if model_ref is None:
        if model.get("status") != "not_applicable":
            raise StageConfigError(f"{stage_id} model must be explicitly not_applicable")
    else:
        canonical_model = manifest["models"][model_ref]
        if model.get("descriptor") != model_ref:
            raise StageConfigError(f"{stage_id} model descriptor mismatch")
        for field in ("checkpoint", "revision"):
            if model.get(field) != canonical_model.get(field):
                raise StageConfigError(f"{stage_id} model {field} mismatch")

    try:
        EndpointID(config["endpoint_id"])
    except ValueError as error:
        raise StageConfigError(f"{stage_id} has an unknown endpoint_id") from error
    try:
        MetricID(config["metric_id"])
    except ValueError as error:
        raise StageConfigError(f"{stage_id} has an unknown metric_id") from error
    for field in ("split_design", "behavioral_object", "neural_object"):
        value = config[field]
        if not isinstance(value, dict):
            raise StageConfigError(f"{field} must be a mapping")
        _sha_or_status(value, field)
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise StageConfigError(f"{stage_id} must record at least one seed")

    defaults = config["scientific_defaults"]
    if not isinstance(defaults, dict) or not defaults.get("path") or not defaults.get("sha256"):
        raise StageConfigError("scientific_defaults must name its frozen path and hash")
    _relative(defaults["path"], "scientific_defaults.path")
    if not _SHA256.fullmatch(defaults["sha256"]):
        raise StageConfigError("scientific_defaults.sha256 must be a lowercase SHA-256")

    execution = config["execution"]
    if not isinstance(execution, dict) or not isinstance(execution.get("rerun_allowed"), bool):
        raise StageConfigError("execution.rerun_allowed must be boolean")
    if execution["rerun_allowed"]:
        if model.get("revision") is None:
            raise StageConfigError(f"{stage_id} cannot be rerunnable without a pinned revision")
        if not isinstance(execution.get("command"), list) or not execution["command"]:
            raise StageConfigError(f"{stage_id} rerun command is required")
    if not isinstance(execution.get("required_inputs"), list):
        raise StageConfigError("execution.required_inputs must be a list")

    output = config["output"]
    frozen = _relative(output.get("frozen_root"), "output.frozen_root")
    rerun = _relative(output.get("rerun_root"), "output.rerun_root")
    if output.get("overwrite_frozen") is not False:
        raise StageConfigError("output.overwrite_frozen must be false")
    if frozen == rerun or frozen in rerun.parents or rerun in frozen.parents:
        raise StageConfigError("rerun output must be separate from frozen output")

    if root is not None:
        root_path = Path(root)
        defaults_path = root_path / _relative(defaults["path"], "scientific_defaults.path")
        if not defaults_path.is_file() or sha256_file(defaults_path) != defaults["sha256"]:
            raise StageConfigError(f"{stage_id} scientific defaults identity mismatch")
        if verify_inputs:
            for entry in execution["required_inputs"]:
                if entry.get("availability") != "committed":
                    raise StageConfigError(
                        f"{stage_id} required input {entry.get('id')!r} is not committed"
                    )
                relative = _relative(entry.get("path"), f"{stage_id} required input")
                expected = entry.get("sha256")
                if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
                    raise StageConfigError(f"{stage_id} required input lacks an exact hash")
                path = root_path / relative
                if not path.is_file() or sha256_file(path) != expected:
                    raise StageConfigError(f"{stage_id} required input identity mismatch: {relative}")


def load_all_stage_configs(root: str | Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(root)
    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "configs/canonical").glob("*.yaml")):
        config = load_stage_config(path)
        validate_stage_config(config, manifest=manifest, root=root)
        stage_id = config["stage_id"]
        if stage_id in loaded:
            raise CanonicalManifestError(f"duplicate canonical config for {stage_id}")
        loaded[stage_id] = config
    missing = set(EXPECTED_CONFIG_STAGES) - set(loaded)
    extra = set(loaded) - set(EXPECTED_CONFIG_STAGES)
    if missing or extra:
        raise CanonicalManifestError(
            f"canonical config inventory mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return loaded

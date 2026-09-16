"""Typed endpoint, metric, model, behavioral, neural, and interface identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .manifest import sha256_file
from .metrics import global_cfr_v1


CORE_QWEN_CONTROLLER_SHA256 = (
    "ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0"
)


class EndpointID(str, Enum):
    COGNITIVE_COUNTERFACTUAL_RECOVERY = "cognitive_counterfactual_recovery"
    NATURAL_EFFECT_RECOVERY = "natural_effect_recovery"


class MetricID(str, Enum):
    GLOBAL_CFR_V1 = "global_cfr_v1"


@dataclass(frozen=True)
class ModelDescriptor:
    model_id: str
    checkpoint: str
    revision: str | None
    identity_status: str | None = None


@dataclass(frozen=True)
class BehavioralObjectDescriptor:
    architecture: str
    model_path: str
    model_sha256: str
    in_memory_sha256: str
    training_condition_sha256: str

    def validate(self) -> "BehavioralObjectDescriptor":
        allowed = {"dual_history", "latent_context", "outcome_history"}
        if self.architecture not in allowed:
            raise ValueError(f"unknown behavioral architecture {self.architecture!r}")
        for name in ("model_sha256", "in_memory_sha256", "training_condition_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"invalid {name}")
        return self


@dataclass(frozen=True)
class NeuralObjectDescriptor:
    path: str
    sha256: str
    layer: int
    rank: int
    target_definition: str


def canonical_metric(metric_id: MetricID | str):
    try:
        parsed = MetricID(metric_id)
    except ValueError as error:
        raise ValueError(f"metric {metric_id!r} is not canonical") from error
    if parsed is MetricID.GLOBAL_CFR_V1:
        return global_cfr_v1
    raise AssertionError(parsed)


def recovery_target(frame, endpoint: EndpointID | str, *, cognitive_column=None):
    try:
        parsed = EndpointID(endpoint)
    except ValueError as error:
        raise ValueError(f"unknown endpoint {endpoint!r}") from error
    column = (
        cognitive_column or "predicted_counterfactual_effect"
        if parsed is EndpointID.COGNITIVE_COUNTERFACTUAL_RECOVERY
        else "natural_effect"
    )
    if column not in frame:
        raise ValueError(f"endpoint {parsed.value} requires column {column!r}")
    values = np.asarray(frame[column], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"endpoint {parsed.value} contains non-finite target values")
    return frame[column]


def persistence_logit(logits: Mapping[str, float], continue_label: str, disengage_label: str) -> float:
    if continue_label == disengage_label:
        raise ValueError("continue and disengage labels must be distinct")
    missing = {continue_label, disengage_label} - set(logits)
    if missing:
        raise ValueError(f"missing semantic logits for {sorted(missing)}")
    return float(logits[continue_label] - logits[disengage_label])


def validate_binary_interface(
    labels: tuple[str, str], *, continue_label: str, interface_id: str
) -> None:
    expected = {"llama_yes_no_v2": {"Yes", "No"}, "qwen_xy_v1": {"X", "Y"}}
    if interface_id not in expected or set(labels) != expected[interface_id]:
        raise ValueError(f"labels {labels!r} do not match interface {interface_id!r}")
    if continue_label not in labels:
        raise ValueError("continue label is not part of the binary interface")


def model_descriptor(manifest: Mapping[str, Any], model_id: str) -> ModelDescriptor:
    try:
        value = manifest["models"][model_id]
    except KeyError as error:
        raise ValueError(f"unknown model descriptor {model_id!r}") from error
    return ModelDescriptor(
        model_id=model_id,
        checkpoint=value["checkpoint"],
        revision=value.get("revision"),
        identity_status=value.get("identity_status"),
    )


def require_model_revision(descriptor: ModelDescriptor) -> ModelDescriptor:
    if descriptor.revision is None:
        raise ValueError(
            f"model {descriptor.model_id} revision is {descriptor.identity_status or 'unknown'}"
        )
    return descriptor


def verify_neural_object(
    descriptor: NeuralObjectDescriptor,
    *,
    root: str | Path,
    expected_layer: int,
    expected_rank: int,
) -> NeuralObjectDescriptor:
    if descriptor.layer != expected_layer:
        raise ValueError(f"neural layer mismatch: {descriptor.layer} != {expected_layer}")
    if descriptor.rank != expected_rank:
        raise ValueError(f"neural rank mismatch: {descriptor.rank} != {expected_rank}")
    path = Path(root) / descriptor.path
    if not path.is_file():
        raise ValueError(f"neural artifact is unavailable: {descriptor.path}")
    observed = sha256_file(path)
    if observed != descriptor.sha256:
        raise ValueError(f"neural artifact SHA-256 mismatch: {observed} != {descriptor.sha256}")
    return descriptor


def verify_behavioral_registry(
    registry_path: str | Path, *, root: str | Path
) -> dict[str, BehavioralObjectDescriptor]:
    import json

    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    architectures = registry.get("architectures", [])
    if set(architectures) != {"dual_history", "latent_context", "outcome_history"}:
        raise ValueError("behavioral registry architecture set does not match the canonical set")
    base = Path(root) / "artifacts/theory_resolution_v1/frozen_models"
    for relative, expected in registry["file_sha256"].items():
        path = base / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"behavioral registry file identity mismatch: {relative}")
    descriptors = {}
    for architecture in architectures:
        prefix = f"{architecture}/"
        descriptor = BehavioralObjectDescriptor(
            architecture=architecture,
            model_path=(base / architecture / "model.pkl").relative_to(root).as_posix(),
            model_sha256=registry["file_sha256"][prefix + "model.pkl"],
            in_memory_sha256=registry["in_memory_model_sha256"][architecture],
            training_condition_sha256=registry["file_sha256"][
                prefix + "training_condition_hashes.json"
            ],
        ).validate()
        descriptors[architecture] = descriptor
    return descriptors

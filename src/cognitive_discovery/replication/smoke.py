"""Short, fail-closed GPU preflight for a prospective replication model.

The smoke protocol exercises the same renderers, token validation, residual
hooks, intervention path, and DAS primitive as the full replication.  Its
outputs are engineering diagnostics only and cannot support scientific claims.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from cognitive_discovery.causal_mechanistic.das import DASAlignment
from cognitive_discovery.data.validation import pilot_gate, validate_records
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.experiments.registry import get_renderer
from cognitive_discovery.pipeline import generate_design, load_config

from .adapters import adapter_registry, resolve_relative_layers
from .config import REVISION_PATTERN
from .workflow import (
    AdapterParticipant,
    _interface_messages,
    _require_full_vocabulary_diagnostics,
)


SMOKE_SCHEMA_VERSION = "replication-gpu-smoke-v2"
SMOKE_PURPOSE = "engineering_preflight_not_scientific_evidence"
REQUIRED_SMOKE_CHECKS = (
    "cuda_available",
    "immutable_revisions",
    "interface_configuration",
    "checkpoint_load",
    "chat_template",
    "seven_task_rendering",
    "response_tokens_and_logits",
    "full_vocabulary_diagnostics",
    "paired_interface_collection",
    "layer_discovery",
    "residual_capture",
    "identity_intervention",
    "nonzero_intervention",
    "das_gradient",
    "eos_tokens",
)
_BANNED_PAYLOAD_KEYS = {
    "activation",
    "activations",
    "hidden_state",
    "hidden_states",
    "residual_state",
    "residual_states",
    "state_vectors",
}


class SmokeConfigurationError(ValueError):
    """Raised before expensive work when a smoke-test input is ambiguous."""


def adapter_for_model_type(model_type: str, architectures: Iterable[str] = ()) -> str:
    """Map only architectures covered by a registered replication adapter."""

    text = " ".join([str(model_type), *map(str, architectures)]).lower()
    if "nemotron" in text:
        raise SmokeConfigurationError(
            "unsupported nemotron architecture: add and test a dedicated adapter; "
            "the smoke harness will not guess that it is Mistral-compatible"
        )
    for marker, adapter in (
        ("qwen", "qwen"),
        ("llama", "llama"),
        ("gemma", "gemma"),
        ("mistral", "mistral"),
    ):
        if marker in text:
            return adapter
    raise SmokeConfigurationError(
        f"unsupported model architecture: model_type={model_type!r}, "
        f"architectures={list(architectures)!r}"
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _find_banned_key(value: Any, path: str = "result") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _BANNED_PAYLOAD_KEYS or "activation" in normalized:
                return f"{path}.{key}"
            found = _find_banned_key(item, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_banned_key(item, f"{path}[{index}]")
            if found:
                return found
    return None


def validate_smoke_result(result: Mapping[str, Any]) -> None:
    """Validate a compact diagnostic without accepting neural-state payloads."""

    if result.get("schema_version") != SMOKE_SCHEMA_VERSION:
        raise ValueError("unknown smoke-result schema")
    if result.get("purpose") != SMOKE_PURPOSE:
        raise ValueError("smoke result must be labeled as non-scientific")
    if result.get("status") not in {"passed", "failed"}:
        raise ValueError("smoke status must be passed or failed")
    checks = result.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(REQUIRED_SMOKE_CHECKS):
        raise ValueError("smoke result does not contain the exact required checks")
    if result["status"] == "passed" and any(
        value.get("status") != "passed" for value in checks.values()
    ):
        raise ValueError("a passing smoke result contains an incomplete check")
    banned = _find_banned_key(result)
    if banned:
        raise ValueError(f"activation/state-vector payload is prohibited: {banned}")
    encoded = json.dumps(result, sort_keys=True, default=str).encode("utf-8")
    if len(encoded) > 256 * 1024:
        raise ValueError("smoke result exceeds the 256 KiB compact-artifact limit")


def _initial_result(
    *,
    model_id: str,
    revision: str,
    adapter: str,
    tokenizer_id: str,
    tokenizer_revision: str,
) -> dict[str, Any]:
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "purpose": SMOKE_PURPOSE,
        "status": "failed",
        "model": {
            "id": str(model_id),
            "revision": str(revision),
            "adapter": str(adapter),
            "tokenizer_id": str(tokenizer_id),
            "tokenizer_revision": str(tokenizer_revision),
        },
        "checks": {
            name: {"status": "not_run"} for name in REQUIRED_SMOKE_CHECKS
        },
        "diagnostics": {},
    }


def _mark(result: dict[str, Any], name: str, **details: Any) -> None:
    result["checks"][name] = {"status": "passed", **details}


def _failure(result: dict[str, Any], check: str, error: Exception) -> None:
    result["checks"][check] = {
        "status": "failed",
        "detail": str(error)[:2000],
    }
    result["error"] = {
        "check": check,
        "type": type(error).__name__,
        "message": str(error)[:4000],
    }


def _prompt_hash(messages: Iterable[Mapping[str, str]]) -> str:
    payload = json.dumps(list(messages), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_interfaces(path: Path) -> list[dict[str, Any]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    candidates = value.get("interface", {}).get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise SmokeConfigurationError("replication config has no interface candidates")
    normalized = []
    for candidate in candidates:
        labels = candidate.get("labels") if isinstance(candidate, Mapping) else None
        if (
            not isinstance(labels, list)
            or len(labels) != 2
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != 2
        ):
            raise SmokeConfigurationError(
                "smoke interface labels must be two distinct nonempty strings"
            )
        normalized.append(dict(candidate))
    return normalized


def _render_smoke_trials(project_root: Path, seed: int):
    ontology = load_config(project_root / "configs" / "discovery_v1.yaml")
    design, _ = generate_design(
        ontology,
        conditions=7,
        seed=int(seed),
        design_id="replication_smoke_v1",
        persist=False,
    )
    trials = []
    observed = set()
    for condition in design:
        if condition.task_family in observed:
            continue
        observed.add(condition.task_family)
        trials.append(get_renderer(condition.task_family).render(condition))
    expected = set(ontology["design"]["task_families"])
    if observed != expected:
        raise RuntimeError(
            f"tiny design did not cover the seven-task battery: {sorted(observed)}"
        )
    return design, trials, ontology


def _compact_gate_preview(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep tiny-sample gate diagnostics without treating them as evidence."""

    value = {
        "approved_on_tiny_sample": bool(gate["approved"]),
        "checks": {key: bool(item) for key, item in gate["checks"].items()},
    }
    for key in (
        "mean_p_continue",
        "persistence_logit_sd",
        "mean_absolute_mapping_gap",
        "top_token_action_rate",
    ):
        item = float(gate[key])
        value[key] = item if math.isfinite(item) else None
    return value


def _resolve_adapter(
    *,
    requested: str,
    model_id: str,
    revision: str,
    local_files_only: bool,
) -> tuple[str, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model_type = str(getattr(config, "model_type", ""))
    architectures = tuple(getattr(config, "architectures", ()) or ())
    detected = adapter_for_model_type(model_type, architectures)
    if requested != "auto" and requested not in adapter_registry.names():
        raise SmokeConfigurationError(
            f"unknown adapter {requested!r}; choose auto or {list(adapter_registry.names())}"
        )
    if requested != "auto" and requested != detected:
        raise SmokeConfigurationError(
            f"requested adapter {requested!r} conflicts with detected {detected!r} "
            f"for model_type={model_type!r}"
        )
    return detected, config


def run_model_smoke(
    *,
    model_id: str,
    revision: str,
    adapter: str,
    output_dir: str | Path,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    config_path: str | Path = "configs/replication/default.yaml",
    project_root: str | Path | None = None,
    local_files_only: bool = True,
    seed: int = 19001,
    relative_depths: Iterable[float] = (0.25, 0.5, 0.75, 0.9),
) -> dict[str, Any]:
    """Run one model's bounded GPU preflight and always write a result JSON."""

    import numpy as np
    import torch

    root = Path(project_root or Path(__file__).resolve().parents[3])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "smoke_result.json"
    if result_path.exists():
        raise FileExistsError(f"smoke result already exists: {result_path}")
    tokenizer_id = str(tokenizer_id or model_id)
    tokenizer_revision = str(tokenizer_revision or revision)
    result = _initial_result(
        model_id=model_id,
        revision=revision,
        adapter=adapter,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    current_check = "cuda_available"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.reset_peak_memory_stats()
        _mark(result, current_check, device=torch.cuda.get_device_name(0))

        current_check = "immutable_revisions"
        if not REVISION_PATTERN.fullmatch(str(revision)):
            raise SmokeConfigurationError("model revision must be an immutable 40-hex commit")
        if not REVISION_PATTERN.fullmatch(tokenizer_revision):
            raise SmokeConfigurationError(
                "tokenizer revision must be an immutable 40-hex commit"
            )
        _mark(result, current_check)

        current_check = "interface_configuration"
        candidates = _load_interfaces(root / Path(config_path))
        _mark(result, current_check, candidates=len(candidates))

        current_check = "checkpoint_load"
        selected_adapter, architecture_config = _resolve_adapter(
            requested=adapter,
            model_id=model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        instance = adapter_registry.create(
            selected_adapter,
            model_id=model_id,
            revision=revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            local_files_only=local_files_only,
        ).load()
        result["model"]["adapter"] = selected_adapter
        result["diagnostics"]["model_type"] = str(
            getattr(architecture_config, "model_type", "")
        )
        _mark(result, current_check)

        current_check = "seven_task_rendering"
        design, trials, ontology = _render_smoke_trials(root, seed)
        result["diagnostics"]["task_prompt_hashes"] = {
            trial.task_family: _prompt_hash(trial.messages) for trial in trials
        }
        _mark(result, current_check, count=len(trials))

        current_check = "response_tokens_and_logits"
        interface_errors = {}
        selected = None
        selected_messages = None
        selected_labels = None
        positive_label = None
        selected_logits = None
        selected_metrics = None
        usable_candidates = []
        candidate_diagnostics = {}
        for candidate in candidates:
            try:
                instance.validate_response_tokens(candidate["labels"])
                messages, labels, mapping = _interface_messages(
                    trials[0].messages,
                    (trials[0].continue_label, trials[0].disengage_label),
                    candidate,
                )
                rendered = instance.render_chat(messages)
                if not isinstance(rendered, str) or not rendered.strip():
                    raise ValueError("chat template returned an empty prompt")
                candidate_positive = mapping[trials[0].continue_label]
                metrics = instance.get_response_metrics(
                    messages,
                    labels,
                    positive_label=candidate_positive,
                )
                logits = {label: metrics[f"logit_{label}"] for label in labels}
                if set(logits) != set(labels) or not all(
                    math.isfinite(float(value)) for value in logits.values()
                ):
                    raise ValueError("response logits are missing or non-finite")
                mass = metrics.get("p_action_mass_raw")
                top_is_action = metrics.get("top_token_is_action")
                if (
                    not isinstance(top_is_action, bool)
                    or mass is None
                    or not math.isfinite(float(mass))
                    or not 0.0 <= float(mass) <= 1.0
                ):
                    raise RuntimeError("full-vocabulary response diagnostics are invalid")
                usable_candidates.append(candidate)
                candidate_diagnostics[candidate["id"]] = {
                    "labels": list(labels),
                    "p_action_mass_raw": float(mass),
                    "top_token_is_action": top_is_action,
                }
                if selected is None:
                    selected = candidate
                    selected_messages = messages
                    selected_labels = tuple(labels)
                    positive_label = candidate_positive
                    selected_logits = logits
                    selected_metrics = metrics
            except (RuntimeError, ValueError) as error:
                interface_errors[str(candidate.get("id", "unnamed"))] = str(error)[:500]
        if selected is None:
            raise RuntimeError(
                "no preregistered response interface passed the chat-token preflight: "
                + json.dumps(interface_errors, sort_keys=True)
            )

        current_check = "chat_template"
        chat_hashes = {}
        for trial in trials:
            messages, _, _ = _interface_messages(
                trial.messages,
                (trial.continue_label, trial.disengage_label),
                selected,
            )
            rendered = instance.render_chat(messages)
            if not isinstance(rendered, str) or not rendered.strip():
                raise ValueError(f"empty chat template for task {trial.task_family}")
            chat_hashes[trial.task_family] = hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest()
        result["diagnostics"]["chat_prompt_hashes"] = chat_hashes
        _mark(result, current_check)

        current_check = "response_tokens_and_logits"
        token_ids = instance._chat_token_ids(selected_messages, selected_labels)
        result["diagnostics"]["interface"] = {
            "id": str(selected.get("id", "unnamed")),
            "labels": list(selected_labels),
            "positive_label": positive_label,
            "token_ids": {key: int(value) for key, value in token_ids.items()},
            "logits": {key: float(value) for key, value in selected_logits.items()},
        }
        _mark(result, current_check)

        current_check = "full_vocabulary_diagnostics"
        result["diagnostics"]["interface_candidates"] = candidate_diagnostics
        result["diagnostics"]["interface"]["p_action_mass_raw"] = float(
            selected_metrics["p_action_mass_raw"]
        )
        result["diagnostics"]["interface"]["top_token_is_action"] = bool(
            selected_metrics["top_token_is_action"]
        )
        _mark(result, current_check, candidates_measured=len(usable_candidates))

        current_check = "paired_interface_collection"
        previews = {}
        for candidate in usable_candidates:
            observations = validate_records(
                collect_conditions(
                    design,
                    AdapterParticipant(instance, candidate),
                    model_revision=revision,
                    sample_actions=False,
                    expand_history_prefixes=False,
                )
            )
            _require_full_vocabulary_diagnostics(observations)
            task_counts = observations.groupby("task_family").size().to_dict()
            if set(task_counts.values()) != {2} or set(task_counts) != {
                trial.task_family for trial in trials
            }:
                raise RuntimeError(
                    f"tiny paired collection is incomplete for {candidate['id']}: "
                    f"{task_counts}"
                )
            previews[candidate["id"]] = {
                task: _compact_gate_preview(pilot_gate(group, ontology))
                for task, group in observations.groupby("task_family")
            }
        result["diagnostics"]["tiny_interface_gate_preview"] = previews
        result["diagnostics"]["tiny_interface_gate_preview_interpretation"] = (
            "engineering_only_not_a_measurement_gate"
        )
        _mark(
            result,
            current_check,
            candidates_collected=len(previews),
            rows_per_candidate=len(design),
        )
        instance.bind_response_interface(selected_labels, positive_label=positive_label)

        current_check = "layer_discovery"
        runner = instance._runner()
        layer_count = runner.layer_count
        layers = resolve_relative_layers(layer_count, tuple(relative_depths))
        result["diagnostics"]["num_layers"] = int(layer_count)
        result["diagnostics"]["resolved_layers"] = list(map(int, layers))
        _mark(result, current_check)

        current_check = "residual_capture"
        baseline = runner.forward(
            selected_messages,
            selected_labels,
            positive_label=positive_label,
            capture_layers=layers,
        )
        if set(baseline.states) != set(layers):
            raise RuntimeError("streamed residual capture omitted a selected layer")
        state_shapes = {
            str(layer): list(baseline.states[layer].shape) for layer in layers
        }
        if any(len(shape) != 1 or shape[0] < 1 for shape in state_shapes.values()):
            raise RuntimeError(f"unexpected residual-state shapes: {state_shapes}")
        result["diagnostics"]["state_shapes"] = state_shapes
        result["diagnostics"]["baseline_persistence_logit"] = float(
            baseline.persistence_logit
        )
        _mark(result, current_check)

        layer = int(layers[-1])
        current_check = "identity_intervention"
        identity = runner.forward(
            selected_messages,
            selected_labels,
            positive_label=positive_label,
            editors={layer: lambda state: state},
            capture_layers=(),
        )
        identity_delta = float(identity.persistence_logit - baseline.persistence_logit)
        tolerance = 1e-4 * max(1.0, abs(float(baseline.persistence_logit)))
        if not math.isfinite(identity_delta) or abs(identity_delta) > tolerance:
            raise RuntimeError(
                f"identity edit changed the logit by {identity_delta} (tolerance={tolerance})"
            )
        result["diagnostics"]["identity_logit_delta"] = identity_delta
        _mark(result, current_check, tolerance=tolerance)

        current_check = "nonzero_intervention"
        direction = runner.choice_output_direction(
            selected_messages,
            selected_labels,
            positive_label=positive_label,
        )
        hidden_size = int(baseline.states[layer].shape[0])
        if np.asarray(direction).shape != (hidden_size,):
            raise RuntimeError(
                f"output direction shape {np.asarray(direction).shape} != ({hidden_size},)"
            )

        def add_output_direction(state):
            vector = torch.as_tensor(direction, dtype=state.dtype, device=state.device)
            vector = vector / vector.float().norm().clamp_min(1e-12).to(vector.dtype)
            return state + vector.unsqueeze(0)

        intervened = runner.forward(
            selected_messages,
            selected_labels,
            positive_label=positive_label,
            editors={layer: add_output_direction},
            capture_layers=(),
        )
        intervention_delta = float(
            intervened.persistence_logit - baseline.persistence_logit
        )
        if not math.isfinite(intervention_delta) or abs(intervention_delta) <= 1e-8:
            raise RuntimeError("nonzero residual edit did not change the response logit")
        result["diagnostics"]["nonzero_intervention_logit_delta"] = intervention_delta
        _mark(result, current_check)

        current_check = "das_gradient"
        other_trial = trials[1]
        source_messages, source_labels, source_mapping = _interface_messages(
            other_trial.messages,
            (other_trial.continue_label, other_trial.disengage_label),
            selected,
        )
        if tuple(source_labels) != selected_labels:
            raise RuntimeError("interface labels changed across smoke trials")
        source_positive = source_mapping[other_trial.continue_label]
        source_forward = runner.forward(
            source_messages,
            source_labels,
            positive_label=source_positive,
            capture_layers=(layer,),
        )
        for parameter in runner.model.parameters():
            parameter.requires_grad_(False)
        model_device = next(runner.model.parameters()).device
        alignment = DASAlignment(
            hidden_size,
            min(2, hidden_size),
            seed=int(seed),
            device=model_device,
        )
        source_state = torch.as_tensor(
            source_forward.states[layer], device=model_device
        ).unsqueeze(0)
        differentiable_logit = runner.differentiable_persistence_logit(
            selected_messages,
            selected_labels,
            positive_label=positive_label,
            editors={layer: lambda state: alignment.edit(state, source_state)},
        )
        differentiable_logit.backward()
        gradient = alignment.parameter.grad
        if gradient is None:
            raise RuntimeError("DAS alignment parameter received no gradient")
        gradient_norm = float(gradient.detach().float().norm().cpu())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError(f"invalid DAS gradient norm: {gradient_norm}")
        if any(parameter.requires_grad for parameter in runner.model.parameters()):
            raise RuntimeError("model parameters were not frozen during DAS preflight")
        result["diagnostics"]["das"] = {
            "rank": int(alignment.rank),
            "gradient_norm": gradient_norm,
            "model_parameters_frozen": True,
        }
        _mark(result, current_check)

        current_check = "eos_tokens"
        eos_ids = tuple(map(int, instance.eos_token_ids()))
        if not eos_ids:
            raise RuntimeError("model/tokenizer expose no EOS token IDs")
        result["diagnostics"]["eos_token_ids"] = list(eos_ids)
        _mark(result, current_check)

        result["diagnostics"]["peak_cuda_memory_bytes"] = int(
            torch.cuda.max_memory_allocated()
        )
        result["status"] = "passed"
    except Exception as error:  # The result must survive expected preflight failures.
        _failure(result, current_check, error)
        if torch.cuda.is_available():
            result["diagnostics"]["peak_cuda_memory_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )

    validate_smoke_result(result)
    _atomic_json(result_path, result)
    return result


def summarize_smoke_runs(
    root: str | Path, *, expected_jobs: int
) -> dict[str, Any]:
    """Aggregate every array slot, including failed and missing GPU tasks."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expected_jobs = int(expected_jobs)
    if expected_jobs < 1:
        raise SmokeConfigurationError("expected_jobs must be positive")
    rows = []
    counts = {"passed": 0, "failed": 0, "missing": 0}
    for index in range(expected_jobs):
        job_name = f"job_{index:04d}"
        path = root / job_name / "smoke_result.json"
        if not path.exists():
            counts["missing"] += 1
            rows.append(
                {
                    "job": job_name,
                    "model": "unknown",
                    "status": "missing",
                    "failed_check": "no result (job may have been killed before Python wrote it)",
                }
            )
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            validate_smoke_result(result)
            status = result["status"]
            counts[status] += 1
            failed_check = result.get("error", {}).get("check", "")
            rows.append(
                {
                    "job": job_name,
                    "model": result.get("model", {}).get("id", "unknown"),
                    "status": status,
                    "failed_check": failed_check,
                }
            )
        except (ValueError, json.JSONDecodeError, KeyError) as error:
            counts["failed"] += 1
            rows.append(
                {
                    "job": job_name,
                    "model": "unknown",
                    "status": "failed",
                    "failed_check": f"invalid result: {error}",
                }
            )
    all_passed = counts == {
        "passed": expected_jobs,
        "failed": 0,
        "missing": 0,
    }
    status = "passed" if all_passed else "failed"
    summary = {
        "schema_version": "replication-gpu-smoke-summary-v2",
        "purpose": SMOKE_PURPOSE,
        "status": status,
        "expected_jobs": expected_jobs,
        "counts": counts,
        "jobs": rows,
    }
    _atomic_json(root / "smoke_summary.json", summary)
    lines = [
        "# Replication GPU smoke report",
        "",
        "This is an engineering preflight and **not scientific evidence**.",
        "It does not evaluate behavioral conclusions, CFR, specificity, or generalization.",
        "",
        f"Overall status: **{status}**",
        "",
        "| Array slot | Model | Status | Failed/missing check |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {job} | {model} | {status} | {failed_check} |".format(
                **{key: str(value).replace("|", "\\|") for key, value in row.items()}
            )
        )
    (root / "SMOKE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

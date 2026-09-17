import json
from pathlib import Path
import subprocess
import sys

import pytest

from cognitive_discovery.replication.smoke import (
    REQUIRED_SMOKE_CHECKS,
    SmokeConfigurationError,
    adapter_for_model_type,
    run_model_smoke,
    summarize_smoke_runs,
    validate_smoke_result,
)


ROOT = Path(__file__).parents[1]


def _passing_result(model_id="example/model"):
    return {
        "schema_version": "replication-gpu-smoke-v1",
        "purpose": "engineering_preflight_not_scientific_evidence",
        "status": "passed",
        "model": {
            "id": model_id,
            "revision": "a" * 40,
            "adapter": "llama",
            "tokenizer_id": model_id,
            "tokenizer_revision": "a" * 40,
        },
        "checks": {name: {"status": "passed"} for name in REQUIRED_SMOKE_CHECKS},
        "diagnostics": {"num_layers": 32, "peak_cuda_memory_bytes": 1234},
    }


@pytest.mark.parametrize(
    ("model_type", "architectures", "expected"),
    [
        ("gemma3", ["Gemma3ForCausalLM"], "gemma"),
        ("llama", ["LlamaForCausalLM"], "llama"),
        ("qwen3", ["Qwen3ForCausalLM"], "qwen"),
        ("mistral", ["MistralForCausalLM"], "mistral"),
    ],
)
def test_auto_adapter_resolution_is_explicit(model_type, architectures, expected):
    assert adapter_for_model_type(model_type, architectures) == expected


def test_unknown_nemotron_architecture_fails_closed_instead_of_guessing_mistral():
    with pytest.raises(SmokeConfigurationError, match="unsupported.*nemotron"):
        adapter_for_model_type("nemotron_h", ["NemotronHForCausalLM"])


def test_protocol_exercises_the_real_neural_boundaries_without_claim_metrics():
    assert set(REQUIRED_SMOKE_CHECKS) == {
        "cuda_available",
        "immutable_revisions",
        "checkpoint_load",
        "chat_template",
        "seven_task_rendering",
        "response_tokens_and_logits",
        "layer_discovery",
        "residual_capture",
        "identity_intervention",
        "nonzero_intervention",
        "das_gradient",
        "eos_tokens",
    }
    assert not any("cfr" in name or "claim" in name for name in REQUIRED_SMOKE_CHECKS)


def test_smoke_result_is_compact_and_rejects_activation_payloads():
    result = _passing_result()
    validate_smoke_result(result)
    result["diagnostics"]["activations"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="activation"):
        validate_smoke_result(result)


def test_preflight_failure_still_writes_a_machine_readable_result(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run_model_smoke(
        model_id="unloaded/model",
        revision="a" * 40,
        adapter="auto",
        output_dir=tmp_path,
    )
    assert result["status"] == "failed"
    assert result["error"]["check"] == "cuda_available"
    saved = json.loads((tmp_path / "smoke_result.json").read_text(encoding="utf-8"))
    assert saved == result


def test_summary_reports_pass_fail_and_missing_jobs(tmp_path):
    passed = tmp_path / "job_0000"
    failed = tmp_path / "job_0001"
    passed.mkdir()
    failed.mkdir()
    (passed / "smoke_result.json").write_text(
        json.dumps(_passing_result("model/passed")), encoding="utf-8"
    )
    failure = _passing_result("model/failed")
    failure["status"] = "failed"
    failure["error"] = {"type": "RuntimeError", "message": "intentional"}
    failure["checks"]["checkpoint_load"] = {
        "status": "failed",
        "detail": "intentional",
    }
    (failed / "smoke_result.json").write_text(json.dumps(failure), encoding="utf-8")

    summary = summarize_smoke_runs(tmp_path, expected_jobs=3)

    assert summary["status"] == "failed"
    assert summary["counts"] == {"passed": 1, "failed": 1, "missing": 1}
    assert (tmp_path / "smoke_summary.json").exists()
    report = (tmp_path / "SMOKE_REPORT.md").read_text(encoding="utf-8")
    assert "model/passed" in report
    assert "model/failed" in report
    assert "job_0002" in report
    assert "not scientific evidence" in report


def test_smoke_cli_help_is_available_without_loading_a_model():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "replication_smoke.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "run-model" in completed.stdout
    assert "summarize" in completed.stdout

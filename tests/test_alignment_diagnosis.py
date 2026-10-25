"""Scientific contracts for the bandit correction proof of concept (CPU only)."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from cognitive_discovery.alignment_diagnosis.design import (
    CORRECTION,
    calibration_design,
    factorial,
    load_config,
    main_design,
    validate_config,
)
from cognitive_discovery.alignment_diagnosis.models import (
    comparison,
    cross_validate,
    features,
    fit,
    predict,
    recovery,
)
from cognitive_discovery.alignment_diagnosis.participant import (
    BanditQwenParticipant,
    SyntheticParticipant,
    parse_response,
)
from cognitive_discovery.alignment_diagnosis.pipeline import (
    _measurement,
    analyze,
    collect,
    freeze_predictions,
    prepare,
    records,
    verify_freeze,
)
from cognitive_discovery.participants.token_mapping import verify_chat_action_tokens


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/alignment_diagnosis/v1.yaml")


def test_factorial_is_matched_valid_and_authorized(config):
    rows = factorial(config)
    assert len(rows) == 256
    frame = pd.DataFrame(rows)
    assert len(frame.groupby(["correction", "history", "incentive"])) == 8
    for _, group in frame.groupby("pair_id"):
        left, right = group.to_dict("records")
        assert left["history_actions"] == right["history_actions"]
        assert left["rewards"] == right["rewards"]
        assert left["messages"][1] == right["messages"][1]
        assert (
            right["messages"][0]["content"]
            == left["messages"][0]["content"] + "\n\n" + CORRECTION
        )
        assert right["messages"][0]["role"] == "system"
        assert set(left["rewards"]) == {-2, 3}
        assert set(left["history_actions"]) <= {"A", "B"}
        assert len(left["rewards"]) == 8
    for _, group in frame.groupby(["variant", "correction", "history"]):
        assert len({tuple(r) for r in group.rewards}) == 1
    assert frame.trial_id.is_unique
    assert factorial(config) == rows


def test_controls_and_stronger_condition_are_independent(config):
    rows = main_design(config, pilot=True)
    assert sum(r["trial_type"] == "choice" for r in rows) == 64
    assert sum(r["trial_type"] == "isolated" for r in rows) == 16
    assert sum(r["trial_type"] == "comprehension" for r in rows) == 64
    for row in rows:
        if row["trial_type"] == "choice":
            assert "identification probe" not in str(row["messages"])
    calibration = calibration_design(config)
    assert {r["correction"] for r in calibration} == {0}
    intervention = factorial(config, phase="intervention", incentive=8)
    assert not {r["trial_id"] for r in calibration} & {
        r["trial_id"] for r in intervention
    }
    assert not {r["prompt_hash"] for r in calibration} & {
        r["prompt_hash"] for r in intervention
    }
    with pytest.raises(ValueError):
        bad = deepcopy(config)
        bad["design"]["stronger_candidates"] = [4]
        validate_config(bad)


def test_model_equations_and_existing_kernel(config):
    rows = factorial(config)[:2]
    beta = [-0.5, 0.7, 0.3]
    logits = features(rows) @ beta
    kernel = sum(
        0.7**lag * value
        for lag, value in enumerate(reversed(rows[0]["history_outcomes"]))
    )
    assert features(rows)[0, 2] == pytest.approx(kernel)
    baseline = predict(rows, "baseline", beta)
    integrated = predict(rows, "integrated", beta, 2.0)
    gate = predict(rows, "gate", beta, 0.8, 0.001)
    assert integrated[0] == pytest.approx(baseline[0])
    assert integrated[1] == pytest.approx(expit(logits[1] - 2))
    assert gate[1] == pytest.approx(0.2 * baseline[1] + 0.8 * 0.001)
    extreme = pd.DataFrame(rows).assign(incentive=100)
    assert predict(extreme, "integrated", beta, 2)[1] > 0.999
    assert predict(extreme, "gate", beta, 0.8)[1] == pytest.approx(0.2008)


@pytest.mark.parametrize("generator,parameter", [("integrated", 2.0), ("gate", 0.8)])
def test_fractional_endpoint_fits_known_generator_without_count_inflation(
    config, generator, parameter
):
    rows = pd.DataFrame(factorial(config))
    rows["p_continue"] = predict(rows, generator, [-0.5, 0.7, 0.3], parameter)
    result = fit(rows, "p_continue")
    assert result["status"] == "complete"
    assert result["n_baseline"] == result["n_corrected"] == 128
    assert result["models"][generator]["correction_parameter"] == pytest.approx(
        parameter, abs=1e-4
    )
    assert result["models"]["baseline"]["beta"] == pytest.approx(
        [-0.5, 0.7, 0.3], abs=1e-4
    )
    cv = cross_validate(rows, "p_continue", epsilon=0.001)
    assert len(cv) == 256
    assert comparison(cv, config)["winner"] == generator


def test_equivalent_candidates_remain_unresolved(config):
    rows = pd.DataFrame(factorial(config))
    rows["p_continue"] = predict(rows, "baseline", [-0.5, 0.7, 0.3])
    cv = cross_validate(rows, "p_continue", epsilon=0.001)
    assert comparison(cv, config)["winner"] is None


def test_actual_budget_recovers_both_synthetic_generators(config):
    result = recovery(factorial(config), config)
    assert result["passed"], result
    assert set(result["generators"]) == {"gate", "integrated"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A", "A"),
        (" C\n", "C"),
        ("A because rewards", None),
        ("STOP", None),
        ("", None),
        ("A B", None),
    ],
)
def test_invalid_responses_are_not_recoded(raw, expected):
    assert parse_response(raw, ("A", "B", "C")) == expected


def test_probabilities_and_failures_are_distinct(config):
    row = factorial(config)[1]
    response = {
        "raw_response": "C",
        "action_logits": {"A": -2.0, "B": -2.0, "C": 0.0},
        "action_probabilities": {"A": 0.1, "B": 0.1, "C": 0.8},
        "p_action_mass_raw": 0.3,
        "top_token_is_action": True,
    }
    result = _measurement(row, response, 10)
    assert result["p_continue"] == pytest.approx(0.2)
    assert result["observed_failure"] is False
    result = _measurement(row, {**response, "raw_response": "B"}, 10)
    assert result["observed_failure"] is True
    result = _measurement(row, {**response, "raw_response": "malformed"}, 10)
    assert result["observed_failure"] is None
    assert result["choice_continue"] is None
    assert not result["valid_response"]


def test_offline_collection_loads_cluster_model_directory(
    config, tmp_path, monkeypatch
):
    output = tmp_path / "run"
    model_path = tmp_path / "models" / "Qwen--Qwen3.5-4B"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}")
    prepare(config, output, pilot=True)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.delenv("DIAGNOSIS_MODEL_PATH", raising=False)
    calls = []

    class LocalParticipant:
        model_id = str(model_path)
        revision = config["model"]["revision"]

        def decision(self, messages, labels, *, seed):
            # Valid responses without loading weights; exercise real-Qwen routing.
            probabilities = dict.fromkeys(labels, 1 / len(labels))
            return {
                "raw_response": labels[-1],
                "action_logits": dict.fromkeys(labels, 0.0),
                "action_probabilities": probabilities,
                "p_action_mass_raw": 1.0,
                "top_token_is_action": True,
            }

    def load(source, **kwargs):
        calls.append((source, kwargs))
        assert source == str(
            model_path
        ), "offline load must use the existing local checkout"
        assert kwargs["local_files_only"] is True
        return LocalParticipant()

    monkeypatch.setattr(BanditQwenParticipant, "from_pretrained", load)
    collect(output, "main", execute=True)
    assert len(calls) == 1
    rows = records(output / "main_observations.jsonl")
    assert {row["model_id"] for row in rows} == {config["model"]["id"]}
    source = json.loads((output / "model_source.json").read_text())
    assert source["local_path"] == str(model_path)
    assert source["local_files_only"] is True


def test_missing_offline_model_never_falls_back_to_hub(config, tmp_path, monkeypatch):
    output = tmp_path / "run"
    prepare(config, output, pilot=True)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing"))
    monkeypatch.delenv("DIAGNOSIS_MODEL_PATH", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("missing local models must be rejected before any hub lookup")

    monkeypatch.setattr(BanditQwenParticipant, "from_pretrained", forbidden)
    with pytest.raises(FileNotFoundError, match="config.json"):
        collect(output, "main", execute=True)


class FakeTokenizer:
    eos_token_id = 7

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if tokenize:
            import torch

            return torch.tensor([[0, 1]])
        return "prefix"

    def encode(self, text, **kwargs):
        return [0, 1] + (
            [{"A": 2, "B": 3, "C": 4}[text[-1]]] if text != "prefix" else []
        )

    def decode(self, ids, **kwargs):
        return "C"


def test_qwen_three_action_interface_samples_without_forcing_actions():
    import torch

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **kwargs):
            assert kwargs["output_hidden_states"] is False
            return SimpleNamespace(
                logits=torch.tensor([[[0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]])
            )

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is True
            assert kwargs["top_k"] == 0
            assert kwargs["top_p"] == 1.0
            return torch.tensor([[0, 1, 4, 7]])

    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "choose"}]
    assert verify_chat_action_tokens(tokenizer, messages, ("A", "B", "C")) == {
        "A": 2,
        "B": 3,
        "C": 4,
    }
    with pytest.raises(ValueError, match="distinct"):
        verify_chat_action_tokens(tokenizer, messages, ("A", "A", "C"))
    result = BanditQwenParticipant(FakeModel(), tokenizer, "test").decision(
        messages, ("A", "B", "C"), seed=1
    )
    assert result["raw_response"] == "C"
    assert sum(result["action_probabilities"].values()) == pytest.approx(1)
    assert result["action_probabilities"]["C"] == pytest.approx(
        expit(3 - np.logaddexp(1, 2))
    )
    assert 0 < result["p_action_mass_raw"] < 1


def fast_config(config):
    result = deepcopy(config)
    result["analysis"]["recovery_replicates"] = 2
    result["analysis"]["bootstrap_replicates"] = 30
    return result


def test_pandas3_nullable_design_collects_and_reports_strict_json(config, tmp_path):
    config = fast_config(config)
    output = tmp_path / "pandas3-run"
    with pd.option_context("future.infer_string", True):
        prepare(config, output, pilot=True, synthetic="integrated")
        collect(output, "main", execute=True)
        collect(output, "calibration", execute=True)
        freeze_predictions(output)
        collect(output, "intervention", execute=True)
        result = analyze(output)
    assert result["data_kind"] == "synthetic"

    def reject_constant(value):
        raise ValueError(f"non-JSON numeric constant: {value}")

    for path in output.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            json.loads(line, parse_constant=reject_constant)
    baseline = records(output / "main_observations.jsonl")[0]
    assert baseline["expected_action"] is None


def test_staged_workflow_freezes_before_intervention_and_resumes(config, tmp_path):
    config = fast_config(config)
    output = tmp_path / "run"
    prepare(config, output, pilot=True, synthetic="integrated")
    assert collect(output, "main")["execute"] is False
    assert not (output / "main_observations.jsonl").exists()
    with pytest.raises(FileNotFoundError):
        collect(output, "intervention")
    for phase in ("main", "calibration"):
        collect(output, phase, execute=True)
        original = (output / f"{phase}_observations.jsonl").read_bytes()
        collect(output, phase, execute=True)
        assert (output / f"{phase}_observations.jsonl").read_bytes() == original
    frozen = freeze_predictions(output)
    assert frozen["attractiveness_gate_passed"] is True
    pending = analyze(output)
    assert pending["counterfactual"]["status"] == "pending_independent_collection"
    prediction_bytes = (output / "preregistered_predictions.jsonl").read_bytes()
    collect(output, "intervention", execute=True)
    result = analyze(output)
    assert result["data_kind"] == "synthetic"
    assert result["go_no_go"] == "synthetic validation only"
    assert result["counterfactual"]["status"] == "complete"
    assert (output / "preregistered_predictions.jsonl").read_bytes() == prediction_bytes
    assert all(
        (output / name).exists()
        for name in (
            "dataset.jsonl",
            "model_comparison.json",
            "recovery.json",
            "diagnostic_figure.png",
            "counterfactual_results.csv",
            "diagnosis.md",
            "analysis_manifest.json",
            "failure_diagnoses.jsonl",
        )
    )
    with pytest.raises(ValueError, match="immutable"):
        freeze_predictions(output)
    with pytest.raises(FileExistsError):
        prepare(config, output)
    (output / "preregistered_predictions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        verify_freeze(output)


def test_resume_preserves_per_trial_randomness_and_rejects_changed_inputs(
    config, tmp_path
):
    output = tmp_path / "run"
    prepare(config, output, pilot=True, synthetic="gate")
    collect(output, "main", execute=True)
    expected = (output / "main_observations.jsonl").read_bytes()
    observed = records(output / "main_observations.jsonl")
    (output / "main_observations.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in observed[:3])
    )
    collect(output, "main", execute=True)
    assert records(output / "main_observations.jsonl") == [
        json.loads(r) for r in expected.splitlines()
    ]
    (output / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="run input changed"):
        collect(output, "main")


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_kind", "qwen"),
        ("model_id", "foreign/checkpoint"),
        ("model_revision", "a" * 40),
        ("sampling_seed", -1),
    ],
)
def test_resume_rejects_checkpoint_identity_mismatch(config, tmp_path, field, value):
    output = tmp_path / "run"
    prepare(config, output, pilot=True, synthetic="integrated")
    collect(output, "main", execute=True)
    checkpoint = output / "main_observations.jsonl"
    rows = records(checkpoint)
    rows[1][field] = value
    checkpoint.write_text("".join(json.dumps(row) + "\n" for row in rows))

    # A completed checkpoint must still be validated before skipping inference.
    with pytest.raises(ValueError, match="observation identity mismatch"):
        collect(output, "main")


@pytest.mark.parametrize(
    "field",
    [
        "raw_response",
        "actual_choice",
        "choice_continue",
        "observed_failure",
        "p_continue",
    ],
)
def test_resume_rejects_inconsistent_derived_measurements(config, tmp_path, field):
    output = tmp_path / "run"
    prepare(config, output, pilot=True, synthetic="integrated")
    collect(output, "main", execute=True)
    checkpoint = output / "main_observations.jsonl"
    rows = records(checkpoint)
    row = rows[1]  # A corrected primary decision.
    if field in ("raw_response", "actual_choice"):
        row[field] = "C" if row[field] != "C" else "A"
    elif field == "p_continue":
        row[field] = 0.123456
    else:
        row[field] = 1 - row[field]
    checkpoint.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="inconsistent observation measurement"):
        collect(output, "main")


@pytest.mark.parametrize("kind", ["ceiling", "invalid"])
def test_ceiling_and_invalid_trials_cannot_produce_diagnoses(config, tmp_path, kind):
    config = fast_config(config)

    class ControlParticipant(SyntheticParticipant):
        def decision_for_row(self, row, *, seed):
            result = super().decision_for_row(row, seed=seed)
            if row["correction"] and row["trial_type"] == "choice":
                result["raw_response"] = "C" if kind == "ceiling" else "unparseable"
            return result

    output = tmp_path / kind
    prepare(config, output, pilot=True, synthetic="integrated")
    collect(output, "main", execute=True, participant=ControlParticipant(config))
    collect(output, "calibration", execute=True)
    freeze_predictions(output)
    result = analyze(output)
    assert result["observed_deviation"]["violations"] == 0
    if kind == "ceiling":
        assert result["computational_diagnosis"].startswith("ceiling effect")
        assert result["observed_deviation"]["valid_corrected_choices"] == 32
    else:
        assert result["observed_deviation"]["valid_corrected_choices"] == 0
        assert not result["controls"]["passed"]
        assert result["computational_diagnosis"].startswith("insufficient evidence")

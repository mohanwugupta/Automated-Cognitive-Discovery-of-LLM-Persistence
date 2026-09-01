import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cognitive_discovery.ood_generation.analysis import (
    cox_alpha_model,
    immediate_random_specificity,
)
from cognitive_discovery.ood_generation.frozen import (
    fit_e_orientation,
    validate_sampling_config,
    verify_frozen_protocol,
)
from cognitive_discovery.ood_generation.generation import (
    GenerationStep,
    QwenFreeGenerationRunner,
    _eos_stats,
    decode_autoregressive,
    edit_final_token,
)
from cognitive_discovery.ood_generation.pipeline import (
    aggregate_ood_run,
    build_evaluation_jobs,
    prepare_ood_run,
)
from cognitive_discovery.pipeline import load_config
from cognitive_discovery.data.storage import write_records


def _step_with_selected_token(token_id, vocabulary=4):
    def step(_tokens, _past, _index):
        logits = torch.full((vocabulary,), -1000.0)
        logits[int(token_id)] = 0.0
        return GenerationStep(logits=logits, past_key_values=object())

    return step


def test_context_exhaustion_is_censored_but_eos_is_an_event():
    common = {
        "initial_token_ids": [1, 2],
        "eos_token_ids": (3,),
        "context_limit": 5,
        "context_safety_margin": 0,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
        "seed": 7,
    }
    eos = decode_autoregressive(step=_step_with_selected_token(3), **common)
    assert eos.eos_emitted
    assert eos.termination_reason == "eos"
    assert not eos.context_censored
    assert eos.generated_token_ids == []

    capacity = decode_autoregressive(step=_step_with_selected_token(0), **common)
    assert not capacity.eos_emitted
    assert capacity.termination_reason == "context_limit"
    assert capacity.context_censored
    assert len(capacity.generated_token_ids) == 3


def test_primary_sampling_rejects_hidden_application_caps_and_keeps_eos_sampleable():
    with pytest.raises(ValueError, match="forbidden output cap"):
        validate_sampling_config({"max_new_tokens": 512})
    frozen = validate_sampling_config({})
    assert frozen["application_token_cap"] is None
    assert (
        frozen["scientific_stop_rule"]
        == "canonical_eos_or_architectural_context_capacity"
    )
    for dose in (-2, -1, 0, 1, 2):
        logits = torch.tensor([0.0, float(dose), -float(dose)])
        probability, _ = _eos_stats(logits, (2,))
        assert 0 < probability < 1


def test_hook_changes_only_new_final_token_and_alpha_zero_is_exact():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    direction = torch.tensor([1.0, -2.0, 3.0, -4.0])
    changed = edit_final_token(hidden, direction, 0.5)
    assert torch.equal(changed[:, :-1], hidden[:, :-1])
    assert torch.equal(changed[:, -1], hidden[:, -1] + 0.5 * direction)
    assert torch.equal(edit_final_token(hidden, direction, 0.0), hidden)


def test_e_orientation_is_restricted_to_frozen_subspace_and_signed_high_to_low():
    rng = np.random.default_rng(8)
    basis, _ = np.linalg.qr(rng.normal(size=(12, 2)))
    coordinates = rng.normal(size=(300, 2))
    evidence = (
        1.5 * coordinates[:, 0]
        - 0.5 * coordinates[:, 1]
        + rng.normal(scale=0.05, size=300)
    )
    result = fit_e_orientation(basis, coordinates, evidence)
    assert result["hidden_direction"].shape == (12,)
    assert np.isclose(np.linalg.norm(result["hidden_direction"]), 1.0)
    assert result["sign_check"]["passed"]
    assert (
        result["sign_check"]["high_projection_mean_E"]
        > result["sign_check"]["low_projection_mean_E"]
    )
    assert result["r2"] > 0.99


class _TinyTokenizer:
    eos_token_id = 3
    model_max_length = 32

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(map(str, token_ids))


class _TinyParticipant:
    def __init__(self, model):
        self.model = model
        self.tokenizer = _TinyTokenizer()

    def _tokenize(self, _messages):
        return {"input_ids": torch.tensor([[0, 1]], dtype=torch.long)}


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        self.embedding = torch.nn.Embedding(4, 5)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.output = torch.nn.Linear(5, 4, bias=False)
        self.config = SimpleNamespace(max_position_embeddings=32, eos_token_id=3)
        self.generation_config = SimpleNamespace(eos_token_id=[3])

    def prepare_inputs_for_generation(
        self, input_ids, *, past_key_values=None, attention_mask=None, use_cache=True
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(
            logits=self.output(hidden),
            past_key_values=torch.tensor([1]) if use_cache else None,
        )


def test_cache_and_uncached_logits_match_at_zero_intervention():
    runner = QwenFreeGenerationRunner(_TinyParticipant(_TinyModel()))
    checks = runner.numerical_checks("anything", layer=0, direction=np.ones(5))
    assert checks["alpha_zero_passed"]
    assert checks["cache_equivalent"]
    assert checks["context_limit"] == 32
    assert checks["eos_token_ids"] == [3]


def test_job_manifest_uses_matched_seeds_for_every_dose():
    prompts = [
        {"prompt_id": "open_ended_01", "prompt_family": "open_ended", "primary": True},
        {"prompt_id": "open_ended_02", "prompt_family": "open_ended", "primary": True},
    ]
    config = {
        "intervention": {"doses": [-2, -1, 0, 1, 2]},
        "design": {"primary_seeds_per_prompt": 4, "seeds_per_job": 2, "seed_start": 10},
        "pulse": {"enabled": False},
        "secondary": {"fixed_topic": False, "continuation": False},
        "controls": {
            "behavioral_random_subspaces": 0,
            "eos_seeds_per_prompt": 2,
            "random_subspaces": 2,
            "neural_random_directions_per_job": 2,
        },
    }
    jobs = build_evaluation_jobs(config, prompts)
    primary = [job for job in jobs if job["control_role"] == "frozen_E"]
    expanded = {}
    for job in primary:
        for prompt_id in job["prompt_ids"]:
            for alpha in job["doses"]:
                expanded.setdefault((prompt_id, alpha), set()).update(job["seeds"])
    for prompt_id in ("open_ended_01", "open_ended_02"):
        seed_sets = [expanded[(prompt_id, alpha)] for alpha in (-2, -1, 0, 1, 2)]
        assert all(value == {10, 11, 12, 13} for value in seed_sets)


def test_synthetic_censored_survival_recovers_negative_positive_e_hazard():
    rng = np.random.default_rng(3)
    rows = []
    for prompt in range(6):
        for alpha in (-2, -1, 0, 1, 2):
            hazard = 0.22 * np.exp(-0.55 * alpha)
            durations = rng.geometric(min(hazard, 0.95), size=80)
            for duration in durations:
                rows.append(
                    {
                        "prompt_id": f"p{prompt}",
                        "alpha": alpha,
                        "survival_time": min(int(duration), 12),
                        "eos_emitted": bool(duration <= 12),
                    }
                )
    result = cox_alpha_model(pd.DataFrame(rows))
    assert result["converged"]
    assert result["coefficient"] < -0.35
    assert result["ci_upper"] < 0


def test_planted_e_direction_exceeds_one_hundred_random_controls():
    rng = np.random.default_rng(9)
    rows = []
    for alpha in (-2, -1, 0, 1, 2):
        rows.append(
            {
                "direction_id": "frozen_E",
                "control_role": "frozen_E",
                "alpha": alpha,
                "eos_logit": -alpha,
            }
        )
    for index in range(100):
        slope = rng.normal(0, 0.08)
        for alpha in (-2, -1, 0, 1, 2):
            rows.append(
                {
                    "direction_id": f"random_{index:03d}",
                    "control_role": "random_subspace",
                    "alpha": alpha,
                    "eos_logit": slope * alpha,
                }
            )
    result = immediate_random_specificity(pd.DataFrame(rows))
    assert result["passed"]
    assert result["random_p"] < 0.05


def test_real_pre_ood_artifacts_freeze_and_hash_verify(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = repository / "artifacts/abstraction_discovery_v1"
    if not source.exists():
        pytest.skip("tracked pre-OOD source artifacts are unavailable")
    config = load_config(repository / "configs/ood_free_generation_v1.yaml")
    result = prepare_ood_run(config, abstraction_output=source, output=tmp_path / "ood")
    manifest = verify_frozen_protocol(tmp_path / "ood")
    assert result["evaluation_jobs"] == 66
    assert result["primary_generation_runs"] == 1500
    assert result["full_generation_runs"] == 2380
    assert result["immediate_control_forwards"] == 3030
    assert result["maximum_generations_per_job"] == 54
    assert result["maximum_guarded_generation_hours_per_job"] == 4.5
    assert result["orientation_rows"] > 1000
    assert manifest["rank"] == 2
    assert manifest["layer"] == 28
    hashes = json.loads((tmp_path / "ood/frozen/hashes.json").read_text())
    assert "e_orientation.pt" in hashes
    assert "evaluation_jobs.json" in hashes
    prompt_manifest = tmp_path / "ood/frozen/prompt_manifest.json"
    prompt_manifest.write_text(prompt_manifest.read_text() + " ")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_frozen_protocol(tmp_path / "ood")


def test_synthetic_shards_complete_ood_gates_figures_and_report(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = repository / "artifacts/abstraction_discovery_v1"
    if not source.exists():
        pytest.skip("tracked pre-OOD source artifacts are unavailable")
    config = load_config(repository / "configs/ood_free_generation_v1.yaml")
    root = tmp_path / "ood"
    prepare_ood_run(config, abstraction_output=source, output=root)
    jobs = json.loads((root / "frozen/evaluation_jobs.json").read_text())
    prompts = {
        row["prompt_id"]: row
        for row in json.loads((root / "frozen/prompt_manifest.json").read_text())[
            "prompts"
        ]
    }
    manifest = json.loads((root / "frozen/das_manifest.json").read_text())
    rng = np.random.default_rng(31)
    for job in jobs:
        shard = root / "shards" / f"job_{job['job_index']:04d}"
        if job["mode"] == "generation":
            summaries, tokens = [], []
            direction_id = (
                f"random_{job['random_index']:03d}"
                if job["control_role"] == "random_subspace"
                else ("frozen_E" if job["control_role"] == "frozen_E" else "direct_eos")
            )
            for prompt_id in job["prompt_ids"]:
                for seed in job["seeds"]:
                    for alpha in job["doses"]:
                        effect = 0.55 if job["control_role"] == "frozen_E" else 0.0
                        if job["regime"] == "pulse":
                            effect = 0.12
                        if job["control_role"] == "direct_eos":
                            effect = 0.8
                        hazard = min(0.9, 0.25 * np.exp(-effect * float(alpha)))
                        duration = (
                            3 + (int(seed) + sum(map(ord, prompt_id))) % 4
                            if float(alpha) == 0.0
                            else int(rng.geometric(hazard))
                        )
                        run_id = f"{job['job_index']}:{prompt_id}:{seed}:{alpha}"
                        common = {
                            "run_id": run_id,
                            "job_index": job["job_index"],
                            "prompt_id": prompt_id,
                            "prompt_family": prompts[prompt_id]["prompt_family"],
                            "control_role": job["control_role"],
                            "direction_id": direction_id,
                            "seed": seed,
                            "alpha": alpha,
                            "regime": job["regime"],
                        }
                        summaries.append(
                            {
                                **common,
                                "prompt_tokens": 4,
                                "generated_tokens": duration - 1,
                                "generated_words": duration - 1,
                                "decision_steps": duration,
                                "survival_time": duration,
                                "eos_emitted": True,
                                "termination_reason": "eos",
                                "context_censored": False,
                                "context_limit": 128,
                                "elapsed_seconds": 0.01,
                                "generated_text": "coherent synthetic continuation",
                                "repeated_4gram_fraction": 0.0,
                                "repeated_8gram_fraction": 0.0,
                                "repeated_sentence_fraction": 0.0,
                                "lexical_diversity": 0.8,
                                "punctuation_fraction": 0.02,
                                "severe_degeneration": False,
                            }
                        )
                        p_eos = float(1 / (1 + np.exp(float(alpha))))
                        tokens.append(
                            {
                                **common,
                                "token_index": 1,
                                "generated_before": 0,
                                "sampled_token_id": 3,
                                "sampled_eos": True,
                                "p_eos": p_eos,
                                "eos_logit": -float(alpha),
                                "continuation_evidence": float(alpha),
                                "baseline_p_eos": None,
                                "baseline_eos_logit": None,
                            }
                        )
            write_records(summaries, shard / "generation_summary.parquet")
            write_records(tokens, shard / "token_events.parquet")
        else:
            immediate = []
            directions = [
                (f"random_{index:03d}", "random_subspace", 0.01 * np.sin(index))
                for index in job["random_indices"]
            ]
            if min(job["random_indices"]) == 0:
                directions.append(("frozen_E", "frozen_E", -1.0))
            for direction_id, role, slope in directions:
                for prompt_id in job["prompt_ids"]:
                    for alpha in job["doses"]:
                        immediate.append(
                            {
                                "job_index": job["job_index"],
                                "prompt_id": prompt_id,
                                "direction_id": direction_id,
                                "control_role": role,
                                "alpha": alpha,
                                "p_eos": float(1 / (1 + np.exp(-slope * alpha))),
                                "eos_logit": slope * alpha,
                            }
                        )
            write_records(immediate, shard / "immediate_controls.parquet")
        audit = {
            "job_index": job["job_index"],
            "job_mode": job["mode"],
            "frozen_before_ood": True,
            "das_sha256": manifest["sha256"],
            "orientation_sha256": manifest["orientation_sha256"],
            "expected_orientation_sha256": manifest["orientation_sha256"],
            "layer": 28,
            "rank": 2,
            "context_limit": 128,
            "eos_token_ids": [3],
            "application_output_cap": None,
            "kv_cache_used": True,
            "final_token_only": True,
            "full_activations_saved": False,
            "cuda_device": "cuda:0",
            "gpu_evaluation_only": True,
            "summaries": 0,
            "token_events": 0,
            "immediate_controls": 0,
        }
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "audit.json").write_text(json.dumps(audit))
    checks = {
        "alpha_zero_passed": True,
        "cache_equivalent": True,
        "alpha_zero_max_abs_logit_error": 0.0,
        "cache_max_abs_logit_error": 0.0,
    }
    (root / "shards/job_0000/numerical_checks.json").write_text(json.dumps(checks))
    result = aggregate_ood_run(config, output=root)
    assert result["passed"]
    assert result["cox_alpha"] < 0
    assert result["random_p"] < 0.05
    assert (root / "report.md").exists()
    assert len(list((root / "figures").glob("*.png"))) == 7

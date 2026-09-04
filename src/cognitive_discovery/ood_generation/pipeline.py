"""End-to-end frozen OOD persistence-generalization workflow."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from cognitive_discovery.data.storage import read_records, write_records

from .analysis import (
    cox_alpha_model,
    dose_response,
    immediate_random_specificity,
    kaplan_meier,
    make_figures,
    prompt_effects,
    quality_summary,
)
from .frozen import (
    freeze_protocol,
    json_write,
    load_orientation,
    resolve_record_path,
    sha256_file,
    verify_frozen_protocol,
)
from .generation import GENERATION_ENGINE_VERSION, QwenFreeGenerationRunner


def _root(config: dict, output=None) -> Path:
    return Path(output or config.get("output_root", "artifacts/ood_free_generation_v1"))


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


def _seed_batches(count: int, batch_size: int, *, start: int) -> list[list[int]]:
    seeds = list(range(int(start), int(start) + int(count)))
    return [
        seeds[index : index + int(batch_size)]
        for index in range(0, len(seeds), int(batch_size))
    ]


def build_evaluation_jobs(config: dict, prompts: list[dict]) -> list[dict]:
    """Freeze restartable GPU work units without inspecting OOD outcomes."""

    jobs = []
    doses = [
        float(value)
        for value in config.get("intervention", {}).get("doses", (-2, -1, 0, 1, 2))
    ]
    design = config.get("design", {})
    batch_size = int(design.get("seeds_per_job", 10))
    seed_start = int(design.get("seed_start", 1))

    def add_prompt_generations(
        prompt_family, *, regime, role, seed_count, selected_doses=doses
    ):
        for prompt in [
            value for value in prompts if value["prompt_family"] == prompt_family
        ]:
            for batch_index, seeds in enumerate(
                _seed_batches(seed_count, batch_size, start=seed_start)
            ):
                jobs.append(
                    {
                        "mode": "generation",
                        "prompt_ids": [prompt["prompt_id"]],
                        "prompt_family": prompt_family,
                        "regime": regime,
                        "control_role": role,
                        "seeds": seeds,
                        "doses": list(selected_doses),
                        "batch_index": batch_index,
                    }
                )

    add_prompt_generations(
        "open_ended",
        regime="continuous",
        role="frozen_E",
        seed_count=int(design.get("primary_seeds_per_prompt", 100)),
    )
    if config.get("pulse", {}).get("enabled", True):
        add_prompt_generations(
            "open_ended",
            regime="pulse",
            role="frozen_E",
            seed_count=int(config.get("pulse", {}).get("seeds_per_prompt", 50)),
        )
    if config.get("secondary", {}).get("fixed_topic", True):
        add_prompt_generations(
            "fixed_topic",
            regime="continuous",
            role="frozen_E",
            seed_count=int(
                config.get("secondary", {}).get("fixed_topic_seeds_per_prompt", 50)
            ),
        )
    if config.get("secondary", {}).get("continuation", False):
        add_prompt_generations(
            "continuation",
            regime="continuous",
            role="frozen_E",
            seed_count=int(
                config.get("secondary", {}).get("continuation_seeds_per_prompt", 50)
            ),
        )

    controls = config.get("controls", {})
    behavioral_count = int(controls.get("behavioral_random_subspaces", 10))
    behavioral_seeds = list(
        range(
            seed_start,
            seed_start + int(controls.get("behavioral_seeds_per_prompt", 10)),
        )
    )
    behavioral_doses = [
        float(value) for value in controls.get("behavioral_random_doses", (-2, 0, 2))
    ]
    primary_prompt_ids = [value["prompt_id"] for value in prompts if value["primary"]]
    for random_index in range(behavioral_count):
        jobs.append(
            {
                "mode": "generation",
                "prompt_ids": primary_prompt_ids,
                "prompt_family": "open_ended",
                "regime": "continuous",
                "control_role": "random_subspace",
                "random_index": random_index,
                "seeds": behavioral_seeds,
                "doses": behavioral_doses,
                "batch_index": 0,
            }
        )
    eos_seeds = list(
        range(seed_start, seed_start + int(controls.get("eos_seeds_per_prompt", 10)))
    )
    for prompt_id in primary_prompt_ids:
        jobs.append(
            {
                "mode": "generation",
                "prompt_ids": [prompt_id],
                "prompt_family": "open_ended",
                "regime": "eos_control",
                "control_role": "direct_eos",
                "seeds": eos_seeds,
                "doses": doses,
                "batch_index": 0,
            }
        )

    random_count = int(controls.get("random_subspaces", 100))
    random_batch = int(controls.get("neural_random_directions_per_job", 10))
    for start in range(0, random_count, random_batch):
        jobs.append(
            {
                "mode": "immediate_control",
                "prompt_ids": primary_prompt_ids,
                "prompt_family": "open_ended",
                "regime": "first_step",
                "control_role": "random_subspace",
                "random_indices": list(
                    range(start, min(start + random_batch, random_count))
                ),
                "doses": doses,
            }
        )
    for index, job in enumerate(jobs):
        job["job_index"] = index
        job["frozen_before_ood"] = True
    if not jobs:
        raise RuntimeError("OOD protocol produced no evaluation jobs")
    return jobs


def prepare_ood_run(
    config: dict,
    *,
    abstraction_output: str | Path | None = None,
    output: str | Path | None = None,
):
    """Freeze direction, scale, prompts, decoding, controls, and analyses on CPU."""

    root = _root(config, output)
    source_root = Path(
        abstraction_output
        or config.get("abstraction_output", "artifacts/abstraction_discovery_v1")
    )
    if any((root / "shards").glob("job_*/audit.json")):
        raise RuntimeError(
            "the OOD protocol cannot be regenerated after outcome collection begins"
        )
    root.mkdir(parents=True, exist_ok=True)
    frozen = freeze_protocol(config, source_root, root)
    jobs = build_evaluation_jobs(config, frozen["prompts"])
    jobs_path = json_write(root / "frozen/evaluation_jobs.json", jobs)
    hashes_path = root / "frozen/hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes[jobs_path.name] = sha256_file(jobs_path)
    json_write(hashes_path, hashes)
    metadata = {
        "protocol_version": config.get("protocol_version", "ood_free_generation_v1"),
        "execution_profile": config.get(
            "execution_profile", "compute_efficient_initial"
        ),
        "git_commit": _git_commit(),
        "source_abstraction_root": str(source_root),
        "primary_artifact_id": frozen["primary"]["artifact_id"],
        "layer": int(frozen["primary"]["layer"]),
        "rank": int(frozen["primary"]["rank"]),
        "das_sha256": frozen["primary"]["sha256"],
        "orientation_sha256": frozen["primary"]["orientation_sha256"],
        "orientation_source": "structured_persistence_only",
        "ood_data_used_to_freeze": False,
        "doses": frozen["doses"],
        "prompts": len(frozen["prompts"]),
        "evaluation_jobs": len(jobs),
        "application_output_cap": None,
        "full_activations_saved": False,
    }
    json_write(
        root / "gates.json",
        {
            "frozen_protocol": {
                "passed": True,
                "das_hash_verified": True,
                "orientation_pre_ood": True,
                "layer_rank_dose_tuned_on_ood": False,
                "application_output_cap_removed": True,
            },
            "ood_generalization": {"status": "awaiting_frozen_evaluation"},
        },
    )
    full_generation_runs = sum(
        len(job.get("prompt_ids", ()))
        * len(job.get("seeds", ()))
        * len(job.get("doses", ()))
        for job in jobs
        if job.get("mode") == "generation"
    )
    maximum_generations_per_job = max(
        (
            len(job.get("prompt_ids", ()))
            * len(job.get("seeds", ()))
            * len(job.get("doses", ()))
            for job in jobs
            if job.get("mode") == "generation"
        ),
        default=0,
    )
    generation_timeout = float(
        config.get("runtime", {}).get("generation_timeout_seconds", 300)
    )
    maximum_guarded_generation_seconds_per_job = (
        maximum_generations_per_job * generation_timeout
    )
    immediate_control_forwards = sum(
        len(job.get("prompt_ids", ()))
        * len(job.get("doses", ()))
        * (
            len(job.get("random_indices", ()))
            + (1 if 0 in job.get("random_indices", ()) else 0)
        )
        for job in jobs
        if job.get("mode") == "immediate_control"
    )
    metadata.update(
        {
            "full_generation_runs": full_generation_runs,
            "immediate_control_forwards": immediate_control_forwards,
            "maximum_generations_per_job": maximum_generations_per_job,
            "maximum_guarded_generation_seconds_per_job": (
                maximum_guarded_generation_seconds_per_job
            ),
        }
    )
    json_write(root / "run_metadata.json", metadata)
    return {
        "output": root,
        "evaluation_jobs": len(jobs),
        "primary_generation_runs": sum(
            len(job.get("seeds", ())) * len(job.get("doses", ()))
            for job in jobs
            if job.get("control_role") == "frozen_E"
            and job.get("regime") == "continuous"
            and job.get("prompt_family") == "open_ended"
        ),
        "full_generation_runs": full_generation_runs,
        "immediate_control_forwards": immediate_control_forwards,
        "maximum_generations_per_job": maximum_generations_per_job,
        "maximum_guarded_generation_hours_per_job": (
            maximum_guarded_generation_seconds_per_job / 3600
        ),
        "orientation_rows": frozen["orientation"]["orientation_rows"],
        "orientation_sha256": frozen["primary"]["orientation_sha256"],
    }


def _load_prompts(root: Path) -> dict[str, dict]:
    value = json.loads(
        (root / "frozen/prompt_manifest.json").read_text(encoding="utf-8")
    )
    return {row["prompt_id"]: row for row in value["prompts"]}


def _runner(config, *, model_path=None, revision=None, online=False):
    return QwenFreeGenerationRunner.from_pretrained(
        model_path or config["model"],
        revision=revision or config.get("model_revision"),
        local_files_only=not online,
    )


def evaluate_ood_job(
    config: dict,
    *,
    job_index: int,
    output: str | Path | None = None,
    model_path=None,
    revision=None,
    online=False,
    limit: int | None = None,
    validate_only: bool = False,
):
    """Run one frozen GPU shard with continuous autoregressive model use."""

    import torch
    import transformers

    root = _root(config, output)
    manifest = verify_frozen_protocol(root)
    jobs = json.loads(
        (root / "frozen/evaluation_jobs.json").read_text(encoding="utf-8")
    )
    job_index = int(job_index)
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"OOD job index must lie in 0..{len(jobs) - 1}")
    if validate_only and job_index != 0:
        raise ValueError("OOD cache validation-only mode requires job index 0")
    job = jobs[job_index]
    shard = root / "shards" / f"job_{job_index:04d}"
    prompts = _load_prompts(root)
    analysis_plan = json.loads(
        (root / "frozen/analysis_plan.json").read_text(encoding="utf-8")
    )
    orientation = load_orientation(root / "frozen/e_orientation.pt")
    random = np.load(root / "frozen/random_subspaces.npz")
    if revision is not None and revision != analysis_plan.get("model_revision"):
        raise RuntimeError(
            "runtime model revision differs from the frozen OOD protocol"
        )
    frozen_model_config = {
        "model": analysis_plan["model"],
        "model_revision": analysis_plan.get("model_revision"),
    }
    runner = _runner(
        frozen_model_config,
        model_path=model_path,
        revision=analysis_plan.get("model_revision"),
        online=online,
    )
    device = next(runner.model.parameters()).device
    if config.get("cluster", {}).get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("OOD evaluation reserved a GPU but the model is not on CUDA")
    layer = int(manifest["layer"])
    if layer >= len(runner.layers):
        raise RuntimeError("frozen OOD layer is outside the loaded model")
    sampling = json.loads(
        (root / "frozen/sampling_config.json").read_text(encoding="utf-8")
    )
    numerical_checks = None
    if job_index == 0:
        first_prompt = prompts[job["prompt_ids"][0]]["text"]
        numerical_checks = runner.numerical_checks(
            first_prompt,
            layer=layer,
            direction=orientation["hidden_direction"],
        )
        # Persist and print the measurements before enforcing the gate.  A
        # failed validation shard must remain diagnosable from its artifact and
        # SLURM log instead of exposing only a generic exception.
        numerical_path = json_write(shard / "numerical_checks.json", numerical_checks)
        print(
            "OOD cache validation: " + json.dumps(numerical_checks, sort_keys=True),
            flush=True,
        )
        if not numerical_checks["alpha_zero_passed"]:
            raise RuntimeError(
                "alpha-zero hook is not numerically identical to baseline; "
                f"diagnostics={numerical_path}"
            )
        if not numerical_checks["cache_state_reuse_equivalent"]:
            replay = numerical_checks["cache_state_replay_comparison"]
            criteria = replay["cache_equivalence_criteria"]
            raise RuntimeError(
                "retained KV cache and fresh recurrent replay are not "
                "equivalent: "
                f"total_variation={replay['cache_total_variation_distance']:.6g} "
                f"(max={criteria['total_variation_max']:.6g}), "
                f"eos_log_odds_abs_error={replay['cache_eos_log_odds_abs_error']:.6g} "
                f"(max={criteria['eos_log_odds_abs_error_max']:.6g}); "
                f"diagnostics={numerical_path}"
            )
        if (
            not numerical_checks["cache_equivalent"]
            and not numerical_checks["native_no_cache_limitation"]
        ):
            criteria = numerical_checks["cache_equivalence_criteria"]
            raise RuntimeError(
                "cached and uncached alpha-zero next-token distributions are "
                "not equivalent: "
                f"total_variation={numerical_checks['cache_total_variation_distance']:.6g} "
                f"(max={criteria['total_variation_max']:.6g}), "
                f"eos_log_odds_abs_error={numerical_checks['cache_eos_log_odds_abs_error']:.6g} "
                f"(max={criteria['eos_log_odds_abs_error_max']:.6g}); "
                f"diagnostics={numerical_path}"
            )
        if numerical_checks["native_no_cache_limitation"]:
            print(
                "WARNING: Qwen3.5 native full-sequence/no-cache logits differ "
                "from recurrent cached decoding. The shard will run because "
                "fresh cache-state replay passed, but the strict native cache "
                "equivalence gate remains failed in the aggregate report.",
                flush=True,
            )
        if validate_only:
            return {
                "job_index": job_index,
                "validation_only": True,
                "numerical_checks": numerical_path,
                "generation_engine_version": GENERATION_ENGINE_VERSION,
            }

    summaries, token_rows, immediate_rows = [], [], []
    if job["mode"] == "generation":
        seeds = job["seeds"][: int(limit)] if limit is not None else job["seeds"]
        if job["control_role"] == "random_subspace":
            direction = random["directions"][int(job["random_index"])]
            direction_id = f"random_{int(job['random_index']):03d}"
        else:
            direction = orientation["hidden_direction"]
            direction_id = (
                "frozen_E" if job["control_role"] == "frozen_E" else "direct_eos"
            )
        timeout = analysis_plan.get("runtime", {}).get(
            "generation_timeout_seconds", 600
        )
        eos_scale = float(analysis_plan.get("controls", {}).get("eos_logit_scale", 2.0))
        for prompt_id in job["prompt_ids"]:
            prompt = prompts[prompt_id]
            for seed in seeds:
                for alpha in job["doses"]:
                    summary, events = runner.generate(
                        prompt["text"],
                        seed=int(seed),
                        alpha=float(alpha),
                        layer=layer,
                        direction=direction,
                        sigma_E=orientation["sigma_E"],
                        regime=job["regime"],
                        sampling=sampling,
                        infrastructure_timeout_seconds=timeout,
                        eos_logit_scale=eos_scale,
                    )
                    run_id = (
                        f"j{job_index:04d}:{prompt_id}:s{int(seed):04d}:"
                        f"a{float(alpha):+g}:{direction_id}"
                    )
                    identifiers = {
                        "run_id": run_id,
                        "job_index": job_index,
                        "prompt_id": prompt_id,
                        "prompt_family": prompt["prompt_family"],
                        "control_role": job["control_role"],
                        "direction_id": direction_id,
                    }
                    summaries.append({**identifiers, **summary})
                    token_rows.extend(
                        {
                            **identifiers,
                            "seed": int(seed),
                            "alpha": float(alpha),
                            "regime": job["regime"],
                            **event,
                        }
                        for event in events
                    )
    elif job["mode"] == "immediate_control":
        directions = [
            (f"random_{index:03d}", "random_subspace", random["directions"][index])
            for index in job["random_indices"]
        ]
        if min(job["random_indices"]) == 0:
            directions.append(("frozen_E", "frozen_E", orientation["hidden_direction"]))
        for direction_id, role, direction in directions:
            for prompt_id in job["prompt_ids"]:
                prompt_results = []
                for alpha in job["doses"]:
                    result = runner.immediate_eos_effect(
                        prompts[prompt_id]["text"],
                        layer=layer,
                        direction=direction,
                        sigma_E=orientation["sigma_E"],
                        alpha=float(alpha),
                    )
                    prompt_results.append(
                        {
                            "job_index": job_index,
                            "prompt_id": prompt_id,
                            "direction_id": direction_id,
                            "control_role": role,
                            **result,
                        }
                    )
                baseline = next(
                    value for value in prompt_results if float(value["alpha"]) == 0.0
                )
                immediate_rows.extend(
                    {
                        **value,
                        "baseline_p_eos": baseline["p_eos"],
                        "baseline_eos_logit": baseline["eos_logit"],
                        "delta_eos_logit": value["eos_logit"] - baseline["eos_logit"],
                    }
                    for value in prompt_results
                )
    else:
        raise RuntimeError(f"unknown frozen OOD job mode: {job['mode']}")

    paths = {}
    if summaries:
        paths["summary"] = write_records(
            summaries, shard / "generation_summary.parquet"
        )
        paths["tokens"] = write_records(token_rows, shard / "token_events.parquet")
    if immediate_rows:
        paths["immediate"] = write_records(
            immediate_rows, shard / "immediate_controls.parquet"
        )
    if numerical_checks is not None:
        json_write(shard / "numerical_checks.json", numerical_checks)
    audit = {
        "job_index": job_index,
        "job_mode": job["mode"],
        "frozen_before_ood": bool(job["frozen_before_ood"]),
        "das_sha256": sha256_file(root / "frozen" / Path(manifest["artifact"]).name),
        "orientation_sha256": sha256_file(root / "frozen/e_orientation.pt"),
        "expected_orientation_sha256": manifest["orientation_sha256"],
        "layer": layer,
        "rank": int(manifest["rank"]),
        "context_limit": int(runner.context_limit),
        "eos_token_ids": list(runner.eos_token_ids),
        "application_output_cap": None,
        "kv_cache_used": True,
        "final_token_only": True,
        "full_activations_saved": False,
        "cuda_device": str(device),
        "gpu_evaluation_only": True,
        "generation_engine_version": GENERATION_ENGINE_VERSION,
        "transformers_version": transformers.__version__,
        "torch_version": str(torch.__version__),
        "model_class": type(runner.model).__name__,
        "summaries": len(summaries),
        "token_events": len(token_rows),
        "immediate_controls": len(immediate_rows),
    }
    if numerical_checks is not None:
        audit.update(
            {
                "cache_state_reuse_equivalent": numerical_checks[
                    "cache_state_reuse_equivalent"
                ],
                "native_no_cache_equivalent": numerical_checks["cache_equivalent"],
                "native_no_cache_limitation": numerical_checks[
                    "native_no_cache_limitation"
                ],
            }
        )
    if audit["orientation_sha256"] != audit["expected_orientation_sha256"]:
        raise RuntimeError("orientation changed during OOD evaluation")
    json_write(shard / "audit.json", audit)
    return {"job_index": job_index, **paths, "audit": shard / "audit.json"}


def _load_job_records(
    root: Path, jobs: list[dict], filename: str, *, modes=None
) -> pd.DataFrame:
    frames = []
    for job in jobs:
        if modes is not None and job["mode"] not in modes:
            continue
        path = resolve_record_path(
            root / "shards" / f"job_{int(job['job_index']):04d}" / filename
        )
        if path.exists():
            frames.append(read_records(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_survival(frame: pd.DataFrame) -> dict:
    try:
        return cox_alpha_model(frame)
    except (ValueError, RuntimeError) as error:
        return {"status": "not_estimable", "reason": str(error)}


def _report(
    root: Path,
    *,
    gates: dict,
    metadata: dict,
    survival: dict,
    dose: dict,
    censoring: pd.DataFrame,
    prompt_table: pd.DataFrame,
    fixed_survival: dict,
    random_result: dict,
    eos_control: pd.DataFrame,
    primary: pd.DataFrame,
    quality: pd.DataFrame,
    pulse_survival: dict,
):
    result = gates["ood_generalization"]
    frozen = gates["frozen_protocol"]
    median_change = dose.get("endpoint_rmst_change", float("nan"))
    eos_count = (
        int(censoring.loc[censoring.termination_reason == "eos", "runs"].sum())
        if len(censoring)
        else 0
    )
    context_count = (
        int(
            censoring.loc[censoring.termination_reason == "context_limit", "runs"].sum()
        )
        if len(censoring)
        else 0
    )
    direction_prompts = (
        int((prompt_table.coefficient < 0).sum()) if len(prompt_table) else 0
    )
    eos_control_note = "not run"
    if len(eos_control):
        direct_positive = eos_control[eos_control.alpha == eos_control.alpha.max()]
        e_positive = primary[primary.alpha == primary.alpha.max()]
        eos_control_note = (
            f"{len(eos_control)} direct-EOS benchmark generations were analyzed; "
            f"at the largest positive dose their median length was "
            f"{direct_positive.generated_tokens.median():.1f} tokens and severe-"
            f"degeneration fraction was "
            f"{direct_positive.severe_degeneration.astype(bool).mean():.3f}, versus "
            f"{e_positive.generated_tokens.median():.1f} and "
            f"{e_positive.severe_degeneration.astype(bool).mean():.3f} for frozen E."
        )
    quality_note = "not estimable"
    if len(quality):
        quality_note = f"maximum severe-degeneration fraction={quality.severe_degeneration.max():.3f}"
    claim = (
        "The frozen integrated persistence-evidence state generalizes causally to voluntary free-form termination."
        if result["passed"]
        else "The frozen structured-task state did not satisfy all preregistered OOD generalization gates."
    )
    text = f"""# OOD free-generation persistence report

## Frozen protocol

1. **DAS hash verified:** yes (`{metadata['das_sha256']}`).
2. **E orientation derived only from pre-OOD data:** yes, from {metadata['orientation_rows']} structured activation rows.
3. **Layer/rank/dose tuned on free generation:** no.
4. **Application output caps removed:** yes; stopping was EOS, context capacity, or separately labeled infrastructure timeout.
5. **Actual model context capacity:** {metadata['context_limit']} tokens.
6. **Retained cache matches fresh recurrent replay:** {frozen['cache_state_reuse_equivalent']}.
7. **Native cached versus full-sequence/no-cache equivalence:** {frozen['cache_equivalent']} (disposition: `{frozen['cache_validation_disposition']}`).

## Primary results

8. **EOS terminations:** {eos_count}.
9. **Context-censored generations:** {context_count}.
10. **Did increasing E lower EOS hazard?** {result['direction_passed']} (Cox beta={survival.get('coefficient', float('nan')):.6g}, 95% CI [{survival.get('ci_lower', float('nan')):.6g}, {survival.get('ci_upper', float('nan')):.6g}]).
11. **Monotonic dose response:** {result['dose_response_passed']} (Spearman alpha–RMST={dose.get('spearman_alpha_rmst', float('nan')):.3f}).
12. **Generation-duration change:** endpoint restricted-mean change={median_change:.3f} decision steps.
13. **Prompt wording generalization:** {result['prompt_generalization_passed']}; {direction_prompts}/{len(prompt_table)} open-ended prompts had negative hazard coefficients.
14. **Fixed-topic generalization:** {result['fixed_topic_direction']} (beta={fixed_survival.get('coefficient', float('nan')):.6g}).
15. **Random-subspace specificity:** {result['random_specificity_passed']} (p={random_result.get('random_p', float('nan')):.6g}; n={random_result.get('random_directions', 0)}).
16. **Direct EOS benchmark:** {eos_control_note}
17. **Degeneration:** {result['nondegeneration_passed']}; {quality_note}.
18. **Single-pulse persistence:** beta={pulse_survival.get('coefficient', float('nan')):.6g}; continuous steering remains primary.

## Conclusion

19. **Strongest justified claim:** {claim}

All outcomes were analyzed under the frozen analysis plan, including negative outcomes. A native Qwen3.5 cache/no-cache discrepancy remains a failed strict protocol gate even when fresh recurrent cache replay passes; it is not reclassified as equivalence. No OOD result was used to redefine the subspace, direction, layer, rank, dose, prompt set, decoding policy, or analysis.
"""
    (root / "report.md").write_text(text, encoding="utf-8")


def aggregate_ood_run(config: dict, *, output: str | Path | None = None):
    """Aggregate frozen shards, run survival analyses, gates, figures, and report."""

    root = _root(config, output)
    manifest = verify_frozen_protocol(root)
    analysis_plan = json.loads(
        (root / "frozen/analysis_plan.json").read_text(encoding="utf-8")
    )
    jobs = json.loads(
        (root / "frozen/evaluation_jobs.json").read_text(encoding="utf-8")
    )
    audits = []
    for job in jobs:
        audit_path = root / "shards" / f"job_{int(job['job_index']):04d}" / "audit.json"
        if not audit_path.exists():
            raise RuntimeError(f"OOD evaluation is incomplete: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            job["mode"] == "generation"
            and audit.get("generation_engine_version") != GENERATION_ENGINE_VERSION
        ):
            raise RuntimeError(
                "OOD shard was produced by an obsolete generation engine: "
                f"{audit_path}"
            )
        if (
            not audit["frozen_before_ood"]
            or audit["application_output_cap"] is not None
            or not audit["kv_cache_used"]
            or not audit["final_token_only"]
            or audit["full_activations_saved"]
        ):
            raise RuntimeError("an OOD shard violated the frozen generation protocol")
        audits.append(audit)
    summaries = _load_job_records(
        root, jobs, "generation_summary.parquet", modes={"generation"}
    )
    token_events = _load_job_records(
        root, jobs, "token_events.parquet", modes={"generation"}
    )
    immediate = _load_job_records(
        root, jobs, "immediate_controls.parquet", modes={"immediate_control"}
    )
    if summaries.empty or token_events.empty or immediate.empty:
        raise RuntimeError("OOD shards do not contain all required outcome families")
    write_records(
        summaries.to_dict("records"), root / "generations/generation_summary.parquet"
    )
    write_records(
        token_events.to_dict("records"), root / "generations/token_events.parquet"
    )

    primary = summaries[
        (summaries.control_role == "frozen_E")
        & (summaries.regime == "continuous")
        & (summaries.prompt_family == "open_ended")
    ].copy()
    pulse = summaries[
        (summaries.control_role == "frozen_E")
        & (summaries.regime == "pulse")
        & (summaries.prompt_family == "open_ended")
    ].copy()
    fixed = summaries[
        (summaries.control_role == "frozen_E")
        & (summaries.regime == "continuous")
        & (summaries.prompt_family == "fixed_topic")
    ].copy()
    random_behavior = summaries[summaries.control_role == "random_subspace"].copy()
    eos_control = summaries[summaries.control_role == "direct_eos"].copy()
    (root / "controls").mkdir(parents=True, exist_ok=True)
    write_records(pulse.to_dict("records"), root / "pulse/generation_summary.parquet")
    immediate.to_csv(root / "controls/random_subspaces.csv", index=False)
    eos_control.to_csv(root / "controls/eos_control.csv", index=False)
    random_behavior.to_csv(
        root / "controls/random_behavioral_generations.csv", index=False
    )
    numerical_path = root / "shards/job_0000/numerical_checks.json"
    numerical = json.loads(numerical_path.read_text(encoding="utf-8"))
    alpha_zero_baseline = primary[primary.alpha == 0][
        [
            "prompt_id",
            "seed",
            "generated_tokens",
            "generated_text",
            "termination_reason",
            "eos_emitted",
        ]
    ].rename(
        columns={
            "generated_tokens": "baseline_generated_tokens",
            "generated_text": "baseline_generated_text",
            "termination_reason": "baseline_termination_reason",
            "eos_emitted": "baseline_eos_emitted",
        }
    )
    alpha_zero_controls = summaries[
        (summaries.alpha == 0)
        & (summaries.prompt_family == "open_ended")
        & ~((summaries.control_role == "frozen_E") & (summaries.regime == "continuous"))
    ]
    alpha_zero_merged = alpha_zero_controls.merge(
        alpha_zero_baseline,
        on=["prompt_id", "seed"],
        validate="many_to_one",
    )
    infrastructure_mask = alpha_zero_merged.termination_reason.astype(
        str
    ).str.startswith(
        "infrastructure_"
    ) | alpha_zero_merged.baseline_termination_reason.astype(
        str
    ).str.startswith(
        "infrastructure_"
    )
    excluded_zero_comparisons = int(infrastructure_mask.sum())
    alpha_zero_merged = alpha_zero_merged[~infrastructure_mask]
    generation_zero_passed = bool(
        len(alpha_zero_merged)
        and (
            alpha_zero_merged.generated_tokens
            == alpha_zero_merged.baseline_generated_tokens
        ).all()
        and (
            alpha_zero_merged.generated_text
            == alpha_zero_merged.baseline_generated_text
        ).all()
        and (
            alpha_zero_merged.termination_reason
            == alpha_zero_merged.baseline_termination_reason
        ).all()
        and (
            alpha_zero_merged.eos_emitted.astype(bool)
            == alpha_zero_merged.baseline_eos_emitted.astype(bool)
        ).all()
    )
    numerical["matched_generation_comparisons"] = int(len(alpha_zero_merged))
    numerical["infrastructure_censored_comparisons_excluded"] = (
        excluded_zero_comparisons
    )
    numerical["matched_generation_alpha_zero_passed"] = generation_zero_passed
    pd.DataFrame([numerical]).to_csv(
        root / "controls/alpha_zero_checks.csv", index=False
    )

    confidence = float(analysis_plan.get("analysis", {}).get("confidence", 0.95))
    survival_model = cox_alpha_model(primary, confidence=confidence)
    survival_curve = kaplan_meier(primary)
    dose_table, dose_summary = dose_response(primary)
    prompt_table = prompt_effects(primary, confidence=confidence)
    fixed_model = _safe_survival(fixed)
    pulse_model = _safe_survival(pulse)
    random_result = immediate_random_specificity(immediate)
    random_slopes = random_result.pop("slopes")
    random_behavior_models = []
    for direction_id, part in random_behavior.groupby("direction_id"):
        random_behavior_models.append(
            {"direction_id": direction_id, **_safe_survival(part)}
        )
    pd.DataFrame(random_behavior_models).to_csv(
        root / "controls/random_behavioral_survival.csv", index=False
    )
    json_write(root / "controls/eos_control_survival.json", _safe_survival(eos_control))
    quality = quality_summary(summaries)
    censoring = (
        primary.groupby("termination_reason", as_index=False)
        .size()
        .rename(columns={"size": "runs"})
    )
    token_primary = token_events[
        (token_events.control_role == "frozen_E")
        & (token_events.regime == "continuous")
        & token_events.prompt_id.isin(primary.prompt_id.unique())
    ].copy()
    token_primary["token_bin"] = (
        (token_primary.token_index.astype(int) - 1) // 50
    ).astype(int)

    analysis_root = root / "analysis"
    json_write(analysis_root / "survival_model.json", survival_model)
    dose_table.to_csv(analysis_root / "dose_response.csv", index=False)
    prompt_table.to_csv(analysis_root / "prompt_effects.csv", index=False)
    censoring.to_csv(analysis_root / "censoring_summary.csv", index=False)
    quality.to_csv(analysis_root / "quality_metrics.csv", index=False)
    if "coarse_topic_terms" in primary:
        topic_summary = (
            primary.groupby("alpha", as_index=False)
            .agg(
                runs=("run_id", "size"),
                unique_topic_signatures=("coarse_topic_terms", "nunique"),
            )
            .sort_values("alpha")
        )
    else:
        topic_summary = pd.DataFrame(
            columns=["alpha", "runs", "unique_topic_signatures"]
        )
    topic_summary.to_csv(analysis_root / "topic_summary.csv", index=False)
    survival_curve.to_csv(analysis_root / "survival_curves.csv", index=False)
    json_write(analysis_root / "fixed_topic_survival_model.json", fixed_model)
    json_write(root / "pulse/survival_model.json", pulse_model)
    random_slopes.to_csv(root / "controls/random_subspace_slopes.csv", index=False)

    prompt_negative = prompt_table.coefficient.dropna() < 0
    prompt_fraction = float(prompt_negative.mean()) if len(prompt_negative) else 0.0
    leave_one_out = []
    for prompt_id in primary.prompt_id.unique():
        leave_one_out.append(
            _safe_survival(primary[primary.prompt_id != prompt_id]).get(
                "coefficient", np.nan
            )
        )
    prompt_passed = bool(
        prompt_fraction > 0.5
        and len(leave_one_out)
        and np.all(np.asarray(leave_one_out, dtype=float) < 0)
    )
    q = analysis_plan.get("gates", {})
    baseline_quality = (
        primary[primary.alpha == 0].severe_degeneration.astype(bool).mean()
    )
    positive_quality = (
        primary[primary.alpha == max(primary.alpha)]
        .severe_degeneration.astype(bool)
        .mean()
    )
    nondegeneration = bool(
        positive_quality <= float(q.get("maximum_positive_dose_degeneration", 0.25))
        and positive_quality - baseline_quality
        <= float(q.get("maximum_degeneration_increase", 0.10))
    )
    direction_passed = bool(
        survival_model["coefficient"] < 0 and survival_model["ci_upper"] < 0
    )
    dose_passed = bool(dose_summary["monotonic"])
    fixed_direction = bool(fixed_model.get("coefficient", np.nan) < 0)
    random_result["passed"] = bool(
        random_result["true_eos_logit_slope"] < 0
        and random_result["random_p"] < float(q.get("random_p_max", 0.05))
    )
    cache_state_reuse_equivalent = numerical.get(
        "cache_state_reuse_equivalent", numerical["cache_equivalent"]
    )
    frozen_passed = bool(
        manifest["orientation"]["ood_data_used"] is False
        and all(audit["das_sha256"] == manifest["sha256"] for audit in audits)
        and len({audit["context_limit"] for audit in audits}) == 1
        and len({tuple(audit["eos_token_ids"]) for audit in audits}) == 1
        and numerical["alpha_zero_passed"]
        and generation_zero_passed
        and cache_state_reuse_equivalent
        and numerical["cache_equivalent"]
    )
    passed = bool(
        direction_passed
        and dose_passed
        and prompt_passed
        and random_result["passed"]
        and nondegeneration
        and frozen_passed
    )
    gates = {
        "frozen_protocol": {
            "passed": frozen_passed,
            "das_hash_verified": True,
            "orientation_pre_ood": True,
            "layer_rank_dose_tuned_on_ood": False,
            "application_output_cap_removed": True,
            "alpha_zero_equivalent": numerical["alpha_zero_passed"],
            "matched_generation_alpha_zero": generation_zero_passed,
            "cache_equivalent": numerical["cache_equivalent"],
            "cache_state_reuse_equivalent": cache_state_reuse_equivalent,
            "native_no_cache_limitation": numerical.get(
                "native_no_cache_limitation", False
            ),
            "cache_validation_disposition": numerical.get(
                "cache_validation_disposition", "legacy_cache_validation"
            ),
        },
        "ood_generalization": {
            "passed": passed,
            "direction_passed": direction_passed,
            "dose_response_passed": dose_passed,
            "prompt_generalization_passed": prompt_passed,
            "prompt_direction_fraction": prompt_fraction,
            "random_specificity_passed": random_result["passed"],
            "nondegeneration_passed": nondegeneration,
            "fixed_topic_direction": fixed_direction,
            "outcome": (
                "strong_ood_generalization"
                if passed
                else "ood_generalization_not_established"
            ),
        },
    }
    json_write(root / "gates.json", gates)
    figures = make_figures(
        root,
        primary,
        token_primary,
        survival_curve,
        dose_table,
        prompt_table,
        random_slopes,
        quality,
        pulse,
    )
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "context_limit": int(audits[0]["context_limit"]),
            "eos_token_ids": audits[0]["eos_token_ids"],
            "primary_runs": int(len(primary)),
            "orientation_rows": int(manifest["orientation"]["orientation_rows"]),
            "ood_generalization_passed": passed,
            "figure_count": len(figures),
            "full_activations_saved": False,
            "generation_engine_version": GENERATION_ENGINE_VERSION,
            "cache_state_reuse_equivalent": cache_state_reuse_equivalent,
            "native_no_cache_equivalent": numerical["cache_equivalent"],
            "native_no_cache_limitation": numerical.get(
                "native_no_cache_limitation", False
            ),
        }
    )
    json_write(root / "run_metadata.json", metadata)
    _report(
        root,
        gates=gates,
        metadata=metadata,
        survival=survival_model,
        dose=dose_summary,
        censoring=censoring,
        prompt_table=prompt_table,
        fixed_survival=fixed_model,
        random_result=random_result,
        eos_control=eos_control,
        primary=primary,
        quality=quality,
        pulse_survival=pulse_model,
    )
    return {
        "passed": passed,
        "outcome": gates["ood_generalization"]["outcome"],
        "cox_alpha": survival_model["coefficient"],
        "random_p": random_result["random_p"],
        "primary_runs": len(primary),
        "figures": len(figures),
        "report": root / "report.md",
    }

"""Aggregate the frozen Sugawara--Katahira Gate-B benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .external_benchmark import validate_factual_dataset


MODEL_PARAMETERS = {
    "standard_rl": ("alpha", "beta"),
    "asymmetry": ("alpha_pos", "alpha_neg", "beta"),
    "perseverance_impulsive": ("alpha", "beta", "phi"),
    "perseverance_gradual": ("alpha", "beta", "tau", "phi"),
    "hybrid_impulsive": ("alpha_pos", "alpha_neg", "beta", "phi"),
    "hybrid_gradual": ("alpha_pos", "alpha_neg", "beta", "tau", "phi"),
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_flags(model: str) -> tuple[bool, bool, bool]:
    return (
        model.startswith("asymmetry") or model.startswith("hybrid"),
        model.startswith("perseverance") or model.startswith("hybrid"),
        model.endswith("gradual"),
    )


def negative_log_posterior(
    parameters: torch.Tensor,
    trials: pd.DataFrame,
    model: str,
) -> torch.Tensor:
    """Published factual likelihood and normalized priors in float64."""

    names = MODEL_PARAMETERS[model]
    values = dict(zip(names, parameters.unbind()))
    asymmetric, perseverance, gradual = _model_flags(model)
    q_values = [parameters.new_zeros(()) for _ in range(8)]
    traces = [parameters.new_zeros(()) for _ in range(8)]
    log_likelihood = parameters.new_zeros(())
    choices = trials.choice.to_numpy(dtype=int)
    rewards = trials.reward.to_numpy(dtype=float)
    for index, (choice, reward) in enumerate(zip(choices, rewards)):
        if index == 96:
            q_values = [parameters.new_zeros(()) for _ in range(8)]
            traces = [parameters.new_zeros(()) for _ in range(8)]
        if choice == 0:
            continue
        chosen = choice - 1
        unchosen = chosen + 1 if choice % 2 == 1 else chosen - 1
        linear = values["beta"] * (q_values[chosen] - q_values[unchosen])
        if perseverance:
            linear = linear + values["phi"] * (traces[chosen] - traces[unchosen])
        log_likelihood = log_likelihood + torch.nn.functional.logsigmoid(linear)
        error = parameters.new_tensor(reward) - q_values[chosen]
        if asymmetric:
            alpha = torch.where(error >= 0, values["alpha_pos"], values["alpha_neg"])
        else:
            alpha = values["alpha"]
        q_values[chosen] = q_values[chosen] + alpha * error
        if perseverance:
            tau = values["tau"] if gradual else parameters.new_tensor(1.0)
            traces[chosen] = traces[chosen] + tau * (1 - traces[chosen])
            traces[unchosen] = traces[unchosen] + tau * (0 - traces[unchosen])

    log_prior = parameters.new_zeros(())
    for name, value in values.items():
        if name.startswith("alpha"):
            a = parameters.new_tensor(1.1)
            b = parameters.new_tensor(1.1)
            log_prior = log_prior + (a - 1) * torch.log(value)
            log_prior = log_prior + (b - 1) * torch.log1p(-value)
            log_prior = log_prior - (
                torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
            )
        elif name == "beta":
            shape = parameters.new_tensor(1.2)
            scale = parameters.new_tensor(5.0)
            log_prior = log_prior + (shape - 1) * torch.log(value)
            log_prior = log_prior - value / scale - torch.lgamma(shape)
            log_prior = log_prior - shape * torch.log(scale)
        elif name == "tau":
            # Beta(1, 1) is normalized and contributes exactly zero.
            log_prior = log_prior + value * 0
        elif name == "phi":
            variance = parameters.new_tensor(5.0)
            log_prior = log_prior - 0.5 * (
                torch.log(2 * torch.pi * variance) + value.square() / variance
            )
    return -(log_likelihood + log_prior)


def _autodiff_laplace(row: pd.Series, trials: pd.DataFrame) -> dict[str, Any]:
    model = str(row.model)
    names = MODEL_PARAMETERS[model]
    parsed = json.loads(row.parameters_json)
    values = torch.tensor(
        [float(parsed[name]) for name in names], dtype=torch.float64, requires_grad=True
    )
    objective = negative_log_posterior(values, trials, model)
    hessian = torch.autograd.functional.hessian(
        lambda vector: negative_log_posterior(vector, trials, model),
        values,
        vectorize=True,
    )
    matrix = hessian.detach().cpu().numpy()
    matrix = (matrix + matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(matrix)
    objective_value = float(objective.detach())
    agreement_error = abs(objective_value + float(row.log_posterior))
    if not np.isfinite(matrix).all():
        return {
            "autodiff_status": "nonfinite_hessian",
            "autodiff_objective": objective_value,
            "reference_abs_error": agreement_error,
            "hessian_min_eigenvalue": np.nan,
            "laplace_log_marginal_likelihood": np.nan,
        }
    if float(eigenvalues.min()) <= 0:
        return {
            "autodiff_status": "hessian_not_positive_definite",
            "autodiff_objective": objective_value,
            "reference_abs_error": agreement_error,
            "hessian_min_eigenvalue": float(eigenvalues.min()),
            "laplace_log_marginal_likelihood": np.nan,
        }
    dimension = len(names)
    lml = -objective_value + dimension / 2 * np.log(2 * np.pi)
    lml -= 0.5 * np.log(eigenvalues).sum()
    return {
        "autodiff_status": "ok",
        "autodiff_objective": objective_value,
        "reference_abs_error": agreement_error,
        "hessian_min_eigenvalue": float(eigenvalues.min()),
        "laplace_log_marginal_likelihood": float(lml),
    }


def _bootstrap(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _execution_provenance(
    *,
    root: Path,
    config_path: Path,
    data_path: Path,
    amendment_path: Path | None = None,
) -> dict[str, Any]:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    provenance = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_sha256": _hash(config_path),
        "data_sha256": _hash(data_path),
        "r_fit_script_sha256": _hash(
            root / "validation/external_benchmark/fit_external_benchmark.R"
        ),
        "python_analysis_sha256": _hash(Path(__file__)),
        "rsolnp_version": "1.16",
        "torch_version": torch.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if amendment_path is not None:
        provenance["nonblocking_amendment_sha256"] = _hash(amendment_path)
    return provenance


def aggregate_benchmark(
    *, root: str | Path, fit_work: str | Path, output: str | Path
) -> dict[str, Any]:
    root, fit_work, output = Path(root).resolve(), Path(fit_work), Path(output)
    config_path = root / "configs/validation/external_benchmark_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["status"] != "owner_frozen":
        raise RuntimeError("external benchmark is not owner-frozen")
    if config["implementation"]["laplace"]["hessian"] != (
        "torch_float64_exact_autodiff_negative_log_posterior"
    ):
        raise RuntimeError("external benchmark Hessian contract changed")
    amendment_path = (
        root
        / "configs/validation/external_benchmark_nonblocking_amendment.yaml"
    )
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    if amendment != {
        "schema_version": "external-benchmark-nonblocking-amendment-v1",
        "status": "owner_frozen",
        "implementation_equivalence": "pass",
        "benchmark_claim_status": "unresolved",
        "benchmark_claim_permission": "stop",
        "pipeline_permission": "continue",
        "v2_boundary_method_status": "not_selected",
    }:
        raise RuntimeError("external benchmark nonblocking amendment changed")
    data_path = (
        root
        / "validation/external_benchmark/raw/figshare_10042319_v3/factual.csv"
    )
    data_contract = validate_factual_dataset(data_path)
    data = pd.read_csv(data_path)
    paths = sorted(fit_work.glob("jobs/participant_*.csv"))
    if len(paths) != 143:
        raise RuntimeError(f"expected 143 participant fit files, observed {len(paths)}")
    fits = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected_rows = 143 * len(config["models"])
    if len(fits) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} model fits, observed {len(fits)}")
    if set(fits.model) != set(config["models"]):
        raise RuntimeError("fit model bank differs from the frozen model bank")
    if not (fits.convergence == 0).all():
        raise RuntimeError("one or more Rsolnp MAP optimizations did not converge")

    diagnostics = []
    for row in fits.itertuples(index=False):
        trials = data[data.subjectid == int(row.subject_id)]
        diagnostics.append(_autodiff_laplace(pd.Series(row._asdict()), trials))
    fits = fits.rename(
        columns={
            "status": "initial_fd_status",
            "laplace_log_marginal_likelihood": "initial_fd_lml",
            "hessian_min_eigenvalue": "initial_fd_hessian_min_eigenvalue",
            "hessian_error": "initial_fd_hessian_error",
        }
    )
    diagnosed = pd.concat(
        [fits.reset_index(drop=True), pd.DataFrame(diagnostics)], axis=1
    )
    output.mkdir(parents=True, exist_ok=True)
    diagnosed.to_csv(
        output / "fit_diagnostics.csv", index=False, float_format="%.17g"
    )
    if diagnosed.reference_abs_error.max() > 1e-7:
        raise RuntimeError(
            "independent R/Python negative-log-posterior implementations disagree"
        )
    if not (diagnosed.autodiff_status == "ok").all():
        failures = diagnosed[diagnosed.autodiff_status != "ok"]
        provenance = _execution_provenance(
            root=root,
            config_path=config_path,
            data_path=data_path,
            amendment_path=amendment_path,
        )
        failure_records = []
        for row in failures.itertuples(index=False):
            parameters = json.loads(row.parameters_json)
            failure_records.append(
                {
                    "subject_id": int(row.subject_id),
                    "model": str(row.model),
                    "status": str(row.autodiff_status),
                    "hessian_min_eigenvalue": float(row.hessian_min_eigenvalue),
                    "tau": float(parameters["tau"]),
                }
            )
        summary = {
            "schema_version": "external-benchmark-execution-status-v1",
            "analysis_id": config["analysis_id"],
            "execution_status": "owner_review",
            "gate_b_status": "not_evaluable",
            "implementation_equivalence": amendment["implementation_equivalence"],
            "benchmark_claim_status": amendment["benchmark_claim_status"],
            "benchmark_claim_permission": amendment["benchmark_claim_permission"],
            "pipeline_permission": amendment["pipeline_permission"],
            "evidence_eligibility": "development_only",
            "reason": "nonregular_active_boundary_laplace_hessians",
            "primary_contrasts_computed": False,
            "bootstrap_computed": False,
            "model_ranking_computed": False,
            "results_csv_emitted": False,
            "participants": 143,
            "model_fits": int(len(diagnosed)),
            "converged_map_fits": int((diagnosed.convergence == 0).sum()),
            "positive_definite_exact_hessians": int(
                (diagnosed.autodiff_status == "ok").sum()
            ),
            "failed_exact_hessians": int(len(failures)),
            "initial_finite_difference_hessian_failures": int(
                (diagnosed.initial_fd_status != "ok").sum()
            ),
            "independent_reference_max_abs_log_posterior_error": float(
                diagnosed.reference_abs_error.max()
            ),
            "failure_records": failure_records,
            "frozen_constraints": {
                "all_143_participants_required": True,
                "positive_definite_hessian_required": True,
                "eigenvalue_floor": None,
                "participant_exclusion_allowed": False,
            },
            "owner_decision_required": (
                "Only if a benchmark claim is pursued: obtain the authors' exact "
                "marginal-likelihood implementation, or preregister a separate v2 "
                "boundary-aware rule that applies to all boundary solutions before "
                "computing any Gate-B contrast or ranking."
            ),
            "provenance": provenance,
        }
        (output / "benchmark_execution_status.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report_lines = [
            "# Gate-B external benchmark execution report",
            "",
            "Gate B: **NOT EVALUABLE — OWNER REVIEW REQUIRED**  ",
            "Benchmark-claim permission: `stop`  ",
            "Downstream pipeline permission: `continue`  ",
            "Implementation equivalence: `pass`  ",
            "Evidence eligibility: `development_only`",
            "",
            "## What completed",
            "",
            "- The frozen Figshare-v3 factual dataset passed its contract: 143 "
            "participants, two 96-trial sessions, and no added filtering.",
            "- All 858 participant/model Rsolnp MAP optimizations converged.",
            f"- {int((diagnosed.autodiff_status == 'ok').sum())} of 858 exact "
            "float64 Hessians were positive definite.",
            "- The independent R and Python log-posteriors agreed to a maximum "
            f"absolute error of {diagnosed.reference_abs_error.max():.3e}.",
            "",
            "## Why classification stopped",
            "",
            f"{len(failures)} gradual-model MAP estimates lie at the active upper "
            "boundary for tau and have a non-positive-definite unconstrained "
            "Hessian. The frozen benchmark requires a positive-definite Hessian "
            "and forbids eigenvalue flooring. A boundary-aware Laplace rule would "
            "therefore be a new analysis convention, not a mechanical repair.",
            "",
            "| Participant | Model | tau | Minimum Hessian eigenvalue |",
            "|---:|---|---:|---:|",
        ]
        for record in failure_records:
            report_lines.append(
                f"| {record['subject_id']} | {record['model']} | "
                f"{record['tau']:.12f} | "
                f"{record['hessian_min_eigenvalue']:.9g} |"
            )
        report_lines.extend(
            [
                "",
                "## Results deliberately not computed",
                "",
                "No primary contrast, bootstrap interval, mean-LML ranking, "
                "Gate-B pass/partial/fail label, or `results.csv` was produced. "
                "Computing any of these from only 137 participants, a jittered "
                "Hessian, an absolute determinant, or a dropped boundary "
                "parameter would violate the owner-frozen specification.",
                "",
                "## Required owner decision",
                "",
                "No decision is required to continue the Qwen/transfer pipeline. "
                "If the benchmark itself will support a scientific claim, first "
                "recover the authors' actual marginal-likelihood implementation. "
                "If that is unavailable, preregister a distinct v2 with one "
                "genuinely boundary-aware integration or independently validated "
                "numerical marginal-likelihood rule applying to every boundary "
                "solution before examining aggregate contrasts or rankings.",
                "",
                "Hessian jitter, absolute determinants, participant exclusion, "
                "and simply dropping a boundary parameter are prohibited.",
                "",
                "The complete participant/model diagnostics are in "
                "`fit_diagnostics.csv`. The earlier finite-difference issue and "
                "its pre-contrast correction are recorded in "
                "`NUMERICAL_IMPLEMENTATION_ERRATUM.md`.",
                "",
            ]
        )
        (output / "REPLICATION_REPORT.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )
        return summary

    pivot = diagnosed.pivot(
        index="subject_id",
        columns="model",
        values="laplace_log_marginal_likelihood",
    ).sort_index()
    rows: list[dict[str, Any]] = []
    contrast_results = []
    samples = int(config["bootstrap"]["samples"])
    seed = int(config["bootstrap"]["seed"])
    for index, contrast in enumerate(config["primary_contrasts"]):
        left, right = contrast["left"], contrast["right"]
        values = (pivot[left] - pivot[right]).to_numpy(dtype=float)
        low, high = _bootstrap(values, samples=samples, seed=seed + index)
        result = {
            "result_type": "primary_contrast",
            "name": f"{left}_minus_{right}",
            "model": left,
            "comparison_model": right,
            "estimate": float(values.mean()),
            "ci_low": low,
            "ci_high": high,
            "n": len(values),
        }
        rows.append(result)
        contrast_results.append(result)

    ranking = (
        diagnosed.groupby("model", as_index=False)
        .laplace_log_marginal_likelihood.mean()
        .sort_values("laplace_log_marginal_likelihood", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    for row in ranking.itertuples(index=False):
        rows.append(
            {
                "result_type": "model_ranking",
                "name": "mean_lml",
                "model": row.model,
                "comparison_model": "",
                "estimate": float(row.laplace_log_marginal_likelihood),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": 143,
                "rank": int(row.rank),
            }
        )

    parameters = diagnosed.parameters_json.map(json.loads)
    asymmetry_bias = np.array(
        [
            value["alpha_pos"] - value["alpha_neg"]
            for value, model in zip(parameters, diagnosed.model)
            if model == "asymmetry"
        ]
    )
    hybrid_bias = np.array(
        [
            value["alpha_pos"] - value["alpha_neg"]
            for value, model in zip(parameters, diagnosed.model)
            if model == "hybrid_gradual"
        ]
    )
    rows.append(
        {
            "result_type": "secondary_nonblocking",
            "name": "mean_bias_reduction_asymmetry_minus_hybrid_gradual",
            "model": "hybrid_gradual",
            "comparison_model": "asymmetry",
            "estimate": float(asymmetry_bias.mean() - hybrid_bias.mean()),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": 143,
        }
    )
    parameter_summary = {}
    for model in ("perseverance_gradual", "hybrid_gradual"):
        subset = diagnosed[diagnosed.model == model]
        parsed = subset.parameters_json.map(json.loads)
        parameter_summary[model] = {
            name: {
                "mean": float(np.mean([value[name] for value in parsed])),
                "sd": float(np.std([value[name] for value in parsed], ddof=1)),
            }
            for name in ("tau", "phi")
        }

    means_positive = all(row["estimate"] > 0 for row in contrast_results)
    intervals_positive = all(row["ci_low"] > 0 for row in contrast_results)
    perseverance_rank = int(
        ranking.loc[ranking.model == "perseverance_gradual", "rank"].iloc[0]
    )
    higher_models = ranking.loc[
        ranking["rank"] < perseverance_rank, "model"
    ].tolist()
    ranking_passed = perseverance_rank <= 2 and set(higher_models).issubset(
        {"hybrid_gradual"}
    )
    if means_positive and intervals_positive and ranking_passed:
        gate_status = "pass"
    elif means_positive:
        gate_status = "partial"
    else:
        gate_status = "fail"

    detailed_path = output / "participant_model_fits.csv"
    diagnosed.to_csv(detailed_path, index=False, float_format="%.17g")
    result_path = output / "results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False, float_format="%.17g")
    provenance = _execution_provenance(
        root=root, config_path=config_path, data_path=data_path
    )
    git_dirty = provenance["git_dirty"]
    summary = {
        "schema_version": "external-benchmark-result-v1",
        "analysis_id": config["analysis_id"],
        "gate_b_status": gate_status,
        "evidence_eligibility": "development_only" if git_dirty else "claim_bearing",
        "primary_contrasts": contrast_results,
        "model_ranking": ranking.to_dict("records"),
        "perseverance_gradual_rank": perseverance_rank,
        "models_above_perseverance_gradual": higher_models,
        "criteria": {
            "both_mean_contrasts_positive": means_positive,
            "both_interval_lower_bounds_positive": intervals_positive,
            "ranking_rule_passed": ranking_passed,
        },
        "secondary": {
            "mean_asymmetry_bias": float(asymmetry_bias.mean()),
            "mean_hybrid_gradual_bias": float(hybrid_bias.mean()),
            "mean_bias_reduction": float(asymmetry_bias.mean() - hybrid_bias.mean()),
            "parameter_summary": parameter_summary,
        },
        "independent_reference_max_abs_log_posterior_error": float(
            diagnosed.reference_abs_error.max()
        ),
        "failed_map_fits": 0,
        "failed_exact_hessians": 0,
        "participants": 143,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "data_contract": data_contract,
        "provenance": provenance,
    }
    summary_path = output / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Gate-B external benchmark report",
        "",
        f"Gate B: **{gate_status.upper()}**  ",
        f"Evidence eligibility: `{summary['evidence_eligibility']}`  ",
        "Dataset: Sugawara & Katahira factual web experiment, Figshare v3  ",
        "Participants: 143; models: 6; endpoint: participant Laplace LML",
        "",
        "## Frozen primary contrasts",
        "",
        "| Contrast | Mean | 95% paired-bootstrap interval |",
        "|---|---:|---:|",
    ]
    for row in contrast_results:
        report_lines.append(
            f"| {row['name']} | {row['estimate']:.6f} | "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] |"
        )
    report_lines.extend(
        [
            "",
            "## Mean-LML ranking",
            "",
            "| Rank | Model | Mean LML |",
            "|---:|---|---:|",
        ]
    )
    for row in ranking.itertuples(index=False):
        report_lines.append(
            f"| {int(row.rank)} | {row.model} | "
            f"{row.laplace_log_marginal_likelihood:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Frozen decision criteria",
            "",
            f"- Both means positive: `{means_positive}`",
            f"- Both interval lower bounds positive: `{intervals_positive}`",
            f"- Perseverance-gradual rank/allowed-higher rule: `{ranking_passed}`",
            "",
            "## Secondary, nonblocking",
            "",
            f"- Mean Asymmetry alpha+ - alpha-: {asymmetry_bias.mean():.6f}",
            f"- Mean Hybrid-gradual alpha+ - alpha-: {hybrid_bias.mean():.6f}",
            f"- Mean bias reduction: {asymmetry_bias.mean() - hybrid_bias.mean():.6f}",
            "",
            "## Numerical validity",
            "",
            "- All 858 Rsolnp MAP optimizations converged.",
            "- All exact float64 Hessians were positive definite without jitter.",
            "- Maximum R/Python log-posterior discrepancy: "
            f"{diagnosed.reference_abs_error.max():.3e}.",
            "- The pre-contrast finite-difference failure and correction are recorded "
            "in `NUMERICAL_IMPLEMENTATION_ERRATUM.md`.",
            "",
            "## Interpretation constraint",
            "",
            "This benchmark validates the isolated literature implementation only. "
            "It does not alter the project's cognitive models or resolve downstream "
            "behavioral-theory ambiguity.",
            "",
        ]
    )
    (output / "REPLICATION_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate the frozen Gate-B benchmark")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--fit-work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate_benchmark(
        root=args.root, fit_work=args.fit_work, output=args.output
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    analyze_run,
    collect_and_evaluate_validation,
    collect_shard,
    finalize_collection,
    generate_design,
    load_config,
    run_residual_discovery,
    prepare_active_round,
    update_after_active_round,
    evaluate_round2_final,
    prepare_theory_resolution,
    evaluate_theory_resolution,
)
from .mechanistic.pipeline import (
    analyze_representations,
    finalize_mechanistic_run,
    prepare_mechanistic_run,
    run_causal_interventions,
    scan_directions,
    scan_projections,
)
from .action_history.pipeline import (
    analyze_action_history_representations,
    finalize_action_history_run,
    fit_action_history_directions,
    prepare_action_history_run,
    project_action_history_directions,
    run_action_history_steering,
    run_matched_projection_patching,
    run_random_direction_null,
)
from .causal_mechanistic.pipeline import (
    finalize_causal_mechanistic_run,
    localize_whole_state_layer,
    prepare_causal_mechanistic_run,
    prepare_diagnostic_information_scan,
    run_circuit_localization,
    select_das_alignments,
    summarize_whole_state_localization,
    train_das_search_job,
    validate_selected_alignments,
)
from .causal_specificity.pipeline import (
    aggregate_specificity_run,
    evaluate_specificity_candidate,
    evaluate_specificity_necessity,
    finalize_specificity_run,
    prepare_specificity_reanalysis,
)
from .causal_abstraction.pipeline import (
    aggregate_abstraction_run,
    evaluate_abstraction_job,
    prepare_abstraction_run,
)
from .ood_generation.pipeline import (
    aggregate_ood_run,
    evaluate_ood_job,
    prepare_ood_run,
)


DEFAULT_CONFIG = "configs/discovery_v1.yaml"


def generate_design_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compile the declarative persistence design"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output")
    parser.add_argument("--conditions", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    _, manifest = generate_design(
        config, output=args.output, conditions=args.conditions, seed=args.seed
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def run_experiments_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect or finalize behavior-only experiments"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output")
    parser.add_argument("--phase", choices=("collect", "finalize"), default="collect")
    parser.add_argument("--conditions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--model-free", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", help="frozen condition manifest to collect")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.phase == "collect":
        result = collect_shard(
            config,
            output=args.output,
            conditions=args.conditions,
            seed=args.seed,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            model_path=args.model,
            revision=args.revision,
            online=args.online,
            model_free=args.model_free,
            resume=args.resume,
            manifest=args.manifest,
        )
        print(result)
    else:
        frame, path, gates = finalize_collection(
            config,
            output=args.output,
            conditions=args.conditions,
            seed=args.seed,
            manifest=args.manifest,
        )
        print(
            json.dumps(
                {"observations": len(frame), "path": str(path), "gates": gates},
                indent=2,
            )
        )


def fit_models_main(argv=None):
    parser = argparse.ArgumentParser(description="Run the cognitive-discovery analyses")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    result = analyze_run(load_config(args.config), output=args.output, smoke=args.smoke)
    print(result["report"])


def validate_predictions_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect independent conditions and score a frozen model"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output")
    parser.add_argument("--conditions", type=int)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--model-free", action="store_true")
    args = parser.parse_args(argv)
    metrics = collect_and_evaluate_validation(
        load_config(args.config),
        output=args.output,
        model_path=args.model,
        revision=args.revision,
        online=args.online,
        model_free=args.model_free,
        conditions=args.conditions,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def discover_residuals_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Discover held-out residual interactions"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output")
    parser.add_argument("--base-model")
    args = parser.parse_args(argv)
    result = run_residual_discovery(
        load_config(args.config), output=args.output, base_model=args.base_model
    )
    print(result.to_string(index=False))


def active_discovery_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Discovery Round-2 active hierarchy workflow"
    )
    parser.add_argument("--config", default="configs/discovery_v2.yaml")
    parser.add_argument("--output")
    parser.add_argument("--round1-output", required=True)
    parser.add_argument("--active-output")
    parser.add_argument("--final-output")
    parser.add_argument(
        "--phase", choices=("prepare", "update", "evaluate"), required=True
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.phase == "prepare":
        result = prepare_active_round(
            config,
            round1_output=args.round1_output,
            output=args.output,
            smoke=args.smoke,
        )
        printable = {
            key: str(value) for key, value in result.items() if key != "hierarchy"
        }
    elif args.phase == "update":
        if not args.active_output:
            parser.error("--active-output is required for update")
        result = update_after_active_round(
            config,
            round1_output=args.round1_output,
            active_output=args.active_output,
            output=args.output,
            smoke=args.smoke,
        )
        printable = {"manifest": str(result["manifest"])}
    else:
        if not args.active_output or not args.final_output:
            parser.error("--active-output and --final-output are required for evaluate")
        result = evaluate_round2_final(
            config,
            round1_output=args.round1_output,
            active_output=args.active_output,
            final_output=args.final_output,
            output=args.output,
        )
        printable = result["metrics"]
    print(json.dumps(printable, indent=2, sort_keys=True))


def theory_resolution_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Discovery Round-3 behavioral theory-resolution workflow"
    )
    parser.add_argument("--config", default="configs/theory_resolution_v1.yaml")
    parser.add_argument("--output")
    parser.add_argument("--round1-output", required=True)
    parser.add_argument("--active-output")
    parser.add_argument("--round2-output")
    parser.add_argument("--round3-output")
    parser.add_argument("--phase", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.phase == "prepare":
        if not args.round2_output:
            parser.error("--round2-output is required for prepare")
        result = prepare_theory_resolution(
            config,
            round1_output=args.round1_output,
            active_output=args.active_output,
            round2_output=args.round2_output,
            output=args.output,
            smoke=args.smoke,
        )
        printable = {
            **result,
            "manifest": str(result["manifest"]),
        }
    else:
        if not args.round3_output:
            parser.error("--round3-output is required for evaluate")
        result = evaluate_theory_resolution(
            config,
            round1_output=args.round1_output,
            active_output=args.active_output,
            round3_output=args.round3_output,
            output=args.output,
            smoke=args.smoke,
        )
        printable = {
            "decision": result["decision"],
            "report": str(result["report"]),
        }
    print(json.dumps(printable, indent=2, sort_keys=True))


def mechanistic_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Mechanistic context-sensitive outcome-history workflow"
    )
    parser.add_argument("--config", default="configs/mechanistic_v1.yaml")
    parser.add_argument("--output")
    parser.add_argument("--theory-output", default="artifacts/theory_resolution_v1")
    parser.add_argument(
        "--phase",
        choices=("prepare", "scan", "project", "analyze", "intervene", "report"),
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.phase == "prepare":
        result = prepare_mechanistic_run(
            config,
            theory_output=args.theory_output,
            output=args.output,
            smoke=args.smoke,
        )
    elif args.phase == "scan":
        result = scan_directions(
            config,
            output=args.output,
            model_path=args.model,
            revision=args.revision,
            online=args.online,
            limit=args.limit,
        )
    elif args.phase == "project":
        result = scan_projections(
            config,
            output=args.output,
            model_path=args.model,
            revision=args.revision,
            online=args.online,
            limit=args.limit,
        )
    elif args.phase == "analyze":
        result = analyze_representations(config, output=args.output)
    elif args.phase == "intervene":
        result = run_causal_interventions(
            config,
            theory_output=args.theory_output,
            output=args.output,
            model_path=args.model,
            revision=args.revision,
            online=args.online,
            limit=args.limit,
        )
    else:
        result = finalize_mechanistic_run(config, output=args.output)
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def action_history_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Disambiguate action-history computation from persistence readout"
    )
    parser.add_argument(
        "--config", default="configs/action_history_disambiguation_v1.yaml"
    )
    parser.add_argument("--output")
    parser.add_argument("--source", default="artifacts/mechanistic_v1")
    parser.add_argument(
        "--phase",
        choices=(
            "prepare",
            "fit",
            "project",
            "analyze",
            "steer",
            "random",
            "patch",
            "report",
        ),
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    common = {"output": args.output}
    gpu = {
        **common,
        "model_path": args.model,
        "revision": args.revision,
        "online": args.online,
        "limit": args.limit,
    }
    if args.phase == "prepare":
        result = prepare_action_history_run(config, source=args.source, **common)
    elif args.phase == "fit":
        result = fit_action_history_directions(config, **gpu)
    elif args.phase == "project":
        result = project_action_history_directions(config, **gpu)
    elif args.phase == "analyze":
        result = analyze_action_history_representations(config, **common)
    elif args.phase == "steer":
        result = run_action_history_steering(config, **gpu)
    elif args.phase == "random":
        result = run_random_direction_null(
            config,
            **gpu,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif args.phase == "patch":
        result = run_matched_projection_patching(config, **gpu)
    else:
        result = finalize_action_history_run(config, **common)
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def causal_mechanistic_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Counterfactual-first causal mechanistic discovery"
    )
    parser.add_argument("--config", default="configs/causal_mech_v1.yaml")
    parser.add_argument("--output")
    parser.add_argument("--theory-output", default="artifacts/theory_resolution_v1")
    parser.add_argument("--mechanistic-output", default="artifacts/mechanistic_v1")
    parser.add_argument(
        "--phase",
        choices=(
            "prepare",
            "diagnostic",
            "localize",
            "summarize",
            "das",
            "select",
            "validate",
            "circuit",
            "report",
        ),
        required=True,
    )
    parser.add_argument("--layer", type=int)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    common = {"output": args.output}
    gpu = {
        **common,
        "model_path": args.model,
        "revision": args.revision,
        "online": args.online,
        "limit": args.limit,
    }
    if args.phase == "prepare":
        result = prepare_causal_mechanistic_run(
            config,
            theory_output=args.theory_output,
            mechanistic_output=args.mechanistic_output,
            **common,
        )
    elif args.phase == "diagnostic":
        result = prepare_diagnostic_information_scan(
            config, mechanistic_output=args.mechanistic_output, **common
        )
    elif args.phase == "localize":
        if args.layer is None:
            parser.error("--layer is required for localize")
        result = localize_whole_state_layer(config, layer=args.layer, **gpu)
    elif args.phase == "summarize":
        result = summarize_whole_state_localization(config, **common)
    elif args.phase == "das":
        if args.job_index is None:
            parser.error("--job-index is required for das")
        result = train_das_search_job(config, job_index=args.job_index, **gpu)
    elif args.phase == "select":
        result = select_das_alignments(config, **common)
    elif args.phase == "validate":
        result = validate_selected_alignments(config, **gpu)
    elif args.phase == "circuit":
        result = run_circuit_localization(config, **gpu)
    else:
        result = finalize_causal_mechanistic_run(config, **common)
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def causal_specificity_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Frozen-DAS causal specificity with stable global CFR"
    )
    parser.add_argument("--config", default="configs/causal_specificity_v2.yaml")
    parser.add_argument("--source", default="artifacts/causal_mech_v1")
    parser.add_argument("--output")
    parser.add_argument(
        "--phase",
        choices=("prepare", "evaluate", "aggregate", "necessity", "finalize"),
        required=True,
    )
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    common = {"source": args.source, "output": args.output}
    if args.phase == "prepare":
        result = prepare_specificity_reanalysis(config, **common)
    elif args.phase == "evaluate":
        if args.job_index is None:
            parser.error("--job-index is required for evaluate")
        result = evaluate_specificity_candidate(
            config,
            job_index=args.job_index,
            model_path=args.model,
            revision=args.revision,
            online=args.online,
            limit=args.limit,
            **common,
        )
    elif args.phase == "aggregate":
        result = aggregate_specificity_run(config, **common)
    elif args.phase == "necessity":
        if args.job_index is None:
            parser.error("--job-index is required for necessity")
        result = evaluate_specificity_necessity(
            config,
            job_index=args.job_index,
            model_path=args.model,
            revision=args.revision,
            online=args.online,
            limit=args.limit,
            **common,
        )
    else:
        result = finalize_specificity_run(config, **common)
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def causal_abstraction_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Identify the abstraction level of the frozen persistence controller"
    )
    parser.add_argument("--config", default="configs/causal_abstraction_v1.yaml")
    parser.add_argument("--output")
    parser.add_argument("--specificity-output")
    parser.add_argument("--theory-output")
    parser.add_argument(
        "--phase", choices=("prepare", "evaluate", "aggregate"), required=True
    )
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.phase == "prepare":
        result = prepare_abstraction_run(
            config,
            specificity_output=args.specificity_output,
            theory_output=args.theory_output,
            output=args.output,
        )
    elif args.phase == "evaluate":
        if args.job_index is None:
            parser.error("--job-index is required for evaluate")
        result = evaluate_abstraction_job(
            config,
            job_index=args.job_index,
            output=args.output,
            model_path=args.model,
            revision=args.revision,
            online=args.online,
            limit=args.limit,
        )
    else:
        result = aggregate_abstraction_run(config, output=args.output)
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def ood_generation_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Test the frozen persistence-evidence subspace on voluntary EOS"
    )
    parser.add_argument("--config", default="configs/ood_free_generation_v1.yaml")
    parser.add_argument("--output")
    parser.add_argument("--abstraction-output")
    parser.add_argument(
        "--phase", choices=("prepare", "evaluate", "aggregate"), required=True
    )
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.phase == "prepare":
        result = prepare_ood_run(
            config,
            abstraction_output=args.abstraction_output,
            output=args.output,
        )
    elif args.phase == "evaluate":
        if args.job_index is None:
            parser.error("--job-index is required for evaluate")
        result = evaluate_ood_job(
            config,
            job_index=args.job_index,
            output=args.output,
            model_path=args.model,
            revision=args.revision,
            online=args.online,
            limit=args.limit,
        )
    else:
        result = aggregate_ood_run(config, output=args.output)
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

from __future__ import annotations

import argparse
import json

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

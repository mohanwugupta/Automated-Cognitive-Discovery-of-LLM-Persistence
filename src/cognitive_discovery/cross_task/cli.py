"""Command line surface for cross-task discovery contracts and analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cognitive_discovery.data.storage import read_records

from .behavior import estimate_behavioral_effects, estimate_preregistered_interactions
from .acquisition import select_active_batch
from .causal import validate_causal_transfer_rows
from .collection import collect_shared_behavior, validate_model_interface
from .computational import evaluate_frozen_validation, fit_computational_map
from .config import load_cross_task_config
from .figures import generate_cross_model_figure, generate_model_figures
from .pipeline import (
    freeze_hypotheses, initialize_run, mark_model_stage, prepare_shared_design,
    validate_final_run,
)
from .provenance import write_provenance
from .reporting import write_cross_model_report, write_model_report
from .representation import persistence_effect_geometry, run_representational_transfer
from .splits import assert_untouched_validation
from .synthesis import cross_model_tables, synthesize_levels


def _parser():
    parser = argparse.ArgumentParser(description="Three-model cross-task persistence discovery")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-config")
    validate.add_argument("--config", type=Path, default=Path("configs/discovery/cross_task_v1.yaml"))
    initialize = sub.add_parser("initialize")
    initialize.add_argument("--config", type=Path, default=Path("configs/discovery/cross_task_v1.yaml"))
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--final", action="store_true")
    design = sub.add_parser("design")
    design.add_argument("--output", type=Path, required=True)
    design.add_argument("--smoke", action="store_true")
    for name in ("interface", "collect-behavior"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--model", choices=("qwen", "gemma", "llama"), required=True)
        command.add_argument("--model-path", type=Path, required=True)
        command.add_argument("--online", action="store_true")
        command.add_argument("--smoke", action="store_true")
    behavior = sub.add_parser("behavior-effects")
    behavior.add_argument("--output", type=Path, required=True)
    behavior.add_argument("--model", choices=("qwen", "gemma", "llama"), required=True)
    behavior.add_argument("--observations", type=Path, required=True)
    computational = sub.add_parser("computational")
    computational.add_argument("--output", type=Path, required=True)
    computational.add_argument("--model", choices=("qwen", "gemma", "llama"), required=True)
    computational.add_argument("--observations", type=Path, required=True)
    validation = sub.add_parser("final-validation")
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument("--model", choices=("qwen", "gemma", "llama"), required=True)
    validation.add_argument("--observations", type=Path, required=True)
    active = sub.add_parser("active-select")
    active.add_argument("--output", type=Path, required=True)
    active.add_argument("--model", choices=("qwen", "gemma", "llama"), required=True)
    active.add_argument("--candidates", type=Path, required=True)
    active.add_argument("--observed-groups", type=Path, required=True)
    active.add_argument("--validation-groups", type=Path, required=True)
    active.add_argument("--batch-size", type=int, required=True)
    active.add_argument("--round", type=int, required=True)
    representation = sub.add_parser("representation")
    representation.add_argument("--output", type=Path, required=True)
    representation.add_argument("--model", choices=("qwen", "gemma", "llama"), required=True)
    representation.add_argument("--features", type=Path, required=True)
    geometry = sub.add_parser("effect-geometry")
    geometry.add_argument("--output", type=Path, required=True)
    geometry.add_argument("--model", choices=("qwen", "gemma", "llama"), required=True)
    geometry.add_argument("--features", type=Path, required=True)
    geometry.add_argument("--rank", type=int, default=2)
    causal = sub.add_parser("validate-causal")
    causal.add_argument("--output", type=Path, required=True)
    causal.add_argument("--model", choices=("qwen", "gemma", "llama"), required=True)
    causal.add_argument("--results", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--manifest", type=Path, required=True)
    synthesize = sub.add_parser("synthesize")
    synthesize.add_argument("--output", type=Path, required=True)
    synthesize.add_argument("--model", choices=("qwen", "gemma", "llama"), required=True)
    cross = sub.add_parser("cross-model")
    cross.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--model", choices=("qwen", "gemma", "llama"))
    figures = sub.add_parser("figures")
    figures.add_argument("--output", type=Path, required=True)
    figures.add_argument("--model", choices=("qwen", "gemma", "llama"))
    final = sub.add_parser("validate-final")
    final.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve()
    if args.command == "validate-config":
        config_path = args.config if args.config.is_absolute() else repository / args.config
        result = {"valid": True, "analysis_id": load_cross_task_config(config_path)["analysis_id"]}
    elif args.command == "initialize":
        config_path = args.config if args.config.is_absolute() else repository / args.config
        result = initialize_run(repository, args.output, config_path, final=args.final)
    elif args.command == "design":
        result = prepare_shared_design(repository, args.output, smoke=args.smoke)
    elif args.command == "interface":
        result = validate_model_interface(
            repository, args.output, model_key=args.model, model_path=args.model_path,
            online=args.online, smoke=args.smoke,
        )
        mark_model_stage(
            args.output, args.model, "interface",
            "complete" if result.get("passed") else "blocked",
            validation="interface/validation.json",
        )
    elif args.command == "collect-behavior":
        result = collect_shared_behavior(
            repository, args.output, model_key=args.model, model_path=args.model_path,
            online=args.online, smoke=args.smoke,
        )
        mark_model_stage(args.output, args.model, "behavior", "complete", observations=result["path"])
    elif args.command == "behavior-effects":
        config = load_cross_task_config(args.output / "effective_config.yaml")
        frame = read_records(args.observations)
        frame["model"] = args.model
        compatibility = pd.read_csv(args.output / "task_variable_compatibility_expanded.csv")
        variables = compatibility[compatibility.manipulable].variable.unique()
        effects = estimate_behavioral_effects(
            frame, variables=variables, bootstrap_samples=1000,
            seed=config["seeds"]["behavior_bootstrap"],
        )
        path = args.output / args.model / "behavior/variable_effects.csv"
        effects.to_csv(path, index=False)
        interactions = estimate_preregistered_interactions(
            frame, config["design"]["preregistered_interactions"]
        )
        interaction_path = args.output / args.model / "behavior/interactions.csv"
        interactions.to_csv(interaction_path, index=False)
        task_matrix = compatibility[compatibility.task.isin(config["tasks"])].copy()
        task_matrix.to_csv(
            args.output / args.model / "behavior/task_variable_matrix.csv", index=False
        )
        result = {
            "rows": len(effects), "interaction_rows": len(interactions),
            "path": str(path),
        }
    elif args.command == "computational":
        config = load_cross_task_config(args.output / "effective_config.yaml")
        frame = read_records(args.observations)
        result = fit_computational_map(
            frame, config, output=args.output / args.model / "computational"
        )
        mark_model_stage(args.output, args.model, "computational", "complete")
    elif args.command == "final-validation":
        frame = evaluate_frozen_validation(
            read_records(args.observations),
            computational_root=args.output / args.model / "computational",
        )
        result = {"rows": len(frame), "refit": False}
    elif args.command == "active-select":
        config = load_cross_task_config(args.output / "effective_config.yaml")
        candidates = read_records(args.candidates)
        observed = json.loads(args.observed_groups.read_text(encoding="utf-8"))
        selected = select_active_batch(
            candidates, batch_size=args.batch_size, observed_groups=observed,
            weights=config["active_discovery"]["weights"],
            seed=config["seeds"]["active_acquisition"] + int(args.round),
        )
        validation = json.loads(args.validation_groups.read_text(encoding="utf-8"))
        assert_untouched_validation(selected.semantic_group, validation)
        selected.insert(0, "round", int(args.round))
        path = args.output / args.model / "computational/active_discrimination.csv"
        if path.is_file():
            selected = pd.concat([pd.read_csv(path), selected], ignore_index=True)
        selected.to_csv(path, index=False)
        mark_model_stage(args.output, args.model, "active_discrimination", "complete", table=str(path))
        result = {"selected": args.batch_size, "round": args.round, "path": str(path)}
    elif args.command == "representation":
        result = run_representational_transfer(
            read_records(args.features), output=args.output / args.model / "representation"
        )
        provenance_path = args.output / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["neural_split_hashes"][args.model] = result["neural_split_sha256"]
        write_provenance(provenance_path, provenance, final=False)
        mark_model_stage(args.output, args.model, "representation", "complete")
    elif args.command == "effect-geometry":
        frame = persistence_effect_geometry(read_records(args.features), rank=args.rank)
        path = args.output / args.model / "representation/persistence_effect_geometry/geometry.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        result = {"rows": len(frame), "path": str(path)}
    elif args.command == "validate-causal":
        frame = read_records(args.results)
        validate_causal_transfer_rows(frame)
        path = args.output / args.model / "causal/transfer_long.csv"
        frame.to_csv(path, index=False)
        frame[frame.source_task == frame.target_task].to_csv(
            args.output / args.model / "causal/within_task.csv", index=False
        )
        provenance_path = args.output / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["subspace_controller_hashes"][args.model] = sorted(
            frame.basis_sha256.astype(str).unique()
        )
        write_provenance(provenance_path, provenance, final=False)
        mark_model_stage(args.output, args.model, "causal", "complete")
        result = {"rows": len(frame), "path": str(path)}
    elif args.command == "freeze":
        result = freeze_hypotheses(
            args.output, json.loads(args.manifest.read_text(encoding="utf-8"))
        )
    elif args.command == "synthesize":
        model_root = args.output / args.model
        result = synthesize_levels(
            pd.read_csv(model_root / "behavior/variable_effects.csv"),
            pd.read_csv(model_root / "representation/transfer_long.csv"),
            pd.read_csv(model_root / "causal/transfer_long.csv"),
            output=model_root / "synthesis",
        )
        mark_model_stage(args.output, args.model, "synthesis", "complete")
    elif args.command == "cross-model":
        result = cross_model_tables(
            {name: args.output / name for name in ("qwen", "gemma", "llama")},
            args.output / "cross_model",
        )
    elif args.command == "report":
        if args.model:
            result = {"path": str(write_model_report(args.output / args.model))}
        else:
            result = {"path": str(write_cross_model_report(args.output / "cross_model"))}
    elif args.command == "figures":
        if args.model:
            result = {"paths": [str(path) for path in generate_model_figures(args.output / args.model)]}
        else:
            result = {"path": str(generate_cross_model_figure(args.output / "cross_model"))}
    else:
        result = validate_final_run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.command == "interface" and result.get("passed") is not True:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point for fresh model replications and Phase-A replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .replication.adapters import adapter_registry
from .replication.frozen_replay import replay_llama_phase_a
from .replication.harness import initialize_replication, plan_replication
from .replication.report import write_replication_report
from .replication.state import ReplicationRunState
from .replication.workflow import execute_all, execute_stage


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-agnostic persistence replication harness")
    parser.add_argument("--root", type=Path, default=_default_root())
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--adapter", choices=adapter_registry.names())
    parser.add_argument("--config", type=Path, default=Path("configs/replication/default.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stage",
        default="all",
        choices=(
            "all",
            "initialize",
            "interface",
            "behavior",
            "model_comparison",
            "freeze_theory",
            "counterfactuals",
            "mechanism",
            "generalization",
            "specificity",
            "report",
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--frozen-llama-replay", action="store_true")
    parser.add_argument("--online", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.frozen_llama_replay:
        if args.output is None:
            parser.error("--output is required for frozen Llama replay")
        result = replay_llama_phase_a(root)
        plan = {
            "mode": "execute" if args.execute else "dry-run",
            "phase": "llama_phase_a_regression",
            "output": str(args.output.resolve()),
            "writes_performed": False,
        }
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        output = args.output.resolve()
        if output.exists():
            parser.error(f"replication output already exists: {output}")
        output.mkdir(parents=True)
        _write_json(output / "phase_a_replay.json", result)
        provenance = {
            "schema_version": "replication-provenance-v1",
            "mode": "frozen_phase_a_replay",
            "input_hashes": result["input_hashes"],
            "endpoint_id": "cognitive_counterfactual_recovery",
            "metric_id": "global_cfr_v1",
        }
        _write_json(output / "provenance.json", provenance)
        write_replication_report(output / "REPLICATION_REPORT.md", result, provenance)
        plan["writes_performed"] = True
        print(json.dumps(plan, indent=2))
        return 0

    if args.resume is not None:
        output = args.resume.resolve()
        state = ReplicationRunState.load(output / "run_state.json")
        summary = {
            "mode": "execute" if args.execute else "dry-run",
            "resume": str(output),
            "requested_stage": args.stage,
            "state": state.to_dict(),
        }
        if args.execute:
            try:
                summary["result"] = (
                    execute_all(root, output, online=args.online)
                    if args.stage == "all"
                    else execute_stage(root, output, args.stage, online=args.online)
                )
                summary["state"] = ReplicationRunState.load(
                    output / "run_state.json"
                ).to_dict()
            except (ValueError, RuntimeError, FileNotFoundError) as error:
                parser.error(str(error))
        print(json.dumps(summary, indent=2))
        return 0

    missing = [
        name
        for name, value in (
            ("--model", args.model),
            ("--revision", args.revision),
            ("--adapter", args.adapter),
            ("--output", args.output),
        )
        if not value
    ]
    if missing:
        parser.error("new replication requires " + ", ".join(missing))
    config_path = args.config if args.config.is_absolute() else root / args.config
    kwargs = {
        "root": root,
        "config_path": config_path,
        "model_id": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "output": args.output,
        "tokenizer_id": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
    }
    try:
        if args.execute:
            result = initialize_replication(**kwargs)
            result["mode"] = "initialized"
            if args.stage != "initialize":
                result["stage"] = args.stage
                result["stage_result"] = (
                    execute_all(root, args.output, online=args.online)
                    if args.stage == "all"
                    else execute_stage(root, args.output, args.stage, online=args.online)
                )
        else:
            result = plan_replication(**kwargs)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Restartable command surface for transfer preparation and aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .neural import evaluate_controller, train_controller
from .pipeline import (
    aggregate_transfer,
    explain_transfer,
    gate_diagonal_transfer,
    prepare_transfer,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task-transfer mapping")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--replication", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=Path("configs/transfer/v1.yaml"))
    prepare.add_argument("--execute", action="store_true")
    prepare.add_argument("--final", action="store_true", help="require a clean worktree")
    prepare.add_argument(
        "--scope",
        choices=("single_task_matrix", "extended"),
        default="single_task_matrix",
    )
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--allow-incomplete", action="store_true")
    aggregate.add_argument("--execute", action="store_true")
    explain = sub.add_parser("explain")
    explain.add_argument("--output", type=Path, required=True)
    explain.add_argument("--predictors", type=Path, required=True)
    explain.add_argument("--alpha", type=float, default=10.0)
    explain.add_argument("--execute", action="store_true")
    diagonal = sub.add_parser("gate-diagonal")
    diagonal.add_argument("--output", type=Path, required=True)
    diagonal.add_argument(
        "--indices",
        help="comma-separated Slurm array indices included in this diagonal pass",
    )
    diagonal.add_argument("--execute", action="store_true")
    for name in ("train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--replication", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--work-id", required=True)
        command.add_argument("--config", type=Path, default=Path("configs/transfer/v1.yaml"))
        command.add_argument("--online", action="store_true")
        command.add_argument("--smoke", action="store_true")
        command.add_argument("--random-count", type=int)
        command.add_argument(
            "--evaluation-scope",
            choices=("all", "diagonal", "off_diagonal"),
            default="all",
        )
        command.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry-run", **vars(args)}, indent=2, default=str))
        return 0
    if args.command == "prepare":
        result = prepare_transfer(
            args.replication,
            args.output,
            args.config,
            final=args.final,
            scope=args.scope,
        )
    elif args.command == "aggregate":
        result = aggregate_transfer(args.output, require_complete=not args.allow_incomplete)
    elif args.command == "explain":
        result = explain_transfer(args.output, args.predictors, alpha=args.alpha)
    elif args.command == "gate-diagonal":
        indices = (
            [int(value) for value in args.indices.split(",") if value]
            if args.indices
            else None
        )
        result = gate_diagonal_transfer(args.output, active_indices=indices)
    elif args.command == "train":
        result = train_controller(args.replication, args.output, args.work_id, args.config, online=args.online, smoke=args.smoke)
    else:
        result = evaluate_controller(
            args.replication,
            args.output,
            args.work_id,
            args.config,
            online=args.online,
            smoke=args.smoke,
            random_count=args.random_count,
            evaluation_scope=args.evaluation_scope,
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

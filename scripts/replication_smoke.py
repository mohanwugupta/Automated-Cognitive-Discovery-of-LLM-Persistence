#!/usr/bin/env python3
"""CLI for the bounded model-agnostic replication GPU smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cognitive_discovery.replication.smoke import run_model_smoke, summarize_smoke_runs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Engineering-only GPU smoke tests for replication checkpoints"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-model", help="smoke-test one model checkpoint")
    run.add_argument("--model", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--adapter", default="auto")
    run.add_argument("--tokenizer")
    run.add_argument("--tokenizer-revision")
    run.add_argument("--config", default="configs/replication/default.yaml")
    run.add_argument("--output", required=True)
    run.add_argument("--seed", type=int, default=19001)
    run.add_argument("--online", action="store_true")
    summary = commands.add_parser("summarize", help="aggregate an entire model array")
    summary.add_argument("--root", required=True)
    summary.add_argument("--expected-jobs", required=True, type=int)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run-model":
        result = run_model_smoke(
            model_id=args.model,
            revision=args.revision,
            adapter=args.adapter,
            tokenizer_id=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            config_path=args.config,
            output_dir=args.output,
            local_files_only=not args.online,
            seed=args.seed,
        )
    else:
        result = summarize_smoke_runs(args.root, expected_jobs=args.expected_jobs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

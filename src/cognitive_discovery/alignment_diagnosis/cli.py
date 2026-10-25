"""CPU design/analysis, explicit GPU execution, and a fully synthetic smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .design import load_config
from .pipeline import analyze, collect, freeze_predictions, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Computational diagnosis of bandit STOP failures"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke"):
        command = commands.add_parser(name)
        command.add_argument(
            "--config", type=Path, default=Path("configs/alignment_diagnosis/v1.yaml")
        )
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--pilot", action="store_true")
        else:
            command.add_argument(
                "--generator", choices=("integrated", "gate"), default="integrated"
            )
    command = commands.add_parser("collect")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--phase", choices=("main", "calibration", "intervention"), required=True
    )
    command.add_argument("--execute", action="store_true")
    command.add_argument(
        "--model-path", type=Path, help="local Qwen checkpoint directory"
    )
    command.add_argument(
        "--online", action="store_true", help="allow pinned model download"
    )
    for name in ("freeze", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(load_config(args.config), args.output, pilot=args.pilot)
    elif args.command == "collect":
        result = collect(
            args.output,
            args.phase,
            execute=args.execute,
            online=args.online,
            model_path=args.model_path,
        )
    elif args.command == "freeze":
        result = freeze_predictions(args.output)
    elif args.command == "analyze":
        result = analyze(args.output)
    else:
        config = load_config(args.config)
        prepare(config, args.output, pilot=True, synthetic=args.generator)
        for phase in ("main", "calibration"):
            collect(args.output, phase, execute=True)
        freeze_predictions(args.output)
        collect(args.output, "intervention", execute=True)
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

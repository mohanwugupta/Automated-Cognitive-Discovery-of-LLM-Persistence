"""CLI for computational-model validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_synthetic_validation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen computational-model validation")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = {"root": str(args.root.resolve()), "output": str((args.output or args.root / 'validation/synthetic').resolve()), "smoke": args.smoke}
    if not args.execute:
        print(json.dumps({**plan, "mode": "dry-run"}, indent=2))
        return 0
    print(json.dumps(run_synthetic_validation(args.root, output=args.output, config_path=args.config, smoke=args.smoke), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

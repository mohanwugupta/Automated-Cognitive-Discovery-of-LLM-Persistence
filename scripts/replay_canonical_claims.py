#!/usr/bin/env python3
"""Replay C01-C13 from committed compact artifacts without GPU or network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_discovery.reproducibility.replay import replay_all_claims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = replay_all_claims(args.root)
    payload = {claim_id: result.to_dict() for claim_id, result in results.items()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for claim_id, result in results.items():
        print(
            f"{claim_id}\t{result.replay_status.value}\t"
            f"frozen_value_reproduced={str(result.frozen_value_reproduced).lower()}"
        )
    if args.check and not all(result.frozen_value_reproduced for result in results.values()):
        raise SystemExit("one or more frozen claim values did not reproduce")


if __name__ == "__main__":
    main()

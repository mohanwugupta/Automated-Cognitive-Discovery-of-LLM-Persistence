#!/usr/bin/env python3
"""Validate the canonical graph, artifact identities, and owner-approved gate semantics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_discovery.reproducibility.claims import validate_claim_gate_consistency
from cognitive_discovery.reproducibility.manifest import (
    load_canonical_manifest,
    validate_canonical_manifest,
    verify_manifest_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = load_canonical_manifest(args.root / "canonical_manifest.yaml")
    summary = validate_canonical_manifest(manifest, root=args.root)
    artifacts = verify_manifest_artifacts(manifest, root=args.root)
    validate_claim_gate_consistency(manifest, root=args.root)
    print(
        f"canonical validation passed: {summary.stage_count} stages, "
        f"{summary.edge_count} edges, {len(artifacts)} verified hashes"
    )


if __name__ == "__main__":
    main()

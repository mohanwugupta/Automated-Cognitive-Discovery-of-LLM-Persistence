"""Manifest-driven, CPU-only navigation and replay of frozen manuscript evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .reproducibility.manifest import (
    CanonicalManifestError,
    load_canonical_manifest,
    stage_dependency_order,
    validate_canonical_manifest,
    verify_stage_artifacts,
)
from .reproducibility.replay import ClaimReplayResult, replay_all_claims


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _claim_record(
    claim_id: str,
    *,
    manifest: Mapping[str, Any],
    replay: ClaimReplayResult,
) -> dict[str, Any]:
    navigation = manifest["claims"][claim_id]
    endpoint = replay.endpoint.value if hasattr(replay.endpoint, "value") else replay.endpoint
    if endpoint != navigation["endpoint_id"]:
        raise CanonicalManifestError(
            f"claim {claim_id} endpoint mismatch: replay={endpoint!r}, "
            f"manifest={navigation['endpoint_id']!r}"
        )
    return {
        "claim_id": claim_id,
        "analysis_id": navigation["analysis_id"],
        "supporting_stage_ids": navigation["supporting_stage_ids"],
        "endpoint_id": endpoint,
        "replay_status": replay.replay_status.value,
        "frozen_value_reproduced": replay.frozen_value_reproduced,
        "values": replay.values,
        "remaining_dependency": replay.remaining_dependency,
        "navigation": {
            "code_path": navigation["code_path"],
            "config_path": navigation["config_path"],
            "artifact_paths": navigation["artifact_paths"],
            "figure_ref": navigation["figure_ref"],
        },
    }


def replay_claim_records(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    manifest = load_canonical_manifest(root / "canonical_manifest.yaml")
    validate_canonical_manifest(manifest, root=root)
    replayed = replay_all_claims(root)
    if set(replayed) != set(manifest["claims"]):
        raise CanonicalManifestError("manifest claim registry and replay implementation differ")
    return [
        _claim_record(claim_id, manifest=manifest, replay=replayed[claim_id])
        for claim_id in manifest["claims"]
    ]


def inspect_stage(root: str | Path, stage_id: str) -> dict[str, Any]:
    root = Path(root)
    manifest = load_canonical_manifest(root / "canonical_manifest.yaml")
    validate_canonical_manifest(manifest, root=root)
    verified = verify_stage_artifacts(manifest, root=root, stage_id=stage_id)
    stage = manifest["stages"][stage_id]
    return {
        "stage_id": stage_id,
        "status": stage["status"],
        "replay_status": stage["replay_status"],
        "role": stage["role"],
        "dependency_order": stage_dependency_order(manifest, stage_id),
        "claims": [
            claim_id
            for claim_id, claim in manifest["claims"].items()
            if claim["analysis_id"] == stage_id or stage_id in claim["supporting_stage_ids"]
        ],
        "verified_artifacts": verified,
    }


def _print_claim(record: Mapping[str, Any]) -> None:
    marker = "PASS" if record["frozen_value_reproduced"] else "FAIL"
    print(
        f"{record['claim_id']} {marker} | {record['analysis_id']} | "
        f"{record['endpoint_id']} | {record['replay_status']}"
    )
    print(f"  code: {record['navigation']['code_path']}")
    print(f"  config: {record['navigation']['config_path'] or 'external / not applicable'}")
    print(f"  figure: {record['navigation']['figure_ref']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay frozen claims or inspect a canonical stage without GPU/network access."
    )
    parser.add_argument("--root", type=Path, default=_default_root())
    subparsers = parser.add_subparsers(dest="command", required=True)
    claims = subparsers.add_parser("claims", help="replay all manuscript claims")
    claims.add_argument("--json", action="store_true")
    claim = subparsers.add_parser("claim", help="replay one manuscript claim")
    claim.add_argument("claim_id", choices=[f"C{index:02d}" for index in range(1, 14)])
    claim.add_argument("--json", action="store_true")
    stage = subparsers.add_parser("stage", help="verify and describe one manifest stage")
    stage.add_argument("stage_id")
    stage.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"claims", "claim"}:
        records = replay_claim_records(args.root)
        if args.command == "claim":
            records = [record for record in records if record["claim_id"] == args.claim_id]
        if args.json:
            print(json.dumps(records if args.command == "claims" else records[0], indent=2))
        else:
            for record in records:
                _print_claim(record)
            passed = sum(record["frozen_value_reproduced"] for record in records)
            print(f"Reproduced {passed}/{len(records)} frozen claim records.")
        return 0 if all(record["frozen_value_reproduced"] for record in records) else 1

    stage = inspect_stage(args.root, args.stage_id)
    if args.json:
        print(json.dumps(stage, indent=2))
    else:
        print(
            f"{stage['stage_id']} | {stage['status']} | {stage['replay_status']} | "
            f"{stage['role']}"
        )
        print("dependency order: " + " -> ".join(stage["dependency_order"]))
        print("claims: " + (", ".join(stage["claims"]) or "none"))
        print(f"verified artifacts: {len(stage['verified_artifacts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

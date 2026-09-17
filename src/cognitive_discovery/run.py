"""Guarded Stage-3 entry point for new experimental reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .reproducibility.manifest import load_canonical_manifest, sha256_file, validate_canonical_manifest
from .reproducibility.provenance import write_run_provenance
from .stages.registry import StageConfigError, load_stage_config, validate_stage_config


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_value(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def _safe_output(root: Path, config: dict[str, Any], override: Path | None) -> Path:
    relative = override or Path(config["output"]["rerun_root"])
    path = relative if relative.is_absolute() else root / relative
    path = path.resolve()
    if root.resolve() not in path.parents:
        raise StageConfigError("rerun output must remain inside the repository")
    frozen_roots = []
    for config_path in (root / "configs/canonical").glob("*.yaml"):
        other = load_stage_config(config_path)
        frozen_roots.append((root / other["output"]["frozen_root"]).resolve())
    if any(
        path == frozen or frozen in path.parents or path in frozen.parents
        for frozen in frozen_roots
    ):
        raise StageConfigError("refusing to write in or above a frozen canonical output root")
    if path.exists():
        raise StageConfigError(f"rerun output already exists: {path}")
    return path


def _provenance_record(
    root: Path, config_path: Path, config: dict[str, Any], output: Path
) -> dict[str, Any]:
    required_inputs = config["execution"]["required_inputs"]
    dataset_hashes = {
        entry["id"]: entry["sha256"]
        for entry in required_inputs
        if entry.get("availability") == "committed"
    }
    if not dataset_hashes:
        dataset_hashes = {"scientific_defaults": config["scientific_defaults"]["sha256"]}
    pair_hash = config["split_design"].get("pair_sha256", "not_applicable")
    split_hash = config["split_design"].get("sha256") or "not_applicable"
    return {
        "schema_version": "run-provenance-v1",
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_value(root, "status", "--porcelain")),
        "model": {
            "checkpoint": config["model"]["checkpoint"],
            "revision": config["model"]["revision"],
        },
        "seeds": config["seeds"],
        "dataset_hashes": dataset_hashes,
        "pair_hash": pair_hash,
        "split_hash": split_hash,
        "behavioral_object": config["behavioral_object"],
        "neural_object": config["neural_object"],
        "endpoint_id": config["endpoint_id"],
        "metric_id": config["metric_id"],
        "config_hash": sha256_file(config_path),
        "output_directory": output.relative_to(root).as_posix(),
        "stage_id": config["stage_id"],
    }


def prepare_rerun(
    *, root: Path, stage_id: str, config_path: Path, output_override: Path | None
) -> tuple[dict[str, Any], Path, list[str], dict[str, Any]]:
    manifest = load_canonical_manifest(root / "canonical_manifest.yaml")
    validate_canonical_manifest(manifest, root=root)
    config = load_stage_config(config_path)
    if config.get("stage_id") != stage_id:
        raise StageConfigError(
            f"--stage {stage_id!r} does not match config stage_id {config.get('stage_id')!r}"
        )
    validate_stage_config(config, manifest=manifest, root=root, verify_inputs=True)
    if not config["execution"]["rerun_allowed"]:
        raise StageConfigError(
            f"{stage_id} is not safely rerunnable: {config['execution'].get('reason', 'unspecified')}"
        )
    if config["model"].get("revision") is None:
        raise StageConfigError(f"{stage_id} has no pinned model revision")
    output = _safe_output(root, config, output_override)
    substitutions = {
        "python": sys.executable,
        "config": str(config_path),
        "output": str(output),
        "revision": config["model"]["revision"],
    }
    command = [piece.format(**substitutions) for piece in config["execution"]["command"]]
    provenance = _provenance_record(root, config_path, config, output)
    return config, output, command, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and execute a canonical experiment rerun")
    parser.add_argument("--root", type=Path, default=_default_root())
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="run after validation; default is dry-run")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        config, output, command, provenance = prepare_rerun(
            root=root,
            stage_id=args.stage,
            config_path=config_path,
            output_override=args.output,
        )
    except StageConfigError as error:
        parser.error(str(error))
    summary = {
        "stage_id": config["stage_id"],
        "output": output.relative_to(root).as_posix(),
        "command": command,
        "provenance": provenance,
        "mode": "execute" if args.execute else "dry-run",
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        return 0
    output.mkdir(parents=True, exist_ok=False)
    write_run_provenance(output / "run_provenance.json", provenance)
    subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

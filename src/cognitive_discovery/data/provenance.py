from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from cognitive_discovery import __version__


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_metadata(config: dict, design_rows, *, root=".") -> dict:
    config_payload = json.dumps(config, sort_keys=True, default=str).encode()
    design_payload = "\n".join(
        json.dumps(row.to_dict(), sort_keys=True, default=str) for row in design_rows
    ).encode()
    dependencies = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "transformers",
        "matplotlib",
        "sweetpea",
    ):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "git_commit": _git_commit(Path(root).resolve()),
        "config_hash": hashlib.sha256(config_payload).hexdigest(),
        "design_hash": hashlib.sha256(design_payload).hexdigest(),
        "model_identifier": config.get("model"),
        "model_revision": config.get("model_revision"),
        "design_seed": config.get("design_seed"),
        "protocol_version": config.get("protocol_version"),
        "task_renderer_version": __version__,
        "dependency_versions": dependencies,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

"""Fail-closed provenance for manuscript-grade task-transfer runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(root: str | Path) -> dict:
    """Hash tracked and untracked differences without mutating the checkout."""

    root = Path(root)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, check=True, capture_output=True, text=True).stdout
    dirty = bool(status.strip())
    if not dirty:
        return {"git_commit": commit, "git_dirty": False, "diff_sha256": None}
    digest = hashlib.sha256()
    diff = subprocess.run(["git", "diff", "--binary", "HEAD", "--"], cwd=root, check=True, capture_output=True).stdout
    digest.update(diff)
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root, check=True, capture_output=True).stdout.split(b"\0")
    for raw in sorted(value for value in untracked if value):
        path = root / raw.decode()
        digest.update(raw + b"\0")
        if path.is_file(): digest.update(path.read_bytes())
    return {"git_commit": commit, "git_dirty": True, "diff_sha256": digest.hexdigest()}


def validate_transfer_provenance(record: dict, *, final: bool = False) -> None:
    required = {
        "schema_version", "git_commit", "git_dirty", "diff_sha256", "model",
        "tokenizer", "environment_lock_sha256", "specification_hashes",
        "pair_manifest_sha256", "transfer_split_sha256", "endpoint_id",
        "metric_id", "seeds", "output_root",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"transfer provenance is incomplete: {sorted(missing)}")
    if record["schema_version"] != "task-transfer-provenance-v1":
        raise ValueError("unsupported task-transfer provenance")
    if not COMMIT.fullmatch(str(record["git_commit"])):
        raise ValueError("git_commit must be an immutable 40-hex commit")
    if final and record["git_dirty"]:
        raise ValueError("final transfer evidence requires a clean worktree")
    if record["git_dirty"] and not SHA256.fullmatch(str(record["diff_sha256"])):
        raise ValueError("dirty development runs require an exact diff hash")
    for key in ("environment_lock_sha256", "pair_manifest_sha256", "transfer_split_sha256"):
        if not SHA256.fullmatch(str(record[key])):
            raise ValueError(f"{key} must be SHA-256")
    if record["endpoint_id"] != "cognitive_counterfactual_recovery" or record["metric_id"] != "global_cfr_v1":
        raise ValueError("task-transfer endpoint or metric identity changed")
    if not record["seeds"]:
        raise ValueError("transfer seeds are required")


def write_provenance(path: str | Path, record: dict, *, final=False) -> Path:
    validate_transfer_provenance(record, final=final)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

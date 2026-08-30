#!/usr/bin/env python3
"""Reject oversized or activation-bank files that are tracked by Git."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
import subprocess


DEFAULT_LIMIT_MB = 10.0
FORBIDDEN_DIRECTORIES = {"temporary_activations", "temporary_batches", "model_cache"}
FORBIDDEN_ACTIVATION_SUFFIXES = {".pt", ".npy", ".npz"}


def _tracked(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    return [root / value.decode() for value in result.stdout.split(b"\0") if value]


def _allowlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def violations(root: Path, *, limit_mb: float, allowlist_path: Path) -> list[str]:
    patterns = _allowlist(allowlist_path)
    maximum = float(limit_mb) * 1024 * 1024
    failures = []
    for path in _tracked(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        allowed = any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)
        if allowed:
            continue
        if FORBIDDEN_DIRECTORIES & set(path.relative_to(root).parts):
            failures.append(f"forbidden scratch artifact is tracked: {relative}")
        if path.suffix in FORBIDDEN_ACTIVATION_SUFFIXES and (
            "activation" in relative.lower() or "mechanistic" in relative.lower()
        ):
            failures.append(f"full activation tensor format is tracked: {relative}")
        if path.stat().st_size > maximum:
            size_mb = path.stat().st_size / 1024 / 1024
            failures.append(
                f"tracked file exceeds {limit_mb:g} MB: {relative} ({size_mb:.2f} MB)"
            )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--max-mb", type=float, default=DEFAULT_LIMIT_MB)
    parser.add_argument("--allowlist", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    allowlist = args.allowlist or root / "config/artifact_allowlist.txt"
    failures = violations(root, limit_mb=args.max_mb, allowlist_path=allowlist)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Artifact size check passed ({args.max_mb:g} MB tracked-file limit).")


if __name__ == "__main__":
    main()

#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT="${OUTPUT:?export OUTPUT to a new qwen-prospective directory}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
RUN_SPEC="$PROJECT_DIR/configs/replication/qwen_prospective_run.yaml"
CONFIG="$PROJECT_DIR/configs/replication/default.yaml"
REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

cd "$PROJECT_DIR"
if [ -n "$(git status --porcelain)" ]; then
  echo "Prospective Qwen requires a completely clean committed worktree" >&2
  git status --short >&2
  exit 2
fi
if [ ! -f uv.lock ]; then
  echo "uv.lock is required" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python -m pytest -q -p no:cacheprovider
if [ -n "$(git status --porcelain)" ]; then
  echo "CPU tests changed the clean baseline; refusing submission" >&2
  git status --short >&2
  exit 2
fi

baseline_record=$(mktemp "${TMPDIR:-/tmp}/qwen-prospective-preflight.XXXXXX.json")
trap 'rm -f "$baseline_record"' EXIT
git_commit=$(git rev-parse HEAD)
uv_hash=$(shasum -a 256 uv.lock | awk '{print $1}')
python -c '
from datetime import datetime, timezone
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "prospective-baseline-preflight-v1",
    "git_commit": sys.argv[2],
    "git_dirty": False,
    "uv_lock_sha256": sys.argv[3],
    "cpu_test_status": "pass",
    "cpu_test_command": "PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$baseline_record" "$git_commit" "$uv_hash"

export MODEL_ID="Qwen/Qwen3.5-4B"
export TOKENIZER_ID="Qwen/Qwen3.5-4B"
export REVISION TOKENIZER_REVISION="$REVISION" ADAPTER=qwen
export OUTPUT CONFIG RUN_SPEC BASELINE_PREFLIGHT="$baseline_record" CONDA_ENV
bash "$PROJECT_DIR/scripts/submit_replication.sh"

echo "Prospective Qwen baseline commit: $git_commit"
echo "Prospective Qwen uv.lock SHA-256: $uv_hash"

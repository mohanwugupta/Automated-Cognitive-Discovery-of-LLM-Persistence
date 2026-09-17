#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPU_SCRIPT="$PROJECT_DIR/slurm/run_replication_smoke.slurm"
CPU_SCRIPT="$PROJECT_DIR/slurm/run_replication_smoke_cpu.slurm"
SMOKE_MODEL_PATHS="${SMOKE_MODEL_PATHS:?space-separated local model paths are required}"
SMOKE_REVISIONS="${SMOKE_REVISIONS:?space-separated immutable 40-hex revisions are required}"
SMOKE_ROOT="${SMOKE_ROOT:?a new SMOKE_ROOT is required}"
SMOKE_MAX_CONCURRENT="${SMOKE_MAX_CONCURRENT:-1}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
  exit 1
fi
if [ ! -f "$GPU_SCRIPT" ] || [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
if [ -e "$SMOKE_ROOT" ]; then
  echo "SMOKE_ROOT already exists; use a new directory to preserve prior diagnostics" >&2
  exit 1
fi
if ! [[ "$SMOKE_MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMOKE_MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

read -r -a model_paths <<< "$SMOKE_MODEL_PATHS"
read -r -a revisions <<< "$SMOKE_REVISIONS"
model_count="${#model_paths[@]}"
if [ "$model_count" -lt 1 ] || [ "${#revisions[@]}" -ne "$model_count" ]; then
  echo "SMOKE_MODEL_PATHS and SMOKE_REVISIONS must be nonempty equal-length lists" >&2
  exit 2
fi
for revision in "${revisions[@]}"; do
  if ! [[ "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Every smoke revision must be an immutable lowercase 40-hex commit: $revision" >&2
    exit 2
  fi
done
for variable in SMOKE_ADAPTERS SMOKE_TOKENIZER_PATHS SMOKE_TOKENIZER_REVISIONS; do
  value="${!variable:-}"
  if [ -n "$value" ]; then
    read -r -a entries <<< "$value"
    if [ "${#entries[@]}" -ne "$model_count" ]; then
      echo "$variable must contain exactly $model_count space-separated values" >&2
      exit 2
    fi
  fi
done

last_index=$((model_count - 1))
export SMOKE_MODEL_PATHS SMOKE_REVISIONS SMOKE_ROOT
export SMOKE_EXPECTED_JOBS="$model_count"

tests_submission=$(sbatch --parsable --job-name=cog_rep_smoke_tests --export=ALL,SMOKE_PHASE=tests "$CPU_SCRIPT")
tests_job="${tests_submission%%;*}"

gpu_submission=$(sbatch --parsable --job-name=cog_rep_smoke --array="0-${last_index}%${SMOKE_MAX_CONCURRENT}" --dependency="afterok:${tests_job}" --export=ALL "$GPU_SCRIPT")
gpu_job="${gpu_submission%%;*}"

summary_submission=$(sbatch --parsable --job-name=cog_rep_smoke_summary --dependency="afterany:${tests_job}:${gpu_job}" --export=ALL,SMOKE_PHASE=summarize "$CPU_SCRIPT")
summary_job="${summary_submission%%;*}"

echo "Replication smoke suite submitted"
echo "  CPU contracts: ${tests_job}"
echo "  GPU models:    ${gpu_job}_[0-${last_index}] (max concurrent: ${SMOKE_MAX_CONCURRENT})"
echo "  CPU summary:   ${summary_job} (runs after any GPU outcome)"
echo "  Report:        ${SMOKE_ROOT}/SMOKE_REPORT.md"

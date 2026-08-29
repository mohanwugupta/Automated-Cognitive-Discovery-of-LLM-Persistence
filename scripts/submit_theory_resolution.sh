#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPU_SCRIPT="$PROJECT_DIR/run_theory_resolution.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_theory_cpu.slurm"
SHARD_COUNT="${SHARD_COUNT:-8}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
  exit 1
fi
if [ ! -f "$GPU_SCRIPT" ] || [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
if ! [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARD_COUNT must be a positive integer" >&2
  exit 2
fi
mkdir -p "$PROJECT_DIR/logs"
last_shard=$((SHARD_COUNT - 1))

tests_submission=$(sbatch --parsable --job-name=cog_v3_tests --time=00:45:00 \
  --export=ALL,PHASE=tests "$CPU_SCRIPT")
tests_job="${tests_submission%%;*}"

prepare_submission=$(sbatch --parsable --job-name=cog_v3_prepare \
  --dependency="afterok:${tests_job}" --export=ALL,PHASE=prepare "$GPU_SCRIPT")
prepare_job="${prepare_submission%%;*}"

collection_submission=$(sbatch --parsable --job-name=cog_v3_collect \
  --array="0-${last_shard}" --dependency="afterok:${prepare_job}" \
  --export="ALL,PHASE=collect,SHARD_COUNT=${SHARD_COUNT}" "$GPU_SCRIPT")
collection_job="${collection_submission%%;*}"

finalize_submission=$(sbatch --parsable --job-name=cog_v3_gate --time=01:00:00 \
  --dependency="afterok:${collection_job}" --export=ALL,PHASE=finalize "$CPU_SCRIPT")
finalize_job="${finalize_submission%%;*}"

evaluate_submission=$(sbatch --parsable --job-name=cog_v3_evaluate \
  --dependency="afterok:${finalize_job}" --export=ALL,PHASE=evaluate "$CPU_SCRIPT")
evaluate_job="${evaluate_submission%%;*}"

echo "Behavioral theory resolution submitted"
echo "  tests:                ${tests_job}"
echo "  audits/freeze/design: ${prepare_job}"
echo "  Round-3 collection:   ${collection_job}_[0-${last_shard}]"
echo "  merge/quality gate:   ${finalize_job}"
echo "  frozen comparison:    ${evaluate_job}"

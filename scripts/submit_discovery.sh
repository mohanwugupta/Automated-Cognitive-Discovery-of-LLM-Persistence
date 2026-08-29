#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_SCRIPT="$PROJECT_DIR/run_discovery.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_discovery_cpu.slurm"
SHARD_COUNT="${SHARD_COUNT:-8}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on the cluster login node" >&2
  exit 1
fi
if [ ! -f "$RUN_SCRIPT" ] || [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
if ! [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARD_COUNT must be a positive integer" >&2
  exit 2
fi

last_shard=$((SHARD_COUNT - 1))

tests_submission=$(sbatch --parsable --job-name=cog_tests --time=00:45:00 \
  --export=ALL,PHASE=tests "$CPU_SCRIPT")
tests_job="${tests_submission%%;*}"

pilot_submission=$(sbatch --parsable --job-name=cog_pilot --array="0-${last_shard}" \
  --dependency="afterok:${tests_job}" \
  --export="ALL,PHASE=pilot_collect,SHARD_COUNT=${SHARD_COUNT}" "$RUN_SCRIPT")
pilot_job="${pilot_submission%%;*}"

pilot_finalize_submission=$(sbatch --parsable --job-name=cog_pilot_gate --time=01:00:00 \
  --dependency="afterok:${pilot_job}" \
  --export=ALL,PHASE=pilot_finalize "$CPU_SCRIPT")
pilot_finalize_job="${pilot_finalize_submission%%;*}"

discovery_submission=$(sbatch --parsable --job-name=cog_collect --array="0-${last_shard}" \
  --dependency="afterok:${pilot_finalize_job}" \
  --export="ALL,PHASE=discovery_collect,SHARD_COUNT=${SHARD_COUNT}" "$RUN_SCRIPT")
discovery_job="${discovery_submission%%;*}"

finalize_submission=$(sbatch --parsable --job-name=cog_finalize --time=01:00:00 \
  --dependency="afterok:${discovery_job}" \
  --export=ALL,PHASE=discovery_finalize "$CPU_SCRIPT")
finalize_job="${finalize_submission%%;*}"

analysis_submission=$(sbatch --parsable --job-name=cog_models --time=12:00:00 \
  --dependency="afterok:${finalize_job}" \
  --export=ALL,PHASE=analyze "$RUN_SCRIPT")
analysis_job="${analysis_submission%%;*}"

validation_submission=$(sbatch --parsable --job-name=cog_validate --time=06:00:00 \
  --dependency="afterok:${analysis_job}" \
  --export=ALL,PHASE=validation "$RUN_SCRIPT")
validation_job="${validation_submission%%;*}"

echo "Automated cognitive discovery submitted"
echo "  tests:              ${tests_job}"
echo "  pilot collection:   ${pilot_job}_[0-${last_shard}]"
echo "  pilot gate:         ${pilot_finalize_job}"
echo "  discovery:          ${discovery_job}_[0-${last_shard}]"
echo "  finalize:           ${finalize_job}"
echo "  model tournament:   ${analysis_job}"
echo "  frozen validation:  ${validation_job}"

#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_SCRIPT="$PROJECT_DIR/run_active_discovery.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_active_cpu.slurm"
SHARD_COUNT="${SHARD_COUNT:-8}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
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
mkdir -p "$PROJECT_DIR/logs"
last_shard=$((SHARD_COUNT - 1))

tests_submission=$(sbatch --parsable --job-name=cog_v2_tests --time=00:45:00 \
  --export=ALL,PHASE=tests "$CPU_SCRIPT")
tests_job="${tests_submission%%;*}"

prepare_submission=$(sbatch --parsable --job-name=cog_v2_prepare \
  --dependency="afterok:${tests_job}" --export=ALL,PHASE=prepare "$RUN_SCRIPT")
prepare_job="${prepare_submission%%;*}"

active_submission=$(sbatch --parsable --job-name=cog_v2_active \
  --array="0-${last_shard}" --dependency="afterok:${prepare_job}" \
  --export="ALL,PHASE=active_collect,SHARD_COUNT=${SHARD_COUNT}" "$RUN_SCRIPT")
active_job="${active_submission%%;*}"

active_finalize_submission=$(sbatch --parsable --job-name=cog_v2_active_gate --time=01:00:00 \
  --dependency="afterok:${active_job}" --export=ALL,PHASE=active_finalize "$CPU_SCRIPT")
active_finalize_job="${active_finalize_submission%%;*}"

update_submission=$(sbatch --parsable --job-name=cog_v2_update \
  --dependency="afterok:${active_finalize_job}" --export=ALL,PHASE=update "$CPU_SCRIPT")
update_job="${update_submission%%;*}"

final_submission=$(sbatch --parsable --job-name=cog_v2_final \
  --array="0-${last_shard}" --dependency="afterok:${update_job}" \
  --export="ALL,PHASE=final_collect,SHARD_COUNT=${SHARD_COUNT}" "$RUN_SCRIPT")
final_job="${final_submission%%;*}"

final_finalize_submission=$(sbatch --parsable --job-name=cog_v2_final_merge --time=01:00:00 \
  --dependency="afterok:${final_job}" --export=ALL,PHASE=final_finalize "$CPU_SCRIPT")
final_finalize_job="${final_finalize_submission%%;*}"

evaluate_submission=$(sbatch --parsable --job-name=cog_v2_evaluate \
  --dependency="afterok:${final_finalize_job}" --export=ALL,PHASE=evaluate "$CPU_SCRIPT")
evaluate_job="${evaluate_submission%%;*}"

echo "Active hierarchical discovery submitted"
echo "  tests:             ${tests_job}"
echo "  hierarchy/prepare: ${prepare_job}"
echo "  active collection: ${active_job}_[0-${last_shard}]"
echo "  active gate:       ${active_finalize_job}"
echo "  hierarchy update:  ${update_job}"
echo "  final collection:  ${final_job}_[0-${last_shard}]"
echo "  final merge:       ${final_finalize_job}"
echo "  final evaluation:  ${evaluate_job}"

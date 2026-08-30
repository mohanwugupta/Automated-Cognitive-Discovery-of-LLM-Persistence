#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPU_SCRIPT="$PROJECT_DIR/run_action_history_disambiguation.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_action_history_cpu.slurm"
RANDOM_SHARD_COUNT="${RANDOM_SHARD_COUNT:-10}"
RANDOM_ARRAY_CONCURRENCY="${RANDOM_ARRAY_CONCURRENCY:-2}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
  exit 1
fi
if [ ! -f "$GPU_SCRIPT" ] || [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
mkdir -p "$PROJECT_DIR/logs"

tests_submission=$(sbatch --parsable --job-name=cog_action_tests --time=00:45:00 \
  --export=ALL,PHASE=tests "$CPU_SCRIPT")
tests_job="${tests_submission%%;*}"

prepare_submission=$(sbatch --parsable --job-name=cog_action_prepare --time=01:00:00 \
  --dependency="afterok:${tests_job}" --export=ALL,PHASE=prepare "$CPU_SCRIPT")
prepare_job="${prepare_submission%%;*}"

fit_submission=$(sbatch --parsable --job-name=cog_action_fit \
  --dependency="afterok:${prepare_job}" --export=ALL,PHASE=fit "$GPU_SCRIPT")
fit_job="${fit_submission%%;*}"

project_submission=$(sbatch --parsable --job-name=cog_action_project \
  --dependency="afterok:${fit_job}" --export=ALL,PHASE=project "$GPU_SCRIPT")
project_job="${project_submission%%;*}"

analyze_submission=$(sbatch --parsable --job-name=cog_action_analyze --time=02:00:00 \
  --dependency="afterok:${project_job}" --export=ALL,PHASE=analyze "$CPU_SCRIPT")
analyze_job="${analyze_submission%%;*}"

steer_submission=$(sbatch --parsable --job-name=cog_action_steer \
  --dependency="afterok:${analyze_job}" --export=ALL,PHASE=steer "$GPU_SCRIPT")
steer_job="${steer_submission%%;*}"

random_submission=$(sbatch --parsable --job-name=cog_action_random \
  --array="0-$((RANDOM_SHARD_COUNT - 1))%${RANDOM_ARRAY_CONCURRENCY}" \
  --dependency="afterok:${analyze_job}" \
  --export=ALL,PHASE=random,RANDOM_SHARD_COUNT="$RANDOM_SHARD_COUNT" "$GPU_SCRIPT")
random_job="${random_submission%%;*}"

patch_submission=$(sbatch --parsable --job-name=cog_action_patch \
  --dependency="afterok:${steer_job}" --export=ALL,PHASE=patch "$GPU_SCRIPT")
patch_job="${patch_submission%%;*}"

report_submission=$(sbatch --parsable --job-name=cog_action_report --time=01:00:00 \
  --dependency="afterok:${patch_job}:${random_job}" \
  --export=ALL,PHASE=report "$CPU_SCRIPT")
report_job="${report_submission%%;*}"

echo "Action-history disambiguation workflow submitted"
echo "  CPU tests:                  ${tests_job}"
echo "  CPU freeze/residual/match:  ${prepare_job}"
echo "  GPU directions/gradients:   ${fit_job}"
echo "  GPU scalar projections:     ${project_job}"
echo "  CPU representation gates:   ${analyze_job}"
echo "  GPU primary steering:       ${steer_job}"
echo "  GPU random-null array:      ${random_job}"
echo "  GPU optional patching:      ${patch_job}"
echo "  CPU report:                 ${report_job}"

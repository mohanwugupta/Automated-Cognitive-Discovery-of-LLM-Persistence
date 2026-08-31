#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPU_SCRIPT="$PROJECT_DIR/run_causal_mechanistic.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_causal_mechanistic_cpu.slurm"
LOCALIZATION_CONCURRENCY="${LOCALIZATION_CONCURRENCY:-2}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
  exit 1
fi
if [ ! -f "$GPU_SCRIPT" ] || [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
mkdir -p "$PROJECT_DIR/logs"

tests_submission=$(sbatch --parsable --job-name=cog_cf_tests --time=00:45:00 \
  --export=ALL,PHASE=tests "$CPU_SCRIPT")
tests_job="${tests_submission%%;*}"

prepare_submission=$(sbatch --parsable --job-name=cog_cf_prepare --time=02:00:00 \
  --dependency="afterok:${tests_job}" --export=ALL,PHASE=prepare "$CPU_SCRIPT")
prepare_job="${prepare_submission%%;*}"

diagnostic_submission=$(sbatch --parsable --job-name=cog_cf_diagnostic --time=00:30:00 \
  --dependency="afterok:${prepare_job}" --export=ALL,PHASE=diagnostic "$CPU_SCRIPT")
diagnostic_job="${diagnostic_submission%%;*}"

localize_submission=$(sbatch --parsable --job-name=cog_cf_localize \
  --array="0-31%${LOCALIZATION_CONCURRENCY}" \
  --dependency="afterok:${diagnostic_job}" --export=ALL,PHASE=localize "$GPU_SCRIPT")
localize_job="${localize_submission%%;*}"

summarize_submission=$(sbatch --parsable --job-name=cog_cf_summarize --time=01:00:00 \
  --dependency="afterok:${localize_job}" --export=ALL,PHASE=summarize "$CPU_SCRIPT")
summarize_job="${summarize_submission%%;*}"

dispatch_submission=$(sbatch --parsable --job-name=cog_cf_dispatch --time=00:15:00 \
  --dependency="afterok:${summarize_job}" --export=ALL,PHASE=dispatch_das "$CPU_SCRIPT")
dispatch_job="${dispatch_submission%%;*}"

echo "Counterfactual causal-mechanistic workflow submitted"
echo "  CPU tests:                         ${tests_job}"
echo "  CPU theory freeze/counterfactuals: ${prepare_job}"
echo "  CPU information diagnostics:       ${diagnostic_job}"
echo "  GPU whole-state localization:      ${localize_job}"
echo "  CPU localization/search plan:      ${summarize_job}"
echo "  CPU conditional DAS dispatcher:    ${dispatch_job}"
echo "The dispatcher submits an exact-size DAS array only if Level 3 passes."

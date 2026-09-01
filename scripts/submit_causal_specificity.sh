#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CPU_SCRIPT="$PROJECT_DIR/run_causal_specificity_cpu.slurm"
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
  exit 1
fi
if [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
mkdir -p "$PROJECT_DIR/logs"

tests_submission=$(sbatch --parsable --job-name=cog_spec_tests --time=00:45:00 \
  --export=ALL,PHASE=tests "$CPU_SCRIPT")
tests_job="${tests_submission%%;*}"
prepare_submission=$(sbatch --parsable --job-name=cog_spec_prepare --time=02:00:00 \
  --dependency="afterok:${tests_job}" --export=ALL,PHASE=prepare "$CPU_SCRIPT")
prepare_job="${prepare_submission%%;*}"
dispatch_submission=$(sbatch --parsable --job-name=cog_spec_dispatch --time=00:15:00 \
  --dependency="afterok:${prepare_job}" --export=ALL,PHASE=dispatch "$CPU_SCRIPT")
dispatch_job="${dispatch_submission%%;*}"

echo "Stable-CFR causal-specificity workflow submitted"
echo "  CPU tests:                    ${tests_job}"
echo "  CPU freeze/corrected metrics: ${prepare_job}"
echo "  CPU exact-array dispatcher:   ${dispatch_job}"
echo "The dispatcher submits one GPU-active evaluation job per frozen DAS candidate."

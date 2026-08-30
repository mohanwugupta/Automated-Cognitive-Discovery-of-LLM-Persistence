#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPU_SCRIPT="$PROJECT_DIR/run_mechanistic.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_mechanistic_cpu.slurm"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
  exit 1
fi
if [ ! -f "$GPU_SCRIPT" ] || [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
mkdir -p "$PROJECT_DIR/logs"

tests_submission=$(sbatch --parsable --job-name=cog_mech_tests --time=00:45:00 \
  --export=ALL,PHASE=tests "$CPU_SCRIPT")
tests_job="${tests_submission%%;*}"

prepare_submission=$(sbatch --parsable --job-name=cog_mech_prepare --time=01:00:00 \
  --dependency="afterok:${tests_job}" --export=ALL,PHASE=prepare "$CPU_SCRIPT")
prepare_job="${prepare_submission%%;*}"

scan_submission=$(sbatch --parsable --job-name=cog_mech_scan \
  --dependency="afterok:${prepare_job}" --export=ALL,PHASE=scan "$GPU_SCRIPT")
scan_job="${scan_submission%%;*}"

project_submission=$(sbatch --parsable --job-name=cog_mech_project \
  --dependency="afterok:${scan_job}" --export=ALL,PHASE=project "$GPU_SCRIPT")
project_job="${project_submission%%;*}"

analyze_submission=$(sbatch --parsable --job-name=cog_mech_analyze --time=02:00:00 \
  --dependency="afterok:${project_job}" --export=ALL,PHASE=analyze "$CPU_SCRIPT")
analyze_job="${analyze_submission%%;*}"

intervene_submission=$(sbatch --parsable --job-name=cog_mech_intervene \
  --dependency="afterok:${analyze_job}" --export=ALL,PHASE=intervene "$GPU_SCRIPT")
intervene_job="${intervene_submission%%;*}"

report_submission=$(sbatch --parsable --job-name=cog_mech_report --time=01:00:00 \
  --dependency="afterok:${intervene_job}" --export=ALL,PHASE=report "$CPU_SCRIPT")
report_job="${report_submission%%;*}"

echo "Mechanistic workflow submitted"
echo "  tests:                   ${tests_job}"
echo "  frozen targets/dataset:  ${prepare_job}"
echo "  paired direction scan:   ${scan_job}"
echo "  scalar projection scan:  ${project_job}"
echo "  representation gates:    ${analyze_job}"
echo "  steering and patching:   ${intervene_job}"
echo "  causal report:           ${report_job}"

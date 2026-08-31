#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CAUSAL_OUTPUT="${CAUSAL_OUTPUT:-artifacts/causal_mech_v1}"
CPU_SCRIPT="$PROJECT_DIR/run_causal_mechanistic_cpu.slurm"
GPU_SCRIPT="$PROJECT_DIR/run_causal_mechanistic.slurm"
status=$(python -c 'import json,os; from pathlib import Path; p=Path(os.environ.get("CAUSAL_OUTPUT","artifacts/causal_mech_v1"))/"gates.json"; print(json.loads(p.read_text()).get("circuit",{}).get("status","not_eligible"))')
if [ "$status" != "eligible" ]; then
  echo "Circuit analysis is not eligible (status=$status); no GPU job was submitted." >&2
  exit 1
fi
circuit_submission=$(sbatch --parsable --job-name=cog_cf_circuit \
  --export=ALL,PHASE=circuit "$GPU_SCRIPT")
circuit_job="${circuit_submission%%;*}"
report_submission=$(sbatch --parsable --job-name=cog_cf_circuit_report --time=01:00:00 \
  --dependency="afterok:${circuit_job}" --export=ALL,PHASE=report "$CPU_SCRIPT")
report_job="${report_submission%%;*}"
echo "Eligible circuit workflow submitted: circuit=${circuit_job}, report=${report_job}"

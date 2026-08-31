#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CAUSAL_OUTPUT="${CAUSAL_OUTPUT:-artifacts/causal_mech_v1}"
CPU_SCRIPT="$PROJECT_DIR/run_causal_mechanistic_cpu.slurm"
GPU_SCRIPT="$PROJECT_DIR/run_causal_mechanistic.slurm"
selected_count=$(python -c 'import json,os; from pathlib import Path; p=Path(os.environ.get("CAUSAL_OUTPUT","artifacts/causal_mech_v1"))/"representations/selected_alignments.json"; print(len(json.loads(p.read_text())))')

if [ "$selected_count" -eq 0 ]; then
  echo "No DAS alignment passed validation; generating the preregistered stop report."
  python scripts/causal_mechanistic.py --config "${CONFIG:-configs/causal_mech_v1.yaml}" \
    --output "$CAUSAL_OUTPUT" --phase report
  exit 0
fi

validation_submission=$(sbatch --parsable --job-name=cog_cf_validate \
  --dependency="afterok:${SLURM_JOB_ID}" --export=ALL,PHASE=validate "$GPU_SCRIPT")
validation_job="${validation_submission%%;*}"
report_submission=$(sbatch --parsable --job-name=cog_cf_report --time=01:00:00 \
  --dependency="afterok:${validation_job}" --export=ALL,PHASE=report "$CPU_SCRIPT")
report_job="${report_submission%%;*}"
echo "Held-out continuation submitted: validation=${validation_job}, report=${report_job}"

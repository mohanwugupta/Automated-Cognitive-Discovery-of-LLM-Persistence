#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SPECIFICITY_OUTPUT="${SPECIFICITY_OUTPUT:-artifacts/causal_specificity_v2}"
GPU_SCRIPT="$PROJECT_DIR/run_causal_specificity.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_causal_specificity_cpu.slurm"
NECESSITY_CONCURRENCY="${NECESSITY_CONCURRENCY:-2}"
job_count=$(python -c 'import json,os; from pathlib import Path; p=Path(os.environ.get("SPECIFICITY_OUTPUT","artifacts/causal_specificity_v2"))/"frozen_models/necessity_jobs.json"; print(len(json.loads(p.read_text())))')

if [ "$job_count" -eq 0 ]; then
  echo "No candidate passed Level 5A; finalizing without a necessity GPU allocation."
  python scripts/causal_specificity.py --config "${CONFIG:-configs/causal_specificity_v2.yaml}" \
    --source "${SOURCE_OUTPUT:-artifacts/causal_mech_v1}" --output "$SPECIFICITY_OUTPUT" \
    --phase finalize
  exit 0
fi

necessity_submission=$(sbatch --parsable --job-name=cog_spec_necessity \
  --array="0-$((job_count - 1))%${NECESSITY_CONCURRENCY}" \
  --dependency="afterok:${SLURM_JOB_ID}" --export=ALL,PHASE=necessity "$GPU_SCRIPT")
necessity_job="${necessity_submission%%;*}"
final_submission=$(sbatch --parsable --job-name=cog_spec_final --time=02:00:00 \
  --dependency="afterok:${necessity_job}" --export=ALL,PHASE=finalize "$CPU_SCRIPT")
final_job="${final_submission%%;*}"
echo "Post-specificity necessity submitted: necessity=${necessity_job}, final=${final_job}"

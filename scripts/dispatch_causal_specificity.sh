#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SPECIFICITY_OUTPUT="${SPECIFICITY_OUTPUT:-artifacts/causal_specificity_v2}"
GPU_SCRIPT="$PROJECT_DIR/run_causal_specificity.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_causal_specificity_cpu.slurm"
EVALUATION_CONCURRENCY="${EVALUATION_CONCURRENCY:-2}"
job_count=$(python -c 'import json,os; from pathlib import Path; p=Path(os.environ.get("SPECIFICITY_OUTPUT","artifacts/causal_specificity_v2"))/"frozen_models/evaluation_jobs.json"; print(len(json.loads(p.read_text())))')

if [ "$job_count" -eq 0 ]; then
  echo "No frozen Level-4 candidates exist; no GPU allocation was submitted." >&2
  exit 1
fi

evaluation_submission=$(sbatch --parsable --job-name=cog_spec_eval \
  --array="0-$((job_count - 1))%${EVALUATION_CONCURRENCY}" \
  --dependency="afterok:${SLURM_JOB_ID}" --export=ALL,PHASE=evaluate "$GPU_SCRIPT")
evaluation_job="${evaluation_submission%%;*}"
aggregate_submission=$(sbatch --parsable --job-name=cog_spec_aggregate --time=06:00:00 \
  --dependency="afterok:${evaluation_job}" --export=ALL,PHASE=aggregate "$CPU_SCRIPT")
aggregate_job="${aggregate_submission%%;*}"
necessity_dispatch_submission=$(sbatch --parsable --job-name=cog_spec_nec_dispatch --time=00:15:00 \
  --dependency="afterok:${aggregate_job}" --export=ALL,PHASE=dispatch_necessity "$CPU_SCRIPT")
necessity_dispatch_job="${necessity_dispatch_submission%%;*}"
echo "Stable-CFR continuation submitted: evaluation=${evaluation_job}, aggregate=${aggregate_job}, necessity-dispatch=${necessity_dispatch_job}"

#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
OOD_OUTPUT="${OOD_OUTPUT:-artifacts/ood_free_generation_v1}"
GPU_SCRIPT="$PROJECT_DIR/run_ood_generation.slurm"
CPU_SCRIPT="$PROJECT_DIR/run_ood_generation_cpu.slurm"
EVALUATION_CONCURRENCY="${EVALUATION_CONCURRENCY:-2}"
job_count=$(python -c 'import json,os; from pathlib import Path; p=Path(os.environ.get("OOD_OUTPUT","artifacts/ood_free_generation_v1"))/"frozen/evaluation_jobs.json"; print(len(json.loads(p.read_text())))')

if [ "$job_count" -eq 0 ]; then
  echo "No frozen OOD jobs exist; no GPU allocation was submitted." >&2
  exit 1
fi

evaluation_submission=$(sbatch --parsable --job-name=cog_ood_eval \
  --array="0-$((job_count - 1))%${EVALUATION_CONCURRENCY}" \
  --dependency="afterok:${SLURM_JOB_ID}" --export=ALL,PHASE=evaluate "$GPU_SCRIPT")
evaluation_job="${evaluation_submission%%;*}"
aggregate_submission=$(sbatch --parsable --job-name=cog_ood_aggregate --time=12:00:00 \
  --dependency="afterok:${evaluation_job}" --export=ALL,PHASE=aggregate "$CPU_SCRIPT")
aggregate_job="${aggregate_submission%%;*}"
echo "OOD continuation submitted: evaluation=${evaluation_job}, aggregate=${aggregate_job}"

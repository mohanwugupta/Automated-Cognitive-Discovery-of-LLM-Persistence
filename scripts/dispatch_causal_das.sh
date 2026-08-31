#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CAUSAL_OUTPUT="${CAUSAL_OUTPUT:-artifacts/causal_mech_v1}"
CPU_SCRIPT="$PROJECT_DIR/run_causal_mechanistic_cpu.slurm"
GPU_SCRIPT="$PROJECT_DIR/run_causal_mechanistic.slurm"
DAS_ARRAY_CONCURRENCY="${DAS_ARRAY_CONCURRENCY:-2}"
job_count=$(python -c 'import json,os; from pathlib import Path; p=Path(os.environ.get("CAUSAL_OUTPUT","artifacts/causal_mech_v1"))/"representations/search_jobs.json"; print(len(json.loads(p.read_text())))')

if [ "$job_count" -eq 0 ]; then
  echo "No layer passed whole-state localization; generating the Level-3 stop report."
  python scripts/causal_mechanistic.py --config "${CONFIG:-configs/causal_mech_v1.yaml}" \
    --output "$CAUSAL_OUTPUT" --phase report
  exit 0
fi

das_submission=$(sbatch --parsable --job-name=cog_cf_das \
  --array="0-$((job_count - 1))%${DAS_ARRAY_CONCURRENCY}" \
  --dependency="afterok:${SLURM_JOB_ID}" --export=ALL,PHASE=das "$GPU_SCRIPT")
das_job="${das_submission%%;*}"
select_submission=$(sbatch --parsable --job-name=cog_cf_select --time=01:00:00 \
  --dependency="afterok:${das_job}" --export=ALL,PHASE=select "$CPU_SCRIPT")
select_job="${select_submission%%;*}"
dispatch_submission=$(sbatch --parsable --job-name=cog_cf_validate_dispatch --time=00:15:00 \
  --dependency="afterok:${select_job}" --export=ALL,PHASE=dispatch_validate "$CPU_SCRIPT")
dispatch_job="${dispatch_submission%%;*}"
echo "DAS continuation submitted: DAS=${das_job}, select=${select_job}, validation-dispatch=${dispatch_job}"

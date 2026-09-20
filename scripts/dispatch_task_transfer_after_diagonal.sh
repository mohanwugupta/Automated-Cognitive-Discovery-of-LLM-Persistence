#!/bin/bash
set -euo pipefail

REPO=${REPO:?export REPO}
TRANSFER_OUTPUT=${TRANSFER_OUTPUT:?export TRANSFER_OUTPUT}
TRANSFER_ACTIVE_INDICES=${TRANSFER_ACTIVE_INDICES:?active diagonal array indices are required}
GPU_SCRIPT="$REPO/slurm/run_task_transfer_gpu.slurm"
CPU_SCRIPT="$REPO/slurm/run_task_transfer_cpu.slurm"
DISPATCH_JOB_ID=${SLURM_JOB_ID:?dispatcher must run under Slurm}

cd "$REPO"
python scripts/task_transfer.py gate-diagonal --output "$TRANSFER_OUTPUT" \
  --indices "$TRANSFER_ACTIVE_INDICES" --execute
eligible=$(python -c '
import json, os
value = json.load(open(os.path.join(os.environ["TRANSFER_OUTPUT"], "diagonal_gate.json")))
indices = value["active_array_indices"] if os.environ.get("TRANSFER_SMOKE") == "1" else value["eligible_array_indices"]
print(",".join(map(str, indices)))
')

if [ -n "$eligible" ]; then
  offdiag_submission=$(sbatch --parsable \
    --dependency="afterok:${DISPATCH_JOB_ID}" \
    --array="${eligible}%${TRANSFER_MAX_CONCURRENT:-2}" \
    --export="ALL,TRANSFER_MODE=evaluate,TRANSFER_EVALUATION_SCOPE=off_diagonal" \
    "$GPU_SCRIPT")
  offdiag="${offdiag_submission%%;*}"
  dependency="afterany:${offdiag}"
else
  offdiag="none-all-sources-unavailable"
  dependency="afterok:${DISPATCH_JOB_ID}"
fi

aggregate_submission=$(sbatch --parsable --dependency="$dependency" \
  --export="ALL,TRANSFER_CPU_MODE=aggregate" "$CPU_SCRIPT")
aggregate="${aggregate_submission%%;*}"
printf 'eligible_array_indices=%s\noff_diagonal=%s\naggregate=%s\n' \
  "$eligible" "$offdiag" "$aggregate"

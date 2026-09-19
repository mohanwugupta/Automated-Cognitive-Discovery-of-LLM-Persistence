#!/bin/bash
#SBATCH --job-name=cog_transfer_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cognitive_transfer_aggregate_%j.out
#SBATCH --error=logs/cognitive_transfer_aggregate_%j.err
set -euo pipefail

REPO=${REPO:?export REPO}
CONDA_ENV=${CONDA_ENV:-llm-cognitive-discovery}
TRANSFER_OUTPUT=${TRANSFER_OUTPUT:?export TRANSFER_OUTPUT}
TRANSFER_CPU_MODE=${TRANSFER_CPU_MODE:-aggregate}
cd "$REPO"
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  echo "CPU aggregation unexpectedly reserved GPU(s): $SLURM_JOB_GPUS" >&2
  exit 3
fi
set +u
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
case "$TRANSFER_CPU_MODE" in
  dispatch_diagonal)
    bash scripts/dispatch_task_transfer_after_diagonal.sh
    ;;
  aggregate)
    EXTRA=()
    if [[ "${TRANSFER_ALLOW_INCOMPLETE:-0}" == 1 ]]; then EXTRA+=(--allow-incomplete); fi
    python scripts/task_transfer.py aggregate --output "$TRANSFER_OUTPUT" "${EXTRA[@]}" --execute
    ;;
  *)
    echo "Unknown TRANSFER_CPU_MODE=$TRANSFER_CPU_MODE" >&2
    exit 2
    ;;
esac

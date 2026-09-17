#!/bin/bash
#SBATCH --job-name=cog_rep_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cognitive_replication_cpu_%j.out
#SBATCH --error=logs/cognitive_replication_cpu_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
STAGE="${STAGE:?STAGE is required}"
case "$STAGE" in
  model_comparison|freeze_theory|counterfactuals|report) ;;
  *) echo "STAGE=$STAGE is not a CPU-only replication stage" >&2; exit 2 ;;
esac
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
  echo "CPU-only stage unexpectedly reserved GPU(s): $SLURM_JOB_GPUS" >&2
  exit 3
fi

cd "$PROJECT_DIR"
# Princeton's module/Conda shell initialization reads interactive variables
# such as PS1, which are unset in Slurm batch shells. Limit nounset relaxation
# to environment activation, then restore fail-fast behavior immediately.
set +u
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
python -m cognitive_discovery.replicate_model \
  --resume "$OUTPUT" --stage "$STAGE" --execute

#!/bin/bash
#SBATCH --job-name=cog_model_validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cognitive_model_validation_%j.out
#SBATCH --error=logs/cognitive_model_validation_%j.err
set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CONDA_ENV=${CONDA_ENV:-llm-cognitive-discovery}
cd "$REPO"
mkdir -p logs
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  echo "CPU validation unexpectedly reserved GPU(s): $SLURM_JOB_GPUS" >&2
  exit 3
fi
set +u
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/validate_computational_models.py --root "$REPO" --execute

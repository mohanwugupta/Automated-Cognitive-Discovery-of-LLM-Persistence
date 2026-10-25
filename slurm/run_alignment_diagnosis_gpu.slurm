#!/bin/bash
#SBATCH --job-name=cog_diagnosis_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=02:00:00
#SBATCH --output=logs/cognitive_diagnosis_gpu_%j.out
#SBATCH --error=logs/cognitive_diagnosis_gpu_%j.err

set -euo pipefail
REPO=${REPO:?REPO is required}
DIAGNOSIS_OUTPUT=${DIAGNOSIS_OUTPUT:?DIAGNOSIS_OUTPUT is required}
DIAGNOSIS_PHASE=${DIAGNOSIS_PHASE:?DIAGNOSIS_PHASE is required}
CONDA_ENV=${CONDA_ENV:-llm-cognitive-discovery}
cd "$REPO"
set +u
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
case "$DIAGNOSIS_PHASE" in
  initial)
    python scripts/alignment_diagnosis.py collect --output "$DIAGNOSIS_OUTPUT" --phase main --execute
    python scripts/alignment_diagnosis.py collect --output "$DIAGNOSIS_OUTPUT" --phase calibration --execute
    ;;
  intervention)
    python scripts/alignment_diagnosis.py collect --output "$DIAGNOSIS_OUTPUT" --phase intervention --execute
    ;;
  *) echo "unknown GPU phase: $DIAGNOSIS_PHASE" >&2; exit 2 ;;
esac

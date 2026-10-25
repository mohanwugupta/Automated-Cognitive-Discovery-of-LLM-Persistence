#!/bin/bash
#SBATCH --job-name=cog_diagnosis_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=00:55:00
#SBATCH --output=logs/cognitive_diagnosis_gpu_%j.out
#SBATCH --error=logs/cognitive_diagnosis_gpu_%j.err

set -euo pipefail
REPO=${REPO:?REPO is required}
DIAGNOSIS_OUTPUT=${DIAGNOSIS_OUTPUT:?DIAGNOSIS_OUTPUT is required}
DIAGNOSIS_PHASE=${DIAGNOSIS_PHASE:?DIAGNOSIS_PHASE is required}
CONDA_ENV=${CONDA_ENV:-llm-cognitive-discovery}
MODEL_PATH=${DIAGNOSIS_MODEL_PATH:-${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}}
[[ -f "$MODEL_PATH/config.json" ]] || { echo "local Qwen model lacks config.json: $MODEL_PATH" >&2; exit 2; }
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
SHARD_ARGS=(--shard-count "${DIAGNOSIS_SHARDS:-1}" --shard-index "${SLURM_ARRAY_TASK_ID:-0}")
case "$DIAGNOSIS_PHASE" in
  initial)
    python scripts/alignment_diagnosis.py collect --output "$DIAGNOSIS_OUTPUT" --phase main --model-path "$MODEL_PATH" --execute "${SHARD_ARGS[@]}"
    python scripts/alignment_diagnosis.py collect --output "$DIAGNOSIS_OUTPUT" --phase calibration --model-path "$MODEL_PATH" --execute "${SHARD_ARGS[@]}"
    ;;
  intervention)
    python scripts/alignment_diagnosis.py collect --output "$DIAGNOSIS_OUTPUT" --phase intervention --model-path "$MODEL_PATH" --execute "${SHARD_ARGS[@]}"
    ;;
  *) echo "unknown GPU phase: $DIAGNOSIS_PHASE" >&2; exit 2 ;;
esac

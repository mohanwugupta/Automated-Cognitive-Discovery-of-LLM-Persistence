#!/bin/bash
#SBATCH --job-name=cog_cross_task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_cross_task_%A_%a.out
#SBATCH --error=logs/cognitive_cross_task_%A_%a.err

set -euo pipefail
REPO=${REPO:?REPO is required}
CROSS_TASK_OUTPUT=${CROSS_TASK_OUTPUT:?CROSS_TASK_OUTPUT is required}
CROSS_TASK_MODEL_ROOT=${CROSS_TASK_MODEL_ROOT:?CROSS_TASK_MODEL_ROOT is required}
CROSS_TASK_MODE=${CROSS_TASK_MODE:?CROSS_TASK_MODE is required}
CONDA_ENV=${CONDA_ENV:-llm-cognitive-discovery}

case "${SLURM_ARRAY_TASK_ID:?array index is required}" in
  0) model=qwen; directory=Qwen--Qwen3.5-4B ;;
  1) model=gemma; directory=google--gemma-4-12b-it ;;
  2) model=llama; directory=meta-llama--Llama-3.1-8B-Instruct ;;
  *) echo "invalid model array index" >&2; exit 2 ;;
esac
model_path="$CROSS_TASK_MODEL_ROOT/$directory"
[[ -f "$model_path/config.json" ]] || { echo "missing model: $model_path" >&2; exit 2; }

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
python -c 'import torch; assert torch.cuda.is_available(); print("CUDA preflight:", torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

smoke_args=()
if [[ "${CROSS_TASK_SMOKE:-0}" == 1 ]]; then smoke_args+=(--smoke); fi
case "$CROSS_TASK_MODE" in
  interface)
    python scripts/cross_task_discovery.py interface --output "$CROSS_TASK_OUTPUT" \
      --model "$model" --model-path "$model_path" "${smoke_args[@]}"
    ;;
  behavior)
    python scripts/cross_task_discovery.py collect-behavior --output "$CROSS_TASK_OUTPUT" \
      --model "$model" --model-path "$model_path" "${smoke_args[@]}"
    ;;
  *) echo "unknown CROSS_TASK_MODE=$CROSS_TASK_MODE" >&2; exit 2 ;;
esac

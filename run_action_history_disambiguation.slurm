#!/bin/bash
#SBATCH --job-name=cog_action_hist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_action_history_%A_%a.out
#SBATCH --error=logs/cognitive_action_history_%A_%a.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/action_history_disambiguation_v1.yaml}"
SOURCE_OUTPUT="${SOURCE_OUTPUT:-artifacts/mechanistic_v1}"
ACTION_OUTPUT="${ACTION_OUTPUT:-artifacts/action_history_disambiguation_v1}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
ACTION_SCRATCH="${ACTION_SCRATCH:-/scratch/gpfs/JORDANAT/${USER}/persistence_action_history}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
PHASE="${PHASE:-tests}"
RANDOM_SHARD_COUNT="${RANDOM_SHARD_COUNT:-10}"
RANDOM_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

cd "$PROJECT_DIR"
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ACTION_SCRATCH/model_cache"
export HF_DATASETS_CACHE="$ACTION_SCRATCH/model_cache/datasets"
export TRANSFORMERS_CACHE="$ACTION_SCRATCH/model_cache"
export TORCH_HOME="$ACTION_SCRATCH/model_cache/torch"
export XDG_CACHE_HOME="$ACTION_SCRATCH/model_cache/cache"
export MPLCONFIGDIR="$ACTION_SCRATCH/model_cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$ACTION_SCRATCH/temporary_batches" "$HF_HOME" \
  "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
JOB_SCRATCH=$(mktemp -d "$ACTION_SCRATCH/temporary_batches/job-XXXXXX")
cleanup() {
  rm -rf -- "$JOB_SCRATCH"
}
trap cleanup EXIT

case "$PHASE" in
  tests|prepare|analyze|report)
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
      echo "PHASE=$PHASE is CPU-only but this job reserved GPU(s): $SLURM_JOB_GPUS" >&2
      echo "Submit it through run_action_history_cpu.slurm." >&2
      exit 3
    fi
    ;;
  fit|project|steer|random|patch)
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA preflight:", torch.cuda.get_device_name(0))'
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac

COMMON=(--config "$CONFIG" --output "$ACTION_OUTPUT" --source "$SOURCE_OUTPUT")
case "$PHASE" in
  tests)
    python -m pytest -q
    python scripts/check_artifact_sizes.py
    ;;
  prepare)
    python scripts/action_history_disambiguation.py "${COMMON[@]}" --phase prepare
    ;;
  fit)
    python scripts/action_history_disambiguation.py "${COMMON[@]}" --phase fit --model "$MODEL_PATH"
    ;;
  project)
    python scripts/action_history_disambiguation.py "${COMMON[@]}" --phase project --model "$MODEL_PATH"
    ;;
  analyze)
    python scripts/action_history_disambiguation.py "${COMMON[@]}" --phase analyze
    ;;
  steer)
    python scripts/action_history_disambiguation.py "${COMMON[@]}" --phase steer --model "$MODEL_PATH"
    ;;
  random)
    python scripts/action_history_disambiguation.py "${COMMON[@]}" --phase random \
      --model "$MODEL_PATH" --shard-index "$RANDOM_SHARD_INDEX" \
      --shard-count "$RANDOM_SHARD_COUNT"
    ;;
  patch)
    python scripts/action_history_disambiguation.py "${COMMON[@]}" --phase patch --model "$MODEL_PATH"
    ;;
  report)
    python scripts/action_history_disambiguation.py "${COMMON[@]}" --phase report
    python scripts/check_artifact_sizes.py
    ;;
esac

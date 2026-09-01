#!/bin/bash
#SBATCH --job-name=cog_specificity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_specificity_%A_%a.out
#SBATCH --error=logs/cognitive_specificity_%A_%a.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/causal_specificity_v2.yaml}"
SOURCE_OUTPUT="${SOURCE_OUTPUT:-artifacts/causal_mech_v1}"
SPECIFICITY_OUTPUT="${SPECIFICITY_OUTPUT:-artifacts/causal_specificity_v2}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
SPECIFICITY_SCRATCH="${SPECIFICITY_SCRATCH:-/scratch/gpfs/JORDANAT/${USER}/persistence_causal_specificity}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
PHASE="${PHASE:-tests}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export CONFIG SOURCE_OUTPUT SPECIFICITY_OUTPUT MODEL_PATH SPECIFICITY_SCRATCH CONDA_ENV

cd "$PROJECT_DIR"
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SPECIFICITY_SCRATCH/model_cache"
export HF_DATASETS_CACHE="$SPECIFICITY_SCRATCH/model_cache/datasets"
export TRANSFORMERS_CACHE="$SPECIFICITY_SCRATCH/model_cache"
export TORCH_HOME="$SPECIFICITY_SCRATCH/model_cache/torch"
export XDG_CACHE_HOME="$SPECIFICITY_SCRATCH/model_cache/cache"
export MPLCONFIGDIR="$SPECIFICITY_SCRATCH/model_cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$SPECIFICITY_SCRATCH/temporary_batches" "$HF_HOME" \
  "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
JOB_SCRATCH=$(mktemp -d "$SPECIFICITY_SCRATCH/temporary_batches/job-XXXXXX")
cleanup() {
  rm -rf -- "$JOB_SCRATCH"
}
trap cleanup EXIT

case "$PHASE" in
  tests|prepare|dispatch|aggregate|dispatch_necessity|finalize)
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
      echo "PHASE=$PHASE is CPU-only but reserved GPU(s): $SLURM_JOB_GPUS" >&2
      echo "Submit it through run_causal_specificity_cpu.slurm." >&2
      exit 3
    fi
    ;;
  evaluate|necessity)
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA preflight:", torch.cuda.get_device_name(0))'
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac

COMMON=(--config "$CONFIG" --source "$SOURCE_OUTPUT" --output "$SPECIFICITY_OUTPUT")
case "$PHASE" in
  tests)
    python -m pytest -q
    python scripts/check_artifact_sizes.py
    ;;
  prepare)
    python scripts/causal_specificity.py "${COMMON[@]}" --phase prepare
    ;;
  dispatch)
    bash scripts/dispatch_causal_specificity.sh
    ;;
  evaluate)
    python scripts/causal_specificity.py "${COMMON[@]}" --phase evaluate \
      --job-index "$ARRAY_INDEX" --model "$MODEL_PATH"
    ;;
  aggregate)
    python scripts/causal_specificity.py "${COMMON[@]}" --phase aggregate
    ;;
  dispatch_necessity)
    bash scripts/dispatch_causal_necessity.sh
    ;;
  necessity)
    python scripts/causal_specificity.py "${COMMON[@]}" --phase necessity \
      --job-index "$ARRAY_INDEX" --model "$MODEL_PATH"
    ;;
  finalize)
    python scripts/causal_specificity.py "${COMMON[@]}" --phase finalize
    python scripts/check_artifact_sizes.py
    ;;
esac

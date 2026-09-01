#!/bin/bash
#SBATCH --job-name=cog_ood_generation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=06:00:00
#SBATCH --output=logs/cognitive_ood_%A_%a.out
#SBATCH --error=logs/cognitive_ood_%A_%a.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/ood_free_generation_v1.yaml}"
OOD_OUTPUT="${OOD_OUTPUT:-artifacts/ood_free_generation_v1}"
ABSTRACTION_OUTPUT="${ABSTRACTION_OUTPUT:-artifacts/abstraction_discovery_v1}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
OOD_SCRATCH="${OOD_SCRATCH:-/scratch/gpfs/JORDANAT/${USER}/persistence_ood_generation}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
PHASE="${PHASE:-tests}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export CONFIG OOD_OUTPUT ABSTRACTION_OUTPUT MODEL_PATH OOD_SCRATCH CONDA_ENV

cd "$PROJECT_DIR"
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$OOD_SCRATCH/model_cache"
export HF_DATASETS_CACHE="$OOD_SCRATCH/model_cache/datasets"
export TRANSFORMERS_CACHE="$OOD_SCRATCH/model_cache"
export TORCH_HOME="$OOD_SCRATCH/model_cache/torch"
export XDG_CACHE_HOME="$OOD_SCRATCH/model_cache/cache"
export MPLCONFIGDIR="$OOD_SCRATCH/model_cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$OOD_SCRATCH/temporary_batches" "$HF_HOME" \
  "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
JOB_SCRATCH=$(mktemp -d "$OOD_SCRATCH/temporary_batches/job-XXXXXX")
cleanup() {
  rm -rf -- "$JOB_SCRATCH"
}
trap cleanup EXIT

case "$PHASE" in
  tests|prepare|dispatch|aggregate)
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
      echo "PHASE=$PHASE is CPU-only but reserved GPU(s): $SLURM_JOB_GPUS" >&2
      echo "Submit it through run_ood_generation_cpu.slurm." >&2
      exit 3
    fi
    ;;
  evaluate)
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA preflight:", torch.cuda.get_device_name(0))'
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac

COMMON=(--config "$CONFIG" --output "$OOD_OUTPUT")
case "$PHASE" in
  tests)
    python -m pytest -q
    python scripts/check_artifact_sizes.py
    ;;
  prepare)
    python scripts/ood_free_generation.py "${COMMON[@]}" --phase prepare \
      --abstraction-output "$ABSTRACTION_OUTPUT"
    ;;
  dispatch)
    bash scripts/dispatch_ood_generation.sh
    ;;
  evaluate)
    python scripts/ood_free_generation.py "${COMMON[@]}" --phase evaluate \
      --job-index "$ARRAY_INDEX" --model "$MODEL_PATH"
    ;;
  aggregate)
    python scripts/ood_free_generation.py "${COMMON[@]}" --phase aggregate
    python scripts/check_artifact_sizes.py
    ;;
esac

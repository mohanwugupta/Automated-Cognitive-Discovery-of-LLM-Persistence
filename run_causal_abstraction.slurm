#!/bin/bash
#SBATCH --job-name=cog_abstraction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=24:00:00
#SBATCH --output=logs/cognitive_abstraction_%A_%a.out
#SBATCH --error=logs/cognitive_abstraction_%A_%a.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/causal_abstraction_v1.yaml}"
ABSTRACTION_OUTPUT="${ABSTRACTION_OUTPUT:-artifacts/abstraction_discovery_v1}"
SPECIFICITY_OUTPUT="${SPECIFICITY_OUTPUT:-artifacts/causal_specificity_v2}"
THEORY_OUTPUT="${THEORY_OUTPUT:-artifacts/theory_resolution_v1}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
ABSTRACTION_SCRATCH="${ABSTRACTION_SCRATCH:-/scratch/gpfs/JORDANAT/${USER}/persistence_causal_abstraction}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
PHASE="${PHASE:-tests}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export CONFIG ABSTRACTION_OUTPUT SPECIFICITY_OUTPUT THEORY_OUTPUT MODEL_PATH
export ABSTRACTION_SCRATCH CONDA_ENV

cd "$PROJECT_DIR"
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ABSTRACTION_SCRATCH/model_cache"
export HF_DATASETS_CACHE="$ABSTRACTION_SCRATCH/model_cache/datasets"
export TRANSFORMERS_CACHE="$ABSTRACTION_SCRATCH/model_cache"
export TORCH_HOME="$ABSTRACTION_SCRATCH/model_cache/torch"
export XDG_CACHE_HOME="$ABSTRACTION_SCRATCH/model_cache/cache"
export MPLCONFIGDIR="$ABSTRACTION_SCRATCH/model_cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$ABSTRACTION_SCRATCH/temporary_batches" "$HF_HOME" \
  "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
JOB_SCRATCH=$(mktemp -d "$ABSTRACTION_SCRATCH/temporary_batches/job-XXXXXX")
cleanup() {
  rm -rf -- "$JOB_SCRATCH"
}
trap cleanup EXIT

case "$PHASE" in
  tests|prepare|dispatch|aggregate)
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
      echo "PHASE=$PHASE is CPU-only but reserved GPU(s): $SLURM_JOB_GPUS" >&2
      echo "Submit it through run_causal_abstraction_cpu.slurm." >&2
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

COMMON=(--config "$CONFIG" --output "$ABSTRACTION_OUTPUT")
case "$PHASE" in
  tests)
    python -m pytest -q
    python scripts/check_artifact_sizes.py
    ;;
  prepare)
    python scripts/causal_abstraction.py "${COMMON[@]}" --phase prepare \
      --specificity-output "$SPECIFICITY_OUTPUT" --theory-output "$THEORY_OUTPUT"
    ;;
  dispatch)
    bash scripts/dispatch_causal_abstraction.sh
    ;;
  evaluate)
    python scripts/causal_abstraction.py "${COMMON[@]}" --phase evaluate \
      --job-index "$ARRAY_INDEX" --model "$MODEL_PATH"
    ;;
  aggregate)
    python scripts/causal_abstraction.py "${COMMON[@]}" --phase aggregate
    python scripts/check_artifact_sizes.py
    ;;
esac

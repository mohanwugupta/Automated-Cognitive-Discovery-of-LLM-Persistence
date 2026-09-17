#!/bin/bash
#SBATCH --job-name=cog_causal_mech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_causal_mech_%A_%a.out
#SBATCH --error=logs/cognitive_causal_mech_%A_%a.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/causal_mech_v1.yaml}"
THEORY_OUTPUT="${THEORY_OUTPUT:-artifacts/theory_resolution_v1}"
MECHANISTIC_OUTPUT="${MECHANISTIC_OUTPUT:-artifacts/mechanistic_v1}"
CAUSAL_OUTPUT="${CAUSAL_OUTPUT:-artifacts/causal_mech_v1}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
CAUSAL_SCRATCH="${CAUSAL_SCRATCH:-/scratch/gpfs/JORDANAT/${USER}/persistence_causal_mech}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
PHASE="${PHASE:-tests}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export CONFIG THEORY_OUTPUT MECHANISTIC_OUTPUT CAUSAL_OUTPUT MODEL_PATH CAUSAL_SCRATCH CONDA_ENV

cd "$PROJECT_DIR"
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CAUSAL_SCRATCH/model_cache"
export HF_DATASETS_CACHE="$CAUSAL_SCRATCH/model_cache/datasets"
export TRANSFORMERS_CACHE="$CAUSAL_SCRATCH/model_cache"
export TORCH_HOME="$CAUSAL_SCRATCH/model_cache/torch"
export XDG_CACHE_HOME="$CAUSAL_SCRATCH/model_cache/cache"
export MPLCONFIGDIR="$CAUSAL_SCRATCH/model_cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$CAUSAL_SCRATCH/temporary_batches" "$HF_HOME" \
  "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
JOB_SCRATCH=$(mktemp -d "$CAUSAL_SCRATCH/temporary_batches/job-XXXXXX")
cleanup() {
  rm -rf -- "$JOB_SCRATCH"
}
trap cleanup EXIT

case "$PHASE" in
  tests|prepare|diagnostic|summarize|select|report|dispatch_das|dispatch_validate)
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
      echo "PHASE=$PHASE is CPU-only but reserved GPU(s): $SLURM_JOB_GPUS" >&2
      echo "Submit it through run_causal_mechanistic_cpu.slurm." >&2
      exit 3
    fi
    ;;
  localize|das|validate|circuit)
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA preflight:", torch.cuda.get_device_name(0))'
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac

COMMON=(--config "$CONFIG" --output "$CAUSAL_OUTPUT" \
  --theory-output "$THEORY_OUTPUT" --mechanistic-output "$MECHANISTIC_OUTPUT")
case "$PHASE" in
  tests)
    python -m pytest -q
    python scripts/check_artifact_sizes.py
    ;;
  prepare)
    python scripts/causal_mechanistic.py "${COMMON[@]}" --phase prepare
    ;;
  diagnostic)
    python scripts/causal_mechanistic.py "${COMMON[@]}" --phase diagnostic
    ;;
  localize)
    python scripts/causal_mechanistic.py "${COMMON[@]}" --phase localize \
      --layer "$ARRAY_INDEX" --model "$MODEL_PATH"
    ;;
  summarize)
    python scripts/causal_mechanistic.py "${COMMON[@]}" --phase summarize
    ;;
  das)
    python scripts/causal_mechanistic.py "${COMMON[@]}" --phase das \
      --job-index "$ARRAY_INDEX" --model "$MODEL_PATH"
    ;;
  select)
    python scripts/causal_mechanistic.py "${COMMON[@]}" --phase select
    ;;
  validate)
    python scripts/causal_mechanistic.py "${COMMON[@]}" --phase validate --model "$MODEL_PATH"
    ;;
  circuit)
    python scripts/causal_mechanistic.py "${COMMON[@]}" --phase circuit --model "$MODEL_PATH"
    ;;
  report)
    python scripts/causal_mechanistic.py "${COMMON[@]}" --phase report
    python scripts/check_artifact_sizes.py
    ;;
  dispatch_das)
    bash scripts/dispatch_causal_das.sh
    ;;
  dispatch_validate)
    bash scripts/dispatch_causal_validation.sh
    ;;
esac

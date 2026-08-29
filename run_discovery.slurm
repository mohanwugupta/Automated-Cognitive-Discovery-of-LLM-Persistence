#!/bin/bash
#SBATCH --job-name=cog_discovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_discovery_%A_%a.out
#SBATCH --error=logs/cognitive_discovery_%A_%a.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/discovery_v1.yaml}"
OUTPUT="${OUTPUT:-artifacts/discovery_v1}"
PILOT_OUTPUT="${PILOT_OUTPUT:-artifacts/pilot_v1}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gpfs/JORDANAT/${USER}/llm-cognitive-discovery}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
PHASE="${PHASE:-tests}"
SHARD_COUNT="${SHARD_COUNT:-8}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
PILOT_CONDITIONS="${PILOT_CONDITIONS:-490}"
DISCOVERY_CONDITIONS="${DISCOVERY_CONDITIONS:-2800}"
VALIDATION_CONDITIONS="${VALIDATION_CONDITIONS:-700}"

cd "$PROJECT_DIR"
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH_ROOT/hf_cache"
export TORCH_HOME="$SCRATCH_ROOT/torch_cache"
export XDG_CACHE_HOME="$SCRATCH_ROOT/cache"
export MPLCONFIGDIR="$SCRATCH_ROOT/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

case "$PHASE" in
  tests)
    python -m pytest -q
    ;;
  design)
    cognitive-discovery-design --config "$CONFIG" --output "$OUTPUT" --conditions "$DISCOVERY_CONDITIONS"
    ;;
  pilot_collect)
    cognitive-discovery-run --config "$CONFIG" --output "$PILOT_OUTPUT" \
      --phase collect --conditions "$PILOT_CONDITIONS" \
      --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX" --model "$MODEL_PATH" --resume
    ;;
  pilot_finalize)
    cognitive-discovery-run --config "$CONFIG" --output "$PILOT_OUTPUT" \
      --phase finalize --conditions "$PILOT_CONDITIONS"
    python -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data["approved"] else 1)' \
      "$PILOT_OUTPUT/validation/pilot_approval.json"
    ;;
  discovery_collect)
    python -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data["approved"] else 1)' \
      "$PILOT_OUTPUT/validation/pilot_approval.json"
    cognitive-discovery-run --config "$CONFIG" --output "$OUTPUT" \
      --phase collect --conditions "$DISCOVERY_CONDITIONS" \
      --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX" --model "$MODEL_PATH" --resume
    ;;
  discovery_finalize)
    cognitive-discovery-run --config "$CONFIG" --output "$OUTPUT" \
      --phase finalize --conditions "$DISCOVERY_CONDITIONS"
    ;;
  analyze)
    cognitive-discovery-fit --config "$CONFIG" --output "$OUTPUT"
    ;;
  validation)
    python scripts/validate_predictions.py --config "$CONFIG" --output "$OUTPUT" \
      --conditions "$VALIDATION_CONDITIONS" --model "$MODEL_PATH"
    ;;
  smoke)
    cognitive-discovery-run --config "$CONFIG" --output artifacts/smoke_v1 \
      --phase collect --conditions 70 --model-free
    cognitive-discovery-run --config "$CONFIG" --output artifacts/smoke_v1 \
      --phase finalize --conditions 70
    cognitive-discovery-fit --config "$CONFIG" --output artifacts/smoke_v1 --smoke
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac

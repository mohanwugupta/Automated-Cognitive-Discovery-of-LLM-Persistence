#!/bin/bash
#SBATCH --job-name=cog_active_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_active_%A_%a.out
#SBATCH --error=logs/cognitive_active_%A_%a.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/discovery_v2.yaml}"
ROUND1_OUTPUT="${ROUND1_OUTPUT:-artifacts/discovery_v1}"
V2_OUTPUT="${V2_OUTPUT:-artifacts/discovery_v2}"
ACTIVE_OUTPUT="${ACTIVE_OUTPUT:-${V2_OUTPUT}/active_collection}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${V2_OUTPUT}/final_collection}"
ACTIVE_MANIFEST="${ACTIVE_MANIFEST:-${V2_OUTPUT}/active_sampling/selected_conditions.jsonl}"
FINAL_MANIFEST="${FINAL_MANIFEST:-${V2_OUTPUT}/final_validation/condition_manifest.jsonl}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gpfs/JORDANAT/${USER}/llm-cognitive-discovery}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
PHASE="${PHASE:-tests}"
SHARD_COUNT="${SHARD_COUNT:-8}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"

cd "$PROJECT_DIR"
module purge
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
  prepare)
    python scripts/active_discovery.py --config "$CONFIG" --output "$V2_OUTPUT" \
      --round1-output "$ROUND1_OUTPUT" --phase prepare
    ;;
  active_collect)
    python scripts/run_experiments.py --config "$CONFIG" --output "$ACTIVE_OUTPUT" \
      --phase collect --manifest "$ACTIVE_MANIFEST" \
      --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX" \
      --model "$MODEL_PATH" --resume
    ;;
  active_finalize)
    python scripts/run_experiments.py --config "$CONFIG" --output "$ACTIVE_OUTPUT" \
      --phase finalize --manifest "$ACTIVE_MANIFEST"
    python -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data["approved"] else 1)' \
      "$ACTIVE_OUTPUT/validation/pilot_approval.json"
    ;;
  update)
    python scripts/active_discovery.py --config "$CONFIG" --output "$V2_OUTPUT" \
      --round1-output "$ROUND1_OUTPUT" --active-output "$ACTIVE_OUTPUT" --phase update
    ;;
  final_collect)
    python scripts/run_experiments.py --config "$CONFIG" --output "$FINAL_OUTPUT" \
      --phase collect --manifest "$FINAL_MANIFEST" \
      --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX" \
      --model "$MODEL_PATH" --resume
    ;;
  final_finalize)
    python scripts/run_experiments.py --config "$CONFIG" --output "$FINAL_OUTPUT" \
      --phase finalize --manifest "$FINAL_MANIFEST"
    ;;
  evaluate)
    python scripts/active_discovery.py --config "$CONFIG" --output "$V2_OUTPUT" \
      --round1-output "$ROUND1_OUTPUT" --active-output "$ACTIVE_OUTPUT" \
      --final-output "$FINAL_OUTPUT" --phase evaluate
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac

#!/bin/bash
#SBATCH --job-name=cog_theory_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_theory_%A_%a.out
#SBATCH --error=logs/cognitive_theory_%A_%a.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/theory_resolution_v1.yaml}"
ROUND1_OUTPUT="${ROUND1_OUTPUT:-artifacts/discovery_v1}"
ROUND2_OUTPUT="${ROUND2_OUTPUT:-artifacts/discovery_v2}"
V2_ACTIVE_OUTPUT="${V2_ACTIVE_OUTPUT:-${ROUND2_OUTPUT}/active_collection}"
V3_OUTPUT="${V3_OUTPUT:-artifacts/theory_resolution_v1}"
ROUND3_OUTPUT="${ROUND3_OUTPUT:-${V3_OUTPUT}/round3_collection}"
ROUND3_MANIFEST="${ROUND3_MANIFEST:-${V3_OUTPUT}/active_sampling/selected_conditions.jsonl}"
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
  tests|finalize|evaluate)
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
      echo "PHASE=$PHASE is CPU-only but this job reserved GPU(s): $SLURM_JOB_GPUS" >&2
      echo "Submit it through run_theory_cpu.slurm." >&2
      exit 3
    fi
    ;;
esac

case "$PHASE" in
  prepare|collect)
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in a GPU phase"; print("CUDA preflight:", torch.cuda.get_device_name(0))'
    ;;
esac

case "$PHASE" in
  tests)
    python -m pytest -q
    ;;
  prepare)
    python scripts/theory_resolution.py --config "$CONFIG" --output "$V3_OUTPUT" \
      --round1-output "$ROUND1_OUTPUT" --active-output "$V2_ACTIVE_OUTPUT" \
      --round2-output "$ROUND2_OUTPUT" --phase prepare
    ;;
  collect)
    python scripts/run_experiments.py --config "$CONFIG" --output "$ROUND3_OUTPUT" \
      --phase collect --manifest "$ROUND3_MANIFEST" \
      --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX" \
      --model "$MODEL_PATH" --resume
    ;;
  finalize)
    python scripts/run_experiments.py --config "$CONFIG" --output "$ROUND3_OUTPUT" \
      --phase finalize --manifest "$ROUND3_MANIFEST"
    python -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data["approved"] else 1)' \
      "$ROUND3_OUTPUT/validation/pilot_approval.json"
    ;;
  evaluate)
    python scripts/theory_resolution.py --config "$CONFIG" --output "$V3_OUTPUT" \
      --round1-output "$ROUND1_OUTPUT" --active-output "$V2_ACTIVE_OUTPUT" \
      --round3-output "$ROUND3_OUTPUT" --phase evaluate
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac

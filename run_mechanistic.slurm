#!/bin/bash
#SBATCH --job-name=cog_mech_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_mech_%A.out
#SBATCH --error=logs/cognitive_mech_%A.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/mechanistic_v1.yaml}"
THEORY_OUTPUT="${THEORY_OUTPUT:-artifacts/theory_resolution_v1}"
MECH_OUTPUT="${MECH_OUTPUT:-artifacts/mechanistic_v1}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
MECH_SCRATCH="${MECH_SCRATCH:-/scratch/gpfs/JORDANAT/${USER}/persistence_mech}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
PHASE="${PHASE:-tests}"

cd "$PROJECT_DIR"
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MECH_SCRATCH
export HF_HOME="$MECH_SCRATCH/model_cache"
export HF_DATASETS_CACHE="$MECH_SCRATCH/model_cache/datasets"
export TRANSFORMERS_CACHE="$MECH_SCRATCH/model_cache"
export TORCH_HOME="$MECH_SCRATCH/model_cache/torch"
export XDG_CACHE_HOME="$MECH_SCRATCH/model_cache/cache"
export MPLCONFIGDIR="$MECH_SCRATCH/model_cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$MECH_SCRATCH/temporary_activations" \
  "$MECH_SCRATCH/temporary_batches" "$HF_HOME" \
  "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
JOB_SCRATCH=$(mktemp -d "$MECH_SCRATCH/temporary_batches/job-XXXXXX")
export MECH_JOB_SCRATCH="$JOB_SCRATCH"
cleanup() {
  rm -rf -- "$JOB_SCRATCH"
}
trap cleanup EXIT

case "$PHASE" in
  tests|prepare|analyze|report)
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
      echo "PHASE=$PHASE is CPU-only but this job reserved GPU(s): $SLURM_JOB_GPUS" >&2
      echo "Submit it through run_mechanistic_cpu.slurm." >&2
      exit 3
    fi
    ;;
  scan|project|intervene)
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in a GPU phase"; print("CUDA preflight:", torch.cuda.get_device_name(0))'
    ;;
esac

case "$PHASE" in
  tests)
    python -m pytest -q
    python scripts/check_artifact_sizes.py
    ;;
  prepare)
    python scripts/mechanistic.py --config "$CONFIG" --output "$MECH_OUTPUT" \
      --theory-output "$THEORY_OUTPUT" --phase prepare
    ;;
  scan)
    python scripts/mechanistic.py --config "$CONFIG" --output "$MECH_OUTPUT" \
      --theory-output "$THEORY_OUTPUT" --phase scan --model "$MODEL_PATH"
    ;;
  project)
    python scripts/mechanistic.py --config "$CONFIG" --output "$MECH_OUTPUT" \
      --theory-output "$THEORY_OUTPUT" --phase project --model "$MODEL_PATH"
    ;;
  analyze)
    python scripts/mechanistic.py --config "$CONFIG" --output "$MECH_OUTPUT" \
      --theory-output "$THEORY_OUTPUT" --phase analyze
    ;;
  intervene)
    python scripts/mechanistic.py --config "$CONFIG" --output "$MECH_OUTPUT" \
      --theory-output "$THEORY_OUTPUT" --phase intervene --model "$MODEL_PATH"
    ;;
  report)
    python scripts/mechanistic.py --config "$CONFIG" --output "$MECH_OUTPUT" \
      --theory-output "$THEORY_OUTPUT" --phase report
    python scripts/check_artifact_sizes.py
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac

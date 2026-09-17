#!/bin/bash
#SBATCH --job-name=cog_rep_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=01:00:00
#SBATCH --output=logs/cognitive_replication_smoke_%A_%a.out
#SBATCH --error=logs/cognitive_replication_smoke_%A_%a.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
SMOKE_ROOT="${SMOKE_ROOT:?SMOKE_ROOT is required}"
SMOKE_MODEL_PATHS="${SMOKE_MODEL_PATHS:?space-separated SMOKE_MODEL_PATHS is required}"
SMOKE_REVISIONS="${SMOKE_REVISIONS:?space-separated immutable SMOKE_REVISIONS is required}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/replication/default.yaml}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array}"

read -r -a model_paths <<< "$SMOKE_MODEL_PATHS"
read -r -a revisions <<< "$SMOKE_REVISIONS"
if [ -n "${SMOKE_ADAPTERS:-}" ]; then
  read -r -a adapters <<< "$SMOKE_ADAPTERS"
else
  adapters=()
  for _ in "${model_paths[@]}"; do adapters+=(auto); done
fi
if [ -n "${SMOKE_TOKENIZER_PATHS:-}" ]; then
  read -r -a tokenizer_paths <<< "$SMOKE_TOKENIZER_PATHS"
else
  tokenizer_paths=("${model_paths[@]}")
fi
if [ -n "${SMOKE_TOKENIZER_REVISIONS:-}" ]; then
  read -r -a tokenizer_revisions <<< "$SMOKE_TOKENIZER_REVISIONS"
else
  tokenizer_revisions=("${revisions[@]}")
fi

expected="${#model_paths[@]}"
for count in "${#revisions[@]}" "${#adapters[@]}" "${#tokenizer_paths[@]}" "${#tokenizer_revisions[@]}"; do
  if [ "$count" -ne "$expected" ]; then
    echo "Smoke model, revision, adapter, and tokenizer lists must have equal lengths" >&2
    exit 2
  fi
done
if ! [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] || [ "$TASK_INDEX" -ge "$expected" ]; then
  echo "SLURM_ARRAY_TASK_ID=$TASK_INDEX is outside the model list" >&2
  exit 2
fi

cd "$PROJECT_DIR"
# Princeton's module/Conda shell initialization reads interactive variables
# such as PS1, which are unset in Slurm batch shells. Limit nounset relaxation
# to environment activation, then restore fail-fast behavior immediately.
set +u
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA preflight:", torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

printf -v job_directory '%s/job_%04d' "$SMOKE_ROOT" "$TASK_INDEX"
online_args=()
if [ "${SMOKE_ONLINE:-0}" = "1" ]; then
  online_args+=(--online)
fi

python scripts/replication_smoke.py run-model \
  --model "${model_paths[$TASK_INDEX]}" \
  --revision "${revisions[$TASK_INDEX]}" \
  --adapter "${adapters[$TASK_INDEX]}" \
  --tokenizer "${tokenizer_paths[$TASK_INDEX]}" \
  --tokenizer-revision "${tokenizer_revisions[$TASK_INDEX]}" \
  --config "$SMOKE_CONFIG" \
  --output "$job_directory" \
  --seed "$((19001 + TASK_INDEX))" \
  "${online_args[@]}"

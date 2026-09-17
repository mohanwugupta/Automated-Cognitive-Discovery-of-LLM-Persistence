#!/bin/bash
#SBATCH --job-name=cog_rep_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_replication_gpu_%j.out
#SBATCH --error=logs/cognitive_replication_gpu_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
STAGE="${STAGE:?STAGE is required}"
case "$STAGE" in
  interface|behavior|mechanism|generalization|specificity) ;;
  *) echo "STAGE=$STAGE is not a model-forward replication stage" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA preflight:", torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

online_args=()
if [ "${ONLINE_FLAG:-}" = "--online" ]; then
  online_args+=(--online)
fi
python -m cognitive_discovery.replicate_model \
  --resume "$OUTPUT" --stage "$STAGE" --execute "${online_args[@]}"

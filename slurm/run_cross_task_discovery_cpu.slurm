#!/bin/bash
#SBATCH --job-name=cog_cross_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_cross_task_cpu_%j.out
#SBATCH --error=logs/cognitive_cross_task_cpu_%j.err

set -euo pipefail
REPO=${REPO:?REPO is required}
CROSS_TASK_OUTPUT=${CROSS_TASK_OUTPUT:?CROSS_TASK_OUTPUT is required}
CONDA_ENV=${CONDA_ENV:-llm-cognitive-discovery}
cd "$REPO"
set +u
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

for model in qwen gemma llama; do
  observations="$CROSS_TASK_OUTPUT/$model/behavior/observations.parquet"
  if [[ ! -f "$observations" ]]; then
    observations="${observations%.parquet}.csv.gz"
  fi
  python scripts/cross_task_discovery.py behavior-effects --output "$CROSS_TASK_OUTPUT" \
    --model "$model" --observations "$observations"
  python scripts/cross_task_discovery.py computational --output "$CROSS_TASK_OUTPUT" \
    --model "$model" --observations "$observations"
done
echo "Behavioral/computational phase complete. Review survivor sets and freeze neural targets."

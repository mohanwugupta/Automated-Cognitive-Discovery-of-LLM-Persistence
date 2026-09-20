#!/bin/bash
#SBATCH --job-name=cog_transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig&gpu40"
#SBATCH --time=08:00:00
#SBATCH --output=logs/cognitive_transfer_%A_%a.out
#SBATCH --error=logs/cognitive_transfer_%A_%a.err
set -euo pipefail

REPO=${REPO:?export REPO}
CONDA_ENV=${CONDA_ENV:-llm-cognitive-discovery}
REPLICATION=${REPLICATION:?export REPLICATION}
TRANSFER_OUTPUT=${TRANSFER_OUTPUT:?export TRANSFER_OUTPUT}
TRANSFER_CONFIG=${TRANSFER_CONFIG:-$REPO/configs/transfer/v1.yaml}
TRANSFER_MODE=${TRANSFER_MODE:?TRANSFER_MODE must be train or evaluate}
TRANSFER_EVALUATION_SCOPE=${TRANSFER_EVALUATION_SCOPE:-all}
cd "$REPO"
mkdir -p logs
set +u
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA preflight:", torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
WORK_ID=$(python -c 'import json,os; p=json.load(open(os.path.join(os.environ["TRANSFER_OUTPUT"],"work_manifest.json"))); print(p["work_units"][int(os.environ["SLURM_ARRAY_TASK_ID"])]["work_id"])')
EXTRA=()
if [[ "${TRANSFER_SMOKE:-0}" == 1 ]]; then EXTRA+=(--smoke); fi
if [[ -n "${TRANSFER_RANDOM_COUNT-}" ]]; then EXTRA+=(--random-count "$TRANSFER_RANDOM_COUNT"); fi
if [[ "$TRANSFER_MODE" == evaluate ]]; then
  EXTRA+=(--evaluation-scope "$TRANSFER_EVALUATION_SCOPE")
fi
started_at=$(date +%s)
python scripts/task_transfer.py "$TRANSFER_MODE" \
  --replication "$REPLICATION" --output "$TRANSFER_OUTPUT" \
  --config "$TRANSFER_CONFIG" --work-id "$WORK_ID" "${EXTRA[@]}" --execute
finished_at=$(date +%s)
job_root="$TRANSFER_OUTPUT/jobs/$WORK_ID"
output_kb=$(du -sk "$job_root" | awk '{print $1}')
resource_name="$TRANSFER_MODE"
if [[ "$TRANSFER_MODE" == evaluate ]]; then
  resource_name="evaluate_${TRANSFER_EVALUATION_SCOPE}"
fi
python -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema_version": "task-transfer-resource-v1",
    "work_id": sys.argv[2],
    "mode": sys.argv[3],
    "evaluation_scope": sys.argv[4],
    "elapsed_seconds": int(sys.argv[6]) - int(sys.argv[5]),
    "output_bytes": int(sys.argv[7]) * 1024,
    "slurm_job_id": sys.argv[8],
    "gpu_count": 1,
    "smoke": sys.argv[9] == "1",
    "random_count": int(sys.argv[10]),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$job_root/resource_${resource_name}.json" "$WORK_ID" "$TRANSFER_MODE" \
  "$TRANSFER_EVALUATION_SCOPE" "$started_at" "$finished_at" "$output_kb" \
  "${SLURM_JOB_ID:-unknown}" "${TRANSFER_SMOKE:-0}" "${TRANSFER_RANDOM_COUNT:-0}"

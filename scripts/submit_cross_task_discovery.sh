#!/bin/bash
# Submit the shared-design behavioral/computational phase. Neural work is
# intentionally not submitted until the scientific freeze and compute estimate.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CROSS_TASK_OUTPUT=${CROSS_TASK_OUTPUT:?export CROSS_TASK_OUTPUT to a new scratch directory}
CROSS_TASK_CONFIG=${CROSS_TASK_CONFIG:-$REPO/configs/discovery/cross_task_v1.yaml}
CROSS_TASK_MODEL_ROOT=${CROSS_TASK_MODEL_ROOT:-/scratch/gpfs/JORDANAT/$USER/models}
CROSS_TASK_MAX_CONCURRENT=${CROSS_TASK_MAX_CONCURRENT:-1}
CROSS_TASK_SMOKE=${CROSS_TASK_SMOKE:-0}
export REPO CROSS_TASK_OUTPUT CROSS_TASK_CONFIG CROSS_TASK_MODEL_ROOT CROSS_TASK_SMOKE

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Della login node" >&2
  exit 1
fi
if [[ -e "$CROSS_TASK_OUTPUT" ]]; then
  echo "CROSS_TASK_OUTPUT already exists; never overwrite a frozen discovery run" >&2
  exit 2
fi
for directory in Qwen--Qwen3.5-4B google--gemma-4-12b-it meta-llama--Llama-3.1-8B-Instruct; do
  if [[ ! -f "$CROSS_TASK_MODEL_ROOT/$directory/config.json" ]]; then
    echo "Pinned local model is absent: $CROSS_TASK_MODEL_ROOT/$directory" >&2
    exit 2
  fi
done

cd "$REPO"
python scripts/cross_task_discovery.py initialize \
  --config "$CROSS_TASK_CONFIG" --output "$CROSS_TASK_OUTPUT" --final
design_args=()
if [[ "$CROSS_TASK_SMOKE" == 1 ]]; then design_args+=(--smoke); fi
python scripts/cross_task_discovery.py design --output "$CROSS_TASK_OUTPUT" "${design_args[@]}"

mkdir -p logs
array="0-2%${CROSS_TASK_MAX_CONCURRENT}"
interface=$(sbatch --parsable --array="$array" \
  --export="ALL,CROSS_TASK_MODE=interface" slurm/run_cross_task_discovery_gpu.slurm)
interface=${interface%%;*}
behavior=$(sbatch --parsable --dependency="afterok:$interface" --array="$array" \
  --export="ALL,CROSS_TASK_MODE=behavior" slurm/run_cross_task_discovery_gpu.slurm)
behavior=${behavior%%;*}
analysis=$(sbatch --parsable --dependency="afterok:$behavior" \
  --export=ALL slurm/run_cross_task_discovery_cpu.slurm)
analysis=${analysis%%;*}

printf 'interface=%s\nbehavior=%s\nbehavioral_analysis=%s\n' "$interface" "$behavior" "$analysis"
echo "This submission stops after behavior/computation; freeze targets before neural jobs."

#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPU_SCRIPT="$PROJECT_DIR/slurm/run_replication_gpu.slurm"
CPU_SCRIPT="$PROJECT_DIR/slurm/run_replication_cpu.slurm"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
REVISION="${REVISION:?REVISION must be an immutable 40-hex commit}"
ADAPTER="${ADAPTER:?ADAPTER is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
CONFIG="${CONFIG:-configs/replication/default.yaml}"
TOKENIZER_ID="${TOKENIZER_ID:-$MODEL_ID}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$REVISION}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on a Della login node" >&2
  exit 1
fi
if [ ! -f "$GPU_SCRIPT" ] || [ ! -f "$CPU_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
if [ -e "$OUTPUT" ]; then
  echo "OUTPUT already exists; resume it explicitly instead of changing its frozen protocol" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p logs
python -m cognitive_discovery.replicate_model \
  --model "$MODEL_ID" --revision "$REVISION" \
  --tokenizer "$TOKENIZER_ID" --tokenizer-revision "$TOKENIZER_REVISION" \
  --adapter "$ADAPTER" --config "$CONFIG" --output "$OUTPUT" \
  --stage initialize --execute

interface=$(sbatch --parsable --export="ALL,STAGE=interface,OUTPUT=$OUTPUT" "$GPU_SCRIPT")
interface="${interface%%;*}"
dispatch=$(sbatch --parsable --dependency="afterany:${interface}" --export="ALL,STAGE=dispatch_interface,OUTPUT=$OUTPUT" "$CPU_SCRIPT")
dispatch="${dispatch%%;*}"

echo "Replication submitted: interface=$interface conditional_dispatch=$dispatch"
echo "Downstream jobs are submitted only if the interface measurement gate passes."
echo "A measurement failure submits a CPU report and reserves no additional GPU."

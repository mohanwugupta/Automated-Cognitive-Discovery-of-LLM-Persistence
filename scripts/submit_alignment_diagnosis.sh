#!/bin/bash
# Submit from Della without importing Python or preparing data on the login node.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DIAGNOSIS_OUTPUT=${DIAGNOSIS_OUTPUT:?export DIAGNOSIS_OUTPUT to a new scratch directory}
DIAGNOSIS_CONFIG=${DIAGNOSIS_CONFIG:-$REPO/configs/alignment_diagnosis/v1.yaml}
DIAGNOSIS_PILOT=${DIAGNOSIS_PILOT:-0}
MODEL_PATH=${DIAGNOSIS_MODEL_PATH:-${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}}
export REPO DIAGNOSIS_OUTPUT DIAGNOSIS_CONFIG DIAGNOSIS_PILOT MODEL_PATH
printf 'Submitting alignment diagnosis jobs from %s\n' "$REPO"
trap 'printf "Submission failed at line %s; check the error above.\n" "$LINENO" >&2' ERR
command -v sbatch >/dev/null || { echo "sbatch is unavailable; run on Della" >&2; exit 1; }
cd "$REPO"
if [[ -e "$DIAGNOSIS_OUTPUT" ]]; then
  echo "DIAGNOSIS_OUTPUT already exists: $DIAGNOSIS_OUTPUT. Choose a new output directory; existing data were left untouched." >&2
  exit 2
fi
[[ -f "$DIAGNOSIS_CONFIG" ]] || { echo "missing config: $DIAGNOSIS_CONFIG" >&2; exit 2; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "local Qwen model lacks config.json: $MODEL_PATH; set MODEL_PATH to the existing cluster model directory" >&2; exit 2; }
printf 'Local Qwen checkpoint: %s\n' "$MODEL_PATH"
mkdir -p logs
printf 'Submitting prepare (CPU)...\n'
prepare=$(sbatch --parsable --export=ALL,DIAGNOSIS_PHASE=prepare slurm/run_alignment_diagnosis_cpu.slurm)
prepare=${prepare%%;*}
printf 'prepare=%s\n' "$prepare"
printf 'Submitting initial collection (GPU)...\n'
initial=$(sbatch --parsable --dependency="afterok:$prepare" \
  --export=ALL,DIAGNOSIS_PHASE=initial slurm/run_alignment_diagnosis_gpu.slurm)
initial=${initial%%;*}
printf 'initial=%s\n' "$initial"
printf 'Submitting prediction freeze (CPU)...\n'
freeze=$(sbatch --parsable --dependency="afterok:$initial" \
  --export=ALL,DIAGNOSIS_PHASE=freeze slurm/run_alignment_diagnosis_cpu.slurm)
freeze=${freeze%%;*}
printf 'freeze=%s\n' "$freeze"
printf 'Submitting intervention collection (GPU)...\n'
intervention=$(sbatch --parsable --dependency="afterok:$freeze" \
  --export=ALL,DIAGNOSIS_PHASE=intervention slurm/run_alignment_diagnosis_gpu.slurm)
intervention=${intervention%%;*}
printf 'intervention=%s\n' "$intervention"
printf 'Submitting analysis (CPU)...\n'
analysis=$(sbatch --parsable --dependency="afterok:$intervention" \
  --export=ALL,DIAGNOSIS_PHASE=analyze slurm/run_alignment_diagnosis_cpu.slurm)
analysis=${analysis%%;*}
printf 'analyze=%s\n' "$analysis"

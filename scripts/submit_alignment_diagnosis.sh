#!/bin/bash
# Run from the activated project environment on Della; pinned weights must be cached.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DIAGNOSIS_OUTPUT=${DIAGNOSIS_OUTPUT:?export DIAGNOSIS_OUTPUT to a new scratch directory}
DIAGNOSIS_CONFIG=${DIAGNOSIS_CONFIG:-$REPO/configs/alignment_diagnosis/v1.yaml}
export REPO DIAGNOSIS_OUTPUT
command -v sbatch >/dev/null || { echo "sbatch is unavailable; run on Della" >&2; exit 1; }
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
pilot_args=()
if [[ "${DIAGNOSIS_PILOT:-0}" == 1 ]]; then pilot_args+=(--pilot); fi
python scripts/alignment_diagnosis.py prepare --config "$DIAGNOSIS_CONFIG" \
  --output "$DIAGNOSIS_OUTPUT" "${pilot_args[@]}"
mkdir -p logs
initial=$(sbatch --parsable --export=ALL,DIAGNOSIS_PHASE=initial slurm/run_alignment_diagnosis_gpu.slurm)
initial=${initial%%;*}
freeze=$(sbatch --parsable --dependency="afterok:$initial" \
  --export=ALL,DIAGNOSIS_PHASE=freeze slurm/run_alignment_diagnosis_cpu.slurm)
freeze=${freeze%%;*}
intervention=$(sbatch --parsable --dependency="afterok:$freeze" \
  --export=ALL,DIAGNOSIS_PHASE=intervention slurm/run_alignment_diagnosis_gpu.slurm)
intervention=${intervention%%;*}
analysis=$(sbatch --parsable --dependency="afterok:$intervention" \
  --export=ALL,DIAGNOSIS_PHASE=analyze slurm/run_alignment_diagnosis_cpu.slurm)
printf 'initial=%s\nfreeze=%s\nintervention=%s\nanalysis=%s\n' "$initial" "$freeze" "$intervention" "$analysis"

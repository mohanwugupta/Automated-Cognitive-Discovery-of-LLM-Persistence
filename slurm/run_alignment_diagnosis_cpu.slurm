#!/bin/bash
#SBATCH --job-name=cog_diagnosis_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cognitive_diagnosis_cpu_%j.out
#SBATCH --error=logs/cognitive_diagnosis_cpu_%j.err

set -euo pipefail
REPO=${REPO:?REPO is required}
DIAGNOSIS_OUTPUT=${DIAGNOSIS_OUTPUT:?DIAGNOSIS_OUTPUT is required}
DIAGNOSIS_PHASE=${DIAGNOSIS_PHASE:?DIAGNOSIS_PHASE is required}
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cognitive-diagnosis-mpl-$SLURM_JOB_ID"
case "$DIAGNOSIS_PHASE" in
  prepare)
    pilot_args=()
    if [[ "${DIAGNOSIS_PILOT:-0}" == 1 ]]; then pilot_args+=(--pilot); fi
    python scripts/alignment_diagnosis.py prepare \
      --config "${DIAGNOSIS_CONFIG:-$REPO/configs/alignment_diagnosis/episodes_v2.yaml}" \
      --output "$DIAGNOSIS_OUTPUT" "${pilot_args[@]}"
    ;;
  freeze|analyze)
    python scripts/alignment_diagnosis.py "$DIAGNOSIS_PHASE" --output "$DIAGNOSIS_OUTPUT"
    ;;
  *) echo "unknown CPU phase: $DIAGNOSIS_PHASE" >&2; exit 2 ;;
esac

#!/bin/bash
#SBATCH --job-name=cog_rep_smoke_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cognitive_replication_smoke_cpu_%j.out
#SBATCH --error=logs/cognitive_replication_smoke_cpu_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-llm-cognitive-discovery}"
SMOKE_PHASE="${SMOKE_PHASE:?SMOKE_PHASE is required}"
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
  echo "CPU smoke phase unexpectedly received GPUs: $SLURM_JOB_GPUS" >&2
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
mkdir -p logs

case "$SMOKE_PHASE" in
  tests)
    pytest -q \
      tests/test_replication_smoke.py \
      tests/test_replication_harness.py \
      tests/test_slurm_resources.py
    ;;
  summarize)
    python scripts/replication_smoke.py summarize \
      --root "${SMOKE_ROOT:?SMOKE_ROOT is required}" \
      --expected-jobs "${SMOKE_EXPECTED_JOBS:?SMOKE_EXPECTED_JOBS is required}"
    ;;
  *)
    echo "Unknown SMOKE_PHASE=$SMOKE_PHASE" >&2
    exit 2
    ;;
esac

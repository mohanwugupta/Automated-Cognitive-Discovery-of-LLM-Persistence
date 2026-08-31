#!/bin/bash
#SBATCH --job-name=cog_causal_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cognitive_causal_mech_cpu_%A.out
#SBATCH --error=logs/cognitive_causal_mech_cpu_%A.err

set -eo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
exec bash "$PROJECT_DIR/run_causal_mechanistic.slurm"

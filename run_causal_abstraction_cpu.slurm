#!/bin/bash
#SBATCH --job-name=cog_abstraction_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_abstraction_cpu_%A.out
#SBATCH --error=logs/cognitive_abstraction_cpu_%A.err

set -eo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
exec bash "$PROJECT_DIR/run_causal_abstraction.slurm"

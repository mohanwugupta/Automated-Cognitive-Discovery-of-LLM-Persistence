#!/bin/bash
#SBATCH --job-name=cog_active_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cognitive_active_cpu_%A_%a.out
#SBATCH --error=logs/cognitive_active_cpu_%A_%a.err

set -eo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
exec bash "$PROJECT_DIR/run_active_discovery.slurm"

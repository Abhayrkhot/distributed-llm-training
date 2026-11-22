#!/bin/bash
# Build the uv-locked venv on a CPU node (no GPU held while pip runs). Submit from a synced code dir.
#SBATCH --job-name=dlt-setup-env
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
set -euo pipefail
export SRC=$SLURM_SUBMIT_DIR
bash "$SRC/scripts/setup_env.sh"

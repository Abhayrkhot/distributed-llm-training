#!/bin/bash
# Download and prepare BIRD on a CPU node (dlt.bird.data): databases, questions, the database-level
# validation split, schemas, gold-SQL execution checks and prompt-length statistics.
#   sbatch slurm/bird_prep.sbatch
#SBATCH --job-name=dlt-bird
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
set -euo pipefail
SRC=$SLURM_SUBMIT_DIR
source "$SRC/scripts/env.sh"
cd "$SRC"
"$VENV/bin/python" -m dlt.bird.data --out "$DLT_HOME/data/bird" --workers 16
du -sh "$DLT_HOME/data/bird"

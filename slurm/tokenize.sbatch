#!/bin/bash
# Tokenize the pinned FineWeb-Edu shard into $DLT_HOME/data/fineweb-edu-000/{train,val}.bin (CPU only).
#SBATCH --job-name=dlt-tokenize
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
set -euo pipefail
SRC=$SLURM_SUBMIT_DIR
source "$SRC/scripts/env.sh"
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$VENV/bin/python" -m dlt.data.tokenize --out "$DLT_HOME/data/fineweb-edu-000" "$@"

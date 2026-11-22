#!/bin/bash
# Download pinned model revisions on a CPU node.  sbatch slurm/fetch_models.sbatch Qwen/Qwen2.5-0.5B ...
#SBATCH --job-name=dlt-fetch
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
set -euo pipefail
SRC=$SLURM_SUBMIT_DIR
source "$SRC/scripts/env.sh"
"$VENV/bin/python" -m dlt.fetch "$@"

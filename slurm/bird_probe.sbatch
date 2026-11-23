#!/bin/bash
# Diagnose BIRD gold-SQL timeouts (scripts/bird_gold_probe.py) on a CPU node.
#SBATCH --job-name=dlt-bird-probe
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
set -euo pipefail
SRC=$SLURM_SUBMIT_DIR
source "$SRC/scripts/env.sh"
cd "$SRC"
"$VENV/bin/python" -u scripts/bird_gold_probe.py "$DLT_HOME/data/bird" "$@"

#!/bin/bash
# Download pinned model revisions on a CPU node and write a verified checksum manifest per model.
#   sbatch slurm/fetch_models.sbatch [--verify] Qwen/Qwen2.5-0.5B ...
#SBATCH --job-name=dlt-fetch
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
set -euo pipefail
SRC=$SLURM_SUBMIT_DIR
source "$SRC/scripts/env.sh"
cd "$SRC"
df -h "$DLT_HOME" | tail -1
(command -v check-quota >/dev/null && timeout 60 check-quota 2>&1 | tail -5) || echo "check-quota unavailable"
"$VENV/bin/python" -m dlt.fetch --workers 8 "$@"
du -sh --apparent-size "$HF_HOME" | tail -1

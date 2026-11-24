#!/bin/bash
# Phase 7: re-store the saved NF4 72B checkpoint with BF16 packed weights for the FSDP1 run
# (scripts/nf4_storage.py). CPU only: reads and writes 38 GiB on /scratch.
#   sbatch slurm/phase7_convert.sbatch
#SBATCH --job-name=dlt-p7-convert
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
SRC=$DLT_HOME/models/qwen2.5-72b-nf4
DST=$DLT_HOME/models/qwen2.5-72b-nf4-bf16
[ -e "$DST" ] && { echo "$DST exists"; exit 0; }
t0=$(date +%s)
dlt_python scripts/nf4_storage.py "$SRC" "$DST.tmp" || { rm -rf "$DST.tmp"; exit 1; }
mv "$DST.tmp" "$DST"
echo "converted in $(($(date +%s) - t0)) s"; du -sh "$DST"

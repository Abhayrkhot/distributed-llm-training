#!/bin/bash
# Run a sweep file (dlt.sweep) on one H200 (gpu partition, 8 h). Finished runs are skipped, so the same
# sweep can be resubmitted or chained until it is complete.
#   sbatch slurm/sweep_1gpu.sbatch configs/sweeps/X.yaml
#SBATCH --job-name=dlt-sweep1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
dlt_python -m dlt.sweep "$@"

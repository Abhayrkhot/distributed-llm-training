#!/bin/bash
# One H200 on `gpu` (8 h max wall). Usage:  sbatch [--time=HH:MM:SS] slurm/h200_1gpu.sbatch <module> [args]
#SBATCH --job-name=dlt-1gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh" "$@"

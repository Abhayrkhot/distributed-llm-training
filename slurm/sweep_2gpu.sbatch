#!/bin/bash
# Run a sweep file (dlt.sweep) inside one 2-GPU allocation. Finished runs are skipped, so the same
# sweep can be resubmitted or chained until it is complete:
#   sbatch slurm/sweep_2gpu.sbatch configs/sweeps/X.yaml
#   slurm/chain.sh 3 slurm/sweep_2gpu.sbatch configs/sweeps/X.yaml
#SBATCH --job-name=dlt-sweep
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
dlt_python -m dlt.sweep "$@"

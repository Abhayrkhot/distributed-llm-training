#!/bin/bash
# Two H200s on `gpu-short` (2 h max wall). Usage:  sbatch [--time=HH:MM:SS] slurm/h200_2gpu.sbatch <module> [args]
# --cpus-per-gpu with --gres-flags=enforce-binding keeps each GPU's CPUs on its own NUMA node
# (Phase 0 saw a 2-GPU job whose CPUs were all on the other GPU's socket).
#SBATCH --job-name=dlt-2gpu
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
dlt_torchrun "$NGPU" "$@"

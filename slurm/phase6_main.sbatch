#!/bin/bash
# One link of the Phase 6 main run (configs/training/p6_main.yaml) on one H200. Chain links with
#   slurm/chain.sh 3 slurm/phase6_main.sbatch [--set key=value ...]
# A link trains until Slurm's USR1 (300 s before the limit), checkpoints and exits; the next resumes; a
# link that finds the run complete exits at once.
#SBATCH --job-name=dlt-p6-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
dlt_torchrun 1 dlt.train --config configs/training/p6_main.yaml "$@"

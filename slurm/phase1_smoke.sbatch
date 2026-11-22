#!/bin/bash
# Phase 1 GPU checks in one 2-GPU allocation (gpu-short allows few jobs per user):
#   1. NCCL bandwidth sweep on 2 GPUs
#   2. Qwen2.5-0.5B smoke training, DDP on 2 GPUs
#   3. the same on 1 GPU
#   4. the wall-time path: train until Slurm's USR1 (300 s before the limit) stops every rank at a step
#      boundary, then exit cleanly well before the limit
#SBATCH --job-name=dlt-phase1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=00:16:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
M=(dlt.bench.smoke_train --model Qwen/Qwen2.5-0.5B)
dlt_torchrun 2 dlt.bench.nccl_bw
dlt_torchrun 2 "${M[@]}" --steps 40
FIRST=${CUDA_VISIBLE_DEVICES%%,*}
CUDA_VISIBLE_DEVICES=$FIRST dlt_torchrun 1 "${M[@]}" --steps 40
if [ "$DLT_USR1_SEEN" = 0 ]; then
  dlt_torchrun 2 "${M[@]}" --steps 1000000
else
  echo "USR1 already arrived; skipping the wall-time test"
fi

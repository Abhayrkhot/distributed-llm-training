#!/bin/bash
# Phase 2 real run across chained jobs: pretrained Qwen2.5-0.5B, FSDP2 on 2 H200s, continued pretraining
# on FineWeb-Edu (configs/training/p2_chain_0p5b.yaml). Each 20 min link trains until Slurm's USR1
# arrives 300 s before the limit, checkpoints and exits 0; the next link resumes from that checkpoint.
#   [CHAIN_AFTER=<jobid>] slurm/chain.sh 2 slurm/phase2_chain.sbatch
#SBATCH --job-name=dlt-p2chain
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
D=$DLT_HOME/data/fineweb-edu-000
dlt_torchrun 2 dlt.train --config configs/training/p2_chain_0p5b.yaml \
  --set "data=$D/train.bin" --set "val_data=$D/val.bin" --set "ckpt_dir=$DLT_HOME/ckpt/p2-chain"

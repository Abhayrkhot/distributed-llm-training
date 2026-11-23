#!/bin/bash
# Phase 5 on two H200s: Qwen2.5-72B in BF16 with its layers split over the GPUs, the same teacher-forced
# logits as slurm/phase5_nf4.sbatch, then the NF4 vs BF16 comparison.
#SBATCH --job-name=dlt-p5-bf16
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
OUT=$DLT_HOME/phase5
mkdir -p "$OUT"
dlt_python -m dlt.qlora.nf4 logits --load bf16 --out "$OUT/bf16.npz" || exit $?
if [ -f "$OUT/nf4.npz" ]; then
  dlt_python -m dlt.qlora.nf4 compare "$OUT/nf4.npz" "$OUT/bf16.npz"
fi

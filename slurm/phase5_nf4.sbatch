#!/bin/bash
# Phase 5 on one H200 (gpu partition): quantize Qwen2.5-72B to NF4, save, reload, then the
# teacher-forced logits on 200 validation examples. Pair with slurm/phase5_bf16.sbatch.
#SBATCH --job-name=dlt-p5-nf4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
OUT=$DLT_HOME/phase5
NF4=$DLT_HOME/models/qwen2.5-72b-nf4
mkdir -p "$OUT"
if [ ! -f "$NF4/config.json" ]; then
  dlt_python -m dlt.qlora.nf4 quantize --out "$NF4" || exit $?
fi
dlt_python -m dlt.qlora.nf4 logits --load nf4-saved --model "$NF4" --out "$OUT/nf4.npz"

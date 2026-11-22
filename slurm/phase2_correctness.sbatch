#!/bin/bash
# Phase 2 correctness on H200s with pretrained Qwen2.5-0.5B and FineWeb-Edu, in one 2-GPU allocation:
#   parity     the same 50 steps (same global batch, same data) on 1 GPU, DDP x2 and FSDP2 x2
#   resume     deterministic mode, eager attention: uninterrupted vs stop at 10 + resume, per backend
#   crash      SIGKILL at step 13 with checkpoints every 5, then recovery from step 10, per backend
# Records go to $DLT_HOME/results/train; checkpoints are deleted at the end.
#SBATCH --job-name=dlt-phase2
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
DATA=$DLT_HOME/data/fineweb-edu-000/train.bin
CK=$DLT_HOME/ckpt/phase2-$SLURM_JOB_ID
C=(dlt.train --config configs/training/p2_0p5b.yaml --set "data=$DATA")
FIRST=${CUDA_VISIBLE_DEVICES%%,*}

CUDA_VISIBLE_DEVICES=$FIRST dlt_torchrun 1 "${C[@]}" --set backend=single --set run_name=parity-single
dlt_torchrun 2 "${C[@]}" --set backend=ddp --set run_name=parity-ddp
dlt_torchrun 2 "${C[@]}" --set backend=fsdp2 --set run_name=parity-fsdp2

for b in ddp fsdp2; do
  D=("${C[@]}" --set backend=$b --set deterministic=true --set attn=eager --set steps=20 --set seq_len=1024)
  dlt_torchrun 2 "${D[@]}" --set run_name=det-$b-full
  dlt_torchrun 2 "${D[@]}" --set "ckpt_dir=$CK/$b" --set stop_at_step=10 --set run_name=det-$b-first
  dlt_torchrun 2 "${D[@]}" --set "ckpt_dir=$CK/$b" --set run_name=det-$b-second
  dlt_torchrun 2 "${D[@]}" --set "ckpt_dir=$CK/$b-crash" --set ckpt_every=5 --set crash_at_step=13 --set run_name=det-$b-crash
  ls "$CK/$b-crash"
  dlt_torchrun 2 "${D[@]}" --set "ckpt_dir=$CK/$b-crash" --set ckpt_every=5 --set run_name=det-$b-recovered
done
du -sh "$CK"/* 2>/dev/null
rm -rf "$CK"

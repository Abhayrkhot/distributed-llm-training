#!/bin/bash
# Phase 8: fault injection on 1.5B FSDP2 (2 H200s), each failure followed by a resume, all compared with
# an uninterrupted run (bit for bit: deterministic mode). Records go to $DLT_HOME/results/train with
# run names p8-*; checkpoints are deleted at the end. Scenarios:
#   reference      40 steps, no checkpoints
#   sync / async   checkpoints every 10 steps, synchronous vs DCP async_save
#   kill-step      SIGKILL every rank before step 25, then resume (from step 20)
#   kill-save      SIGKILL after the step-30 shards are written but before publish, then resume (from 20)
#   kill-eval      SIGKILL at the start of the end-of-run eval, then a further link (finds the run complete)
#   term-graceful  SIGTERM before step 25 with stop_on_term: checkpoint at 26, resume, nothing lost
#   term-default   SIGTERM before step 25 without the handler: dies, resume from 20
#SBATCH --job-name=dlt-p8
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=01:45:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
DATA=$DLT_HOME/data/fineweb-edu-000
CK=$DLT_HOME/ckpt/phase8-$SLURM_JOB_ID
C=(dlt.train --config configs/training/p8_1p5b_fsdp2.yaml --set "data=$DATA/train.bin")
E=(--set "val_data=$DATA/val.bin" --set eval_samples=32)
K=(--set ckpt_every=10 --set keep_ckpts=3)

dlt_torchrun 2 "${C[@]}" --set run_name=p8-reference
dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/sync" --set run_name=p8-sync
dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/async" --set async_ckpt=true --set run_name=p8-async

dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/kstep" --set crash_at_step=25 --set run_name=p8-kill-step
dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/kstep" --set run_name=p8-kill-step-resume

dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/ksave" --set stop_at_step=20 --set run_name=p8-kill-save-pre
dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/ksave" --set crash_in_save=true --set run_name=p8-kill-save
ls "$CK/ksave"
dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/ksave" --set run_name=p8-kill-save-resume

dlt_torchrun 2 "${C[@]}" "${K[@]}" "${E[@]}" --set "ckpt_dir=$CK/keval" --set crash_in_eval=true --set run_name=p8-kill-eval
dlt_torchrun 2 "${C[@]}" "${K[@]}" "${E[@]}" --set "ckpt_dir=$CK/keval" --set run_name=p8-kill-eval-resume

dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/tgrace" --set term_at_step=25 --set stop_on_term=true --set run_name=p8-term-graceful
dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/tgrace" --set run_name=p8-term-graceful-resume

dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/tdef" --set term_at_step=25 --set run_name=p8-term-default
dlt_torchrun 2 "${C[@]}" "${K[@]}" --set "ckpt_dir=$CK/tdef" --set run_name=p8-term-default-resume

du -sh "$CK"/* 2>/dev/null
rm -rf "$CK"

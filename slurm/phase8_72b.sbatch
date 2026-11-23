#!/bin/bash
# Phase 8 on the real workload: 72B QLoRA (one H200), checkpoints every 5 steps, SIGKILL before step 12,
# then a resume in a new process (as after a node failure). Measures adapter checkpoint size, save and
# restore time, base reload time and lost work; also SIGTERM with stop_on_term on the same run.
#SBATCH --job-name=dlt-p8-72b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
CK=$DLT_HOME/ckpt/phase8-72b-$SLURM_JOB_ID
C=(dlt.train --config configs/training/p6_overfit.yaml --set sft_limit=0 --set global_batch=8 --set steps=24
   --set "ckpt_dir=$CK" --set ckpt_every=5 --set keep_ckpts=2)
dlt_torchrun 1 "${C[@]}" --set crash_at_step=12 --set run_name=p8-72b-kill
dlt_torchrun 1 "${C[@]}" --set term_at_step=17 --set stop_on_term=true --set run_name=p8-72b-resume-then-term
dlt_torchrun 1 "${C[@]}" --set run_name=p8-72b-final
du -sh "$CK"/* 2>/dev/null
rm -rf "$CK"

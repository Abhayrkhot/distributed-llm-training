#!/bin/bash
# Phase 8: a real scancel against a running job, from the login node. Submits a 1.5B FSDP2 run (one H200,
# the gpu partition, so it does not wait for two free GPUs) with
# stop_on_term and checkpoints every 10 steps, waits until it has passed step 15, scancels it, then
# submits the same run again to resume. The two jobs' records (run names p8-scancel, p8-scancel-resume)
# show whether the SIGTERM-to-SIGKILL window was long enough for the graceful checkpoint.
#   bash scripts/phase8_scancel.sh [term|usr1]     (from /scratch/$USER/dlt/code/<sha>)
#   term: plain scancel (SIGTERM to every process of the job, torchrun included)
#   usr1: scancel --batch --signal=USR1 (only the batch shell, which forwards USR1 to the workers)
set -euo pipefail
MODE=${1:-term}
DLT_HOME=/scratch/$USER/dlt
CK=$DLT_HOME/ckpt/phase8-scancel-$MODE
ARGS=(dlt.train --config configs/training/p8_1p5b_fsdp2.yaml --set "data=$DLT_HOME/data/fineweb-edu-000/train.bin"
      --set "ckpt_dir=$CK" --set ckpt_every=10 --set stop_on_term=true --set steps=60)
echo "KillWait: $(scontrol show config | grep -i '^KillWait' || true)"
rm -rf "$CK"
j=$(sbatch --parsable --time=00:40:00 --mem=96G --job-name=dlt-p8-scancel slurm/h200_1gpu.sbatch "${ARGS[@]}" --set run_name=p8-scancel-$MODE)
echo "run job $j"
log=$DLT_HOME/logs/dlt-p8-scancel-$j.out
until [ -f "$log" ] && grep -q "^step 20 " "$log"; do sleep 15; done
echo "$(date -Is) scancel $j at: $(grep '^step ' "$log" | tail -1)"
if [ "$MODE" = usr1 ]; then scancel --batch --signal=USR1 "$j"; else scancel "$j"; fi
while squeue -h -j "$j" | grep -q .; do sleep 5; done
sacct -j "$j" -X -n -o JobID,State,ExitCode,Elapsed
ls "$CK"
r=$(sbatch --parsable --time=00:40:00 --mem=96G --job-name=dlt-p8-scancel slurm/h200_1gpu.sbatch "${ARGS[@]}" --set run_name=p8-scancel-$MODE-resume)
echo "resume job $r"

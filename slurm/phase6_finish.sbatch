#!/bin/bash
# The rest of Phase 6 in one 1-GPU allocation, run back to back so no step waits in the queue: resume the
# main run (configs/training/p6_main.yaml) from its latest checkpoint to the end, then evaluate the saved
# adapter on val and dev, then reproduce the dev evaluation in a fresh process that reloads the adapter
# from disk. Submitted after the first main-run job (which stops on Slurm's USR1), so the run still spans
# two Slurm jobs.
#   sbatch --dependency=afterany:<link 1> slurm/phase6_finish.sbatch
#SBATCH --job-name=dlt-p6-finish
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
NF4=$DLT_HOME/models/qwen2.5-72b-nf4
ADAPTER=$DLT_HOME/adapters/p6-main
dlt_torchrun 1 dlt.train --config configs/training/p6_main.yaml || exit $?
if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "no adapter at $ADAPTER: the run did not finish in this job"
  exit 4
fi
EV=(dlt.bird.evaluate --mode finetuned --model "$NF4" --adapter "$ADAPTER")
dlt_torchrun 1 "${EV[@]}" --split val --name finetuned-val
dlt_torchrun 1 "${EV[@]}" --split dev --name finetuned-dev
dlt_torchrun 1 "${EV[@]}" --split dev --name finetuned-dev-repro

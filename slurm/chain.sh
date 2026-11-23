#!/bin/bash
# Submit N links of one job, each starting after the previous one ends in any state, so a long run
# continues across wall-time limits by resuming from its latest checkpoint. A link that finds the run
# complete exits before loading the model.
#   [CHAIN_AFTER=<jobid>] slurm/chain.sh N slurm/h200_2gpu.sbatch <module> [args]
# CHAIN_AFTER makes the first link wait for that job to succeed (afterok).
set -euo pipefail
n=$1
shift
prev=""
for i in $(seq 1 "$n"); do
  if [ -n "$prev" ]; then
    prev=$(sbatch --parsable --dependency="afterany:$prev" "$@")
  elif [ -n "${CHAIN_AFTER:-}" ]; then
    prev=$(sbatch --parsable --dependency="afterok:$CHAIN_AFTER" "$@")
  else
    prev=$(sbatch --parsable "$@")
  fi
  echo "link $i: job $prev"
done

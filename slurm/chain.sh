#!/bin/bash
# Submit N links of one job, each starting after the previous one ends in any state, so a long run
# continues across wall-time limits by resuming from its latest checkpoint (resume arrives in Phase 2).
#   slurm/chain.sh N slurm/h200_1gpu.sbatch <module> [args]
set -euo pipefail
n=$1
shift
prev=""
for i in $(seq 1 "$n"); do
  if [ -n "$prev" ]; then
    prev=$(sbatch --parsable --dependency="afterany:$prev" "$@")
  else
    prev=$(sbatch --parsable "$@")
  fi
  echo "link $i: job $prev"
done

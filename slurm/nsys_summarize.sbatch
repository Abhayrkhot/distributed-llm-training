#!/bin/bash
# Re-summarize existing Nsight Systems reports on a CPU node (after changing scripts/nsys_summary.py).
#   sbatch slurm/nsys_summarize.sbatch NAME [NAME ...]     (reports $DLT_HOME/nsys/NAME-rank<R>.nsys-rep)
#SBATCH --job-name=dlt-nsys-sum
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
set -uo pipefail
[ -n "$(type -t module)" ] || source /usr/share/Modules/init/bash
SRC=$SLURM_SUBMIT_DIR
source "$SRC/scripts/env.sh"
cd "$SRC" || exit 3
module load cuda/12.8.0
NSYS=$(command -v nsys) || { echo "nsys not found"; exit 3; }
for name in "$@"; do
  for rep in "$DLT_HOME/nsys/$name"-rank*.nsys-rep; do
    r=${rep##*-rank}
    r=${r%.nsys-rep}
    "$VENV/bin/python" scripts/nsys_summary.py "$NSYS" "$rep" --name "$name" --rank "$r" | tail -40
  done
done

#!/bin/bash
# One training run under Nsight Systems, one report per rank (Phase 4 profiling).
#   sbatch slurm/nsys_2gpu.sbatch NAME [--set key=value ...]
# Captures only the profiled steps (the engine calls cudaProfilerStart/Stop around them), with NVTX
# ranges for forward, backward and optimizer. Reports go to $DLT_HOME/nsys/NAME-rank<R>.nsys-rep and
# are summarized by scripts/nsys_summary.py into results/profile/.
#SBATCH --job-name=dlt-nsys
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
module load cuda/12.8.0
NSYS=$(command -v nsys) || { echo "nsys not found"; exit 3; }
"$NSYS" --version
NAME=$1
shift
OUT=$DLT_HOME/nsys
mkdir -p "$OUT"
echo "$(date -Is) nsys run $NAME $*"
"$VENV/bin/torchrun" --standalone --nproc-per-node="$NGPU" --no-python \
  "$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-memory-usage=false --force-overwrite=true --output="$OUT/$NAME-rank%q{RANK}" \
  "$VENV/bin/python" -m dlt.train --set profiler=nsys --set nvtx=true --set "run_name=$NAME" "$@"
rc=$?
echo "$(date -Is) exit=$rc"
for r in $(seq 0 $((NGPU - 1))); do
  "$VENV/bin/python" scripts/nsys_summary.py "$NSYS" "$OUT/$NAME-rank$r.nsys-rep" --name "$NAME" --rank "$r"
done
exit $rc

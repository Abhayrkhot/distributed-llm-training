# Common job setup for slurm/*.sbatch (sourced): environment, topology log, and dlt_torchrun.
#
#   source "$SLURM_SUBMIT_DIR/slurm/launch.sh"
#   dlt_torchrun NPROC <python module> [args...]
#   dlt_python <python args...>        (a process that handles USR1 itself, e.g. dlt.sweep)
#
# The sbatch files ask Slurm for SIGUSR1 to the batch shell 300 s before the wall-time limit. torchrun
# does not handle USR1 (its default action would kill it), so the shell traps it and forwards it to the
# worker processes, which stop at the next step boundary (and, from Phase 2, checkpoint first).
set -uo pipefail
[ -n "$(type -t module)" ] || source /usr/share/Modules/init/bash
SRC=$SLURM_SUBMIT_DIR
source "$SRC/scripts/env.sh"
if [ ! -x "$VENV/bin/python" ]; then
  echo "venv $VENV missing: sbatch slurm/setup_env.sbatch first"
  exit 3
fi
NGPU=$(nvidia-smi -L | wc -l)
echo "$(date -Is) job=$SLURM_JOB_ID node=$(hostname -s) gpus=$NGPU sha=$DLT_GIT_SHA lock=$DLT_LOCK_HASH"
nvidia-smi topo -m | head -n $((NGPU + 1))
echo "cpus: $(taskset -cp $$ | cut -d: -f2)"
cd "$SRC" || exit 3

DLT_PID=""
DLT_USR1_SEEN=0
DLT_USR1_SELF=0

forward_usr1() {
  DLT_USR1_SEEN=1
  if [ -z "$DLT_PID" ]; then
    echo "$(date -Is) USR1: no workers running"
    return
  fi
  if [ "$DLT_USR1_SELF" = 1 ]; then
    echo "$(date -Is) USR1: forwarding to $DLT_PID"
    kill -USR1 "$DLT_PID"
    return
  fi
  # torchrun's direct children are the workers (read from /proc: no procps dependency)
  local kids
  kids=$(cat /proc/"$DLT_PID"/task/*/children 2>/dev/null)
  echo "$(date -Is) USR1: forwarding to workers $kids"
  [ -n "$kids" ] && kill -USR1 $kids
}
trap forward_usr1 USR1

dlt_torchrun() {
  local nproc=$1 rc
  shift
  echo "$(date -Is) torchrun nproc=$nproc $*"
  "$VENV/bin/torchrun" --standalone --nproc-per-node="$nproc" -m "$@" &
  DLT_PID=$!
  while true; do
    wait "$DLT_PID"
    rc=$?
    kill -0 "$DLT_PID" 2>/dev/null || break
  done
  DLT_PID=""
  echo "$(date -Is) exit=$rc"
  return "$rc"
}

dlt_python() {
  local rc
  echo "$(date -Is) python $*"
  "$VENV/bin/python" "$@" &
  DLT_PID=$!
  DLT_USR1_SELF=1
  while true; do
    wait "$DLT_PID"
    rc=$?
    kill -0 "$DLT_PID" 2>/dev/null || break
  done
  DLT_PID=""
  DLT_USR1_SELF=0
  echo "$(date -Is) exit=$rc"
  return "$rc"
}

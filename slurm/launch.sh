# Common body for slurm/h200_*.sbatch (sourced): environment, topology log, torchrun, USR1 forwarding.
# Arguments: a python module and its arguments, e.g.  dlt.bench.smoke_train --steps 30
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
echo "$(date -Is) job=$SLURM_JOB_ID node=$(hostname -s) gpus=$NGPU sha=$DLT_GIT_SHA lock=$DLT_LOCK_HASH cmd=$*"
nvidia-smi topo -m | head -n $((NGPU + 1))
echo "cpus: $(taskset -cp $$ | cut -d: -f2)"

cd "$SRC"
"$VENV/bin/torchrun" --standalone --nproc-per-node="$NGPU" -m "$@" &
PID=$!
forward_usr1() {
  # torchrun's direct children are the workers (read from /proc: no procps dependency)
  local kids
  kids=$(cat /proc/"$PID"/task/*/children 2>/dev/null)
  echo "$(date -Is) USR1: forwarding to workers $kids"
  [ -n "$kids" ] && kill -USR1 $kids
}
trap forward_usr1 USR1
while true; do
  wait "$PID"
  rc=$?
  kill -0 "$PID" 2>/dev/null || break
done
echo "$(date -Is) exit=$rc"
exit "$rc"

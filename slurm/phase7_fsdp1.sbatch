#!/bin/bash
# Phase 7, sharded run: the 72b-fsdp1 entry of configs/sweeps/p7_72b.yaml with the BF16 checkpoint
# staged in node memory first. Reading it from /scratch inside from_pretrained (two ranks, page faults on
# memory-mapped files over NFS) took 6 min in job 12100515 and was 64% done after 90 min in job 12107944;
# a parallel sequential copy into /dev/shm reads each byte once, and both ranks then load from memory.
#   sbatch slurm/phase7_fsdp1.sbatch
#SBATCH --job-name=dlt-p7-fsdp1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/slurm/launch.sh"

REV=efba10c8e54e91e0d9570ab5f7b51a958474d4cb
SRC_SNAP=$HF_HOME/hub/models--Qwen--Qwen2.5-72B/snapshots/$REV
STAGE=/dev/shm/dlt-hf-$SLURM_JOB_ID
trap 'rm -rf "$STAGE"' EXIT
df -h /dev/shm
need=$(du -sLk "$SRC_SNAP" | cut -f1)
free=$(df -k --output=avail /dev/shm | tail -1)
echo "checkpoint $((need / 1048576)) GiB, /dev/shm free $((free / 1048576)) GiB"
[ "$free" -gt $((need + 1048576)) ] || { echo "not enough /dev/shm"; exit 5; }

# the hub cache layout from_pretrained resolves for a pinned revision, with real files in place of links
DST=$STAGE/hub/models--Qwen--Qwen2.5-72B/snapshots/$REV
mkdir -p "$DST"
t0=$(date +%s)
find "$SRC_SNAP"/ -mindepth 1 -maxdepth 1 -printf '%f\n' | xargs -P 8 -I{} cp -L "$SRC_SNAP/{}" "$DST/{}" || exit 6
t1=$(date +%s)
echo "staged $((need / 1048576)) GiB in $((t1 - t0)) s ($((need / 1024 / (t1 - t0 + 1))) MiB/s)"

export HF_HUB_CACHE=$STAGE/hub HF_HUB_OFFLINE=1
dlt_python -m dlt.sweep configs/sweeps/p7_72b.yaml --only 72b-fsdp1 --retry-failed

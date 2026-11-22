#!/bin/bash
# Phase 0 CPU-side probe: storage quota and Hugging Face download speed from a compute node.
# Submit from a synced code dir:  sbatch slurm/probe_cpu.sbatch
#SBATCH --job-name=dlt-probe-cpu
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
set -uo pipefail

OUT=/scratch/$USER/dlt/results/probe/cpu-$SLURM_JOB_ID
mkdir -p "$OUT"
cd "$OUT"

{
  echo "date=$(date -Is)"
  echo "host=$(hostname -s)"
  echo "job=$SLURM_JOB_ID"
} > meta.txt

# check-quota is RC's tool (works from compute nodes); fall back to df.
(check-quota "/scratch/$USER" || df -h "/scratch/$USER") > quota_scratch.txt 2>&1
(check-quota "/home/$USER" || df -h "/home/$USER") > quota_home.txt 2>&1

# Download speed: 512 MiB range of one pinned Qwen2.5-72B shard, three times.
REV=efba10c8e54e91e0d9570ab5f7b51a958474d4cb
URL=https://huggingface.co/Qwen/Qwen2.5-72B/resolve/$REV/model-00010-of-00037.safetensors
for i in 1 2 3; do
  curl -sL --max-time 300 -r 0-536870911 -o /dev/null -w "try=$i http=%{http_code} bytes=%{size_download} secs=%{time_total} bytes_per_s=%{speed_download}\n" "$URL"
done > hf_speed.txt 2>&1

echo "done $(date -Is)" >> meta.txt

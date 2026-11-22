#!/bin/bash
# Phase 0 hardware probe: GPU variant, topology, clocks/power, CPU/NUMA, profiler availability, and
# quick GEMM / HBM / P2P microbenchmarks. Short on purpose (does work and exits; IdleBot-safe).
# Submit from a synced code dir:  sbatch slurm/probe_h200.sbatch
#SBATCH --job-name=dlt-probe-h200
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/dlt/logs/%x-%j.out
set -uo pipefail
[ -n "$(type -t module)" ] || source /usr/share/Modules/init/bash

OUT=/scratch/$USER/dlt/results/probe/h200-$SLURM_JOB_ID
mkdir -p "$OUT"
cd "$OUT"
SRC=${SLURM_SUBMIT_DIR:-$PWD}

{
  echo "date=$(date -Is)"
  echo "host=$(hostname -s)"
  echo "job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "git_sha=$(cat "$SRC/GIT_SHA" 2>/dev/null || echo unknown)"
  echo "kernel=$(uname -r)"
  echo "os=$(. /etc/os-release && echo "$PRETTY_NAME")"
} > meta.txt

nvidia-smi -q -x > smi.xml 2>&1
nvidia-smi -q > smi.txt 2>&1
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,memory.total,power.limit,power.max_limit,clocks.max.sm,clocks.max.memory,compute_cap,persistence_mode \
  --format=csv > gpus.csv 2>&1
nvidia-smi topo -m > topo.txt 2>&1
nvidia-smi nvlink -s > nvlink.txt 2>&1
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > clocks.txt 2>&1
lscpu > lscpu.txt 2>&1
(numactl -H || echo "numactl missing") > numa.txt 2>&1
free -g > free.txt 2>&1
scontrol show job "$SLURM_JOB_ID" > job.txt 2>&1

# Profilers: is nsys/ncu on PATH after module load, and do they actually work in a job?
{
  for m in cuda/12.8.0 Nsight/2024.7.1; do
    echo "== module $m"
    module load "$m" 2>&1 && { which nsys ncu 2>&1; nsys --version 2>&1 | head -1; ncu --version 2>&1 | tail -1; }
    module unload "$m" 2>/dev/null
  done
} > tools.txt 2>&1

# Probe Python env: an existing torch 2.11 (cu128) venv on scratch, used read-only for this probe only.
module load python/3.13.5 cuda/12.8.0 >/dev/null 2>&1
PY=/scratch/$USER/ctk/venv/bin/python
export PYTHONNOUSERSITE=1
$PY "$SRC/scripts/probe_gpu.py" > torch_probe.json 2> torch_probe.err
echo "probe_exit=$?" >> meta.txt

# Do profilers work without extra permissions?
SNIP='import torch; a=torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16); [a@a for _ in range(20)]; torch.cuda.synchronize()'
timeout 300 nsys profile -t cuda,nvtx --force-overwrite true -o "$OUT/nsys_test" $PY -c "$SNIP" > nsys_test.log 2>&1
echo "nsys_exit=$? report=$(ls nsys_test.nsys-rep 2>/dev/null || echo missing)" >> meta.txt
timeout 300 ncu --metrics sm__cycles_elapsed.avg -c 1 $PY -c "$SNIP" > ncu_test.log 2>&1
echo "ncu_exit=$? perm_error=$(grep -c ERR_NVGPUCTRPERM ncu_test.log)" >> meta.txt
rm -f nsys_test.nsys-rep nsys_test.sqlite

echo "done $(date -Is)" >> meta.txt

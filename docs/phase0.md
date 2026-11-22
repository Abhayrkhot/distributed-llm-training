# Phase 0: resource manifest and decisions

Checked 2026-11-22. Machine-readable version: [manifests/h200.json](../manifests/h200.json); scrubbed
raw probe output in [manifests/raw/](../manifests/raw/).

## 1. Slurm limits

From `sacctmgr show qos`, `scontrol show partition` and the RC docs.

| Partition / QOS | Max wall | GPUs per job | GPUs per user | H200 nodes |
|---|---|---|---|---|
| `gpu` | 8 h | 1 | 4 | d4052-d4055 |
| `gpu-short` | 2 h | 2 | 2 | d4052-d4055 |
| `gpu-interactive` | 2 h | 1 | none set | d4052-d4055 |
| `sharing` | 1 h | 1 | 2 | d4077 |
| `courses-gpu` | 1 day | 1 | 2 | none |
| `multigpu` | 24 h | 8 | 8 | approval required |

The available accounts have QOS `normal` only, so one job gets at most 2 H200s today. The per-user
GPU limits are shared with every other job the same user runs (other projects' jobs count).

## 2. H200 nodes

| Node | GPUs | CPU threads | RAM (Slurm) | OS kernel | Partitions |
|---|---|---|---|---|---|
| d4052-d4055 | 8x H200 | 128 (2 sockets x 32 cores x 2) | 500 GiB | 5.14.0-362 (EL 9.3) | gpu, gpu-short, gpu-interactive |
| d4077 | 8x H200 | 192 (2 x 48 x 2) | 1,159,000 MiB | 5.14.0-611 (EL 9.7) | sharing |

Load when checked: 31 of the 32 GPUs on d4052-d4055 were allocated and about 130 one-GPU H200 jobs
were pending. Even so, the 20-minute probe jobs started within two minutes (backfill), so short time
limits pay off.

## 3. Probe results

Jobs 12066488 (2 GPUs) and 12066498 (1 GPU) on d4052, plus CPU job 12066489 on d0012, all on
2026-11-22. Scripts: [slurm/probe_h200.sbatch](../slurm/probe_h200.sbatch),
[scripts/probe_gpu.py](../scripts/probe_gpu.py), [slurm/probe_cpu.sbatch](../slurm/probe_cpu.sbatch).

**Hardware**

| Item | Value |
|---|---|
| GPU | NVIDIA H200 SXM (700 W power limit, 132 SMs, 1,980 MHz max SM clock), compute capability 9.0 |
| GPU memory | 143,771 MiB per `nvidia-smi`; 139.8 GiB visible to PyTorch, 139.3 GiB free at job start |
| Driver | 570.86.15, CUDA 12.8 driver API. CUDA 12.8 builds run natively; CUDA 13 builds need forward-compat libraries |
| GPU to GPU | NV18: 18 NVLink links at 26.6 GB/s (478 GB/s per direction) through NVSwitch, so any GPU pair Slurm assigns is NVLink-connected |
| CPU | 2x Intel Xeon Gold 6548Y+ (32 cores each, 2 threads per core), 2 NUMA nodes. The `cascadelake` Slurm feature tag is stale |
| Host RAM | 1.5 TiB physical, but Slurm allocates at most 500 GiB per node |
| GPU/CPU locality | In the 2-GPU job, GPU0 sat on NUMA node 0 and GPU1 on NUMA node 1, while all 8 allocated CPUs were on node 0. Phase 1 job scripts request CPUs per GPU with binding |

**Microbenchmarks** (CUDA events, median of 5 timed repeats after warmup; probe environment torch
2.11.0+cu128, NCCL 2.28.9)

| Measurement | Result | Reference |
|---|---|---|
| BF16 GEMM, 8192 x 8192 x 8192 | 680-702 TFLOPS (median 687) | 69-71% of the 989 TFLOPS dense BF16 peak |
| Device copy (read + write) | 4,284-4,296 GB/s | ~89% of the 4.8 TB/s HBM3e spec |
| Peer copy GPU0 to GPU1 (and back) | 395 GB/s each way | 83% of the 478 GB/s link total |

MFU in this project uses the 989 TFLOPS datasheet peak. The 687 TFLOPS GEMM number is reported next
to it as the practical cuBLAS ceiling.

**Profilers**

- Nsight Systems works inside jobs: `cuda/12.8.0` ships nsys 2024.6.2, and the `Nsight/2024.7.1`
  module ships 2024.7.1. Test captures exited 0 with a report.
- Nsight Compute fails with `ERR_NVGPUCTRPERM`: GPU performance counters are restricted to admins.
  Getting `ncu` would need an RC ticket. The plan relies on Nsight Systems and PyTorch Profiler, so
  this only limits kernel-level drill-downs.

**Network and quota**

- Hugging Face from a compute node: 114, 182 and 193 MB/s on three 512 MiB ranges of a pinned 72B
  shard (median 182 MB/s). The 145 GB BF16 checkpoint should take about 13-21 min in a CPU job.
- RC's `check-quota` failed from the compute node (its backend refused the connection), so the
  per-user /scratch quota is still unknown. /scratch had 131 TB free. Retry before the 72B download
  in Phase 5.

## 4. Storage

- **/scratch** is a VAST NFS filesystem (1.5 PB, 92% full when checked). RC purges it monthly and does
  not back it up. `quota` does not work against it; RC's `check-quota` runs from a compute node.
- **/home** is 75 GB, fixed.
- **Consequence:** weights and datasets on /scratch are a cache, not storage. Each one gets a manifest
  (source URL, pinned revision, sha256) and a re-fetch script, so a purge costs download time, not
  results. Results files are pulled off the cluster after every run. Adapter checkpoints (about 2.4 GiB with FP32 LoRA weights and Adam state for r=16)
 are small enough to copy off when they matter.
- **Space needed at peak:** about 250 GB (72B BF16 145 GB, NF4 ~41 GB, BIRD 9.3 GB zipped, small
  models ~55 GB, checkpoints).

## 5. Pinned models

Revisions from the Hugging Face API. All ungated.

| Model | Revision | License | Safetensors |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B | `060db6499f32faf8b98477b0a26969ef7d8b9987` | Apache 2.0 | 1.0 GB |
| Qwen/Qwen2.5-1.5B | `8faed761d45a263340a0528343f099c05c9a4323` | Apache 2.0 | 3.1 GB |
| Qwen/Qwen2.5-3B | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | Qwen Research | 6.2 GB |
| Qwen/Qwen2.5-7B | `d149729398750b98c0af14eb82c78cfe92750796` | Apache 2.0 | 15.2 GB |
| Qwen/Qwen2.5-14B | `97e1e76335b7017d8f67c08a19d103c0504298c9` | Apache 2.0 | 29.5 GB |
| Qwen/Qwen2.5-72B | `efba10c8e54e91e0d9570ab5f7b51a958474d4cb` | Qwen | 145.4 GB (37 shards) |
| Qwen/Qwen2.5-72B-Instruct (eval reference only) | `495f39366efef23836d0cfae4fbe635880d2be31` | Qwen | 145.4 GB |

Qwen2.5-3B's research license and the 72B's Qwen license both allow this project's use. Whether to
publish the trained 72B adapter weights is decided in Phase 10 after reading the 72B license; code,
results and metrics are published regardless.

## 6. 72B task: BIRD text-to-SQL

- **Data:** BIRD train, 9,428 questions over 69 SQLite databases (`train.zip`, 8.9 GB). Dev: the
  cleaned `bird-sql-dev-1106` split released 2025-11 (1,534 questions; HF
  `birdsql/bird_sql_dev_20251106` at `3c11fb193e5439b338e23677fa0aae11e8b85db9`), with databases from
  `dev.zip` (346 MB). License CC BY-SA 4.0.
- **Metric:** execution accuracy (EX): the predicted and gold SQL return the same result set on the
  database, following the official evaluator (`llm/src/evaluation.py` in DAMO-ConvAI/bird), with a
  per-query timeout. Reported with a bootstrap CI.
- **Model selection without touching dev:** hold out whole databases from BIRD train as a validation
  split (dev databases never appear in train, so a database-level split mimics that shift). All sweeps
  and checkpoint choices use the validation split; dev is scored once per reported model.
- **Claim:** the gain over the base model under the same prompt and scorer. Qwen's pretraining data is
  unknown, so absolute scores carry a contamination caveat.
- **Reported as:** "BIRD dev (1106 split)", as the leaderboard asks.

## 7. `multigpu`

Process (RC docs): partition application form, then a temporary reservation to run 1/2/4/8-GPU tests;
scaling efficiency should be above 0.5 at the maximum GPU count. RC describes the partition as for
research workflows, so this project may not qualify. A draft question to RC is in
[rc-multigpu-request.md](rc-multigpu-request.md); it has not been sent. Nothing in Phases 1-8 waits on
the answer.

## 8. Workflow

- Git lives on a dev box; edits are synced there and committed. `slurm/*.sbatch` jobs are submitted
  from `/scratch/$USER/dlt/code/<sha>`, an exported copy of a committed HEAD, so each job runs exactly
  one commit (recorded as `GIT_SHA`).
- Logs go to `/scratch/$USER/dlt/logs`, results to `/scratch/$USER/dlt/results`; results are pulled,
  distilled into `results/` and `manifests/`, then committed.
- GPU jobs do work and exit. Debugging happens on CPU or on 0.5B models (IdleBot emails after 15 min
  of low GPU utilization and cancels after 1 h).

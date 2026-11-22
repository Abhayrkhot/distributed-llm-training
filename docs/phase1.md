# Phase 1: environment, run records, job templates, smoke tests

## 1. Environment

**Pinned stack** ([pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock)): Python 3.12, torch
2.14.1+cu126, transformers 5.24.0, peft 0.22.0, bitsandbytes 0.50.2, accelerate 1.16.0, NCCL 2.29.3
(the `nvidia-nccl-cu12` wheel torch pulls in).

**Why CUDA 12.6 wheels:** the H200 nodes run driver 570.86.15, whose newest native CUDA is 12.8. torch
2.13 and later no longer publish cu128 wheels, and of the CUDA 12.6-13.2 wheel indexes, torch 2.15.0 is
published only for CUDA 13.2. CUDA 13
builds would need NVIDIA's forward-compatibility libraries on this driver; cu126 runs natively, so
2.14.1+cu126 is the newest torch that needs no shim.

**Why a locked venv instead of an Apptainer image** (a change from the first plan):
- The lockfile pins every package with hashes, so the same environment resolves on the dev box, in CI
  and on Explorer. An image digest only identifies a build that happened once on one machine.
- A venv builds in a CPU job ([slurm/setup_env.sbatch](../slurm/setup_env.sbatch)) with no root,
  fakeroot or image transfer. /scratch is purged monthly, so rebuilding cheaply matters.
- The venv lives at `/scratch/$USER/dlt/venv/<first 12 hex of sha256(uv.lock)>`, so every commit with
  the same lockfile shares one venv, and each run record carries the lock hash.
- uv (0.13.9) installs its own CPython 3.12 (`--managed-python`), so the cluster's Python modules and
  anaconda never leak in. `PYTHONNOUSERSITE=1` blocks `~/.local` packages.

## 2. Run records

[src/dlt/results.py](../src/dlt/results.py): one JSON file per run under `results/<kind>/`, holding the
resolved config, git SHA, lock hash, Slurm job, node, GPU name, driver, package versions, per-step or
per-size metrics and a summary. `validate()` checks the schema; a test validates every committed record
and checks that cluster paths are scrubbed.

Numbers reach the README only through [scripts/gen_readme.py](../scripts/gen_readme.py), which
rewrites marked sections from `manifests/` and `results/`. CI runs it with `--check`.

Pulled records go through [scripts/import_results.py](../scripts/import_results.py) (validate,
scrub, copy) before they are committed.

## 3. Job templates

- [slurm/launch.sh](../slurm/launch.sh), sourced by every GPU job: sets up the environment, logs the
  node, GPU topology and CPU binding, and defines `dlt_torchrun NPROC module [args]`.
- [slurm/h200_1gpu.sbatch](../slurm/h200_1gpu.sbatch) (`gpu`, 8 h) and
  [slurm/h200_2gpu.sbatch](../slurm/h200_2gpu.sbatch) (`gpu-short`, 2 h) take a module and arguments.
- **Wall-time signal:** jobs request `--signal=B:USR1@300`. torchrun does not handle SIGUSR1 (its
  default action would kill it and orphan the workers), so the batch shell traps it and forwards it to
  torchrun's direct children, read from `/proc/<pid>/task/*/children`. The workers set a flag; every
  rank all-reduces it each step, so all ranks stop at the same step boundary. Phase 2 adds a checkpoint
  at that point.
- **CPU binding:** the 2-GPU template adds `--cpus-per-gpu=8 --gres-flags=enforce-binding`, because
  the Phase 0 probe got GPU1 on NUMA node 1 with all its CPUs on node 0.
- [slurm/chain.sh](../slurm/chain.sh) submits N links of a job with `--dependency=afterany`.
- **Queue limits met in practice:** `gpu-short` refused a fourth Phase 1 job
  (`QOSMaxSubmitJobPerUserLimit`) while other projects' jobs were also queued, and the per-user GPU limits
  count every job of the user. So the Phase 1 GPU checks run as one 2-GPU job,
  [slurm/phase1_smoke.sbatch](../slurm/phase1_smoke.sbatch).

## 4. Checks before GPU time

On the dev box (CPU, gloo):
- 10 unit tests: FLOP and parameter counts against the model cards (Qwen2.5-72B 72.7B, 0.5B 0.494B;
  the QLoRA count reproduces the 3.18e11 FLOPs/token in PLAN.md 4.4), nccl-tests bus-bandwidth
  factors, record schema, a tiny CPU training run whose loss must fall, README in sync.
- `torchrun --nproc-per-node=2` DDP on CPU: 8 steps, loss 6.23 -> 5.91.
- USR1 path: SIGUSR1 sent to torchrun's children after 8 s stopped both ranks at the same step (20)
  and the run exited 0 with its record written.

PHASE1_RESULTS

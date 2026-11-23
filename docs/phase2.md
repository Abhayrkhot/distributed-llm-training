# Phase 2: training engine with correctness and resume

## 1. Engine

[src/dlt/train/engine.py](../src/dlt/train/engine.py), run as `python -m dlt.train --config X.yaml
--set key=value` under torchrun (config fields: [src/dlt/train/config.py](../src/dlt/train/config.py)).

- **Backends:** `single`, `ddp` (DistributedDataParallel) and `fsdp2` (`fully_shard` on each decoder
  layer, then the root; `MixedPrecisionPolicy` with BF16 parameters for compute and FP32 gradient
  reduction).
- **Precision:** FP32 master parameters, gradients and AdamW moments. single and DDP compute under BF16
  autocast; FSDP2 casts parameters to BF16. Cross-entropy runs in FP32 on the logits.
- **Data:** FineWeb-Edu `sample/10BT/000_00000.parquet` at a pinned revision, tokenized once with the
  Qwen2.5 tokenizer into a flat uint32 file ([src/dlt/data/tokenize.py](../src/dlt/data/tokenize.py)):
  746,085,019 training tokens from 724,000 documents, and 2,096,082 validation tokens from the last
  2,000 documents. Sample i is the 2,049-token window starting at token i x 2,048, so samples never
  overlap and every target is a real token (no padding). Documents are joined with end-of-text tokens;
  attention is not masked across them.
- **Data order:** a pure function of (seed, step) ([src/dlt/data/packed.py](../src/dlt/data/packed.py)).
  Epoch e is a seeded permutation of all samples; step s takes the next `global_batch` of them; rank r
  takes a contiguous block and splits it into micro-batches. Resuming needs only the step.
- **Loss:** mean token cross-entropy over the global batch. Every sequence has exactly `seq_len`
  targets, so averaging micro-batch means over ranks and accumulation steps is exact.
- **Checkpoints** ([src/dlt/train/checkpoint.py](../src/dlt/train/checkpoint.py)): model and optimizer
  through DCP (`get_state_dict` / `dcp.save`), each rank's RNG state (python, numpy, torch, CUDA), the
  scheduler, step and resolved config. Written to `step_N.tmp`, then renamed by rank 0 after a barrier,
  so a checkpoint either exists completely or not at all; the newest `keep_ckpts` are kept. A stale
  `.tmp` from a crash is deleted by the next save.
- **Stopping:** SIGUSR1 (Slurm, 300 s before the limit, forwarded by
  [slurm/launch.sh](../slurm/launch.sh)), `max_seconds` or `stop_at_step` set a flag; every rank
  all-reduces it each step, so all ranks stop at the same step boundary, checkpoint and exit 0. A run
  that reaches `steps` checkpoints too, and a later job pointed at the same directory exits before
  loading the model.
- **Validation:** with `val_data` and `eval_samples`, the mean token loss over the first
  `eval_samples` validation sequences at step 0 and at the end of every job.

## 2. Tests

On CPU with gloo, in CI ([tests/test_train_cpu.py](../tests/test_train_cpu.py),
[tests/test_data.py](../tests/test_data.py)), on a 2-layer Qwen2 in deterministic mode:

| Property | How it is checked |
|---|---|
| Sampler | every sample exactly once per epoch across ranks and micro-batches; 1 and 2 ranks cover the same global batch |
| Stop and resume | stop at step 4, resume in a new process: later losses and gradient norms bit-identical to an uninterrupted run (single, DDP, FSDP2) |
| SIGKILL | kill every rank at step 5 with checkpoints every 2 steps; resume from step 4, bit-identical |
| Kill during a checkpoint write | SIGKILL after the write, before the rename: the run resumes from the previous checkpoint and the next save removes the `.tmp` |
| Gradient accumulation | micro-batch 2 x accumulation 4 matches micro-batch 8 x 1 (rtol 1e-5) |
| Backend parity | DDP x2 matches one process (rtol 1e-5); FSDP2 x2 within 2e-3 (BF16 compute vs the FP32 CPU run) |
| Validation | the same val loss on 1 and 2 ranks, unchanged by a resume; a further chain link exits without a record |

## 3. On H200s

Jobs 12066972 (parity and the DDP set, cancelled during the FSDP2 set) and 12073204 (the FSDP2 set),
2026-11-22, with [slurm/phase2_correctness.sbatch](../slurm/phase2_correctness.sbatch). Tables: the
README's "Engine correctness" section, generated from [results/train/](../results/train/).

- **Parity:** pretrained Qwen2.5-0.5B, 50 steps, global batch 16 x 2,048 tokens, micro-batch 4. Against
  the single-GPU run, DDP's per-step loss differs by at most 4.7e-4 (1.7e-4 relative) and FSDP2's by at
  most 8.0e-4 (2.9e-4 relative). Both are BF16-compute differences (summation order changes with the
  split across ranks and accumulation steps). They stay at the same order of magnitude over the run:
  DDP 3.8e-4 in the first 10 steps and 4.7e-4 in the last 20, FSDP2 6.2e-4 and 8.0e-4.
- **Resume:** deterministic mode with eager attention, 20 steps of 1,024 tokens. For DDP and FSDP2, stop
  at step 10 plus resume, and SIGKILL at step 13 plus recovery from the step-10 checkpoint, both
  reproduce every later loss and gradient norm of the uninterrupted run exactly.
- **Checkpoint cost:** 6.03 GiB for 0.5B: FP32 parameters and both Adam moments (12 bytes per
  parameter), plus the tied embedding stored a second time under its `lm_head` name. Saves took 9-27 s and restores 11-16 s on VAST NFS /scratch. The
  spread between identical saves (14 s vs 27 s for FSDP2) is the shared filesystem, so checkpoint
  timings are reported as ranges.
- **Throughput at this size:** 67k tokens/s on 1 GPU, 132k with DDP on 2 (98% scaling) and 125k with
  FSDP2 on 2. MFU is 22-24%; Phase 3 measures the ladder properly.

**Differences from the plan.** The plan asked for SIGTERM and SIGKILL kill tests and a final parameter
hash. Slurm's wall-time path is USR1 (tested above and in Phase 1), and SIGKILL covers abrupt death, so
SIGTERM was not tested separately. Resume is checked through every later per-step loss and gradient
norm rather than a parameter hash; the gradient norm at each step depends on every parameter.

## 4. Training across chained jobs

One run of 4,500 steps of pretrained Qwen2.5-0.5B (FSDP2, 2 H200s, global batch 16 x 2,048 tokens,
147M tokens in all; [configs/training/p2_chain_0p5b.yaml](../configs/training/p2_chain_0p5b.yaml))
submitted as two chained 20-minute jobs with [slurm/chain.sh](../slurm/chain.sh)
([slurm/phase2_chain.sbatch](../slurm/phase2_chain.sbatch)), 2026-11-22.

| | Job 12073205 | Job 12073206 |
|---|---|---|
| Steps | 0-2,986 | 2,986-4,500 |
| Ended by | Slurm USR1, 300 s before the limit | reaching step 4,500 |
| Checkpoint at the boundary | step 2,986, written in 16.2 s; job exited 21 s after the signal | restored in 8.1 s |
| Throughput | 123,666 tokens/s | 126,759 tokens/s |
| Validation loss | 2.6886 at step 0, 2.7151 at step 2,986 | 2.6965 at step 4,500 |

- **The boundary is invisible in the run:** the learning rate continues from 1.0149e-5 to 1.0140e-5 (the
  cosine schedule's next value), and the mean training loss is 2.699 over the last 100 steps of job 1
  and 2.691 over the first 100 of job 2. Bit-for-bit equality across a resume is shown in section 3; this
  run shows the same mechanism under a real Slurm wall-time signal, with the queue wait (10 minutes)
  between the jobs.
- **The 300 s margin is ample at this size:** stopping, saving 6 GiB and evaluating took 21 s. For
  larger models the margin matters more; save times are logged in every record.
- **Validation loss does not improve:** it rises from 2.689 to 2.715 while the learning rate is near its
  3e-5 peak and recovers to 2.697 as it decays. This is the usual re-warming bump when a fully trained
  model's learning rate is raised again (Gupta et al., "Continual Pre-Training of Large Language Models:
  How to (re)warm your model?", 2023): 147M tokens of FineWeb-Edu do not move a model trained on 18T
  tokens. The run's purpose is the job boundary, not a better model.

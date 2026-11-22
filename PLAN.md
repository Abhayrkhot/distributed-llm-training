# Distributed LLM Training on Explorer H200s: Project Plan

**Primary objective:** a reproducible Qwen2.5-72B QLoRA fine-tune on a real task, with measured memory, throughput and held-out quality, that survives being split across Slurm jobs.

**Secondary objective:** a full-parameter DDP vs FSDP2 scaling and memory study on the Qwen2.5 size ladder (0.5B to 14B), with MFU, communication and profiler evidence.

**Hardware:** Northeastern Explorer H200s. 1 GPU per job for up to 8 h, 2 GPUs per job for up to 2 h. 4+ GPUs only if `multigpu` access is granted, which is not assumed.

**Schedule:** milestone-based. The binding constraint is GPU-hours and queue time, not coding time.

**Repo name:** `distributed-llm-training` (working name).

---

## 1. What the finished project shows

1. A training engine (DDP and FSDP2) whose correctness is tested, not assumed: loss parity across GPU counts, gradient-accumulation equivalence, and bit-exact resume after a kill.
2. A scaling and memory study on 0.5B-14B models with MFU, peak memory, step-time breakdown and exposed communication time, plus a memory predictor checked against every run.
3. Qwen2.5-72B fine-tuned with QLoRA on a task with an objective metric, beating the base model on a held-out set, with the main run spanning multiple Slurm jobs through checkpoint/resume.
4. A measured answer to "should you shard a 4-bit 72B base across GPUs?" (replicated vs sharded on 2 H200s).

Every number in the README comes from a results file produced by a run in this repo.

## 2. Changes from the first draft

| Change | Reason |
|---|---|
| 1-2 GPUs is the baseline; 4+ is a gated bonus | A job gets at most 2 GPUs (`gpu-short`, 2 h); `gpu` is 1 GPU per job; `multigpu` needs RC approval and is meant for research workflows |
| Checkpoint/resume moved into the engine phase | Any 2-GPU run longer than 2 h must resume across jobs, so resume is a prerequisite, not a final phase |
| The 72B fine-tune has a task and a metric | A decreasing training loss is not evidence that the model learned anything useful |
| MFU and HFU reported with every throughput number | Tokens/sec is not comparable across model sizes or hardware |
| A100 reproduction phase removed | No code or logs for the 39K/105K/133K tokens/s numbers were found, and a 4-A100 job is not available anyway |
| DeepSpeed ZeRO-3 cut to one optional matched comparison | Most plumbing for the least new information |
| Full-parameter 72B reduced to a gated memory experiment | CPU offload cannot fit in available node RAM; only low-bit optimizer states across 4+ GPUs could fit |
| Qwen2.5 size ladder for the small models | Same architecture family as the 72B, so sizes compare cleanly |

## 3. Cluster facts (checked 2026-11-22)

From `sacctmgr`, `scontrol`, `sinfo` and the RC docs:

| Partition / QOS | Max wall | GPUs per job | GPUs per user | H200 nodes |
|---|---|---|---|---|
| `gpu` | 8 h | 1 | 4 | yes |
| `gpu-short` | 2 h | 2 | 2 | yes |
| `gpu-interactive` | 2 h | 1 | none set | yes |
| `sharing` | 1 h | 1 | 2 | d4077 only |
| `courses-gpu` | 1 day | 1 | 2 | no |
| `multigpu` | 24 h | 8 | 8 | yes, approval required |

- **H200 nodes:** d4052-d4055 have 8x H200, 128 cores and 500 GiB RAM each, shared by all users. d4077 (in `sharing`) has 8x H200 and 1.1 TiB RAM.
- **Accounts:** the available Slurm accounts have QOS `normal` only, so no `multigpu` access today.
- **`multigpu` access:** partition application form (ServiceNow). RC gives a temporary reservation to run 1/2/4/8-GPU tests; scaling efficiency should be above 0.5 at the maximum GPU count. RC describes the partition as for research workflows, so a personal project may not qualify. Ask before planning around it.
- **IdleBot:** emails after 15 min of low GPU utilization and cancels the job after 1 h. Logs via `gpu-logs`. Keep model loads and checkpoint writes well under 15 min, and never hold a GPU for interactive debugging.
- **Storage and network:** /scratch is NFS (a file can take ~30 s to appear on other nodes). Compute nodes set `http_proxy`. Login nodes are often overloaded: one SSH connection per step, tar over ssh, poll at 90 s or more.
- **Queueing:** requesting `--gres=gpu:h200:N` specifically can mean longer waits. Node RAM, not GPUs, was the limiting request in past H200 jobs, so set `--mem` explicitly.

**Still unknown (Phase 0):** the exact H200 variant (SXM or NVL, which sets peak FLOPS and topology), which GPU pairs share NVLink (`nvidia-smi topo -m`), driver and CUDA versions on the H200 nodes, scratch quota, and whether Nsight Systems/Compute work without RC help.

## 4. Feasibility math

All memory is in GiB, which is what PyTorch and `nvidia-smi` report. An H200 shows about 140 GiB.

### 4.1 Qwen2.5-72B

72.7B parameters; 80 layers, hidden 8,192, MLP 29,568, 64 query / 8 KV heads, vocab 152,064, untied embeddings.

| State | Bytes per param | GiB |
|---|---|---|
| BF16 weights | 2 | 135 |
| BF16 gradients | 2 | 135 |
| FP32 Adam m + v | 8 | 542 |
| FP32 master weights | 4 | 271 |
| Full training, no master copy | 12 | 812 |
| Full training, with master copy | 16 | 1,083 |
| NF4 linear layers (double quant) + BF16 embeddings and lm_head | ~4.1 bits + 4.6 GiB | ~38 |

These exclude activations, workspace, collective buffers and fragmentation.

What follows from the table:

- **BF16 on one GPU:** the BF16 weights alone (135 GiB) leave about 5 GiB on one H200. No BF16 training on one GPU; BF16 inference needs 2.
- **NF4 on one GPU:** the NF4 base (~38 GiB) leaves about 100 GiB on one H200 for activations, LoRA states and workspace. Single-GPU QLoRA is comfortable, so multi-GPU QLoRA is about throughput, not memory.
- **LoRA size:** LoRA r=16 on all 7 linear projections per layer is 210.5M trainable parameters (0.29%). Their BF16 gradients are 0.39 GiB per step to all-reduce, negligible next to a multi-second step. **Prediction:** replicated NF4 + DDP beats a sharded NF4 base on 2 GPUs. Phase 7 tests it.
- **Full-parameter 72B:** CPU offload needs more than 812 GiB of host RAM, and d4052-d4055 have 500 GiB shared by 8 GPUs, so offload is out. The only path is FSDP2 with BF16 parameters and gradients plus 8-bit optimizer states: 6 bytes per param, 406 GiB in total, about 102 GiB per GPU on 4 GPUs before activations (51 GiB on 8). It needs `multigpu` access and stochastic rounding (or similar) for BF16 updates. Gated stretch, Phase 9.

### 4.2 Qwen2.5 size ladder, full-parameter

FP32 master parameters and gradients with BF16 autocast, plus Adam: 16 bytes per param, before activations.

| Model | Params | State (GiB) | 1 GPU | 2 GPUs, DDP | 2 GPUs, FSDP2 (per GPU) |
|---|---|---|---|---|---|
| 0.5B | 0.49B | 7.3 | yes | yes | 3.7 |
| 1.5B | 1.54B | 23 | yes | yes | 11.5 |
| 3B | 3.09B | 46 | yes | yes | 23 |
| 7B | 7.61B | 113 | borderline | borderline | 57 |
| 14B | 14.7B | 219 | no | no | 110 |

- 7B on one GPU is a measured outcome, not an assumption. 14B has no 1-GPU baseline, so its 2-GPU result is reported as MFU, not scaling efficiency.
- Licenses differ across the family: Qwen2.5-3B uses the Qwen Research license, Qwen2.5-72B the Qwen license, and the rest Apache 2.0. Confirm on the model cards at the pinned revisions.

### 4.3 MFU convention

- **Full training:** 6N + 12·L·d·T FLOPs per token (PaLM convention). N is the non-embedding parameters plus lm_head, L the layers, d the attention width (heads × head dim), T the sequence length. This counts the full attention matrix even with causal masking; every table states the convention.
- **QLoRA:** 4N + 12·L·d·T (the frozen base has no weight-gradient GEMMs).
- **HFU:** with activation checkpointing on, HFU adds the recomputed forward pass, 2N + 4·L·d·T.
- **Peak:** the datasheet dense BF16 number for the exact variant `nvidia-smi` reports (989 TFLOPS for H200 SXM).

### 4.4 Throughput estimate for budgeting (not a target)

For 72B QLoRA at T = 4,096, the convention above gives about 3.2e11 FLOPs per token. At 15-30% MFU on an H200 SXM, that is roughly 450-950 tokens/s on one GPU, so a 10M-token run costs about 3-6 GPU-hours: one or two chained 8 h `gpu` jobs. The measured MFU is a result of the project.

## 5. GPU-hour budget (rough)

| Phase | GPU-hours |
|---|---|
| 0-1: manifest, smoke tests, NCCL bandwidth | 2 |
| 2: engine correctness and resume tests | 8 |
| 3: scaling and memory study (5 sizes, 1 and 2 GPUs, 3+ repeats) | 30 |
| 4: profiling and tuning | 10 |
| 5: 72B load and NF4 vs BF16 check | 4 |
| 6: 72B QLoRA overfit, sweeps, main run, eval | 25 |
| 7: 72B replicated vs sharded on 2 GPUs | 8 |
| 8: fault-injection report | 6 |
| **Total** | **~95** |

Track actual usage per phase in the results log and re-plan if a phase runs 2x over.

## 6. Scope

**In scope**
- Slurm execution on 1 and 2 H200s, with 4 and 8 only if `multigpu` is granted.
- Pinned containers, deterministic configs, structured results, generated README.
- Full-parameter DDP and FSDP2 on the Qwen2.5 ladder.
- Qwen2.5-72B NF4 QLoRA on one task, evaluated against the base model.
- Profiling with PyTorch Profiler and Nsight Systems where allowed.
- Distributed checkpoint/resume with tested correctness.

**Out of scope**
- Pretraining or full-parameter training of 72B (beyond the gated memory experiment).
- Reproducing the old A100 numbers.
- DeepSpeed as a primary backend.
- Any throughput target set before measurement.
- Serving or deploying the fine-tuned model beyond what evaluation needs.

## 7. Phases

### Phase 0: Decisions and resource manifest

Already done: QOS limits, H200 nodes and the `multigpu` process (Section 3).

**Tasks**
- In a short H200 job that does work and exits, record:
  - `nvidia-smi -q`: variant, clocks, power limit, driver.
  - `nvidia-smi topo -m`: which GPU pairs share NVLink, since Slurm may hand you a pair that doesn't.
  - CUDA and NCCL versions, and the CPU/NUMA layout.
  - Commit it all as `manifests/h200.json`.
- Check scratch space and file-count quotas. Budget about 250 GB: 72B BF16 (145 GB), NF4 (~41 GB), datasets and checkpoints.
- Check whether `nsys` runs inside a job and whether `ncu` needs RC to enable counters.
- Ask RC whether this project qualifies for `multigpu`. Don't block on the answer.
- Confirm the 72B task (default in Phase 6) and the licenses of the models and dataset.

**Exit:** manifest committed, task chosen, quota known.

### Phase 1: Environment, results schema, smoke tests

**Tasks**
- **Containers:**
  - Training: one Apptainer image with pinned torch, transformers, peft, bitsandbytes, torchao and accelerate (deepspeed only if the optional Phase 4 comparison runs).
  - Eval: a separate vLLM image; reuse the existing vLLM 0.34 image if its versions fit.
  - Build images on node-local /tmp, not /scratch.
- **Results schema, before any benchmark:**
  - One JSON per run: resolved config, git SHA, image digest, node ID, GPU variant, software versions, per-step metrics and summary stats.
  - `scripts/gen_readme.py` builds the README tables from `results/`; CI runs it with `--check`.
- **Slurm scripts:**
  - `h200_1gpu.sbatch` (`gpu`, 8 h) and `h200_2gpu.sbatch` (`gpu-short`, 2 h), both with `--signal=B:USR1@300` for a checkpoint before the wall-time limit.
  - `chain.sh` submits follow-on jobs with `--dependency=afterany`.
- **Smoke tests:**
  - A 1-GPU training step.
  - A 2-GPU NCCL all-reduce / all-gather / reduce-scatter bandwidth sweep (nccl-tests or a torch script).
  - A 2-rank training step.

**Exit:** smoke results in `results/`, with the README table generated from them.

### Phase 2: Training engine with correctness and resume

**Models:** Qwen2.5-0.5B and 1.5B.

**Data:** a fixed FineWeb-Edu shard, pre-tokenized once on CPU into a uint32 memmap and packed into fixed-length sequences. With no padding, every counted token is real.

**Tasks**
- **Backends:** single GPU, DDP, and FSDP2 (`fully_shard` per decoder block; mixed-precision policy with BF16 compute and FP32 reduce).
- **Deterministic data order:** the global sample order comes from (seed, epoch), each rank takes its slice of each global batch, and the data cursor is just the global step.
- **Checkpointing:**
  - Model and optimizer through DCP (`torch.distributed.checkpoint` with `get_state_dict` / `set_state_dict`).
  - Plus the scheduler, step, per-rank RNG (python, numpy, torch, cuda), data cursor and resolved config.
  - Write to a temp directory and rename; keep the last 2; save on USR1.
- **Tests (all must pass before Phase 3):**
  - Loss parity: single GPU vs DDP vs FSDP2 at the same global batch, first 50 steps within a stated BF16 tolerance.
  - Gradient accumulation: microbatch 4 × accumulation 2 matches microbatch 8 × accumulation 1 within tolerance.
  - Sampler: across ranks and steps, every sample is seen exactly once per epoch.
  - Resume parity: in deterministic mode, kill at step k (SIGTERM and SIGKILL) and resume. Per-step losses after k and the final parameter hash must be bit-identical to an uninterrupted run.
  - Kill during a checkpoint write: resume falls back to the previous checkpoint.
  - One real run that spans 2 chained Slurm jobs.

**Exit:** test suite green on 1 and 2 H200s.

### Phase 3: Full-parameter scaling and memory study

**Tasks**
- **Ladder:** 0.5B, 1.5B, 3B, 7B, 14B. DDP where it fits, FSDP2 wherever it fits, on 1 and 2 GPUs.
- **Fixed per comparison:** model, precision, data, sequence length and global batch. Any exception is stated in the table.
- **Per run:**
  - Tokens/s, MFU, HFU.
  - Step-time split: forward, backward, optimizer, exposed communication.
  - Peak allocated and reserved memory per rank, and host RAM.
  - 3+ repeats, reported as median and range.
- **Activation checkpointing on vs off:** memory saved vs throughput lost.
- **Scaling efficiency** = throughput_2 / (2 × throughput_1) where a 1-GPU run fits. For 14B, report 2-GPU MFU instead and say why.
- **Memory predictor:**
  - Formula for parameters, gradients, optimizer state and activations.
  - Log predicted vs measured peak for every run, including OOMs.
  - An OOM run saves its config and a memory snapshot.

**Exit:** tables and plots generated from results; predictor error reported.

### Phase 4: Profiling and tuning

**Tasks**
- **Workloads:** 7B FSDP2 on 2 GPUs (the most communication) and 1.5B DDP.
- **Profilers:** PyTorch Profiler and Nsight Systems, covering the kernel timeline, NCCL kernels, exposed (non-overlapped) communication, CPU gaps and dataloader stalls.
- **Candidates,** one at a time, each with before/after numbers over 3+ repeats:
  - FSDP2 prefetch and `reshard_after_forward`.
  - DDP bucket size.
  - Microbatch × accumulation, and sequence length.
  - SDPA backends vs FlashAttention (FA3 if it builds; the H200 is Hopper).
  - Chunked or fused cross-entropy (the 152K vocab makes logits a large share of activation memory).
  - Where activation checkpointing is applied.
- **Optional:** one matched ZeRO-3 vs FSDP2 point (7B, 2 GPUs).

**Exit:** profiler-backed report covering root cause, change, measured effect and variance.

### Phase 5: Qwen2.5-72B integration

**Tasks**
- **Download and verify:** pin a commit of `Qwen/Qwen2.5-72B`, download it in a CPU job, and write a checksum manifest. Check that the tokenizer round-trips text and that special tokens are correct.
- **Quantize once:** NF4 with double quant and BF16 compute, then save. Record load time from /scratch for BF16 (145 GB) vs the saved NF4.
- **Correctness:** on 200 held-out task examples, compare NF4 (1 GPU) against BF16 (2 GPUs, layers split) on next-token agreement and the perplexity gap.
- **Record:** peak GPU memory, load time and host RAM.

**Exit:** validated NF4 forward pass with recorded numbers.

### Phase 6: 72B QLoRA on 1 H200 (primary milestone)

**Task (default; change it in Phase 0 if preferred):** text-to-SQL on BIRD.
- Train on BIRD train and report execution accuracy on BIRD dev (1,534 questions; the test set is hidden).
- The metric is objective and the base model is not saturated on it.
- Qwen's pretraining data is unknown, so the claim is the gain over the base model, not the absolute score.

**Tasks**
- **Baselines:** the base 72B with few-shot prompts, and Qwen2.5-72B-Instruct zero-shot as a reference ceiling. Same prompt format and scorer.
- **Eval path:**
  - Use vLLM with the NF4 base + adapter if the pinned vLLM supports bitsandbytes with LoRA for Qwen2.
  - Otherwise use a BF16 base with tensor parallel 2 on `gpu-short`.
  - Either way, check greedy-output agreement against HF + PEFT on 200 examples.
- **Trainable-parameter audit:** list every trainable tensor and assert the base is frozen (checksum the base weights before and after a step).
- **Overfit test:** 64 examples, trained to near-zero loss.
- **Short sweeps at a fixed token budget:** LoRA rank 16 vs 64, sequence length, activation checkpointing on/off, attention backend.
- **Main run:** spans at least 2 chained `gpu` jobs on purpose, so resume is exercised on the real workload. Adapter checkpoints include optimizer, scheduler, RNG and data cursor.
- **Eval:** report confidence intervals (bootstrap over questions), then reload the saved adapter in a fresh job and reproduce the eval.

**Exit:** held-out gain over the base model with a CI, a reloadable adapter, and memory/MFU traces.

### Phase 7: 72B on 2 H200s, replicated vs sharded

**Tasks**
- **Strategy A (replicated):** each rank loads the full NF4 base (~38 GiB); DDP covers the adapter parameters only.
- **Strategy B (sharded):** shard the NF4 base, using whichever of these works with the pinned versions, and record what failed:
  - FSDP1 with `bnb_4bit_quant_storage=bfloat16` (the documented HF/PEFT path).
  - FSDP2 with torchao NF4 tensors.
- **Setup:** same global batch; loss parity over the first N steps between A, B and the 1-GPU run.
- **Report:** per-rank memory, step time, communication time and MFU. The prediction is that A is faster and B saves memory that isn't needed.
- **Wall time:** runs fit in 2 h or chain via resume.

**Exit:** measured answer with traces.

### Phase 8: Fault-tolerance report

Resume has worked since Phase 2; this phase quantifies it.

**Tasks**
- **Kill points:** mid-step, mid-checkpoint-write, during eval, SIGTERM vs SIGKILL, `scancel`, and wall-time expiry.
- **Measure:** checkpoint size, save time (synchronous vs DCP `async_save`), restore time, and work lost per failure.
- **Cadence:** save on USR1 before the wall-time limit, plus every K steps, with K sized so lost work stays under a stated bound (Young's formula for random failures).

**Exit:** resilience table for small-model FSDP2 and for 72B QLoRA.

### Phase 9 (gated): 4+ GPUs

Only if RC approves `multigpu`.

**Tasks**
- **Application evidence:** the Phase 3 2-GPU efficiency numbers. In RC's test reservation, run 1/2/4/8-GPU tests of 7B and 14B FSDP2 and of 72B replicated QLoRA.
- **If granted:** extend the Phase 3 and Phase 7 tables. Run the full-parameter 72B memory experiment (FSDP2, BF16 parameters and gradients, 8-bit optimizer states, 4 and 8 GPUs) for a few steps to measure memory and step time; it is not a training run.

**Exit:** either 4/8-GPU results, or a note that access was not granted.

### Phase 10: Report and release

**Tasks**
- README tables and plots generated from `results/`, with CI `--check`.
- A manifest for every run; median and range for every benchmark.
- **Plots:**
  - Throughput and MFU by model size and GPU count.
  - Peak memory, measured vs predicted.
  - Step-time breakdown and exposed communication.
  - Checkpoint overhead.
  - Training and eval loss.
  - BIRD accuracy, base vs fine-tuned.
- Failed configurations and dead ends logged in `docs/RESULTS_LOG.md`.
- Every resume/GitHub claim links to a results file.

## 8. Definition of done

- [ ] Resource manifest for the H200 nodes committed.
- [ ] DDP and FSDP2 pass the loss-parity, accumulation, sampler and bit-exact resume tests on 1 and 2 H200s.
- [ ] Scaling and memory study on 0.5B-14B covers MFU, HFU, memory, step breakdown and communication, with 3+ repeats and the predictor error.
- [ ] At least one profiler-backed optimization with before/after numbers.
- [ ] 72B NF4 forward pass validated against BF16.
- [ ] 72B QLoRA run passes the frozen-base audit, spans 2+ Slurm jobs, and beats the base model on the held-out set (with a CI); a reloaded adapter reproduces the eval.
- [ ] Replicated vs sharded 72B on 2 GPUs measured.
- [ ] Fault-tolerance table published.
- [ ] README generated from results; no number without a run behind it.

**Stretch:** `multigpu` results; the full-parameter 72B memory experiment.

## 9. Measurement rules

- **Timing:** synchronize (CUDA events) before timing; exclude the first N steps and report N.
- **Token counts:** tokens are counted after packing, with no padding; say so.
- **MFU:** state the MFU/HFU convention and peak FLOPS in every table.
- **Memory:** report GiB per rank from `torch.cuda.max_memory_allocated` / `max_memory_reserved`.
- **Repeats:** 3+ repeats with median and min-max. Record the node ID and other jobs on the node, since shared nodes are noisy.
- **Fair comparisons:**
  - Hold global batch, precision, sequence length and data fixed when comparing strategies; list exceptions in the table.
  - Never present results from different model sizes as a scaling result.
  - Never mix training and inference throughput.
- **Failures:** OOMs and failed runs are results; log them with their configs.

## 10. Experiment matrix

| Study | GPUs | Model | Main metrics |
|---|---|---|---|
| Smoke and NCCL bandwidth | 1, 2 | none / 0.5B | bus bandwidth, working step |
| Engine correctness | 1, 2 | 0.5B, 1.5B | loss parity, resume parity |
| DDP vs FSDP2 scaling | 1, 2 | 0.5B-7B (14B FSDP2 on 2 only) | tokens/s, MFU, efficiency, memory |
| Activation checkpointing | 1, 2 | 1.5B-14B | memory saved vs throughput cost |
| Profiling and tuning | 2 | 7B FSDP2, 1.5B DDP | exposed comm, kernel time, before/after |
| ZeRO-3 (optional) | 2 | 7B | matched vs FSDP2 |
| 72B load and NF4 check | 1, 2 | 72B | load time, memory, agreement vs BF16 |
| 72B QLoRA | 1 | 72B | BIRD execution accuracy, loss, MFU, memory |
| 72B replicated vs sharded | 2 | 72B | memory, step time, comm |
| Fault injection | 1, 2 | 1.5B, 72B | restore correctness, save/restore time, lost work |
| Gated scaling | 4, 8 | 7B, 14B, 72B | efficiency, comm |

## 11. Repository layout

```
distributed-llm-training/
  PLAN.md
  README.md                  tables generated from results/
  pyproject.toml
  containers/                Apptainer definitions (train, eval)
  configs/
    models/
    training/
    benchmarks/
  src/dlt/
    data/                    tokenize, pack, deterministic sampler
    models/                  loading, NF4, LoRA attach and audit
    distributed/             ddp.py, fsdp2.py, deepspeed.py (optional)
    training/                loop, accumulation, precision
    checkpointing/           DCP, adapter checkpoints, USR1 handling
    metrics/                 MFU/HFU, memory, timing, run manifest
    evaluation/              BIRD execution accuracy, vLLM client
  slurm/                     h200_1gpu.sbatch, h200_2gpu.sbatch, chain.sh
  scripts/                   gen_readme.py, explorer push/pull, pack_results
  tests/                     CPU unit tests and the GPU correctness suite
  results/                   small JSON results (committed)
  manifests/                 hardware and artifact manifests with checksums
  docs/                      RESULTS_LOG.md, figures
```

Not committed: weights, datasets, Hugging Face tokens, large profiler traces, cluster connection details. Manifests with checksums and paths stand in for them.

## 12. Risks

| Risk | Mitigation |
|---|---|
| `multigpu` not granted (likely, since it is research-only) | The plan is complete at 1-2 GPUs; Phase 9 is a bonus |
| 2 h wall time on 2-GPU jobs | Resume from Phase 2, USR1 save before the wall-time limit, chained jobs |
| IdleBot cancels jobs during long loads or idle time | Pre-quantized NF4 checkpoint; no idle dev jobs; debug on CPU or 0.5B |
| H200 queue waits | 1-GPU work on `gpu` (8 h); short jobs; small models where possible |
| Quantized sharding incompatible with pinned versions | Replicated path A is primary; B is documented either way |
| vLLM lacks bitsandbytes + LoRA for Qwen2 | BF16 TP=2 eval on `gpu-short`, with an agreement check vs HF |
| BIRD contamination in Qwen pretraining | Claim the gain over the base model, not the absolute score |
| Scratch quota | Check in Phase 0; one copy of each weight set; delete BF16 weights after Phases 5-6 |
| Noisy shared nodes | 3+ repeats, node ID recorded, median and range |
| Login-node overload | One SSH connection per step, tar over ssh, polling at 90 s or more |
| Nsight not permitted | PyTorch Profiler fallback; document what could not be seen |

## 13. Old resume numbers

The earlier bullet numbers (4x A100 DDP, 39K/105K/133K tokens/s, 85% efficiency) are retired: no code or logs back them, and a 4-A100 job is not available to reproduce them. New claims come only from this repo's results files.

## 14. References

- PyTorch FSDP2: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- PyTorch Distributed Checkpoint: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
- PyTorch Profiler: https://docs.pytorch.org/docs/stable/profiler.html
- torchao (NF4 tensors, low-bit optimizers): https://github.com/pytorch/ao
- PEFT: https://huggingface.co/docs/peft/index
- PEFT with FSDP (including QLoRA): https://huggingface.co/docs/peft/accelerate/fsdp
- Transformers quantization: https://huggingface.co/docs/transformers/quantization/overview
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- Qwen2.5-72B: https://huggingface.co/Qwen/Qwen2.5-72B
- BIRD benchmark: https://bird-bench.github.io/
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Explorer GPU overview: https://rc-docs.northeastern.edu/en/latest/gpus/gpuoverview.html
- Explorer `multigpu` access: https://rc-docs.northeastern.edu/en/latest/gpus/multigpu-partition-access.html
- Explorer IdleBot: https://rc-docs.northeastern.edu/en/latest/gpus/idlebot.html

**Guiding principle:** measured engineering quality over parameter count. A tested distributed engine, an honest scaling study and a 72B adapter that measurably beats its base model are a complete project at 1-2 GPUs.

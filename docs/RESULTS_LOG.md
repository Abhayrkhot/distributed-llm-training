# Results log: failures, dead ends and fixes

Every configuration that failed and every bug the measurements exposed, in the order they were found,
with the job that showed it. Failed runs stay in `results/` (OOM and failure records are results);
records invalidated by a bug are kept and listed in [results/invalid.json](../results/invalid.json).

## Out-of-memory configurations (by design: probes at the 139.8 GiB limit)

| Configuration | Predicted (v3) | Outcome | Phase 3 entry |
|---|---|---|---|
| 7B single GPU, no checkpointing | 145.1 GiB | OOM in step 2 | `probe-7b-single` |
| 7B DDP on 2 GPUs, checkpointing | 160.4 GiB | OOM in step 1 | `probe-7b-ddp-ac` |
| 14B FSDP2 on 2 GPUs, checkpointing, no-sync accumulation | 148.5 GiB | OOM in step 2 | `probe-14b-fsdp2-ac-last` |
| 14B FSDP2 on 2 GPUs, no checkpointing, per-micro-batch sync | 140.5 GiB | OOM in step 2 | `probe-14b-fsdp2-every` |

## Bugs the measurements found

| Phase | What happened | How it showed | Fix |
|---|---|---|---|
| 1 | The README generator never matched an empty `gen` section, so every table stayed blank and `--check` passed | blank README tables after the first GPU results | regex no longer needs a body line; a test renders an empty section |
| 2 | A periodic save and a stop at the same step wrote the same checkpoint twice; the rename failed | `os.replace` "Directory not empty" on CPU | saved-step set dedupes |
| 2 | YAML read `1e-3` as a string | AdamW TypeError | config values are coerced by the field's type |
| 3 | `from_pretrained` returns the model in eval mode and transformers checkpoints only in train mode, so activation checkpointing was silently off for single-GPU and DDP | 1.5B and 3B single-GPU runs with and without checkpointing had identical memory and throughput (job 12073550) | `model.train()` after loading; a test counts layer recomputes; the 9 records invalidated and rerun (job 12075847) |
| 3 | The 7B single-GPU "checkpointing" OOM was the same bug (the run was the no-checkpointing configuration) | OOM at 136 GiB with a 132 GiB prediction | with checkpointing working it fits at 129.2 GiB |
| 3 | Predictor v1 counted FSDP2's sharded gradients and its unsharded no-sync accumulation together | FSDP2 runs over-predicted by up to 21% | v2: the two never coexist at the peak (read from the FSDP2 source) |
| 3 | Predictor v2 kept autocast's BF16 weight copies until backward under checkpointing | single-GPU checkpointed runs over-predicted by 8-10% | v3: peak = max(end of forward, start of backward) |
| 3 | Two sweep runs finishing in the same second wrote the same record file | the sweep's own CPU test | run IDs carry the process ID |
| 3 | FSDP2-vs-autocast scaling rows above 100% compared two BF16 mechanisms | 100.4-100.5% efficiencies | FSDP2 1-GPU baselines; like-for-like table |
| 3 | A 1-GPU FSDP2 run started without torchrun had no process group for the device mesh | all 18 FSDP2 1-GPU runs failed in seconds (job 12076199) | `dlt.dist.ensure_group` creates a one-process group; rerun in job 12082451 |
| 4 | The Nsight summary grouped kernels by short name and put FSDP2 copies and gradient-norm kernels under "optimizer" (all launch as `multi_tensor_apply_kernel`) | 0.57 s/step "optimizer" vs 0.05 s measured | categories from the demangled name (template arguments) |
| 4 | PyTorch Profiler traces with CPU events would have been hundreds of MB per rank at 16-32 micro-batches per step | estimate before the sweep ran | GPU kernels only, 2 steps, 2-GPU runs only |
| 5 | `train.zip` holds macOS metadata (`__MACOSX/._*.zip`) that the nested extraction tried to open | BIRD prep failed after the download (job 12076688) | metadata skipped |
| 5 | BIRD gold SQL "timed out" on a 64 KB database: SQLite file locks over VAST NFS, 16 queries at a time | 801 train and 82 dev golds at 30 s (job 12076960); probe job 12077462 | node-local copies opened with `immutable=1` (median 1.8 s and 0.6 s instead of 34.9 s and 27.4 s) |
| 5 | The first validation split drew 3 databases, one with a 5,000-token schema (dev's largest: 1,861) | prompt statistics in the manifest | val databases limited to schemas of at most 2,500 tokens, at least 6 |
| 6 | Passing `quantization_config=None` overrode the saved NF4 checkpoint's own config and built an unquantized model | overfit test and LoRA sweep failed at load (jobs 12077474, 12077475) | the argument is omitted for pre-quantized checkpoints |
| 7 | The adapter audit counted FSDP1's FlatParameters at their local size, which is this rank's shard, so the 72B FSDP1 run reported 105.3M of the expected 210.5M LoRA parameters and stopped | job 12100515 failed after loading and sharding the model | the audit counts the original parameters each FlatParameter stands for; rerun in job 12107944 |
| 7 | The FSDP1 rerun spent its whole 90 min run timeout loading: it quantizes while reading the 145 GB BF16 checkpoint through memory-mapped safetensors over NFS, which on a busy filesystem ran at about 17 MB/s (the same read took 6 min in the previous job), and it needs a 300 GB host-memory request that the H200 nodes rarely have free | job 12107944 was 64% through `Loading weights` when the sweep stopped it | [scripts/nf4_storage.py](../scripts/nf4_storage.py) re-stores the saved 38 GiB NF4 checkpoint with its packed weights as BF16 bytes (what FSDP1 needs to flatten a layer): converted in 152 s (job 12111554), loaded in 2.5 min with a 128 GB request, and the sharded run uses exactly the 4-bit values of the replicated runs |
| 7 | The frozen-base fingerprint read FSDP1 FlatParameters after the first step, when their storage (the full-size buffer, and also `_local_shard`) had been freed by the reshard | jobs 12111555 and 12111943 failed right after step 0 with `storage of size 0` | under FSDP1 the fingerprint runs inside `FSDP.summon_full_params(writeback=False)`, before and after the step, and every rank's result is combined |
| 7 | The FSDP1 frozen-base check gathers the whole base onto every rank, and that one-off allocation became the run's peak memory | job 12112017 reported 61.2 GiB per rank for a sharded run, more than the replicated 52.4 GiB | the check's peak goes into the audit (`check_peak_gib`), its cache is released and the peak statistics restart; the record is in `results/invalid.json`, rerun in job 12112395 |
| 7 | Under Nsight Systems (CUDA API tracing on) the FSDP1 run hung at its first profiled step: rank 0 stopped issuing collectives two all-gathers before rank 1, and the NCCL watchdog ended the job after its 10 min timeout. The DDP run under the same Nsight setup completed | job 12116345 (FSDP1) failed; job 12116346 (DDP) is in `results/profile/` | not investigated; the partial FSDP1 reports are set aside, and the split of idle time into host-bound and queued is measured in-process from the PyTorch Profiler's CUDA API events instead (`dlt.metrics.overlap.host_wait`, sweep `p7h`) |
| 8 | DCP `async_save` on the default process group deadlocked DDP: its background thread and training ran collectives on one group | the async-checkpoint CPU test hung under DDP | a dedicated gloo group for the background write |
| 8 | A plain `scancel` lost a checkpoint interval despite the SIGTERM handler and a 600 s KillWait: Slurm signals torchrun too, and torchrun tears its workers down before an 18 GB save finishes | job 12085605 left `step_00000040.tmp`; the resume restarted at step 30 | stop jobs early with `scancel --batch --signal=USR1` (job 12085783: clean checkpoint, nothing lost) |

## Dead ends and open questions

- **DDP bucket size and `gradient_as_bucket_view` at 1.5B**: no throughput effect (communication is
  already hidden, 0.01 s exposed per step). The bucket view should save the 5.7 GiB bucket copy, but
  measured peak memory did not change; not resolved.
- **FSDP2 explicit prefetch of 2 layers**: +0.7%; the default one-layer prefetch already hides the
  all-gathers.
- **FlashAttention-3**: not tried; no prebuilt wheel for the pinned torch/CUDA.
- **Chunked cross-entropy**: costs 2-6% throughput; worth it only where memory, not compute, limits the
  micro-batch.
- **Checkpoint interval of the 72B main run** was set from the r=16 adapter's 12 s save; the r=64
  adapter's saves took a median 53 s, so 7.5% of the training time went to checkpoint stalls (Young's
  optimum at one failure a day is about 106 steps, not 25). Not rerun.
- **Allocator headroom on the 72B runs**: the main run peaks at 61.7 GiB allocated but up to 121.6 GiB
  reserved, from micro-batches of many different lengths, and the sharded Phase 7 run at 36.4-36.6 GiB
  allocated and 96.9-99.9 GiB reserved (DDP: 52.4-52.6 and 62.6-68.6). Harmless with one process per
  GPU, since the allocator releases its cache and retries before it reports out of memory; not tuned,
  and no run under a memory cap shows where each setup would actually fail (`expandable_segments` would
  be the first thing to try).
- **Explorer's `multigpu` partition** (more than 2 GPUs per job) needs an application for research
  use; not pursued. Scale-out is Phase 9, on rented GPUs.
- **Nsight Compute** is blocked for users (`ERR_NVGPUCTRPERM`); kernel-level analysis uses Nsight
  Systems and the PyTorch Profiler only.

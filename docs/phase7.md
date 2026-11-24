# Phase 7: Qwen2.5-72B QLoRA on 2 H200s, replicated vs sharded

## 1. Setup

[configs/sweeps/p7_72b.yaml](../configs/sweeps/p7_72b.yaml): the Phase 6 QLoRA step (NF4 base, LoRA
r=16 on all seven projections, 16 BIRD train questions per optimizer step, activation checkpointing,
learning rate 1e-4 after 5 warmup steps) for 20 steps, in three ways. Every run sees the same questions
in the same steps (the data order is a function of the seed and the step), so per-step losses compare
directly.

| Run | GPUs | What each GPU holds | Communication |
|---|---|---|---|
| 72b-single | 1 | the whole NF4 base (38.4 GiB, the checkpoint saved in Phase 5), the adapter, its Adam state | none |
| 72b-ddp (replicated) | 2 | the same as 1 GPU; each rank takes 8 of the 16 questions | one all-reduce of the 210.5M FP32 adapter gradients (842 MB) per step, after the last micro-batch |
| 72b-fsdp1 (sharded) | 2 | half of the base and half of the adapter and its Adam state. The base is the same NF4 checkpoint with its packed 4-bit bytes stored as BF16 ([scripts/nf4_storage.py](../scripts/nf4_storage.py)), so FSDP1 can flatten each unit into one BF16 tensor and shard it (the PEFT/bitsandbytes FSDP-QLoRA setup). PEFT's wrap policy makes 1,201 FSDP units: 80 decoder layers, 1,120 LoRA modules (`lora_A` and `lora_B` of 7 projections in 80 layers), and the root | for every micro-batch, an all-gather of each unit before its forward and again before its backward; a reduce-scatter of the adapter gradients after the last micro-batch |

The NF4 runs need about 40 GB of host memory per rank and ran in one job (12100514, `--mem=128G`, node
d4052). The FSDP1 entry ran in jobs of its own (the sweep runner's `--only` option runs one entry of a
sweep file), and the sixth is the one reported. The PEFT recipe quantizes the 145 GB BF16 checkpoint while loading it, which needed a 300 GB
host-memory request and, from a busy /scratch, did not finish loading in 90 minutes (job 12107944).
Re-storing the saved NF4 checkpoint's packed bytes as BF16 instead (152 s on a CPU node) gives FSDP1
exactly the 4-bit values the replicated runs use, in a file it loads like theirs. The other failures
were in this project's audit and its frozen-base check under FSDP1, and are fixed and listed in
[RESULTS_LOG.md](RESULTS_LOG.md). The FSDP1 numbers below are from job 12112395 (node d4053). Throughput
is measured over steps 3-19, without steps 12-13 on 2 GPUs, which run under the PyTorch Profiler to
measure communication.

## 2. Replicated (DDP) against 1 GPU

| | 1 GPU | DDP, 2 GPUs |
|---|---|---|
| Tokens/s | 891 | 1,735 (97.4% scaling) |
| MFU (QLoRA convention) | 26.9% | 26.2% |
| Median step | 28.33 s | 15.17 s |
| Peak GPU memory allocated per rank | 51.8 GiB | 52.4 GiB, 52.6 GiB |
| Peak GPU memory reserved per rank | 96.5 GiB | 62.6 GiB, 68.6 GiB |
| NCCL kernel time per step (exposed) | - | rank 0: 2.60 s (0.53 s); rank 1: 0.013 s (0.003 s) |
| GPU idle per step (no kernel running) | - | 0.89 s, 0.84 s |
| Loss: step 0 / largest gap over 20 steps / median gap | - | 6e-8 / 1.48% / 0.51% |

- **Scaling is 97.4%.** Replication moves very little data: only the adapter's gradients (842 MB) are
  all-reduced, once per step, against about 12 s of compute per rank. The all-reduce itself shows up on
  rank 1: 13 ms of NCCL time per step.
- **What looks like exposed communication is load imbalance.** Rank 0's all-reduce kernel starts when its
  backward pass ends and spins inside NCCL until rank 1 arrives: 2.6 s of kernel time, 0.53 s of it with
  no compute running. Each rank gets 8 of the 16 questions, and question lengths differ, so in the
  profiled steps rank 0 had 0.57 s less compute (11.62 s against 12.19 s). The step lasts as long as the
  slower rank: 12.19 s of compute plus 0.84 s of idle GPU time (kernel launches, the per-step loss
  sync), 13.0 s, which matches the measured step times of steps 12-13. Averaged over the two ranks the
  wait is about 2% of the step, most of the 2.6% scaling loss.
- **Memory per GPU does not drop.** Each rank holds the whole 38.4 GiB base plus the adapter and its Adam
  state, so a 2-GPU DDP rank needs what one GPU needs (52.4-52.6 GiB against 51.8 GiB; the difference is
  DDP's gradient buckets). Replication buys throughput, not memory. The reserved memory is lower on 2
  GPUs (62.6-68.6 GiB against 96.5 GiB), each process running 8 micro-batches instead of 16.
- **The loss matches to run-to-run noise.** Step 0, before any update, agrees to 6e-8. After that the
  2-GPU losses stay within 1.48% of the 1-GPU losses (median 0.51%). The runs are not in deterministic
  mode, and the same 1-GPU configuration run twice (the r=16 sweep run of Phase 6, job 12082456, and this
  one) differs by up to 0.34% over steps 1-5, where the two schedules are identical, against 0.41%
  between 1 and 2 GPUs over the same steps. In deterministic mode (Phase 2) DDP matched one GPU within
  1.7e-4.
- **Start-up:** the DDP ranks loaded the base in 16.8 s, the 1-GPU run in 80.8 s just before it on the
  same node (the files were then in the page cache).

## 3. Sharded (FSDP1) against replicated

| | DDP, 2 GPUs | FSDP1, 2 GPUs |
|---|---|---|
| Tokens/s | 1,735 (97.4% scaling) | 1,226 (68.8% scaling) |
| MFU (QLoRA convention) | 26.2% | 18.5% |
| Median step (forward + backward) | 15.17 s (4.53 + 10.27 s) | 20.40 s (7.61 + 12.86 s) |
| Peak GPU memory allocated per rank | 52.4 GiB, 52.6 GiB | 36.4 GiB, 36.6 GiB |
| Peak GPU memory reserved per rank | 62.6 GiB, 68.6 GiB | 96.9 GiB, 99.9 GiB |
| Loss vs 1 GPU: step 0 / largest gap / median gap | 6e-8 / 1.48% / 0.51% | 6e-8 / 1.97% / 0.38% |
| Model load | 16.8 s | 275 s with the files not in the node's page cache, 23 s with them |

The profiler figures below come from a repeat of both 2-GPU runs in one job (sweep `p7h`, job 12118523,
[configs/sweeps/p7_host.yaml](../configs/sweeps/p7_host.yaml)), which also records when each kernel's
launch call was made (`dlt.metrics.overlap.host_wait`; every kernel in both traces matched its launch
call). The repeats reproduce the throughput: 1,738 tokens/s for DDP and 1,232 for FSDP1.

| Profiled steps 12-13, per step, rank 0 / rank 1 | DDP | FSDP1 |
|---|---|---|
| Compute kernel time | 11.21 s / 12.19 s | 10.95 s / 13.14 s |
| NCCL kernel time (exposed: no compute running) | 3.03 s (1.04 s) / 0.01 s (0.00 s) | 5.15 s (3.02 s) / 11.93 s (6.19 s) |
| GPU idle: no kernel running | 0.78 s / 0.84 s | 7.48 s / 2.13 s |
| ... of which before the host had launched the next kernel | 0.23 s / 0.21 s | 6.49 s / 1.67 s |
| Kernels launched | 240,009 / 240,089 | 270,842 / 270,922 |

- **Sharding saves 16.0 GiB per GPU (30%), not half.** Each rank holds 19.2 GiB of the 38.4 GiB base
  instead of all of it, plus half of the adapter and its Adam state. Two things stay whole. FSDP1 does
  not free the root unit between each micro-batch's forward and its backward (`_post_forward_reshard` in
  torch 2.14 skips the root), and here the root holds the embeddings and the output head, 4.6 GiB in
  BF16. And the activations, about 10 GiB per rank with checkpointing (51.8 GiB on one GPU less the
  38.4 GiB base and 3.1 GiB of adapter weights, gradients and Adam state), are not sharded at all. From
  sizes rather than a per-buffer measurement: 52.4 less 19.2 GiB of base and 1.2 GiB of adapter state,
  plus 4.6 GiB of root, is 36.6 GiB, against 36.4-36.6 GiB measured.
- **Reserved memory goes up, not down**: 96.9-99.9 GiB against 62.6-68.6 GiB for DDP, about what one
  GPU reserves (96.5 GiB). PyTorch's allocator releases its cache and retries before it reports out of
  memory, so the allocated peak is the number that has to fit, but no run here is under a memory cap;
  this is an open question in [RESULTS_LOG.md](RESULTS_LOG.md).
- **FSDP1 is slower because the GPUs wait for the CPU, not for the network.** Compute is the same as
  DDP's (10.95-13.14 s per profiled step against 11.21-12.19 s). What FSDP1 adds is 8.3-10.5 s per step
  per rank with no compute kernel running, against 0.8-1.8 s for DDP. The transfers account for little
  of it: per micro-batch FSDP1 gathers each decoder layer twice (0.45 GB, about 0.9 ms at the
  all-gather bandwidth measured in Phase 1), the root once (5.0 GB, 9 ms), and each of the 1,120 LoRA
  units twice (0.07-1.9 MB each, about 25 us apiece, the all-gather's latency floor), 0.21 s per
  micro-batch or about 1.7 s per step. Most of the rest is host-bound: on rank 0, 6.49 s of its 7.48 s of
  idle GPU time per step passed before the host had even called the API that launched the next kernel,
  and rank 1 spent about the same time (6.19 s of exposed NCCL time) in all-gather kernels that had
  started and were waiting for rank 0 to join. With DDP the host-bound part is 0.21-0.23 s per step.
- **The host time is in FSDP1's bookkeeping, not in kernel launches.** FSDP1 launches only 13% more
  kernels per step than DDP (270.8k against 240.0k). What it adds on the host is the work around them:
  hooks that run before and after each of the 1,201 units, twice per micro-batch, 19,216 unit passes per
  step. Spread evenly, 6.5 s is about 0.34 ms per pass. The traces record GPU kernels and CUDA API calls,
  not Python, so which part of that host code costs the most is not measured.
- **Which rank falls behind changes from run to run**: rank 0 in job 12118523 (7.5 s idle), rank 1 in
  job 12112017 (7.7 s idle), both partly in job 12112395 (5.8 s and 3.9 s). The step lasts as long as the
  slower host either way, and the three FSDP1 runs agree on throughput within 1.1% (1,226, 1,232 and
  1,239 tokens/s). Why one host lags is not investigated.
- **The loss matches to run-to-run noise.** Step 0 agrees with one GPU to 6e-8, as for DDP. After that
  the largest gap is 1.97% (median 0.38%), and two FSDP1 runs of the same configuration (jobs 12112017 and
  12112395) differ from each other by up to 2.27%.
- **Start-up:** each rank reads the whole 39 GB checkpoint (39.6 GiB of host memory per rank) and FSDP1
  shards it on the GPU afterwards. From /scratch that took 275 s; on a node that had just read the files
  it took 23 s, as for DDP.

## 4. Which to use

- **Replicate when the base fits on one GPU.** The NF4 72B does: a DDP rank needs 52.4 GiB of a 140 GiB
  H200. Replication gives 42% more tokens/s than sharding (1,735 against 1,226) for the same losses, and
  needs none of the FSDP1 machinery: no BF16-storage copy of the checkpoint, no FlatParameter-aware
  audit, no `summon_full_params` to check the frozen base.
- **Shard when a replica does not fit, or when the memory is needed for something else.** A DDP rank
  needs 52.4 GiB however many GPUs there are; the sharded base's share halves with each doubling (19.2
  GiB on 2 GPUs here, 9.6 GiB on 4 by the same arithmetic, not measured: Phase 9). The 16 GiB per GPU that
  sharding frees on 2 GPUs could hold longer sequences or larger micro-batches.
- **If sharding, the cost to work on is host time per unit, not bandwidth.** Not tried here: fewer,
  larger units (FSDP1 can put LoRA and frozen weights in one unit only with `use_orig_params=True`, which
  its documentation advises against because of higher gradient memory), FSDP2's per-parameter sharding,
  or CUDA graphs to take the per-unit launches off the host.

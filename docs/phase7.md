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
| 72b-fsdp1 (sharded) | 2 | half of the base and half of the adapter; the base is quantized to NF4 while loading the BF16 checkpoint and stored as BF16 bytes, so FSDP1 can flatten and shard it (the PEFT/bitsandbytes FSDP-QLoRA setup) | an all-gather of each decoder layer before its forward and again before its backward, and a reduce-scatter of the adapter gradients |

The NF4 runs need about 40 GB of host memory per rank and ran in one job (12100514, `--mem=128G`); the
FSDP1 run reads the 145 GB BF16 checkpoint and ran in its own job with `--mem=300G` (the sweep runner's
`--only` option splits a sweep file across jobs). All three ran on node d4052. Throughput is measured
over steps 3-19, without steps 12-13 on 2 GPUs, which run under the PyTorch Profiler to measure
communication.

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

PHASE7_FSDP1

## 4. Which to use

PHASE7_CHOICE

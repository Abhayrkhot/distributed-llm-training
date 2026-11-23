# Phase 4: profiling and tuning

Two workloads from Phase 3, both on 2 H200s: Qwen2.5-7B with FSDP2 (micro-batch 1, accumulation 16:
the most communication) and Qwen2.5-1.5B with DDP (micro-batch 4, accumulation 4). Same setup as
Phase 3 (pretrained weights, FineWeb-Edu, 65,536 tokens per optimizer step, 25 steps, warmup and
profiled steps not timed). The README's "Profiling and tuning" tables are generated from the records.

## 1. Where the step goes (Nsight Systems)

[slurm/nsys_2gpu.sbatch](../slurm/nsys_2gpu.sbatch) runs one baseline step range under `nsys` on each
rank (capture limited to the profiled steps through `cudaProfilerStart/Stop`, NVTX ranges for forward,
backward and optimizer), and [scripts/nsys_summary.py](../scripts/nsys_summary.py) groups every kernel
by its demangled name. (The first summary grouped by short name and put FSDP2's copy kernels and the
gradient-norm kernels under "optimizer", because all of them launch as `multi_tensor_apply_kernel`;
only the template argument tells them apart.)

| Kernel time per step | 7B FSDP2 | 1.5B DDP |
|---|---|---|
| GEMM | 35-39% | 34% |
| elementwise and reductions (RMSNorm, SiLU, residual adds, casts, gradient accumulation) | 22% | 33% |
| NCCL (includes waiting for the other rank) | 17-25% | 21% (rank 0) |
| FSDP2 copy-in (foreach casts into the all-gather buffer) and copy-out | 14% | - |
| attention (FlashAttention-2 through SDPA) | 4-5% | 8% |
| cross-entropy | 1% | 3% |
| fused AdamW | 0.6% | 1% |

Two root causes follow:

1. **7B FSDP2 re-gathers every layer 32 times per step.** With accumulation 16 and FSDP2's defaults
   (reshard after forward and after backward), each micro-batch all-gathers every layer in forward and
   again in backward: 912 all-gathers per step in the trace (28 layers x 32 + the root). The NCCL time
   mostly overlaps compute (0.15 s exposed of 4.4 s), but each gather also pays copy-in and copy-out
   kernels on the GPU (0.75 s per step).
2. **A third of the 1.5B step, and a fifth of the 7B step, is unfused elementwise work.** transformers'
   RMSNorm and SwiGLU run as several memory-bound kernels each, and the BF16 casts and FP32 gradient
   accumulation add more.

DDP's communication is already hidden at 1.5B (0.01 s exposed per step), so bucket tuning has little
to win.

## 2. One change at a time

[configs/sweeps/p4_tuning.yaml](../configs/sweeps/p4_tuning.yaml), job 12075850, 3 repeats each, every
change in the same allocation as its baseline. Repeat ranges are under 0.5% wide, so differences above
about 1% are real.

**7B FSDP2** (baseline 14,570 tokens/s, 33.1% MFU, 89.4 GiB):

| Change | Tokens/s | Memory | Why |
|---|---|---|---|
| micro-batch 2 (accumulation 8) | **+20.3%** (39.8% MFU) | +16 GiB | half the per-micro-batch gathers, copies and accumulation adds; larger GEMMs |
| sequence length 4,096 (same tokens per step) | +15.8% | +16 GiB | same effect through fewer, longer micro-batches |
| no reshard after forward | +9.8% | +12 GiB | the backward reuses the forward's gathered layer: 16 gathers per layer per step instead of 32 |
| cuDNN SDPA | +2.6% | - | faster attention kernels than FlashAttention-2 on H200 |
| prefetch 2 layers | +0.7% | - | the default one-layer prefetch already hides the gathers |
| FlashAttention-2 forced | +0.5% (what auto picks) | - | |
| chunked cross-entropy | -1.9% | -0.6 GiB | recomputes the lm_head; little to save at micro-batch 1 |
| memory-efficient SDPA | -3.8% | - | |
| checkpoint every 2nd layer | -7.1% | -6 GiB | the memory/throughput point between none and all layers |

micro-batch 2 with the chunked loss gives +16.8% at 100.8 GiB, 4.6 GiB under micro-batch 2 alone.

**1.5B DDP** (baseline 65,074 tokens/s, 34.0% MFU, 73.5 GiB):

| Change | Tokens/s | Memory |
|---|---|---|
| micro-batch 8 (accumulation 2) | +4.8% | +42 GiB |
| cuDNN SDPA | +3.3% | - |
| bucket 5 MB / 100 MB, gradient bucket view | +0.2% / +0.3% / +0.5% (within or near noise) | - |
| chunked cross-entropy | -5.7% | -8.8 GiB |
| micro-batch 8 + chunked loss | -1.3% | +17 GiB (vs +42 without chunking) |
| memory-efficient SDPA | -8.0% | - |
| sequence length 4,096 (same tokens per step) | -6.7% (MFU +0.9 points: attention FLOPs grow) | - |

FlashAttention-3 was not tried: no build for this torch/CUDA pin exists without compiling it.

## 3. Profiler-guided second round

PHASE4_ROUND2

## 4. The memory predictor out of sample

The Phase 4 configurations were never used to correct the predictor (Phase 3 was), so they test it:
median absolute error 1.6% over the 66 runs. The one miss is `1p5b-bucket-view` (-7.1%): with
`gradient_as_bucket_view` the gradients live in DDP's buckets, so the predictor drops the separate
bucket copy (5.7 GiB), but measured peak memory did not change (73.5 GiB with and without). The peak
is reached somewhere the bucket copy does not matter; this is not resolved.

# Draft: question to Research Computing about `multigpu` access

Not sent. Paste into the partition application form (https://bit.ly/NURC-PartitionAccess, Partition
Type: Multigpu) or a general RC ticket. Edit freely.

---

Hello,

I'd like to ask whether a self-directed project would qualify for the `multigpu` partition, before I
spend a test reservation on it.

The project benchmarks distributed LLM training on the H200 nodes: PyTorch DDP and FSDP2 on Qwen2.5
models from 0.5B to 14B parameters, plus QLoRA fine-tuning of Qwen2.5-72B. Everything currently
runs within the `gpu` (1 GPU) and `gpu-short` (2 GPUs) limits, and the 2-GPU scaling results will be
available before any multigpu testing.

If it qualifies, the test plan for the reservation would be short runs (under 2 hours in total) of
the same workloads on 1, 2, 4 and 8 H200s in a single node, recording step time, throughput and
scaling efficiency. Jobs checkpoint and exit; nothing sits idle.

If it does not qualify, no problem: the project is designed to be complete at 1-2 GPUs.

Thank you,

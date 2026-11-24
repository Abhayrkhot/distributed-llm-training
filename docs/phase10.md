# Phase 10: report and release

The README is the report: every table in it is generated from the run records in
[results/](../results/) by [scripts/gen_readme.py](../scripts/gen_readme.py), and CI fails if the
committed README differs from what the records produce (`gen_readme.py --check`). The figures come from
the same records ([scripts/plots.py](../scripts/plots.py)). Each run record carries its git commit, lock
hash, package versions, GPU, driver, Slurm job and node; benchmarks report the median and range over 3
repeats.

## 1. Definition of done (PLAN.md section 8)

| Item | Status | Evidence |
|---|---|---|
| Resource manifest for the H200 nodes | done | [manifests/h200.json](../manifests/h200.json), [docs/phase0.md](phase0.md) |
| DDP and FSDP2 pass loss-parity, accumulation, sampler and bit-exact resume tests on 1 and 2 H200s | done; sampler and accumulation are tested in CPU CI, parity and resume on H200s | [docs/phase2.md](phase2.md), [tests/test_train_cpu.py](../tests/test_train_cpu.py), jobs 12066972 and 12073204 |
| Scaling and memory study, 0.5B-14B: MFU, HFU, memory, step breakdown, communication, 3+ repeats, predictor error | done | [docs/phase3.md](phase3.md); `p3.*` records in [results/train/](../results/train/) |
| At least one profiler-backed optimization with before/after numbers | done: +58% tokens/s on 7B FSDP2, +36% on 1.5B DDP | [docs/phase4.md](phase4.md); `p4*` records |
| 72B NF4 forward pass validated against BF16 | done | [docs/phase5.md](phase5.md); [results/eval/eval-12077471-20261123T112802-p69101.json](../results/eval/eval-12077471-20261123T112802-p69101.json) |
| 72B QLoRA run passes the frozen-base audit, spans 2+ Slurm jobs, beats the base model on held-out data with a CI; a reloaded adapter reproduces the eval | done | [docs/phase6.md](phase6.md); records listed in section 2 |
| Replicated vs sharded 72B on 2 GPUs measured | PHASE10_P7 | [docs/phase7.md](phase7.md) |
| Fault-tolerance table published | done | [docs/phase8.md](phase8.md); `p8-*` records |
| README generated from results; no number without a run behind it | done | [scripts/gen_readme.py](../scripts/gen_readme.py), CI `--check` |

The Phase 10 figures the plan lists are all in the README: throughput and MFU by model and GPU count
([p3_mfu.png](img/p3_mfu.png)), measured vs predicted memory ([p3_memory.png](img/p3_memory.png)), the
step-time split ([p3_step_split.png](img/p3_step_split.png)) and exposed communication
([p3_comm.png](img/p3_comm.png)), checkpoint cost ([p8_checkpoint.png](img/p8_checkpoint.png)),
training and validation loss ([p2_chain.png](img/p2_chain.png), [p6_main.png](img/p6_main.png)), and
BIRD accuracy ([p6_bird.png](img/p6_bird.png)). Failures and dead ends are in
[docs/RESULTS_LOG.md](RESULTS_LOG.md).

## 2. Headline numbers and the records behind them

| Claim | Number | Records |
|---|---|---|
| 72B QLoRA on one H200, BIRD dev execution accuracy | 46.1% (base, 3-shot) -> 61.3% (fine-tuned), +15.2 points, paired 95% CI +12.8 to +17.5; val 55.5% -> 72.0% | base [dev](../results/eval/eval-12082458-20261123T181803-p3000467.json), [val](../results/eval/eval-12077472-20261123T112007-p4168167.json); fine-tuned [dev](../results/eval/eval-12085545-20261124T032331-p2745064.json), [val](../results/eval/eval-12085545-20261124T022934-p2542734.json) |
| The reloaded adapter reproduces the eval | 1,534 of 1,534 dev questions score the same in a fresh process | [repeat](../results/eval/eval-12085545-20261124T041449-p3137604.json) |
| The 72B run spans Slurm jobs | USR1 stop at step 337, resume, 500 steps; 873 tokens/s, 26.3% MFU | [job 12084761](../results/train/train-12084761-20261123T213221-p3801882.json), [job 12085545](../results/train/train-12085545-20261124T015211-p1905176.json) |
| Frozen-base audit | 842.1M trainable LoRA parameters (r=64), exactly the formula; all 963 base tensors byte-identical after the first step | `adapter_audit` in the two records above |
| NF4 72B against BF16 | next-token agreement 97.8% on SQL tokens, perplexity +0.32% | [record](../results/eval/eval-12077471-20261123T112802-p69101.json) |
| Profiler-guided tuning, 7B FSDP2 on 2 H200s | 14,384 -> 22,720 tokens/s (+58%), 32.6% -> 51.6% MFU | `p4c.7b-base.r*`, `p4c.7b-best.r*` (job 12082452) |
| Profiler-guided tuning, 1.5B DDP on 2 H200s | 64,295 -> 87,692 tokens/s (+36%), 33.5% -> 45.8% MFU | `p4c.1p5b-base.r*`, `p4c.1p5b-best.r*` (job 12082452) |
| DDP scaling 1 -> 2 H200s, 0.5B-3B | 96.7-97.3% | `p3.*-single.r*`, `p3.*-ddp.r*` (job 12073550) |
| 72B QLoRA replicated on 2 H200s | 97.4% scaling (1,735 vs 891 tokens/s) | [DDP](../results/train/train-12100514-20261124T101736-p3213835.json), [1 GPU](../results/train/train-12100514-20261124T101107-p3102381.json) |
| Peak-memory predictor | median error 1.6% on 72 Phase 3 runs (worst 4.0%), 1.6% out of sample on 66 Phase 4 runs; fit vs OOM right for all 28 configurations | `p3.*` and `p4.*` records; README "Memory" section |
| Fault tolerance | every injected failure resumes bit-exact; async checkpoint stalls 1.9 s instead of 35.2 s per 18 GiB save | [sync](../results/train/train-12082454-20261123T180838-p3754793.json), [async](../results/train/train-12082454-20261123T181151-p3761021.json), `p8-*` records |
| NCCL on one node | all-reduce 328 GB/s bus bandwidth between 2 H200s | [record](../results/nccl_bw/nccl_bw-12066735-20261122T214613.json) |

## 3. Not done

- **Phase 9** (4 and 8 GPUs, the full-parameter 72B memory experiment) needs rented GPUs: Explorer gives
  at most 2 GPUs per job. Everything above is measured on 1-2 H200s.
- **Open questions** from the measurements (DDP bucket view memory, the predictor's miss on it, the
  72B checkpoint interval, allocator headroom) are listed in [docs/RESULTS_LOG.md](RESULTS_LOG.md).

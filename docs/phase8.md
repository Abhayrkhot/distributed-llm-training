# Phase 8: fault tolerance

Resume has been tested since Phase 2 (bit-exact on CPU and on H200s, and across a real Slurm wall-time
stop). This phase measures what each kind of failure costs and sets the checkpoint cadence. Tables: the
README's "Fault tolerance" section, generated from [results/train/](../results/train/) (run names p8-*).

## 1. Failure scenarios on the small model

[slurm/phase8_faults.sbatch](../slurm/phase8_faults.sbatch), job 12082454: pretrained Qwen2.5-1.5B,
FSDP2 on 2 H200s, deterministic mode with eager attention, 40 steps, checkpoints every 10 steps (18.1
GiB each: FP32 parameters and both Adam moments, sharded over the two ranks). Each failure is followed
by a resume in a new process; the resumed run's per-step losses are compared with an uninterrupted run.

| Failure | What survives | Lost work | Matches the uninterrupted run |
|---|---|---|---|
| SIGKILL of every rank mid-step (as a node failure) | the step-20 checkpoint | 5 steps | yes, bit for bit |
| SIGKILL after the step-30 shards are written but before the checkpoint is published | the step-20 checkpoint; the half-written `.tmp` is ignored and cleaned up | 10 steps | yes |
| SIGKILL during the end-of-run evaluation | the final checkpoint (written before evaluation) | only the evaluation; the next job finds the run complete and exits | - |
| SIGTERM with `stop_on_term` (what `scancel` sends) | a checkpoint at the next step boundary | none | yes |
| SIGTERM without the handler | the step-20 checkpoint | 5 steps | yes |
| Slurm wall-time limit (USR1 300 s before, Phase 2) | a checkpoint at the next step | none | yes (Phase 2) |

At this size a restart costs 9-16 s of model load and 6-50 s of checkpoint restore (the spread is the
shared filesystem), more than the lost steps themselves (0.39 s each).

## 2. Synchronous vs asynchronous checkpoints

| | Training blocked per save | Save complete | Median step |
|---|---|---|---|
| synchronous `dcp.save` | 35.2 s | 35.2 s | 0.390 s |
| `dcp.async_save` (stage to host memory, write in the background) | 1.9 s | 40.6 s | 0.378 s |

Asynchronous saves cut the stall per 18.1 GiB checkpoint from 35 s to 2 s, and the background write did
not slow the training steps. The checkpoint is published (meta.json written, directory renamed) only
after every rank's write has finished, so a crash during the background write leaves the previous
checkpoint as the latest (the kill-during-save scenario above exercises the same rule).

## 3. Checkpoint cadence

Wall-time stops cost nothing (USR1 checkpoint). Random failures (a node lost, an OOM kill, a cancelled
job without the handler) cost the work since the last checkpoint. Young's approximation gives the
interval that minimizes the expected loss, tau = sqrt(2 C M), for a checkpoint stall C and a mean time
between failures M; the expected overhead at that interval (stalls plus recomputed work) is about
sqrt(2 C / M):

| Small model (1.5B FSDP2) | C | M = 6 h | M = 24 h | M = 7 days |
|---|---|---|---|---|
| synchronous | 35 s | 20.5 min (5.7%) | 41 min (2.8%) | 1.8 h (1.1%) |
| asynchronous | 1.9 s | 4.8 min (1.3%) | 9.6 min (0.66%) | 25 min (0.25%) |

With asynchronous saves, checkpointing about every 10 minutes keeps the expected cost of one failure a
day (stalls plus recomputed work, restart time aside) near 0.7%, a quarter of the synchronous figure.
The 72B adapter run is in section 4.

## 4. 72B QLoRA

[slurm/phase8_72b.sbatch](../slurm/phase8_72b.sbatch), job 12082461, on one H200: the QLoRA setup of the
overfit test (r=16, 8 questions per step), checkpoints every 5 steps.

- **What a checkpoint holds:** the 210.5M LoRA parameters, their Adam moments, the scheduler, RNG state
  and step (3.4 GiB); the 38.4 GiB NF4 base is not saved, it is reloaded from its own checkpoint.
  Saving takes 10-15 s against 14-15 s training steps.
- **SIGKILL before step 12:** the next process resumed from the step-10 checkpoint: 2 steps (27 s) lost;
  restarting cost 43 s of base load (the saved NF4 checkpoint, warm in the page cache) and 23 s of
  adapter restore.
- **SIGTERM before step 17 with `stop_on_term`:** stopped at step 18 with a checkpoint (12.9 s), nothing
  lost; the next process resumed at step 18 (load 39 s, restore 25 s) and finished at step 24.
- **Wall-time stops** are exercised by the Phase 6 main run, which spans several 3 h jobs on purpose.

For the 72B adapter, C is about 12 s per synchronous save. At M = 24 h Young's interval is 24 min (sqrt(2 x
12 x 86,400) s), about 48 steps of the main run (about 30 s per 16-question step); the main run saves every
25 steps (about 12.5 min), shorter than optimal, for an expected overhead of about 2.0% (1.7% at the
optimum).

## 5. A real `scancel`

[scripts/phase8_scancel.sh](../scripts/phase8_scancel.sh), from the login node, against a live 1.5B FSDP2
job (one H200, checkpoints every 10 steps, `stop_on_term` on). Explorer's KillWait (the time between
Slurm's SIGTERM and its SIGKILL) is 600 s.

| How the job was stopped | What happened | Lost |
|---|---|---|
| `scancel <job>` (job 12085605) | Slurm sent SIGTERM to every process of the job, torchrun included. torchrun shut its worker down at once, in the middle of the periodic step-40 save (18 GB, about 50 s): the job ended CANCELLED, `step_00000040.tmp` was left half-written and ignored, and the resume (job 12085753) restarted from step 30. | 10 steps and a checkpoint write |
| `scancel --batch --signal=USR1 <job>` (job 12085783) | only the batch shell got the signal; `slurm/launch.sh` forwarded USR1 to the worker, which finished the step-30 checkpoint (37 s), recorded `stopped_by=signal` and exited 0, 32 s after the signal; the job ended COMPLETED and the resume (job 12085851) continued at step 30. | nothing |

So the 600 s KillWait does not protect a torchrun job: the SIGTERM handler in the training processes
never gets to finish, because torchrun itself tears them down first. Stopping a job early without
losing work means sending the same signal the wall-time limit sends, to the batch shell only:
`scancel --batch --signal=USR1 <job>`. Plain `scancel` still loses at most one checkpoint interval,
since an interrupted save is never published.

# Phase 6: Qwen2.5-72B QLoRA on BIRD text-to-SQL, one H200

## 1. Setup

- **Base:** Qwen2.5-72B in NF4 (the checkpoint saved in Phase 5, 38.4 GiB), frozen. **Adapters:** LoRA on
  all seven projections of every layer (q, k, v, o, gate, up, down), FP32, dropout 0.05 in the main run.
- **Data:** BIRD train after preparation (docs/phase5.md), one example per question: the prompt (schema
  with three sample rows per table when the example fits 4,096 tokens, else the plain schema; external
  knowledge; question) followed by the gold SQL and end-of-text. The loss covers only the SQL and the
  end-of-text token, normalized by the number of target tokens in the step. 464 of the 8,184 training
  questions (5.7%) do not fit 4,096 tokens even with the plain schema and are left out.
- **Training:** [src/dlt/train/engine.py](../src/dlt/train/engine.py) (the same engine as Phases 2-4,
  with `quant=nf4`, `peft=lora`, `task=sft`), 16 questions per optimizer step, AdamW, cosine schedule,
  activation checkpointing, BF16 compute under autocast. Adapter-only checkpoints (LoRA weights, Adam
  state, scheduler, RNG, step).
- **Evaluation:** [src/dlt/bird/evaluate.py](../src/dlt/bird/evaluate.py): batched greedy decoding, at
  most 320 new tokens, the same schema text for every model; execution accuracy by BIRD's criterion
  (same set of result rows as the gold SQL), on node-local read-only database copies with a 30 s limit
  per query, and a 95% bootstrap interval over questions. "val" is the 880-question database-level
  holdout from BIRD train used for every decision; "dev" is BIRD dev (1106 split, 1,534 questions),
  scored once per reported model. 3 dev gold queries exceed 30 s and count as wrong for every model.

## 2. Checks before the main run

**Adapter audit** (every run; job 12082455 shown): 210,534,400 trainable parameters in 1,120 tensors,
exactly r x (in + out) summed over the 560 adapted projections for r=16; every trainable tensor is a
LoRA tensor; the optimizer holds exactly those; all 963 base tensors are byte-for-byte unchanged after
the first optimizer step (two integer checksums over each tensor's raw bytes, computed on the GPU).

**Overfit test** ([configs/training/p6_overfit.yaml](../configs/training/p6_overfit.yaml), job
12082455): the first 64 training questions, 8 per step, 120 steps at learning rate 2e-4. Training loss
0.1225 at step 10, 0.0081 at 50, 0.0001 at 100, 0.0000 at 120: the adapter can memorize what it sees,
so the masking, the loss and the update all reach the model. 944 tokens/s, 28.7% MFU (QLoRA convention
4N + 12LdT), 42.8% HFU, 47.7 GiB peak.

**Short sweep** ([configs/sweeps/p6_lora.yaml](../configs/sweeps/p6_lora.yaml), job 12082456): the same
640 training questions (40 steps x 16) in every run, judged by the loss on 128 validation questions.

| Run | Validation loss after 640 questions (start 0.659) | Tokens/s | Peak GPU memory |
|---|---|---|---|
| r=16 | 0.1531 | 905 | 51.9 GiB |
| **r=64** (chosen) | **0.1466** | 905 | 61.4 GiB |
| r=16, no activation checkpointing | OOM (133.4 GiB allocated) | - | - |
| r=16, 2,048-token limit (939 questions dropped, plain schemas more often) | 0.1537 | 836 | 47.6 GiB |
| r=16, memory-efficient SDPA instead of FlashAttention | 0.1511 | 867 | 51.9 GiB |

The two r=16 runs that differ only in the attention kernel land 0.002 apart, a measure of run-to-run
noise; r=64's 0.0065 lower loss is three times that, at the same throughput and 9.5 GiB more memory,
so the main run uses r=64. 4,096 tokens keeps the schema-with-rows prompts dev is evaluated with (dev
prompts reach 4,246 tokens).

## 3. Baselines

Same prompt format and scorer for every model; the base model gets 3 worked examples (question,
knowledge, SQL) from other training databases, the Instruct model none (chat template).

| Model | val (880) | dev (1,534) | dev: simple / moderate / challenging | dev SQL that fails to execute |
|---|---|---|---|---|
| Qwen2.5-72B base, 3-shot | 55.5% (52.2-58.8) | 46.1% (43.6-48.6) | 56.9 / 38.6 / 20.3 | 332 |
| Qwen2.5-72B-Instruct, zero-shot (reference) | 70.2% (67.2-73.3) | 58.9% (56.5-61.4) | 70.2 / 53.3 / 27.7 | 91 |

Both run in NF4 on one H200. The Instruct model is a reference for what Qwen's own post-training
reaches, not a baseline the fine-tune has to beat; the claim is the gain over the base model under the
same prompt and scorer. Qwen's pretraining data is not published, so absolute scores carry a
contamination caveat (BIRD dev has been public since 2023; the 2025-11 cleanup rewrote many questions
and SQL).

## 4. Main run

PHASE6_MAIN

## 5. Fine-tuned results

PHASE6_RESULTS

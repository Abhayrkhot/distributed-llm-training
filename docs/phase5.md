# Phase 5: Qwen2.5-72B integration, and the BIRD data

## 1. Weights

[src/dlt/fetch.py](../src/dlt/fetch.py) downloads the pinned revisions in a CPU job and writes a
`model` run record per checkpoint ([results/model/](../results/model/)): every file's size, the sha256
the Hub publishes and the sha256 computed on Explorer.

| Model | Revision | Size | Download | Verified |
|---|---|---|---|---|
| Qwen/Qwen2.5-72B | `efba10c8e5` | 135.4 GiB, 37 safetensors shards | 240 s | 37/37 match |
| Qwen/Qwen2.5-72B-Instruct (eval reference) | `495f39366e` | 135.4 GiB | 228 s | 37/37 match |

## 2. NF4, and the check against BF16

[src/dlt/qlora/nf4.py](../src/dlt/qlora/nf4.py), jobs 12077470 (1 H200) and 12077471 (2 H200s),
2026-11-23. NF4 with double quantization and BF16 compute; the 560 linear projections (7 per layer) are quantized,
the embeddings, norms, biases and `lm_head` stay BF16.

| | Time | Peak GPU memory | Host RSS |
|---|---|---|---|
| quantize while loading the BF16 checkpoint (1 GPU) | 391 s | 38.8 GiB | 136.4 GiB |
| save the NF4 checkpoint (38.4 GiB on disk, 37.3 GiB of parameter storage) | 73 s | - | - |
| reload the saved NF4 checkpoint | 114 s | 66.2 GiB (transient) | - |
| BF16 with the layers split over 2 GPUs | 329 s | 71.3 + 68.6 GiB | 136.3 GiB |

Host RSS counts the memory-mapped safetensors pages, so it reaches the size of the BF16 checkpoint
while loading; the reload of the saved NF4 copy is 3.4x faster than quantizing again, so every later
job loads the saved copy.

**Tokenizer.** `<|endoftext|>` (151643), `<|im_start|>` (151644) and `<|im_end|>` (151645) have their
documented ids; all 200 check texts (prompt + SQL) decode back to themselves exactly; the 72B
vocabulary is identical to the 0.5B tokenizer the data statistics were computed with.

**NF4 vs BF16.** Teacher-forced forward passes over the first 200 validation questions (prompt with
schema and sample rows, then the gold SQL and end-of-text): 301,661 predicted positions, 12,133 of
them inside the SQL.

| | NF4 (1 GPU) | BF16 (2 GPUs) |
|---|---|---|
| argmax next token agrees with the other model | 93.44% of all positions, 97.81% in the SQL | |
| perplexity of the gold SQL | 1.9367 | 1.9305 (NF4 +0.32%) |
| perplexity over all tokens | | NF4 +0.68% |
| forward throughput | 3,107 tokens/s | 3,889 tokens/s |

Most disagreements fall outside the SQL (in the schema, sample rows and question text, where the next
token is often not predictable); on the SQL the two pick the same token 97.8% of the time and the
perplexity differs by 0.3%. NF4 is a usable base for the fine-tune.

## 3. BIRD preparation

[src/dlt/bird/data.py](../src/dlt/bird/data.py) in a CPU job (final version: job 12077469, 11 min):
the official `train.zip` (8.3 GiB) and `dev.zip`, the cleaned dev questions of 2025-11-06 from Hugging
Face, CREATE-statement schemas with and without three sample rows per table, a gold-SQL execution
check, a database-level validation split and a manifest with the sha256 of every source and database.

| | Questions | Databases | Prompt tokens (plain schema), median / p95 |
|---|---|---|---|
| train | 8,184 (of 9,428; 364 with broken gold SQL dropped) | 63 | 536 / 5,040 |
| val (held-out databases) | 880 | 6 | 440 / 538 |
| dev (1106 split) | 1,534 (3 golds do not finish in 30 s) | 11 | 728 / 1,628 |

**What went wrong first, and the fixes:**

- **Gold SQL "timed out" on a 64 KB database.** The first preparation ran 16 queries at a time
  against the databases on /scratch and marked 801 train and 82 dev golds as timed out at 30 s, 330 of
  them on `retail_world`, a 64 KB database. A probe (job 12077462,
  [scripts/bird_gold_probe.py](../scripts/bird_gold_probe.py)) re-ran samples one at a time: on /scratch
  with SQLite's default locking, 7 of 25 train golds finished within 60 s (median 34.9 s); on node-local
  copies opened with `immutable=1`, all 25 finished within 30 s (median 1.8 s), and 79 of 82 dev golds
  (median 0.6 s). SQLite's file locks over the VAST NFS mount were the cost. Every query (preparation
  and evaluation) now runs on node-local immutable copies; 3 dev golds still exceed 30 s and count as
  unanswerable for every model.
- **Train golds are dropped only when they error** (missing tables or columns: known BIRD train
  noise), not when they are slow; 13 slow ones are kept.
- **The first validation split was 3 databases, one with a 5,000-token schema** (dev's largest is
  1,861), so it measured something dev does not. Validation databases are now drawn only from those
  whose schema is at most 2,500 tokens, at least 6 of them.
- **macOS metadata** (`__MACOSX/._*.zip`) inside `train.zip` broke the nested extraction in the first
  attempt.

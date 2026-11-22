"""Tokenize one pinned FineWeb-Edu parquet shard into flat uint32 token files.

Documents are joined with the tokenizer's end-of-text token. The last ``--val-docs`` documents of the
shard go to ``val.bin``, the rest to ``train.bin``; ``meta.json`` records the source revision and
sha256, the tokenizer revision and the token counts. Runs on a CPU node (slurm/tokenize.sbatch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

DEFAULTS = {
    "dataset": "HuggingFaceFW/fineweb-edu",
    "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "file": "sample/10BT/000_00000.parquet",
    "tokenizer": "Qwen/Qwen2.5-0.5B",
    "tokenizer_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
}


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def tokenize_texts(tokenizer, texts: list[str], eos_id: int) -> np.ndarray:
    ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    out = np.empty(sum(len(x) + 1 for x in ids), dtype=np.uint32)
    pos = 0
    for x in ids:
        out[pos : pos + len(x)] = x
        out[pos + len(x)] = eos_id
        pos += len(x) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    ap.add_argument("--parquet", type=Path, default=None, help="local parquet instead of downloading")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-docs", type=int, default=2000)
    ap.add_argument("--batch-docs", type=int, default=2000)
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if args.parquet is None:
        from huggingface_hub import hf_hub_download

        src = Path(hf_hub_download(args.dataset, args.file, repo_type="dataset", revision=args.revision))
    else:
        src = args.parquet
    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    eos_id = tok.convert_tokens_to_ids("<|endoftext|>") if tok.eos_token_id is None else tok.eos_token_id

    pf = pq.ParquetFile(src)
    n_docs = pf.metadata.num_rows if args.max_docs is None else min(args.max_docs, pf.metadata.num_rows)
    n_val = min(args.val_docs, n_docs // 10)
    n_train = n_docs - n_val
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    seen = 0
    t0 = time.time()
    with open(args.out / "train.bin.tmp", "wb") as ftrain, open(args.out / "val.bin.tmp", "wb") as fval:
        for batch in pf.iter_batches(batch_size=args.batch_docs, columns=["text"]):
            texts = batch.column(0).to_pylist()[: n_docs - seen]
            if not texts:
                break
            split = max(0, min(len(texts), n_train - seen))
            for name, part, f in (("train", texts[:split], ftrain), ("val", texts[split:], fval)):
                if part:
                    arr = tokenize_texts(tok, part, eos_id)
                    f.write(arr.tobytes())
                    counts[name] += int(arr.size)
            seen += len(texts)
            if seen % (args.batch_docs * 50) < args.batch_docs:
                rate = (counts["train"] + counts["val"]) / (time.time() - t0)
                print(
                    f"{seen}/{n_docs} docs, {counts['train'] + counts['val']:,} tokens, {rate:,.0f} tok/s",
                    flush=True,
                )
    for name in ("train", "val"):
        (args.out / f"{name}.bin.tmp").replace(args.out / f"{name}.bin")
    meta = {
        "source": {
            "dataset": args.dataset,
            "revision": args.revision,
            "file": args.file if args.parquet is None else str(args.parquet.name),
            "sha256": sha256_file(src),
            "docs": n_docs,
        },
        "tokenizer": {
            "repo": args.tokenizer,
            "revision": args.tokenizer_revision,
            "eos_id": int(eos_id),
            "vocab_size": len(tok),
        },
        "dtype": "uint32",
        "train_docs": n_train,
        "val_docs": n_val,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "train_sha256": sha256_file(args.out / "train.bin"),
        "val_sha256": sha256_file(args.out / "val.bin"),
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

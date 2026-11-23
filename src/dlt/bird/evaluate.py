"""Generate SQL for BIRD questions and score it by execution (Phase 6).

    python -m dlt.bird.evaluate --mode base      --split val          (3-shot, the pretrained base)
    python -m dlt.bird.evaluate --mode finetuned --adapter DIR        (zero-shot, base + LoRA adapter)
    python -m dlt.bird.evaluate --mode instruct                       (zero-shot chat, reference ceiling)
    python -m dlt.bird.evaluate --mode oracle                         (gold SQL as the prediction: checks
                                                                       the scorer and the databases)

All modes use the same schema text, decoding (greedy, at most ``--max-new`` tokens) and scorer. The base
model's few-shot examples come from the training split (other databases). The 72B models load in NF4
on one GPU (``--model`` may point at the saved NF4 checkpoint, which loads faster). Each question's
prediction, correctness and error are written to ``--out`` (JSONL), and an ``eval`` run record holds
EX with a bootstrap CI, EX by difficulty, generation time and the per-question correctness vector
(for paired comparisons between models).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from dlt.bird import execute
from dlt.bird.prompt import extract_sql, few_shot_prefix, instruct_messages, prompt
from dlt.results import new_record, write_record
from dlt.train.config import PINNED

BASE = "Qwen/Qwen2.5-72B"
INSTRUCT = "Qwen/Qwen2.5-72B-Instruct"
STOP = ["\n\n", "\n-- "]


def load_split(data: Path, split: str, n: int) -> list[dict]:
    rows = [json.loads(line) for line in (data / f"{split}.jsonl").read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["question_id"])
    return rows[:n] if n else rows


def shots(data: Path, k: int) -> list[dict]:
    """k short training questions from k different databases (deterministic)."""
    rows = load_split(data, "train", 0)
    rows.sort(key=lambda r: (len(r["sql"]) + len(r["question"]), r["question_id"]))
    out, seen = [], set()
    for r in rows:
        if r["db_id"] not in seen and len(r["sql"]) > 40:
            out.append(r)
            seen.add(r["db_id"])
        if len(out) == k:
            break
    return out


def build_inputs(
    mode: str, rows: list[dict], schemas: dict, tok, k_shots: list[dict], max_prompt: int
) -> list:
    prefix = few_shot_prefix(k_shots) if mode == "base" else ""
    texts = []
    for r in rows:
        for key in ("ddl_rows", "ddl"):
            s = schemas[r["db_id"]][key]
            if mode == "instruct":
                text = tok.apply_chat_template(
                    instruct_messages(r, s), tokenize=False, add_generation_prompt=True
                )
            else:
                text = prefix + prompt(r, s)
            if len(tok(text).input_ids) <= max_prompt or key == "ddl":
                texts.append(text)
                break
    return texts


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dlt.qlora.nf4 import bnb_config

    model_id = args.model or (INSTRUCT if args.mode == "instruct" else BASE)
    rev = PINNED.get(model_id)
    saved_quantized = Path(model_id).is_dir() and "quantization_config" in json.loads(
        (Path(model_id) / "config.json").read_text()
    )
    kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa", "device_map": {"": 0}}
    if args.quant == "nf4" and not saved_quantized:
        kw["quantization_config"] = bnb_config()
    if not torch.cuda.is_available():
        kw.pop("device_map")
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, **kw)
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tok, model_id


@torch.no_grad()
def generate(model, tok, texts: list[str], batch: int, max_new: int, mode: str) -> tuple[list[str], int]:
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos = [tok.eos_token_id]
    if (im_end := tok.convert_tokens_to_ids("<|im_end|>")) not in (None, tok.unk_token_id):
        eos.append(im_end)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[str] = [""] * len(texts)
    new_tokens = 0
    device = next(model.parameters()).device
    for start in range(0, len(order), batch):
        idx = order[start : start + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=tok.pad_token_id,
            stop_strings=STOP if mode == "base" else None,  # the fine-tuned model ends with EOS
            tokenizer=tok,
        )
        tail = gen[:, enc.input_ids.shape[1] :]
        new_tokens += int((tail != tok.pad_token_id).sum())
        for i, seq in zip(idx, tail, strict=True):
            out[i] = tok.decode(seq, skip_special_tokens=True)
        print(f"generated {min(start + batch, len(order))}/{len(order)}", flush=True)
    return out, new_tokens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    home = os.environ.get("DLT_HOME", ".")
    ap.add_argument("--mode", choices=["base", "finetuned", "instruct", "oracle"], required=True)
    ap.add_argument("--split", default="val", choices=["val", "dev"])
    ap.add_argument("--n", type=int, default=0, help="first N questions by id (0: all)")
    ap.add_argument("--data", type=Path, default=Path(f"{home}/data/bird"))
    ap.add_argument("--model", default="", help="model id or local dir (default: the pinned base/instruct)")
    ap.add_argument("--quant", default="nf4", choices=["nf4", "bf16"])
    ap.add_argument("--adapter", default="")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--max-prompt", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=execute.DEFAULT_TIMEOUT_S)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.mode == "finetuned" and not args.adapter:
        ap.error("--mode finetuned needs --adapter")

    rows = load_split(args.data, args.split, args.n)
    schemas = json.loads((args.data / "schemas.json").read_text())
    dbs = execute.local_copies({r["db_id"]: args.data / "databases" / f"{r['db_id']}.sqlite" for r in rows})
    name = args.name or f"{args.mode}-{args.split}"
    t0 = time.time()
    model_id, new_tokens, load_s, gen_s = None, 0, 0.0, 0.0
    if args.mode == "oracle":
        preds = [r["sql"] for r in rows]
        raw = preds
    else:
        model, tok, model_id = load_model(args)
        load_s = time.time() - t0
        k = shots(args.data, args.shots) if args.mode == "base" else []
        texts = build_inputs(args.mode, rows, schemas, tok, k, args.max_prompt)
        t1 = time.time()
        raw, new_tokens = generate(model, tok, texts, args.batch, args.max_new, args.mode)
        gen_s = time.time() - t1
        preds = [extract_sql(t) for t in raw]
    t2 = time.time()
    pred_res = execute.run_many(
        [(dbs[r["db_id"]], p) for r, p in zip(rows, preds, strict=True)],
        workers=args.workers,
        timeout=args.timeout,
    )
    gold_res = execute.run_many(
        [(dbs[r["db_id"]], r["sql"]) for r in rows], workers=args.workers, timeout=args.timeout
    )
    score_s = time.time() - t2
    correct = [execute.ex_correct(p, g) for p, g in zip(pred_res, gold_res, strict=True)]
    lo, hi = execute.bootstrap_ci(correct)
    out = args.out or Path(home) / "eval" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r, p, t, c, pr, gr in zip(rows, preds, raw, correct, pred_res, gold_res, strict=True):
            f.write(
                json.dumps(
                    {
                        "question_id": r["question_id"],
                        "db_id": r["db_id"],
                        "pred": p,
                        "raw": t,
                        "correct": c,
                        "pred_error": pr.error,
                        "gold_error": gr.error,
                    }
                )
                + "\n"
            )
    summary = {
        "name": name,
        "mode": args.mode,
        "split": args.split,
        "model": model_id,
        "adapter": os.path.basename(args.adapter.rstrip("/")) if args.adapter else None,
        "questions": len(rows),
        "ex": sum(correct) / len(correct) if correct else None,
        "ex_ci95": [lo, hi],
        "by_difficulty": execute.by_group(correct, [r.get("difficulty") or "-" for r in rows]),
        "pred_errors": sum(not p.ok for p in pred_res),
        "gold_errors": sum(not g.ok for g in gold_res),
        "shots": args.shots if args.mode == "base" else 0,
        "load_s": round(load_s, 1),
        "generate_s": round(gen_s, 1),
        "new_tokens": new_tokens,
        "score_s": round(score_s, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else None,
    }
    metrics = {"question_id": [r["question_id"] for r in rows], "correct": [int(c) for c in correct]}
    write_record(
        new_record(
            "eval", config={k: str(v) for k, v in vars(args).items()}, metrics=metrics, summary=summary
        )
    )
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 5: quantize Qwen2.5-72B to NF4 once, and check it against BF16.

    python -m dlt.qlora.nf4 quantize --out $DLT_HOME/models/qwen2.5-72b-nf4        (1 GPU)
    python -m dlt.qlora.nf4 logits --load nf4 --model $DLT_HOME/models/qwen2.5-72b-nf4 --out nf4.npz
    python -m dlt.qlora.nf4 logits --load bf16 --out bf16.npz                        (2 GPUs, layers split)
    python -m dlt.qlora.nf4 compare nf4.npz bf16.npz

quantize: loads the pinned BF16 checkpoint with bitsandbytes NF4 (double quantization, BF16 compute;
embeddings, norms and lm_head stay BF16), saves it, and reloads the saved copy, recording load times,
peak GPU memory and host RAM. It also checks the tokenizer: special-token ids, a round trip over every
example text, and that the 72B vocabulary matches the 0.5B tokenizer the data was measured with.

logits: teacher-forced forward over the first N validation examples (prompt + gold SQL + end-of-text),
saving each position's argmax token and the log-probability of the true next token. compare reports
next-token agreement and the perplexity gap, over the SQL tokens and over all tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from dlt.bird.prompt import prompt, target
from dlt.results import new_record, write_record
from dlt.train.config import PINNED

MODEL = "Qwen/Qwen2.5-72B"
GIB = 2**30
SPECIAL = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}


def bnb_config(storage: torch.dtype | None = None):
    """NF4, double quantization, BF16 compute. ``storage=torch.bfloat16`` packs the 4-bit weights in BF16
    tensors so FSDP1 can flatten them with the BF16 norms (the documented FSDP-QLoRA setup)."""
    from transformers import BitsAndBytesConfig

    extra = {"bnb_4bit_quant_storage": storage} if storage is not None else {}
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        **extra,
    )


def host_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB


def gpu_peaks() -> list[float]:
    return [round(torch.cuda.max_memory_allocated(i) / GIB, 2) for i in range(torch.cuda.device_count())]


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def load(kind: str, model: str, revision: str | None):
    """kind: bf16 (layers split over the visible GPUs), nf4 (quantize on load, 1 GPU), nf4-saved."""
    from transformers import AutoModelForCausalLM

    common = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if kind == "bf16":
        n = torch.cuda.device_count()
        return AutoModelForCausalLM.from_pretrained(
            model, revision=revision, device_map="auto", max_memory={i: "125GiB" for i in range(n)}, **common
        )
    if kind == "nf4":
        return AutoModelForCausalLM.from_pretrained(
            model, revision=revision, quantization_config=bnb_config(), device_map={"": 0}, **common
        )
    return AutoModelForCausalLM.from_pretrained(model, device_map={"": 0}, **common)


def examples(data: Path, n: int, schema_key: str) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in (data / "val.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: r["question_id"])
    schemas = json.loads((data / "schemas.json").read_text())
    return rows[:n], {k: v[schema_key] for k, v in schemas.items()}


def check_tokenizer(tok, rows: list[dict], schemas: dict) -> dict:
    from transformers import AutoTokenizer

    ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIAL}
    bad_special = {t: i for t, i in ids.items() if i != SPECIAL[t]}
    texts = [prompt(r, schemas[r["db_id"]]) + target(r) for r in rows]
    roundtrip_fail = sum(tok.decode(tok(t).input_ids) != t for t in texts)
    small = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", revision=PINNED["Qwen/Qwen2.5-0.5B"])
    same_vocab = tok.get_vocab() == small.get_vocab()
    return {
        "special_ids": ids,
        "special_ok": not bad_special,
        "eos_token": tok.eos_token,
        "roundtrip_texts": len(texts),
        "roundtrip_failures": roundtrip_fail,
        "vocab_matches_0p5b": same_vocab,
    }


def cmd_quantize(args) -> int:
    from transformers import AutoTokenizer

    rev = PINNED[MODEL]
    out = Path(args.out)
    t0 = time.time()
    model = load("nf4", MODEL, rev)
    quantize_s = time.time() - t0
    peak_quant = gpu_peaks()
    t0 = time.time()
    model.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    tok.save_pretrained(out)
    save_s = time.time() - t0
    nf4_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    reloaded = load("nf4-saved", str(out), None)
    reload_s = time.time() - t0
    reloaded_params = sum(1 for _ in reloaded.parameters())
    peak_reload = gpu_peaks()
    rows, schemas = examples(Path(args.data), args.n, args.schema)
    summary = {
        "name": "nf4-quantize",
        "model": MODEL,
        "revision": rev,
        "quantize_on_load_s": round(quantize_s, 1),
        "save_s": round(save_s, 1),
        "reload_saved_s": round(reload_s, 1),
        "saved_gib": round(dir_bytes(out) / GIB, 2),
        "param_storage_gib": round(nf4_param_bytes / GIB, 2),
        "reloaded_param_tensors": reloaded_params,
        "peak_gpu_gib_quantize": peak_quant,
        "peak_gpu_gib_reload": peak_reload,
        "host_rss_gib": round(host_rss_gib(), 2),
        "tokenizer": check_tokenizer(tok, rows, schemas),
    }
    write_record(new_record("bench", config=vars(args), metrics={}, summary=summary))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


@torch.no_grad()
def cmd_logits(args) -> int:
    from transformers import AutoTokenizer

    rev = PINNED[MODEL] if args.model == MODEL else None
    t0 = time.time()
    model = load(args.load, args.model, rev)
    load_s = time.time() - t0
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model, revision=rev)
    rows, schemas = examples(Path(args.data), args.n, args.schema)
    device = next(model.parameters()).device
    argmax, gold_lp, offsets, starts, qids = [], [], [0], [], []
    t0 = time.time()
    tokens = 0
    for r in rows:
        p = tok(prompt(r, schemas[r["db_id"]])).input_ids
        full = p + tok(target(r)).input_ids + [tok.eos_token_id]
        x = torch.tensor([full], device=device)
        logits = model(input_ids=x).logits[0, :-1].float()  # predictions for positions 1..T-1
        lp = torch.log_softmax(logits, dim=-1)
        nxt = x[0, 1:]
        argmax.append(logits.argmax(-1).to(torch.int32).cpu().numpy())
        gold_lp.append(lp.gather(1, nxt[:, None])[:, 0].cpu().numpy())
        offsets.append(offsets[-1] + len(full) - 1)
        starts.append(len(p) - 1)  # first prediction of a SQL token, within this example
        qids.append(r["question_id"])
        tokens += len(full)
    fwd_s = time.time() - t0
    np.savez_compressed(
        args.out,
        argmax=np.concatenate(argmax),
        gold_logprob=np.concatenate(gold_lp),
        offsets=np.array(offsets),
        sql_start=np.array(starts),
        question_id=np.array(qids),
    )
    summary = {
        "name": f"logits-{args.load}",
        "load": args.load,
        "model": args.model if args.load != "bf16" else MODEL,
        "gpus": torch.cuda.device_count(),
        "load_s": round(load_s, 1),
        "examples": len(rows),
        "tokens": tokens,
        "forward_s": round(fwd_s, 1),
        "tokens_per_s": round(tokens / fwd_s, 1),
        "peak_gpu_gib": gpu_peaks(),
        "host_rss_gib": round(host_rss_gib(), 2),
        "out": os.path.basename(args.out),
    }
    write_record(new_record("bench", config=vars(args), metrics={}, summary=summary))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def _split(npz) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    off = npz["offsets"]
    am = [npz["argmax"][off[i] : off[i + 1]] for i in range(len(off) - 1)]
    lp = [npz["gold_logprob"][off[i] : off[i + 1]] for i in range(len(off) - 1)]
    return am, lp, list(npz["sql_start"])


def cmd_compare(args) -> int:
    a, b = np.load(args.a), np.load(args.b)
    if not np.array_equal(a["question_id"], b["question_id"]) or not np.array_equal(
        a["offsets"], b["offsets"]
    ):
        raise SystemExit("the two files cover different examples or tokenizations")
    am_a, lp_a, starts = _split(a)
    am_b, lp_b, _ = _split(b)
    agree_all = float(np.mean(np.concatenate([x == y for x, y in zip(am_a, am_b, strict=True)])))
    agree_sql = float(
        np.mean(np.concatenate([x[s:] == y[s:] for x, y, s in zip(am_a, am_b, starts, strict=True)]))
    )

    def ppl(lps, sql_only):
        v = np.concatenate([lp[s:] if sql_only else lp for lp, s in zip(lps, starts, strict=True)])
        return float(np.exp(-v.mean()))

    summary = {
        "name": "nf4-vs-bf16",
        "a": os.path.basename(args.a),
        "b": os.path.basename(args.b),
        "examples": len(starts),
        "positions": int(a["offsets"][-1]),
        "sql_positions": int(sum(len(x) - s for x, s in zip(am_a, starts, strict=True))),
        "argmax_agreement_all": round(agree_all, 5),
        "argmax_agreement_sql": round(agree_sql, 5),
        "ppl_all": {"a": round(ppl(lp_a, False), 5), "b": round(ppl(lp_b, False), 5)},
        "ppl_sql": {"a": round(ppl(lp_a, True), 5), "b": round(ppl(lp_b, True), 5)},
    }
    summary["ppl_sql_gap"] = round(summary["ppl_sql"]["a"] / summary["ppl_sql"]["b"] - 1, 5)
    summary["ppl_all_gap"] = round(summary["ppl_all"]["a"] / summary["ppl_all"]["b"] - 1, 5)
    write_record(new_record("eval", config=vars(args), metrics={}, summary=summary))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    home = os.environ.get("DLT_HOME", ".")
    for name in ("quantize", "logits"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=f"{home}/data/bird")
        p.add_argument("--n", type=int, default=200)
        p.add_argument("--schema", default="ddl_rows", choices=["ddl", "ddl_rows"])
        p.add_argument("--out", required=True)
    sub.choices["logits"].add_argument("--load", choices=["bf16", "nf4", "nf4-saved"], required=True)
    sub.choices["logits"].add_argument("--model", default=MODEL)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)
    return {"quantize": cmd_quantize, "logits": cmd_logits, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

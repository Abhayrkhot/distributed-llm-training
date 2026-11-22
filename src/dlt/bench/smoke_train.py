"""Smoke training run: proves the environment, DDP and the stop signal path work end to end.

Builds a Qwen2.5 model from its pinned config with random weights (no checkpoint download), then
trains on one fixed batch per rank so the loss must fall. FP32 parameters with BF16 autocast, AdamW.
Reports step time, tokens/s, MFU (PLAN.md 4.3 convention) and peak memory per rank.

SIGUSR1 (forwarded by slurm/launch.sh before the wall-time limit) makes every rank stop at the same
step boundary; Phase 2 turns that stop into a checkpoint.
"""

from __future__ import annotations

import argparse
import math
import signal
import statistics
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from dlt import dist as dlt_dist
from dlt.metrics.flops import ModelShape, mfu, train_flops_per_token
from dlt.results import new_record, write_record

PINNED = {
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "Qwen/Qwen2.5-1.5B": "8faed761d45a263340a0528343f099c05c9a4323",
}
_STOP = False


def _on_usr1(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True


def build_config(args):
    from transformers import AutoConfig, Qwen2Config

    if args.tiny:
        return Qwen2Config(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=max(args.seq_len, 128),
            tie_word_embeddings=True,
        )
    return AutoConfig.from_pretrained(args.model, revision=args.revision or PINNED.get(args.model))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tiny", action="store_true", help="2-layer toy config (CPU tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--activation-checkpointing", action="store_true")
    ap.add_argument("--step-sleep", type=float, default=0.0, help="test hook: pause per step")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGUSR1, _on_usr1)
    ctx = dlt_dist.setup(args.device)
    cuda = ctx.device.type == "cuda"
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM

    cfg = build_config(args)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").to(ctx.device)
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    shape = ModelShape.from_hf_config(cfg)
    if ctx.distributed:
        model = DDP(model, device_ids=[ctx.local_rank] if cuda else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=cuda)

    gen = torch.Generator().manual_seed(args.seed * 1000 + ctx.rank)
    batch = torch.randint(0, cfg.vocab_size, (args.micro_batch, args.seq_len), generator=gen).to(ctx.device)

    losses, step_s = [], []
    stopped_at = None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        if dlt_dist.any_rank(ctx, _STOP):
            stopped_at = step
            break
        t0 = time.perf_counter()
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        with torch.autocast(ctx.device.type, dtype=torch.bfloat16, enabled=cuda):
            loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if cuda:
            end.record()
            end.synchronize()
            step_s.append(start.elapsed_time(end) / 1e3)
        else:
            step_s.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if args.step_sleep:
            time.sleep(args.step_sleep)

    timed = step_s[args.warmup :] if len(step_s) > args.warmup else step_s
    step_med = statistics.median(timed) if timed else float("nan")
    tokens_per_step = ctx.world_size * args.micro_batch * args.seq_len
    tok_s = tokens_per_step / step_med if timed else float("nan")
    fpt = train_flops_per_token(shape, args.seq_len, "full", recompute=args.activation_checkpointing)
    gpu_name = torch.cuda.get_device_name(ctx.device) if cuda else None
    model_mfu = mfu(train_flops_per_token(shape, args.seq_len, "full"), tok_s, ctx.world_size, gpu_name)
    hw_fu = mfu(fpt, tok_s, ctx.world_size, gpu_name)
    mem = [
        torch.cuda.max_memory_allocated(ctx.device) / 2**30 if cuda else 0.0,
        torch.cuda.max_memory_reserved(ctx.device) / 2**30 if cuda else 0.0,
    ]
    mem_by_rank = dlt_dist.gather_floats(ctx, mem)

    if ctx.is_main:
        summary = {
            "model": "tiny" if args.tiny else args.model,
            "world_size": ctx.world_size,
            "mode": "ddp" if ctx.distributed else "single",
            "steps_run": len(losses),
            "stopped_by_signal_at_step": stopped_at,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "step_time_s_median": round(step_med, 4) if timed else None,
            "step_time_s_min": round(min(timed), 4) if timed else None,
            "step_time_s_max": round(max(timed), 4) if timed else None,
            "tokens_per_s": round(tok_s, 1) if timed else None,
            "mfu": round(model_mfu, 4) if model_mfu is not None else None,
            "hfu": round(hw_fu, 4) if hw_fu is not None else None,
            "flops_per_token": fpt,
            "matmul_params": shape.matmul_params(),
            "total_params": shape.total_params(),
            "peak_alloc_gib_by_rank": [round(m[0], 2) for m in mem_by_rank],
            "peak_reserved_gib_by_rank": [round(m[1], 2) for m in mem_by_rank],
        }
        cfg_out = {**vars(args), "revision": None if args.tiny else (args.revision or PINNED.get(args.model))}
        rec = new_record(
            "smoke_train",
            config=cfg_out,
            metrics={"loss": losses, "step_time_s": [round(s, 5) for s in step_s]},
            summary=summary,
        )
        path = write_record(rec)
        print(
            f"steps={len(losses)} loss {summary['loss_first']:.3f} -> {summary['loss_last']:.3f} "
            f"step {step_med:.3f}s tok/s {tok_s:,.0f} mfu {summary['mfu']} stopped_at={stopped_at} -> {path}"
        )
        if losses and not math.isfinite(losses[-1]):
            raise SystemExit("non-finite loss")
    dlt_dist.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

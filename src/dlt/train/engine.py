"""Training loop for causal LMs on one or more GPUs: single, DDP or FSDP2, with resumable checkpoints.

Run under torchrun:  python -m dlt.train --config configs/training/X.yaml --set steps=200

Precision: FP32 master parameters and optimizer state. single/DDP compute under BF16 autocast; FSDP2
casts parameters to BF16 for compute (MixedPrecisionPolicy) and reduces gradients in FP32.

Each optimizer step consumes one global batch (``global_batch`` sequences) from the deterministic order
in dlt.data.packed: every rank takes its slice and splits it into micro-batches, so the same config
sees the same tokens per step on 1 or 2 GPUs. The loss is the mean token cross-entropy over the global
batch (every sequence has exactly ``seq_len`` targets, so averaging micro-batch means is exact).

Stopping: SIGUSR1 (forwarded by slurm/launch.sh before the wall-time limit), ``max_seconds`` or
``stop_at_step`` make every rank agree to stop at the same step boundary, write a checkpoint and exit 0.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dlt import dist as dlt_dist
from dlt.data.packed import GlobalOrder, PackedTokens
from dlt.metrics.flops import ModelShape, mfu, train_flops_per_token
from dlt.results import git_sha, new_record, write_record
from dlt.train import checkpoint as ckpt
from dlt.train.config import TrainConfig, parse

_STOP_SIGNAL = False


def _on_usr1(signum, frame):  # noqa: ARG001
    global _STOP_SIGNAL
    _STOP_SIGNAL = True


def tiny_config(seq_len: int):
    from transformers import Qwen2Config

    return Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max(seq_len, 128),
        tie_word_embeddings=True,
    )


def build_model(cfg: TrainConfig):
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(cfg.seed)  # identical random init on every rank
    if cfg.init == "tiny":
        model = AutoModelForCausalLM.from_config(tiny_config(cfg.seq_len), attn_implementation=cfg.attn)
    elif cfg.init == "random":
        hf_cfg = AutoConfig.from_pretrained(cfg.model, revision=cfg.resolved_revision())
        model = AutoModelForCausalLM.from_config(hf_cfg, attn_implementation=cfg.attn)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.resolved_revision(), dtype=torch.float32, attn_implementation=cfg.attn
        )
    model.config.use_cache = False
    if cfg.activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def wrap(model, cfg: TrainConfig, ctx: dlt_dist.DistCtx):
    model.to(ctx.device)
    if cfg.backend == "single":
        if ctx.distributed:
            raise ValueError("backend=single needs exactly one rank")
        return model
    if cfg.backend == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        if not ctx.distributed:
            return model
        return DDP(model, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh(ctx.device.type, (ctx.world_size,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    return model


def make_optimizer(model, cfg: TrainConfig, cuda: bool):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), fused=cuda)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _no_grad_sync(model, cfg: TrainConfig, ctx: dlt_dist.DistCtx, last: bool):
    """Skip the gradient all-reduce / reduce-scatter on all but the last micro-batch of a step."""
    if not ctx.distributed or last:
        if cfg.backend == "fsdp2":
            model.set_requires_gradient_sync(True)
        return contextlib.nullcontext()
    if cfg.backend == "ddp":
        return model.no_sync()
    model.set_requires_gradient_sync(False)
    return contextlib.nullcontext()


def _grad_norm(total) -> float:
    return float(total.full_tensor() if hasattr(total, "full_tensor") else total)


def main(argv: list[str] | None = None) -> int:
    cfg = parse(argv)
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    signal.signal(signal.SIGUSR1, _on_usr1)
    ctx = dlt_dist.setup()
    cuda = ctx.device.type == "cuda"
    autocast = cuda and cfg.backend != "fsdp2"

    model = build_model(cfg)
    shape = ModelShape.from_hf_config(model.config)
    model = wrap(model, cfg, ctx)
    opt, sched = make_optimizer(model, cfg, cuda)

    data = PackedTokens(cfg.data, cfg.seq_len)
    order = GlobalOrder(len(data), cfg.global_batch, cfg.seed)
    accum = cfg.global_batch // (ctx.world_size * cfg.micro_batch)
    if accum * ctx.world_size * cfg.micro_batch != cfg.global_batch:
        raise ValueError("global_batch must be a multiple of world_size x micro_batch")

    root = Path(cfg.ckpt_dir) if cfg.ckpt_dir else None
    start_step, resumed_from, restore_s = 0, None, None
    if root and cfg.resume and (found := ckpt.latest(root)):
        t0 = time.perf_counter()
        meta = ckpt.load(found, model, opt, sched, ctx)
        restore_s = time.perf_counter() - t0
        start_step, resumed_from = meta["step"], found.name
        if ctx.is_main:
            print(f"resumed from {found} (step {start_step}) in {restore_s:.1f}s", flush=True)

    saves: list[dict] = []
    saved_steps = {start_step} if resumed_from else set()

    def save_ckpt(step: int) -> None:
        if step in saved_steps:  # e.g. a periodic save and a stop at the same step
            return
        saved_steps.add(step)
        info = ckpt.save(
            root,
            step,
            model,
            opt,
            sched,
            ctx,
            meta={"config": cfg.to_dict(), "git_sha": git_sha()},
            keep=cfg.keep_ckpts,
            crash_before_publish=cfg.crash_in_save,
        )
        if ctx.is_main:
            print(
                f"checkpoint step {step}: {info['total_s']:.1f}s {info['bytes'] / 2**30:.2f} GiB", flush=True
            )
        saves.append({"step": step, **info})

    losses, grad_norms, lrs, step_s = [], [], [], []
    stopped_by, t_start = None, time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    step = start_step
    while step < cfg.steps:
        want_stop = _STOP_SIGNAL or step == cfg.stop_at_step
        want_stop |= bool(cfg.max_seconds) and time.time() - t_start > cfg.max_seconds
        if dlt_dist.any_rank(ctx, want_stop):
            stopped_by = (
                "signal" if _STOP_SIGNAL else ("stop_at_step" if step == cfg.stop_at_step else "max_seconds")
            )
            if root:
                save_ckpt(step)
            break
        if step == cfg.crash_at_step:
            os.kill(os.getpid(), signal.SIGKILL)

        if cuda:
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=ctx.device)
        mbs = order.micro_batches(step, ctx.rank, ctx.world_size, cfg.micro_batch)
        for i, idx in enumerate(mbs):
            x, y = data.batch(idx)
            x, y = x.to(ctx.device, non_blocking=True), y.to(ctx.device, non_blocking=True)
            with _no_grad_sync(model, cfg, ctx, last=i == len(mbs) - 1):
                with torch.autocast(ctx.device.type, dtype=torch.bfloat16, enabled=autocast):
                    logits = model(input_ids=x).logits
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)) / accum
                loss.backward()
            loss_sum += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if ctx.distributed:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG if cuda else dist.ReduceOp.SUM)
            if not cuda:
                loss_sum /= ctx.world_size
        if cuda:
            ev1.record()
            ev1.synchronize()
            step_s.append(ev0.elapsed_time(ev1) / 1e3)
        else:
            step_s.append(time.perf_counter() - t0)
        losses.append(loss_sum.item())
        grad_norms.append(_grad_norm(gnorm))
        lrs.append(sched.get_last_lr()[0])
        step += 1
        if ctx.is_main and (step % 10 == 0 or step == cfg.steps):
            print(
                f"step {step} loss {losses[-1]:.4f} gnorm {grad_norms[-1]:.3f} {step_s[-1]:.3f}s", flush=True
            )
        if root and cfg.ckpt_every and step % cfg.ckpt_every == 0:
            save_ckpt(step)

    warm = min(3, max(0, len(step_s) - 1))
    timed = step_s[warm:]
    step_med = statistics.median(timed) if timed else None
    tok_s = cfg.global_batch * cfg.seq_len / step_med if step_med else None
    gpu = torch.cuda.get_device_name(ctx.device) if cuda else None
    fpt = train_flops_per_token(shape, cfg.seq_len, "full")
    hfpt = train_flops_per_token(shape, cfg.seq_len, "full", recompute=cfg.activation_checkpointing)
    mem = [
        torch.cuda.max_memory_allocated(ctx.device) / 2**30 if cuda else 0.0,
        torch.cuda.max_memory_reserved(ctx.device) / 2**30 if cuda else 0.0,
    ]
    mem_by_rank = dlt_dist.gather_floats(ctx, mem)
    if ctx.is_main and cfg.write_record:
        m = mfu(fpt, tok_s, ctx.world_size, gpu) if tok_s else None
        h = mfu(hfpt, tok_s, ctx.world_size, gpu) if tok_s else None
        summary = {
            "run_name": cfg.run_name,
            "backend": cfg.backend,
            "world_size": ctx.world_size,
            "accum_steps": accum,
            "start_step": start_step,
            "end_step": step,
            "resumed_from": resumed_from,
            "restore_s": round(restore_s, 2) if restore_s is not None else None,
            "stopped_by": stopped_by,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "step_time_s_median": round(step_med, 4) if step_med else None,
            "tokens_per_s": round(tok_s, 1) if tok_s else None,
            "mfu": round(m, 4) if m is not None else None,
            "hfu": round(h, 4) if h is not None else None,
            "peak_alloc_gib_by_rank": [round(x[0], 2) for x in mem_by_rank],
            "peak_reserved_gib_by_rank": [round(x[1], 2) for x in mem_by_rank],
            "checkpoints": saves,
        }
        metrics = {
            "first_step": start_step,
            "loss": losses,
            "grad_norm": grad_norms,
            "lr": lrs,
            "step_time_s": [round(s, 5) for s in step_s],
        }
        path = write_record(new_record("train", config=cfg.to_dict(), metrics=metrics, summary=summary))
        print(f"steps {start_step}->{step} stopped_by={stopped_by} record {path}", flush=True)
    dlt_dist.teardown()
    return 0

"""Training loop for causal LMs on one or more GPUs: single, DDP or FSDP2, with resumable checkpoints.

Run under torchrun:  python -m dlt.train --config configs/training/X.yaml --set steps=200

Precision: FP32 master parameters and optimizer state. single/DDP compute under BF16 autocast; FSDP2
casts parameters to BF16 for compute (MixedPrecisionPolicy) and reduces gradients in FP32.

Each optimizer step consumes one global batch (``global_batch`` sequences) from the deterministic order
in dlt.data.packed: every rank takes its slice and splits it into micro-batches, so the same config
sees the same tokens per step on 1 or 2 GPUs. The loss is the mean token cross-entropy over the global
batch (every sequence has exactly ``seq_len`` targets, so averaging micro-batch means is exact).

Gradient sync under accumulation (``accum_sync``): ``last`` skips it on all but the last micro-batch
(DDP ``no_sync``; FSDP2 then keeps unsharded FP32 gradients for the whole model), ``every`` syncs each
micro-batch (more communication; FSDP2 keeps only sharded gradients).

Loading: FSDP2 with pretrained weights builds the model on the meta device and each rank reads only its
own shard from the safetensors files (dlt.train.loading), so no rank holds the full model.

Stopping: SIGUSR1 (forwarded by slurm/launch.sh before the wall-time limit), ``max_seconds`` or
``stop_at_step`` make every rank agree to stop at the same step boundary, write a checkpoint and exit 0.
A run that reaches ``steps`` also checkpoints, so a later link of a job chain finds it complete and exits
before loading the model.

Validation: with ``val_data`` and ``eval_samples``, the mean token loss over the first ``eval_samples``
validation sequences is measured at step 0 (fresh runs) and at the end of every run, including stops.

Measurement: each step is split into forward, backward and optimizer time with CUDA events (exposed
communication lands in the phase that waits for it). ``profile_steps`` > 0 runs the PyTorch Profiler
for that many steps and reports compute, communication and exposed communication per step from the
kernel timeline; profiled steps are excluded from timing. Out of memory writes a record with the phase,
memory statistics and the predicted peak, then exits with code 75 (torchrun stops the other ranks).
"""

from __future__ import annotations

import contextlib
import math
import os
import resource
import shutil
import signal
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dlt import dist as dlt_dist
from dlt.data.packed import GlobalOrder, PackedTokens
from dlt.metrics.flops import ModelShape, mfu, train_flops_per_token
from dlt.metrics.memory import GIB, predict_for_config
from dlt.metrics.memory import VERSION as PREDICTOR_VERSION
from dlt.metrics.overlap import comm_overlap, kernels_from_trace
from dlt.results import git_sha, new_record, write_record
from dlt.train import checkpoint as ckpt
from dlt.train import loading
from dlt.train.config import TrainConfig, parse

_STOP_SIGNAL = False
OOM_EXIT = 75


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


def sharded_load(cfg: TrainConfig) -> bool:
    return cfg.backend == "fsdp2" and cfg.init == "pretrained" and cfg.sharded_load


def build_model(cfg: TrainConfig):
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(cfg.seed)  # identical random init on every rank
    if sharded_load(cfg):
        model = loading.build_on_meta(cfg.model, cfg.resolved_revision(), cfg.attn)
    elif cfg.init == "tiny":
        model = AutoModelForCausalLM.from_config(tiny_config(cfg.seq_len), attn_implementation=cfg.attn)
    elif cfg.init == "random":
        hf_cfg = AutoConfig.from_pretrained(cfg.model, revision=cfg.resolved_revision())
        model = AutoModelForCausalLM.from_config(hf_cfg, attn_implementation=cfg.attn)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.resolved_revision(), dtype=torch.float32, attn_implementation=cfg.attn
        )
    model.config.use_cache = False
    # from_pretrained returns the model in eval mode, and transformers only checkpoints layers in train
    # mode: without this, activation checkpointing was silently off for single/DDP in Phase 3
    model.train()
    if cfg.activation_checkpointing and cfg.ac_every == 1:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif cfg.activation_checkpointing:
        checkpoint_layers(model, cfg.ac_every)
    if cfg.loss_chunk:
        ChunkedLossHead.install(model.lm_head, cfg.loss_chunk)
    if cfg.compile:
        # one compiled graph serves every layer (parameters are graph inputs, so the identical layers
        # share it); applied before DDP/FSDP2 wrapping, whose hooks stay outside the compiled region
        for layer in model.model.layers:
            layer.compile(backend=cfg.compile_backend)
    return model


def checkpoint_layers(model, every: int) -> list[int]:
    """Checkpoint decoder layers 0, every, 2*every, ... by wrapping their forward (names unchanged)."""
    from torch.utils.checkpoint import checkpoint

    chosen = []
    for i, layer in enumerate(model.model.layers):
        if i % every:
            continue
        inner = layer.forward

        def forward(*args, _inner=inner, **kwargs):
            return checkpoint(_inner, *args, use_reentrant=False, **kwargs)

        layer.forward = forward
        chosen.append(i)
    return chosen


def _chunk_ce_sum(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(F.linear(h, w).float(), y, reduction="sum")


class ChunkedLossHead(torch.nn.Linear):
    """lm_head that returns the summed token cross-entropy instead of logits.

    The output projection and the loss run over ``chunk`` tokens at a time, each under activation
    checkpointing, so neither the full logits nor their FP32 copy and gradient ever exist (14 V bytes
    per token otherwise). Costs one extra lm_head forward in backward. Targets are set per micro-batch
    through ``targets``; each chunk's targets are passed to the checkpointed function explicitly.
    The class is swapped in place, so the parameter keeps its name and its tie to the embedding.
    """

    chunk: int
    targets: torch.Tensor | None

    @classmethod
    def install(cls, head: torch.nn.Linear, chunk: int) -> None:
        head.__class__ = cls
        head.chunk, head.targets = chunk, None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        if self.targets is None:
            raise RuntimeError("set lm_head.targets before the forward")
        hs, ys = h.reshape(-1, h.size(-1)), self.targets.reshape(-1)
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        for hc, yc in zip(hs.split(self.chunk), ys.split(self.chunk), strict=True):
            total = total + checkpoint(_chunk_ce_sum, hc, self.weight, yc, use_reentrant=False)
        return total


def token_loss_sum(model, x: torch.Tensor, y: torch.Tensor, cfg: TrainConfig, autocast: bool) -> torch.Tensor:
    """Summed token cross-entropy of one micro-batch (FP32), through full logits or the chunked head."""
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=autocast):
        if cfg.loss_chunk:
            getattr(model, "module", model).lm_head.targets = y
            return model(input_ids=x).logits
        logits = model(input_ids=x).logits
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum")


def sdpa_context(cfg: TrainConfig):
    """Restrict scaled_dot_product_attention to one backend (forward, recompute and backward)."""
    if cfg.sdpa_backend == "auto":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }
    return sdpa_kernel([backends[cfg.sdpa_backend]])


def nvtx(cfg: TrainConfig, name: str):
    if not (cfg.nvtx and torch.cuda.is_available()):
        return contextlib.nullcontext()
    return torch.cuda.nvtx.range(name)


def wrap(model, cfg: TrainConfig, ctx: dlt_dist.DistCtx):
    on_meta = any(p.is_meta for p in model.parameters())
    if not on_meta:
        model.to(ctx.device)
    if cfg.backend == "single":
        if ctx.distributed:
            raise ValueError("backend=single needs exactly one rank")
        return model
    if cfg.backend == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        if not ctx.distributed:
            return model
        return DDP(
            model,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            bucket_cap_mb=cfg.ddp_bucket_mb,
            gradient_as_bucket_view=cfg.ddp_bucket_view,
        )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh(ctx.device.type, (ctx.world_size,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    layers = list(model.model.layers)
    reshard = cfg.reshard_after_forward and not cfg.fsdp_hold_params
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    if (
        cfg.fsdp_prefetch > 0
    ):  # explicit all-gather prefetch of the next (forward) / previous (backward) layers
        n = cfg.fsdp_prefetch
        for i, layer in enumerate(layers):
            if nxt := layers[i + 1 : i + 1 + n]:
                layer.set_modules_to_forward_prefetch(nxt)
            if prev := layers[max(0, i - n) : i][::-1]:
                layer.set_modules_to_backward_prefetch(prev)
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
    if cfg.backend == "fsdp2" and cfg.fsdp_hold_params:
        # with reshard_after_forward off, this keeps every layer gathered from the first micro-batch's
        # forward to the last one's backward: one all-gather per layer per step instead of 2 per micro-batch
        model.set_reshard_after_backward(last)
    if not ctx.distributed or last or cfg.accum_sync == "every":
        if cfg.backend == "fsdp2":
            model.set_requires_gradient_sync(True)
        return contextlib.nullcontext()
    if cfg.backend == "ddp":
        return model.no_sync()
    model.set_requires_gradient_sync(False)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, val: PackedTokens, cfg: TrainConfig, ctx: dlt_dist.DistCtx, autocast: bool) -> float:
    """Mean token cross-entropy over val samples [0, eval_samples), each rank taking a contiguous block."""
    per_rank, rem = divmod(cfg.eval_samples, ctx.world_size)
    if rem or per_rank % cfg.micro_batch or cfg.eval_samples > len(val):
        raise ValueError(
            f"eval_samples={cfg.eval_samples} must be a multiple of world_size x micro_batch "
            f"and at most {len(val)}"
        )
    total = torch.zeros(2, dtype=torch.float64, device=ctx.device)  # loss sum, token count
    model.eval()
    for idx in torch.arange(ctx.rank * per_rank, (ctx.rank + 1) * per_rank).split(cfg.micro_batch):
        x, y = val.batch(idx)
        x, y = x.to(ctx.device), y.to(ctx.device)
        with sdpa_context(cfg):
            total[0] += token_loss_sum(model, x, y, cfg, autocast)
        total[1] += y.numel()
    model.train()
    if ctx.distributed:
        dist.all_reduce(total)
    return float(total[0] / total[1])


def _grad_norm(total) -> float:
    return float(total.full_tensor() if hasattr(total, "full_tensor") else total)


class Clock:
    """Timestamps for step phases: CUDA events on GPU (read after the step's final sync), else wall time."""

    def __init__(self, cuda: bool):
        self.cuda = cuda

    def now(self):
        if not self.cuda:
            return time.perf_counter()
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def dt(self, a, b) -> float:
        return a.elapsed_time(b) / 1e3 if self.cuda else b - a


def _host_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB  # ru_maxrss is KiB on Linux


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 5) if xs else None


def _predicted(shape: ModelShape, cfg: TrainConfig, world: int, accum: int) -> dict[str, float]:
    terms = predict_for_config(shape, cfg, world, accum)
    return {k: round(v / GIB, 3) for k, v in terms.items()}


def _oom_exit(cfg: TrainConfig, ctx: dlt_dist.DistCtx, state: dict, err: BaseException) -> None:
    """Record what ran out of memory where, then exit hard (collectives on other ranks would hang)."""
    cuda = ctx.device.type == "cuda"
    mem = {}
    if cuda:
        free, total_b = torch.cuda.mem_get_info(ctx.device)
        mem = {
            "allocated_gib": round(torch.cuda.memory_allocated(ctx.device) / GIB, 3),
            "reserved_gib": round(torch.cuda.memory_reserved(ctx.device) / GIB, 3),
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated(ctx.device) / GIB, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(ctx.device) / GIB, 3),
            "device_free_gib": round(free / GIB, 3),
            "device_total_gib": round(total_b / GIB, 3),
        }
    summary = {
        "run_name": cfg.run_name,
        "status": "oom",
        "backend": cfg.backend,
        "world_size": ctx.world_size,
        "rank": ctx.rank,
        "phase": state.get("phase"),
        "accum_steps": state.get("accum"),
        "error": str(err).splitlines()[0][:400] if str(err) else type(err).__name__,
        "memory": mem,
        "host_rss_gib": round(_host_rss_gib(), 2),
        "predicted_gib": state.get("predicted"),
        "predictor_version": PREDICTOR_VERSION,
    }
    metrics = {"memory_summary": torch.cuda.memory_summary(ctx.device) if cuda else ""}
    if cfg.write_record:
        rec = new_record("train", config=cfg.to_dict(), metrics=metrics, summary=summary)
        rec["run_id"] += f"-rank{ctx.rank}"
        path = write_record(rec)
        print(f"OOM on rank {ctx.rank} during {state.get('phase')}: record {path}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(OOM_EXIT)


def main(argv: list[str] | None = None) -> int:
    cfg = parse(argv)
    if cfg.alloc_conf:  # read when the CUDA caching allocator initializes, which is after this
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cfg.alloc_conf
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    signal.signal(signal.SIGUSR1, _on_usr1)
    ctx = dlt_dist.setup()
    root = Path(cfg.ckpt_dir) if cfg.ckpt_dir else None
    if root and cfg.resume and (done := ckpt.latest(root)) and ckpt.read_meta(done)["step"] >= cfg.steps:
        if ctx.is_main:
            print(f"{done} has reached steps={cfg.steps}: run complete, nothing to do", flush=True)
        dlt_dist.teardown()
        return 0
    state: dict = {"phase": "load"}
    try:
        train(cfg, ctx, root, state)
    except torch.OutOfMemoryError as err:
        _oom_exit(cfg, ctx, state, err)
    dlt_dist.teardown()
    return 0


def train(cfg: TrainConfig, ctx: dlt_dist.DistCtx, root: Path | None, state: dict) -> None:
    cuda = ctx.device.type == "cuda"
    autocast = cuda and cfg.backend != "fsdp2"
    accum = cfg.global_batch // (ctx.world_size * cfg.micro_batch)
    if accum * ctx.world_size * cfg.micro_batch != cfg.global_batch:
        raise ValueError("global_batch must be a multiple of world_size x micro_batch")
    state["accum"] = accum

    t_load = time.perf_counter()
    model = build_model(cfg)
    shape = ModelShape.from_hf_config(model.config)
    state["predicted"] = _predicted(shape, cfg, ctx.world_size, accum)
    model = wrap(model, cfg, ctx)
    load_info: dict = {}
    if sharded_load(cfg):
        directory = loading.snapshot_dir(cfg.model, cfg.resolved_revision())
        load_info = loading.materialize_and_load(model, directory, ctx.device, ctx.rank, ctx.world_size)
    if cuda:
        torch.cuda.synchronize(ctx.device)
    load_s = time.perf_counter() - t_load
    opt, sched = make_optimizer(model, cfg, cuda)

    data = PackedTokens(cfg.data, cfg.seq_len)
    order = GlobalOrder(len(data), cfg.global_batch, cfg.seed)

    start_step, resumed_from, restore_s = 0, None, None
    if root and cfg.resume and (found := ckpt.latest(root)):
        state["phase"] = "restore"
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
        state["phase"] = f"save:{step}"
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

    val = PackedTokens(cfg.val_data, cfg.seq_len) if cfg.eval_samples else None
    val_log: list[dict] = []

    def run_eval(step: int) -> None:
        state["phase"] = f"eval:{step}"
        t0 = time.perf_counter()
        loss = evaluate(model, val, cfg, ctx, autocast)
        val_log.append({"step": step, "loss": loss, "seconds": round(time.perf_counter() - t0, 2)})
        if ctx.is_main:
            print(f"val step {step} loss {loss:.4f} ({val_log[-1]['seconds']}s)", flush=True)

    if val is not None and start_step == 0:
        run_eval(0)

    clock = Clock(cuda)
    losses, grad_norms, lrs = [], [], []
    times: dict[str, list[float]] = {"step": [], "fwd": [], "bwd": [], "opt": []}
    prof_first = start_step + cfg.profile_start if cfg.profile_steps else None
    prof, prof_dir, profiled = None, None, []
    stopped_by, t_start = None, time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    step = start_step
    while step < cfg.steps:
        state["phase"] = f"step:{step}"
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
        if step == cfg.oom_at_step and ctx.is_main:
            raise torch.OutOfMemoryError(f"simulated out of memory at step {step} (oom_at_step)")
        if step == prof_first and cfg.profiler == "nsys":
            torch.cuda.profiler.start()  # nsys --capture-range=cudaProfilerApi records from here
            prof = "nsys"
        elif step == prof_first:
            prof_dir = Path(tempfile.mkdtemp(prefix="dlt-prof-"))
            # GPU kernels only: CPU op events would make traces of 16-32 micro-batch steps huge
            activity = torch.profiler.ProfilerActivity.CUDA if cuda else torch.profiler.ProfilerActivity.CPU
            prof = torch.profiler.profile(activities=[activity])
            prof.start()

        t0 = clock.now()
        fwd_bwd = []
        loss_sum = torch.zeros((), device=ctx.device)
        mbs = order.micro_batches(step, ctx.rank, ctx.world_size, cfg.micro_batch)
        for i, idx in enumerate(mbs):
            x, y = data.batch(idx)
            x, y = x.to(ctx.device, non_blocking=True), y.to(ctx.device, non_blocking=True)
            with _no_grad_sync(model, cfg, ctx, last=i == len(mbs) - 1), sdpa_context(cfg):
                f0 = clock.now()
                with nvtx(cfg, "forward"):
                    if cfg.loss_chunk:
                        loss = token_loss_sum(model, x, y, cfg, autocast) / y.numel() / accum
                    else:  # the Phase 2/3 path, kept op for op
                        with torch.autocast(ctx.device.type, dtype=torch.bfloat16, enabled=autocast):
                            logits = model(input_ids=x).logits
                        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)) / accum
                f1 = clock.now()
                with nvtx(cfg, "backward"):
                    loss.backward()
                b1 = clock.now()
            fwd_bwd.append((f0, f1, b1))
            loss_sum += loss.detach()
        o0 = clock.now()
        with nvtx(cfg, "optimizer"):
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        o1 = clock.now()
        if ctx.distributed:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG if cuda else dist.ReduceOp.SUM)
            if not cuda:
                loss_sum /= ctx.world_size
        t1 = clock.now()
        if cuda:
            t1.synchronize()
        times["step"].append(clock.dt(t0, t1))
        times["fwd"].append(sum(clock.dt(a, b) for a, b, _ in fwd_bwd))
        times["bwd"].append(sum(clock.dt(b, c) for _, b, c in fwd_bwd))
        times["opt"].append(clock.dt(o0, o1))
        losses.append(loss_sum.item())
        grad_norms.append(_grad_norm(gnorm))
        lrs.append(sched.get_last_lr()[0])
        if prof == "nsys":
            profiled.append(step)
            if len(profiled) == cfg.profile_steps:
                torch.cuda.profiler.stop()
                prof = None
        elif prof is not None:
            profiled.append(step)
            if len(profiled) == cfg.profile_steps:
                prof.stop()
                trace = prof_dir / f"rank{ctx.rank}.json"
                prof.export_chrome_trace(str(trace))
                state["comm"] = comm_overlap(kernels_from_trace(trace), n_steps=len(profiled))
                if cfg.profile_dir:
                    dst = Path(cfg.profile_dir)
                    dst.mkdir(parents=True, exist_ok=True)
                    trace.replace(dst / f"{cfg.run_name or 'run'}-rank{ctx.rank}.json")
                shutil.rmtree(prof_dir, ignore_errors=True)
                prof = None
        step += 1
        if ctx.is_main and (step % 10 == 0 or step == cfg.steps):
            print(
                f"step {step} loss {losses[-1]:.4f} gnorm {grad_norms[-1]:.3f} {times['step'][-1]:.3f}s",
                flush=True,
            )
        if root and cfg.ckpt_every and step % cfg.ckpt_every == 0:
            save_ckpt(step)
    if prof == "nsys":  # the run ended inside the profiling window
        torch.cuda.profiler.stop()
    elif prof is not None:
        prof.stop()
        shutil.rmtree(prof_dir, ignore_errors=True)
    if root and stopped_by is None:
        save_ckpt(step)  # the run is complete; later links of a chain see it and exit
    if val is not None:
        run_eval(step)
    state["phase"] = "report"

    # timing: skip warmup steps and any profiled steps
    warm = min(cfg.timing_warmup, max(0, len(losses) - 1))
    keep = [k for k in range(warm, len(losses)) if start_step + k not in profiled]
    med = {k: _median([v[j] for j in keep]) for k, v in times.items()}
    step_med = med["step"]
    tok_s = cfg.global_batch * cfg.seq_len / step_med if step_med else None
    gpu = torch.cuda.get_device_name(ctx.device) if cuda else None
    fpt = train_flops_per_token(shape, cfg.seq_len, "full")
    hfpt = train_flops_per_token(shape, cfg.seq_len, "full", recompute=cfg.activation_checkpointing)
    mem = [
        torch.cuda.max_memory_allocated(ctx.device) / GIB if cuda else 0.0,
        torch.cuda.max_memory_reserved(ctx.device) / GIB if cuda else 0.0,
        _host_rss_gib(),
        load_s,
    ]
    by_rank = dlt_dist.gather_floats(ctx, mem)
    comm = state.get("comm")
    comm_keys = ("exposed_comm_s", "comm_s", "compute_s", "gpu_idle_s")
    comm_by_rank = (
        dlt_dist.gather_floats(ctx, [comm[k] for k in comm_keys])
        if cfg.profile_steps and comm is not None
        else None
    )
    if ctx.is_main and cfg.write_record:
        m = mfu(fpt, tok_s, ctx.world_size, gpu) if tok_s else None
        h = mfu(hfpt, tok_s, ctx.world_size, gpu) if tok_s else None
        pred = state["predicted"]["total"]
        peak = max(r[0] for r in by_rank)
        summary = {
            "run_name": cfg.run_name,
            "status": "ok",
            "backend": cfg.backend,
            "world_size": ctx.world_size,
            "accum_steps": accum,
            "accum_sync": cfg.accum_sync,
            "activation_checkpointing": cfg.activation_checkpointing,
            "start_step": start_step,
            "end_step": step,
            "resumed_from": resumed_from,
            "restore_s": round(restore_s, 2) if restore_s is not None else None,
            "stopped_by": stopped_by,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "val_loss_start": next((v["loss"] for v in val_log if v["step"] == 0), None),
            "val_loss_end": val_log[-1]["loss"] if val_log else None,
            "timed_steps": len(keep),
            "timing_warmup": warm,
            "step_time_s_median": med["step"],
            "step_time_s_min": round(min(times["step"][j] for j in keep), 5) if keep else None,
            "step_time_s_max": round(max(times["step"][j] for j in keep), 5) if keep else None,
            "fwd_s_median": med["fwd"],
            "bwd_s_median": med["bwd"],
            "opt_s_median": med["opt"],
            "tokens_per_s": round(tok_s, 1) if tok_s else None,
            "mfu": round(m, 4) if m is not None else None,
            "hfu": round(h, 4) if h is not None else None,
            "peak_alloc_gib_by_rank": [round(r[0], 2) for r in by_rank],
            "peak_reserved_gib_by_rank": [round(r[1], 2) for r in by_rank],
            "host_rss_gib_by_rank": [round(r[2], 2) for r in by_rank],
            "load_s_by_rank": [round(r[3], 1) for r in by_rank],
            "sharded_load": sharded_load(cfg),
            **({"load_read_gib": round(load_info["load_read_bytes"] / GIB, 2)} if load_info else {}),
            "predicted_gib": pred,
            "predictor_version": PREDICTOR_VERSION,
            "predicted_error": round((pred - peak) / peak, 4) if cuda and peak else None,
            "comm_by_rank": (
                [{k: round(v, 5) for k, v in zip(comm_keys, c, strict=True)} for c in comm_by_rank]
                if comm_by_rank
                else None
            ),
            "profiled_steps": profiled,
            "checkpoints": saves,
        }
        metrics = {
            "first_step": start_step,
            "loss": losses,
            "grad_norm": grad_norms,
            "lr": lrs,
            "step_time_s": [round(s, 5) for s in times["step"]],
            "fwd_s": [round(s, 5) for s in times["fwd"]],
            "bwd_s": [round(s, 5) for s in times["bwd"]],
            "opt_s": [round(s, 5) for s in times["opt"]],
            "val": val_log,
            "predicted_terms_gib": state["predicted"],
        }
        path = write_record(new_record("train", config=cfg.to_dict(), metrics=metrics, summary=summary))
        print(f"steps {start_step}->{step} stopped_by={stopped_by} record {path}", flush=True)

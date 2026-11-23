"""Distributed checkpoints: DCP for model and optimizer, plus run state, published by atomic rename.

Layout under ``ckpt_dir``::

    step_00000120/            complete checkpoint = state after 120 optimizer steps
      .metadata, __*.distcp   DCP model + optimizer shards (any backend: single, DDP, FSDP2)
      rank0.pt, rank1.pt      per-rank RNG states
      meta.json               step, scheduler, config, git SHA, world size, save time (written last)
    step_00000180.tmp/        an interrupted save: ignored, removed by the next save

A checkpoint is written into ``*.tmp`` and renamed only after every rank has finished, so a crash
during a save leaves the previous checkpoint as the latest complete one.

With ``frozen=False`` (LoRA on a frozen base) only the trainable parameters and their optimizer state
are saved and restored; the base comes from its own pinned checkpoint.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from dlt.dist import DistCtx


def _barrier(ctx: DistCtx) -> None:
    if ctx.distributed:
        dist.barrier()


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def complete_checkpoints(root: Path) -> list[Path]:
    if not root.exists():
        return []
    done = [p for p in root.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp")]
    return sorted((p for p in done if (p / "meta.json").exists()), key=lambda p: p.name)


def latest(root: Path) -> Path | None:
    done = complete_checkpoints(root)
    return done[-1] if done else None


def read_meta(path: Path) -> dict[str, Any]:
    return json.loads((path / "meta.json").read_text())


_GLOO = None


def _async_group():
    """A dedicated CPU group for dcp.async_save: its background thread runs collectives while training
    runs its own, and sharing one group between the two threads deadlocks (seen with gloo on CPU)."""
    global _GLOO
    if not dist.is_initialized():
        return None
    if _GLOO is None:
        _GLOO = dist.new_group(backend="gloo")
    return _GLOO


class PendingSave:
    """A checkpoint whose shards are still being written in the background (DCP async_save). It is
    published (meta.json + rename) only by ``finish``, so an interrupted write is never the latest."""

    def __init__(self, root, step, tmp, final, future, meta, ctx, keep, t0, blocking_s):
        self.root, self.step, self.tmp, self.final = root, step, tmp, final
        self.future, self.meta, self.ctx, self.keep = future, meta, ctx, keep
        self.t0, self.blocking_s = t0, blocking_s

    def finish(self, crash_before_publish: bool = False) -> dict[str, float]:
        if self.future is not None:
            self.future.result()
        return _publish(self, crash_before_publish)


def _publish(p: PendingSave, crash_before_publish: bool) -> dict[str, float]:
    _barrier(p.ctx)
    written = time.perf_counter()
    if crash_before_publish:
        os.kill(os.getpid(), signal.SIGKILL)
    if p.ctx.is_main:
        (p.tmp / "meta.json").write_text(json.dumps(p.meta, indent=2, default=str))
        os.replace(p.tmp, p.final)
        for old in complete_checkpoints(p.root)[: -p.keep]:
            shutil.rmtree(old, ignore_errors=True)
    _barrier(p.ctx)
    size = sum(f.stat().st_size for f in p.final.rglob("*") if f.is_file()) if p.ctx.is_main else 0
    return {
        "write_s": written - p.t0,
        "total_s": time.perf_counter() - p.t0,
        "blocking_s": p.blocking_s if p.future is not None else time.perf_counter() - p.t0,
        "async": p.future is not None,
        "bytes": size,
    }


def begin_save(
    root: Path,
    step: int,
    model,
    optimizer,
    scheduler,
    ctx: DistCtx,
    meta: dict[str, Any],
    keep: int = 2,
    frozen: bool = True,
    asynchronous: bool = False,
) -> PendingSave:
    """Write the shards (synchronously, or staged to host memory and written in the background)."""
    t0 = time.perf_counter()
    tmp = root / f"step_{step:08d}.tmp"
    final = root / f"step_{step:08d}"
    if ctx.is_main:
        root.mkdir(parents=True, exist_ok=True)
        for stale in root.glob("step_*.tmp"):
            shutil.rmtree(stale, ignore_errors=True)
        tmp.mkdir()
    _barrier(ctx)
    model_sd, optim_sd = get_state_dict(model, optimizer, options=_options(frozen))
    state = {"model": model_sd, "optim": optim_sd}
    future = None
    if asynchronous:
        future = dcp.async_save(state, checkpoint_id=str(tmp), process_group=_async_group())
    else:
        dcp.save(state, checkpoint_id=str(tmp))
    torch.save(rng_state(), tmp / f"rank{ctx.rank}.pt")
    meta = {**meta, "step": step, "world_size": ctx.world_size, "scheduler": scheduler.state_dict()}
    return PendingSave(root, step, tmp, final, future, meta, ctx, keep, t0, time.perf_counter() - t0)


def save(
    root: Path,
    step: int,
    model,
    optimizer,
    scheduler,
    ctx: DistCtx,
    meta: dict[str, Any],
    keep: int = 2,
    crash_before_publish: bool = False,
    frozen: bool = True,
) -> dict[str, float]:
    pending = begin_save(root, step, model, optimizer, scheduler, ctx, meta, keep=keep, frozen=frozen)
    return pending.finish(crash_before_publish)


def _options(frozen: bool) -> StateDictOptions:
    return StateDictOptions(ignore_frozen_params=not frozen)


def load(path: Path, model, optimizer, scheduler, ctx: DistCtx, frozen: bool = True) -> dict[str, Any]:
    meta = read_meta(path)
    model_sd, optim_sd = get_state_dict(model, optimizer, options=_options(frozen))
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optim"],
        options=StateDictOptions(ignore_frozen_params=not frozen, strict=frozen),
    )
    scheduler.load_state_dict(meta["scheduler"])
    rank_file = path / f"rank{ctx.rank}.pt"
    if meta["world_size"] == ctx.world_size and rank_file.exists():
        set_rng_state(torch.load(rank_file, weights_only=False))
    return meta

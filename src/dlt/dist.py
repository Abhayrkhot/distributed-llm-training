"""Process-group setup from torchrun's environment, plus small collective helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistCtx:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def setup(device: str | None = None) -> DistCtx:
    """Read RANK/WORLD_SIZE/LOCAL_RANK (set by torchrun; default: one process); init NCCL or gloo."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    kind = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device("cuda", local) if kind == "cuda" else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    if world > 1 and not dist.is_initialized():
        if dev.type == "cuda":
            dist.init_process_group(backend="nccl", device_id=dev)
        else:
            dist.init_process_group(backend="gloo")
    return DistCtx(rank=rank, world_size=world, local_rank=local, device=dev)


def ensure_group(ctx: DistCtx) -> None:
    """A one-process group (in-memory store, no rendezvous) when none exists: FSDP2's device mesh needs a
    process group even on a single rank, and a process started without torchrun has none."""
    if dist.is_initialized():
        return
    if ctx.device.type == "cuda":
        dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1, device_id=ctx.device)
    else:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)


def teardown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def gather_floats(ctx: DistCtx, values: list[float]) -> list[list[float]]:
    """Every rank contributes the same-length list; every rank gets all of them, indexed by rank."""
    t = torch.tensor(values, dtype=torch.float64, device=ctx.device)
    if not ctx.distributed:
        return [t.tolist()]
    out = [torch.empty_like(t) for _ in range(ctx.world_size)]
    dist.all_gather(out, t)
    return [o.tolist() for o in out]


def any_rank(ctx: DistCtx, flag: bool) -> bool:
    """True if the flag is set on any rank (used so every rank stops at the same step)."""
    if not ctx.distributed:
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())

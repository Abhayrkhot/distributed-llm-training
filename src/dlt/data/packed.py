"""Packed fixed-length samples over a flat uint32 token file, and the deterministic global batch order.

Sample i is tokens[i*T : i*T + T + 1]: inputs are the first T tokens, targets the last T, so samples
never overlap and every target token is real (no padding). Document boundaries are end-of-text
tokens inside a sample; attention is not masked across them (standard packed pretraining).

The global order is a pure function of (seed, step): epoch e uses randperm(seed, e) over all samples,
global step s takes the next ``global_batch`` indices, and rank r takes a contiguous block of them,
split into micro-batches. Resuming at step s therefore needs no data-loader state beyond s itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


class PackedTokens:
    def __init__(self, path: str | Path, seq_len: int):
        self.path = Path(path)
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.n_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.seq_len
        rows = np.stack([self.tokens[i * t : i * t + t + 1] for i in indices.tolist()]).astype(np.int64)
        x = torch.from_numpy(rows)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()


@dataclass(frozen=True)
class GlobalOrder:
    n_samples: int
    global_batch: int
    seed: int

    def __post_init__(self):
        if self.global_batch > self.n_samples:
            raise ValueError(f"global batch {self.global_batch} > {self.n_samples} samples")

    @property
    def steps_per_epoch(self) -> int:
        return self.n_samples // self.global_batch

    def epoch_perm(self, epoch: int) -> torch.Tensor:
        return _perm(self.n_samples, self.seed, epoch)

    def global_indices(self, step: int) -> torch.Tensor:
        epoch, k = divmod(step, self.steps_per_epoch)
        off = k * self.global_batch
        return self.epoch_perm(epoch)[off : off + self.global_batch]

    def micro_batches(self, step: int, rank: int, world: int, micro: int) -> list[torch.Tensor]:
        per_rank, rem = divmod(self.global_batch, world)
        if rem or per_rank % micro:
            raise ValueError(
                f"global batch {self.global_batch} not divisible by world {world} x micro {micro}"
            )
        mine = self.global_indices(step)[rank * per_rank : (rank + 1) * per_rank]
        return list(mine.split(micro))


@lru_cache(maxsize=2)
def _perm(n: int, seed: int, epoch: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(n, generator=g)

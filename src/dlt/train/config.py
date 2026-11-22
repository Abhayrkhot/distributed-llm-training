"""Training configuration: a dataclass, optionally loaded from YAML, overridden with --set key=value."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PINNED = {
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "Qwen/Qwen2.5-1.5B": "8faed761d45a263340a0528343f099c05c9a4323",
    "Qwen/Qwen2.5-3B": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796",
    "Qwen/Qwen2.5-14B": "97e1e76335b7017d8f67c08a19d103c0504298c9",
}
BACKENDS = ("single", "ddp", "fsdp2")


@dataclass
class TrainConfig:
    # model
    model: str = "Qwen/Qwen2.5-0.5B"
    revision: str | None = None  # None: the pinned revision from PINNED
    init: str = "pretrained"  # pretrained | random | tiny (2-layer toy, CPU tests)
    backend: str = "ddp"  # single | ddp | fsdp2
    attn: str = "sdpa"  # sdpa | eager (eager is deterministic)
    activation_checkpointing: bool = False
    # data
    data: str = ""  # train.bin (flat uint32 tokens)
    seq_len: int = 2048
    global_batch: int = 32  # sequences per optimizer step, across all ranks
    micro_batch: int = 4  # sequences per forward/backward on one rank
    seed: int = 0
    # optimization
    steps: int = 100
    lr: float = 3e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 10
    weight_decay: float = 0.1
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    deterministic: bool = False
    # checkpointing
    ckpt_dir: str = ""  # empty: no checkpoints
    ckpt_every: int = 0  # 0: only when stopping
    keep_ckpts: int = 2
    resume: bool = True
    # run control and test hooks
    run_name: str = ""
    max_seconds: float = 0.0  # stop (checkpoint, exit) after this much wall time; 0 = off
    stop_at_step: int = -1  # behave as if SIGUSR1 arrived before this step
    crash_at_step: int = -1  # SIGKILL every rank before this step (simulated node failure)
    crash_in_save: bool = False  # SIGKILL after the checkpoint is written but before it is published
    write_record: bool = True

    def resolved_revision(self) -> str | None:
        return self.revision or PINNED.get(self.model)

    def validate(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if self.init not in ("pretrained", "random", "tiny"):
            raise ValueError("init must be pretrained, random or tiny")
        if not self.data:
            raise ValueError("data (path to train.bin) is required")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["revision"] = self.resolved_revision()
        return d


def coerce(key: str, value):
    """Convert a YAML value or a --set string to the type of the field's default."""
    defaults = {f.name: getattr(TrainConfig(), f.name) for f in dataclasses.fields(TrainConfig)}
    if key not in defaults:
        raise ValueError(f"unknown config key {key!r}")
    default = defaults[key]
    if isinstance(value, str):
        text = value.strip()
        if isinstance(default, bool):
            if text.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"{key}: expected a boolean, got {value!r}")
            return text.lower() in ("true", "1", "yes")
        if isinstance(default, int):
            return int(text)
        if isinstance(default, float):
            return float(text)
        if isinstance(default, list):
            return yaml.safe_load(text)
        return None if text.lower() in ("none", "null") else text
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def parse(argv: list[str] | None = None) -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train a causal LM (see dlt.train.engine)")
    ap.add_argument("--config", type=Path, default=None, help="YAML file with TrainConfig fields")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a field")
    args = ap.parse_args(argv)
    cfg = TrainConfig()
    if args.config:
        for k, v in (yaml.safe_load(args.config.read_text()) or {}).items():
            setattr(cfg, k, coerce(k, v))
    for item in args.set:
        k, _, v = item.partition("=")
        setattr(cfg, k, coerce(k, v))
    cfg.validate()
    return cfg
